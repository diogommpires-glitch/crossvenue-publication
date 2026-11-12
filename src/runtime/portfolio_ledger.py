"""Multi-asset paper-account ledger with explicit instrument conventions.

The ledger is accounting only: it never routes orders or retrieves provider data.
Callers supply normalized fills, marks, fees, financing/borrow costs, and FX
conversion marks. Amounts in public methods are expressed in the units named by
the registered :class:`InstrumentSpec`.

Accounting assumptions
----------------------
* Equities and crypto are cash securities: a fill exchanges quote cash for a
  marked position. Shorts are rejected unless explicitly enabled and the spec
  provides a positive borrow limit and an explicit borrow fee rate.
* Deliverable FX exchanges quote cash for base cash. The base balance is already
  included in portfolio equity, so the FX position is not added a second time.
* Rolling FX and futures do not exchange their notional principal. Fills realize
  P&L into settlement cash when reducing a position; open P&L is marked through
  equity. Futures can be variation-settled with :meth:`settle_future`.
* Fees and other costs must be supplied explicitly and are debited from the
  stated currency. Foreign cash and P&L require a fresh direct conversion mark
  to the reporting currency; this module does not invent triangulation rates.
* Reservations are holds for risk checks. Callers must resize/release them as
  orders partially fill, fill, or cancel. A reservation is not a cash posting.

The equity reconciliation is `equity = starting_equity + contributions +
total_pnl`. P&L also reports a translation residual because changes in the
reporting value of foreign cash need not be attributable to an open instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable, Dict, Mapping, Optional, Tuple

from ..market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from ..utils.types import Side


def _number(value: float, name: str, *, positive: bool = False, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be finite".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be finite".format(name))
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    if positive and result <= 0:
        raise ValueError("{} must be positive".format(name))
    if non_negative and result < 0:
        raise ValueError("{} must be non-negative".format(name))
    return result


def _currency(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("currency must be a non-empty string")
    return value.strip().upper()


def _enum_value(value) -> str:
    return str(getattr(value, "value", value)).lower()


def _instrument_fingerprint(spec: InstrumentSpec) -> Dict[str, object]:
    """Canonical accounting identity stored with every durable position."""
    return {
        "instrument_id": spec.instrument_id,
        "asset_class": spec.asset_class.value,
        "symbol": spec.symbol,
        "venue": spec.venue,
        "base_currency": spec.base_currency,
        "quote_currency": spec.quote_currency,
        "settlement_currency": spec.settlement_currency,
        "quantity_unit": spec.quantity_unit,
        "tick_size": spec.tick_size,
        "quantity_step": spec.quantity_step,
        "min_quantity": spec.min_quantity,
        "contract_multiplier": spec.contract_multiplier,
        "session_timezone": spec.session_timezone,
        "expiry": spec.expiry.isoformat() if spec.expiry else None,
        "fx_accounting_mode": (spec.fx_accounting_mode.value
                               if spec.fx_accounting_mode else None),
        "initial_margin_rate": spec.initial_margin_rate,
        "maintenance_margin_rate": spec.maintenance_margin_rate,
        "initial_margin_per_contract": spec.initial_margin_per_contract,
        "maintenance_margin_per_contract": spec.maintenance_margin_per_contract,
        "borrow_fee_rate": spec.borrow_fee_rate,
        "capabilities": sorted(spec.capabilities),
    }


def _event_time(value: float, name: str = "timestamp") -> float:
    return _number(value, name, non_negative=True)


@dataclass
class _Position:
    spec: InstrumentSpec
    allow_short: bool = False
    grandfathered_short: bool = False
    quantity: float = 0.0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0  # Quote currency, or settlement currency for derivatives.
    mark_price: Optional[float] = None
    mark_timestamp: Optional[float] = None
    settlement_timestamp: float | None = None
    sourced_settlement_price: float | None = None
    last_fill_timestamp: float | None = None
    fill_history_verified: bool = True


@dataclass(frozen=True)
class _Reservation:
    reservation_id: str
    kind: str
    currency: str
    amount: float
    instrument_id: Optional[str] = None


@dataclass(frozen=True)
class PositionSnapshot:
    instrument_id: str
    asset_class: str
    quantity: float
    quantity_unit: str
    average_entry_price: Optional[float]
    mark_price: Optional[float]
    mark_timestamp: Optional[float]
    pnl_currency: str
    realized_pnl: float
    unrealized_pnl: float
    market_value: Optional[float]
    initial_margin_required: float
    maintenance_margin_required: float
    mark_basis: str = "observed_quote"
    valuation_valid_until: Optional[float] = None
    last_fill_timestamp: float | None = None
    fill_history_verified: bool = True


@dataclass(frozen=True)
class PortfolioSnapshot:
    as_of: float
    reporting_currency: str
    cash_by_currency: Mapping[str, float]
    available_cash_by_currency: Mapping[str, float]
    reserved_cash_by_currency: Mapping[str, float]
    reserved_margin_by_currency: Mapping[str, float]
    fees_paid_by_currency: Mapping[str, float]
    costs_paid_by_currency: Mapping[str, float]
    income_by_currency: Mapping[str, float]
    positions: Mapping[str, PositionSnapshot]
    starting_equity: float
    contributions: float
    equity: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    translation_pnl: float
    fees_paid: float
    costs_paid: float
    income: float
    initial_margin_required: float
    maintenance_margin_required: float
    accrued_costs_by_currency: Mapping[str, float] = field(default_factory=dict)
    accrued_costs: float = 0.0


class PortfolioLedger:
    """Account balances, positions, marks, costs, and reservations for paper use.

    Args:
        initial_cash: Positive opening cash in ``reporting_currency``.
        reporting_currency: Currency used for consolidated equity/P&L.
        max_mark_age_seconds: Maximum permitted age for any required asset or
            conversion mark. Snapshots fail closed if a required mark is absent,
            from the future, or older than this threshold.

    Register each instrument before recording fills. The spec's margin fields
    are mandatory for futures (enforced by ``InstrumentSpec``) and rolling FX
    (enforced here). Equities/crypto can only be shorted when ``allow_short`` is
    true and the spec includes ``borrow_limit`` and ``borrow_fee_rate``.
    """

    def __init__(self, initial_cash: float, reporting_currency: str = "USD",
                 max_mark_age_seconds: float = 30.0, *, clock: Callable[[], float] = time.time):
        if not callable(clock):
            raise TypeError("ledger clock must be callable")
        self.clock = clock
        self.reporting_currency = _currency(reporting_currency)
        initial = _number(initial_cash, "initial_cash", positive=True)
        self.max_mark_age_seconds = _number(max_mark_age_seconds, "max_mark_age_seconds", positive=True)
        self._cash: Dict[str, float] = {self.reporting_currency: initial}
        self._opening_balances: Optional[Dict[str, float]] = dict(self._cash)
        self._starting_equity = initial
        self._contributions = 0.0
        self._positions: Dict[str, _Position] = {}
        self._fx_marks: Dict[str, Tuple[float, float, float]] = {}
        self._reservations: Dict[str, _Reservation] = {}
        self._fees: Dict[str, float] = {}
        self._costs: Dict[str, float] = {}
        self._income: Dict[str, float] = {}
        self._fill_signatures: Dict[str, tuple] = {}
        self._cost_ids: Dict[str, tuple] = {}
        self._cash_adjustment_ids: Dict[str, tuple] = {}
        self._corporate_action_ids: Dict[str, tuple] = {}
        self._signature_lookup = None
        self._borrow_state: Dict[str, Dict] = {}
        self._valuation_marks: Dict[str, Dict] = {}
        self._lock = threading.RLock()

    def set_signature_lookup(self, lookup: Callable[[str, str], Optional[tuple]]) -> None:
        """Attach durable idempotency lookup before bounding hot signature maps."""
        if not callable(lookup):
            raise TypeError("signature lookup must be callable")
        self._signature_lookup = lookup

    def _signature(self, namespace: str, key: str):
        prior = getattr(self, "_" + namespace).get(key)
        if prior is None and self._signature_lookup is not None:
            prior = self._signature_lookup(namespace, key)
        return None if prior is None else tuple(prior)

    def _remember(self, namespace: str, key: str, signature: tuple) -> None:
        values = getattr(self, "_" + namespace)
        values[key] = signature
        if self._signature_lookup is not None:
            while len(values) > 4096:
                values.pop(next(iter(values)))

    def clone(self) -> "PortfolioLedger":
        """Return an isolated candidate ledger, including working cash holds."""
        with self._lock:
            candidate = PortfolioLedger(self._starting_equity, self.reporting_currency,
                                        self.max_mark_age_seconds, clock=self.clock)
            for position in self._positions.values():
                candidate.register_instrument(position.spec, allow_short=position.allow_short)
            candidate.restore_state(self.export_state())
            candidate._reservations = dict(self._reservations)
            candidate._signature_lookup = self._signature_lookup
            return candidate

    def set_valuation_mark(self, instrument_id: str, price: float, timestamp: float,
                           *, valid_until: float, source: str, basis: str,
                           stress_fraction: float,
                           processing_timestamp: Optional[float] = None) -> None:
        """Supply a sourced closure valuation without making a quote tradable."""
        stamp = _event_time(timestamp)
        expiry = _event_time(valid_until)
        stress = _number(stress_fraction, "closed-market stress", non_negative=True)
        if expiry <= stamp or stress > 1 or not isinstance(source, str) or not source.strip():
            raise ValueError("valuation requires source, later validity end and stress in [0,1]")
        if basis not in {"official_close", "official_settlement"}:
            raise ValueError("closure valuation must be an official close or settlement")
        with self._lock:
            state = self._get_position(instrument_id)
            value = _number(price, "valuation price")
            if state.spec.asset_class != AssetClass.FUTURE and value <= 0:
                raise ValueError("non-futures valuation price must be positive")
            # The close's source time is distinct from the local posting time.
            # A normal late official-close import must not rewind an accrual
            # cursor already advanced by a periodic account timer.
            processed = (stamp if processing_timestamp is None else
                         _event_time(processing_timestamp, "valuation processing time"))
            if processed < stamp:
                raise ValueError("valuation processing time precedes source time")
            prior = self._valuation_marks.get(instrument_id)
            if prior is not None and stamp < prior["timestamp"]:
                return
            cursor = self._borrow_state.get(instrument_id, {}).get("last_ts", stamp)
            self._accrue_position_borrow(state, max(processed, cursor, state.mark_timestamp or stamp))
            if state.mark_timestamp is None or stamp >= state.mark_timestamp:
                state.mark_price, state.mark_timestamp = value, stamp
            self._valuation_marks[instrument_id] = {
                "price": float(price), "timestamp": stamp, "valid_until": expiry,
                "source": source.strip(), "basis": basis, "stress_fraction": stress,
                "processed_ts": processed}

    def update_instrument_policy(self, spec: InstrumentSpec, timestamp: float) -> None:
        """Apply prospective sourced terms without changing accounting identity."""
        with self._lock:
            state = self._get_position(spec.instrument_id)
            for name in ("asset_class", "base_currency", "quote_currency", "settlement_currency",
                         "contract_multiplier", "expiry", "fx_accounting_mode"):
                if getattr(state.spec, name) != getattr(spec, name):
                    raise ValueError("accounting identity changes require a new instrument")
            self._accrue_position_borrow(state, _event_time(timestamp))
            state.spec = spec

    def closed_market_stress(self, timestamp: float) -> float:
        with self._lock:
            result = 0.0
            for key, mark in self._valuation_marks.items():
                state = self._positions[key]
                if (state.quantity and self._matching_valuation_mark(state) is not None
                        and timestamp - mark["timestamp"] > self.max_mark_age_seconds
                        and timestamp <= mark["valid_until"]):
                    currency = (state.spec.settlement_currency if state.spec.asset_class == AssetClass.FUTURE
                                else state.spec.quote_currency)
                    result += (abs(state.quantity * mark["price"] * state.spec.contract_multiplier)
                               * mark["stress_fraction"] * self._fx_rate(currency, timestamp))
            return result

    def accrue_borrow(self, timestamp: float) -> Dict[str, float]:
        """Accrue time-weighted estimated borrow, posting UTC midnight charges."""
        stamp = _event_time(timestamp)
        with self._lock:
            for state in self._positions.values():
                self._accrue_position_borrow(state, stamp)
            return self._accrued_borrow_by_currency(stamp)

    def _borrow_currency(self, state: _Position) -> str:
        spec = state.spec
        return (spec.base_currency if spec.asset_class == AssetClass.FX
                and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE else spec.quote_currency)

    def _borrow_per_second(self, state: _Position) -> float:
        spec = state.spec
        if state.quantity >= 0 or spec.borrow_fee_rate is None or not spec.borrow_rate_source:
            return 0.0
        if spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE:
            basis = abs(state.quantity) * spec.contract_multiplier
        elif spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}:
            basis = abs(state.quantity) * abs(state.mark_price or 0) * spec.contract_multiplier
        else:
            return 0.0
        return basis * spec.borrow_fee_rate / (365.0 * 86400.0)

    def _accrue_position_borrow(self, state: _Position, timestamp: float) -> None:
        spec = state.spec
        if not spec.borrow_rate_source:
            return
        entry = self._borrow_state.setdefault(spec.instrument_id, {
            "last_ts": timestamp, "unpaid": 0.0, "source": spec.borrow_rate_source})
        previous = entry["last_ts"]
        if timestamp < previous:
            raise ValueError("borrow accrual time cannot move backwards")
        cursor = previous
        while cursor < timestamp:
            boundary = (math.floor(cursor / 86400.0) + 1) * 86400.0
            end = min(timestamp, boundary)
            start = max(cursor, spec.borrow_rate_effective_from or cursor)
            eligible_end = min(end, spec.borrow_rate_effective_to or end)
            entry["unpaid"] += max(0.0, eligible_end - start) * self._borrow_per_second(state)
            cursor = end
            if end == boundary and entry["unpaid"]:
                amount = entry["unpaid"]
                currency = self._borrow_currency(state)
                self._apply_cash_delta(currency, -amount)
                self._costs[currency] = self._costs.get(currency, 0.0) + amount
                entry["unpaid"] = 0.0
        entry["last_ts"] = timestamp
        entry["source"] = spec.borrow_rate_source

    def _accrued_borrow_by_currency(self, timestamp: float) -> Dict[str, float]:
        totals = {}
        for key, entry in self._borrow_state.items():
            state = self._positions[key]
            spec = state.spec
            start = max(entry["last_ts"], spec.borrow_rate_effective_from or entry["last_ts"])
            end = min(timestamp, spec.borrow_rate_effective_to or timestamp)
            value = entry["unpaid"] + max(0, end - start) * self._borrow_per_second(state)
            currency = self._borrow_currency(state)
            totals[currency] = totals.get(currency, 0.0) + value
        return totals

    @property
    def cash_by_currency(self) -> Dict[str, float]:
        """Return a defensive copy of unreserved and reserved cash balances."""
        with self._lock:
            return dict(self._cash)

    def register_instrument(self, spec: InstrumentSpec, *, allow_short: bool = False) -> None:
        """Register a validated instrument and its explicit short/margin policy."""
        if not isinstance(spec, InstrumentSpec):
            raise TypeError("spec must be an InstrumentSpec")
        if not isinstance(allow_short, bool):
            raise ValueError("allow_short must be a boolean")
        asset = spec.asset_class
        rolling_fx = (asset == AssetClass.FX
                      and spec.fx_accounting_mode == FXAccountingMode.ROLLING)
        if asset == AssetClass.FUTURE or rolling_fx:
            self._validate_margin(spec, rolling_fx=rolling_fx)
        if allow_short and asset != AssetClass.FUTURE and not rolling_fx:
            if spec.borrow_limit is None or spec.borrow_limit <= 0:
                raise ValueError("shorting {} requires a positive borrow_limit".format(spec.instrument_id))
            if spec.borrow_fee_rate is None:
                raise ValueError("shorting {} requires an explicit borrow_fee_rate (zero is allowed)".format(
                    spec.instrument_id))
        with self._lock:
            existing = self._positions.get(spec.instrument_id)
            if existing is not None:
                if existing.spec != spec or existing.allow_short != allow_short:
                    raise ValueError("instrument_id is already registered with different metadata")
                return
            self._positions[spec.instrument_id] = _Position(spec=spec, allow_short=allow_short)

    @staticmethod
    def _require_margin(spec: InstrumentSpec) -> Tuple[float, float]:
        initial = spec.initial_margin_rate
        maintenance = spec.maintenance_margin_rate
        if initial is None or initial <= 0:
            raise ValueError("{} requires an explicit positive initial_margin_rate".format(spec.instrument_id))
        if maintenance is None or maintenance <= 0:
            raise ValueError("{} requires an explicit positive maintenance_margin_rate".format(spec.instrument_id))
        if maintenance > initial:
            raise ValueError("maintenance margin cannot exceed initial margin")
        return float(initial), float(maintenance)

    @classmethod
    def _validate_margin(cls, spec: InstrumentSpec, *, rolling_fx: bool = False) -> None:
        """Require complete margin inputs; futures may use contract amounts."""
        if spec.asset_class == AssetClass.FUTURE:
            if (spec.initial_margin_per_contract is not None
                    and spec.maintenance_margin_per_contract is not None
                    and spec.initial_margin_per_contract > 0
                    and spec.maintenance_margin_per_contract > 0):
                return
            raise ValueError("{} requires explicit positive per-contract futures margins".format(
                spec.instrument_id))
        cls._require_margin(spec)

    @classmethod
    def _margin_amounts(cls, spec: InstrumentSpec, quantity: float, price: float
                        ) -> Tuple[float, float]:
        """Compute price-independent CME contract margin where supplied."""
        qty = abs(float(quantity))
        if spec.asset_class == AssetClass.FUTURE:
            cls._validate_margin(spec)
            return (qty * spec.initial_margin_per_contract,
                    qty * spec.maintenance_margin_per_contract)
        initial_rate, maintenance_rate = cls._require_margin(spec)
        notional = qty * abs(float(price)) * spec.contract_multiplier
        return notional * initial_rate, notional * maintenance_rate

    def fund_currency(self, currency: str, amount: float, timestamp: float,
                      fx_rate_to_reporting: Optional[float] = None) -> None:
        """Add opening/deposit cash and include its reporting value in capital.

        Non-reporting deposits require an explicit direct FX rate. The supplied
        rate is recorded as a fresh mark at ``timestamp``; no implicit par rate
        or cross-rate triangulation is used.
        """
        code = _currency(currency)
        value = _number(amount, "amount", positive=True)
        now = _event_time(timestamp)
        if code == self.reporting_currency:
            rate = 1.0
        elif fx_rate_to_reporting is None:
            raise ValueError("funding in {} requires fx_rate_to_reporting".format(code))
        else:
            rate = _number(fx_rate_to_reporting, "fx_rate_to_reporting", positive=True)
        with self._lock:
            if code != self.reporting_currency:
                self._fx_marks[code] = (rate, now, self.max_mark_age_seconds)
            self._cash[code] = self._cash.get(code, 0.0) + value
            contribution = value * rate
            self._contributions += contribution

    def set_fx_mark(self, currency: str, rate_to_reporting: float, timestamp: float,
                    max_age_seconds: Optional[float] = None) -> None:
        """Set a direct quote for one unit of ``currency`` in reporting units."""
        code = _currency(currency)
        if code == self.reporting_currency:
            raise ValueError("the reporting currency has an implicit rate of one")
        rate = _number(rate_to_reporting, "rate_to_reporting", positive=True)
        stamp = _event_time(timestamp)
        max_age = (self.max_mark_age_seconds if max_age_seconds is None else
                   _number(max_age_seconds, "max_age_seconds", positive=True))
        with self._lock:
            previous = self._fx_marks.get(code)
            if previous is not None and stamp < previous[1]:
                raise ValueError("FX marks must not move backward in time")
            self._fx_marks[code] = (rate, stamp, max_age)

    def set_initial_balances(self, balances: Mapping[str, float], timestamp: float) -> None:
        """Allocate opening equity across currencies without creating P&L."""
        if not isinstance(balances, Mapping) or not balances:
            raise ValueError("initial balances must be a non-empty currency mapping")
        stamp = _event_time(timestamp)
        normalized = {_currency(code): _number(amount, "initial balance", non_negative=True)
                      for code, amount in balances.items()}
        normalized = {code: amount for code, amount in normalized.items() if amount > 0}
        if not normalized:
            raise ValueError("at least one initial balance must be positive")
        with self._lock:
            if any(state.quantity != 0 or state.realized_pnl != 0 for state in self._positions.values()):
                raise ValueError("opening balances cannot change after trading begins")
            if (self._fill_signatures or self._contributions or self._fees or self._costs
                    or self._income or self._cash_adjustment_ids or self._corporate_action_ids):
                raise ValueError("opening balances cannot change after account activity")
            rates = {code: self._fx_rate(code, stamp) for code in normalized}
            equity = sum(amount * rates[code] for code, amount in normalized.items())
            if not math.isclose(equity, self._starting_equity, rel_tol=1e-8, abs_tol=0.01):
                raise ValueError("opening currency balances must reconcile to starting equity")
            self._cash = normalized
            self._opening_balances = dict(normalized)

    def mark_instrument(self, instrument_id: str, price: float, timestamp: float,
                        *, processing_timestamp: Optional[float] = None) -> None:
        """Update a mark. Non-futures require positive prices; futures may be negative."""
        stamp = _event_time(timestamp)
        with self._lock:
            state = self._get_position(instrument_id)
            value = _number(price, "price")
            if state.spec.asset_class != AssetClass.FUTURE and value <= 0:
                raise ValueError("price must be positive for non-futures")
            # A valuation mark may be the midpoint of a valid bid and ask,
            # which can fall halfway between tradable ticks. Order and fill
            # prices remain tick-validated at their execution boundaries.
            if state.mark_timestamp is not None and stamp < state.mark_timestamp:
                raise ValueError("instrument marks must not move backward in time")
            accrual_time = stamp
            if processing_timestamp is not None:
                accrual_time = max(stamp, _event_time(processing_timestamp, "mark processing time"),
                    self._borrow_state.get(instrument_id, {}).get("last_ts", stamp))
            self._accrue_position_borrow(state, accrual_time)
            state.mark_price = value
            state.mark_timestamp = stamp

    def record_fill(self, fill_id: str, instrument_id: str, side, quantity: float,
                    price: float, *, fee: float, fee_currency: str,
                    timestamp: float, fill_instrument_id: Optional[str] = None,
                    allow_risk_reduction: bool = False,
                    processing_timestamp: Optional[float] = None) -> bool:
        """Apply one fill once; return ``False`` for an identical duplicate ID.

        ``fee`` and ``fee_currency`` are required even when the fee is zero.
        If the upstream Fill model carries an instrument ID, pass it as
        ``fill_instrument_id`` so mismatched fills fail closed.
        """
        if not isinstance(fill_id, str) or not fill_id.strip():
            raise ValueError("fill_id must be a non-empty string")
        if fill_instrument_id is not None and fill_instrument_id != instrument_id:
            raise ValueError("fill instrument_id does not match ledger instrument_id")
        normalized_side = self._side(side)
        qty = _number(quantity, "quantity", positive=True)
        px = _number(price, "price")
        fee_value = _number(fee, "fee", non_negative=True)
        fee_ccy = _currency(fee_currency)
        stamp = _event_time(timestamp)

        with self._lock:
            state = self._get_position(instrument_id)
            spec = state.spec
            if spec.asset_class != AssetClass.FUTURE and px <= 0:
                raise ValueError("fill price must be positive for non-futures")
            if not spec.is_valid_price(px):
                raise ValueError("fill price is not aligned to the instrument tick")
            if not spec.is_valid_quantity(qty):
                raise ValueError("fill quantity is not aligned to the instrument lot")
            if fee_ccy not in {spec.quote_currency, spec.settlement_currency,
                               spec.base_currency, self.reporting_currency}:
                raise ValueError("fee_currency must be an instrument or account currency")

            signature = (instrument_id, normalized_side, qty, px, fee_value, fee_ccy, stamp)
            prior = self._signature("fill_signatures", fill_id)
            if prior is not None:
                if prior != signature:
                    raise ValueError("fill_id was reused with different fill data")
                return False
            if state.settlement_timestamp is not None and stamp < state.settlement_timestamp:
                raise ValueError("fill timestamp precedes a committed settlement; reconciliation is required")

            accrual_time = stamp
            if processing_timestamp is not None:
                accrual_time = max(stamp, _event_time(processing_timestamp, "fill processing time"),
                    self._borrow_state.get(instrument_id, {}).get("last_ts", stamp))
            self._accrue_position_borrow(state, accrual_time)
            sign = 1.0 if normalized_side == "buy" else -1.0
            delta = sign * qty
            new_quantity = state.quantity + delta
            self._validate_short(state, new_quantity,
                                 allow_risk_reduction=allow_risk_reduction)
            asset = spec.asset_class
            mode = spec.fx_accounting_mode if asset == AssetClass.FX else None
            notional = qty * px * spec.contract_multiplier

            if asset in {AssetClass.EQUITY, AssetClass.CRYPTO}:
                cash_delta = -sign * notional
                self._apply_cash_delta(spec.quote_currency, cash_delta)
            elif asset == AssetClass.FX and mode == FXAccountingMode.DELIVERABLE:
                self._apply_cash_delta(spec.base_currency, delta * spec.contract_multiplier)
                self._apply_cash_delta(spec.quote_currency, -delta * px * spec.contract_multiplier)
            elif asset == AssetClass.FX and mode == FXAccountingMode.ROLLING:
                self._apply_derivative_realized(state, delta, px)
            elif asset == AssetClass.FUTURE:
                self._apply_derivative_realized(state, delta, px)
            else:
                raise ValueError("unsupported instrument accounting mode")

            self._apply_fee(fee_ccy, fee_value)
            self._update_position(state, delta, px)
            if state.mark_timestamp is None or stamp >= state.mark_timestamp:
                state.mark_price = px
                state.mark_timestamp = stamp
            state.last_fill_timestamp = (stamp if state.last_fill_timestamp is None
                                         else max(stamp, state.last_fill_timestamp))
            self._remember("fill_signatures", fill_id, signature)
            return True

    def record_cost(self, cost_id: str, instrument_id: str, amount: float,
                    currency: str, *, kind: str, timestamp: float) -> bool:
        """Debit an explicitly known borrow, rollover, or other account cost."""
        if not isinstance(cost_id, str) or not cost_id.strip():
            raise ValueError("cost_id must be a non-empty string")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind must be a non-empty cost description")
        value = _number(amount, "amount", non_negative=True)
        code = _currency(currency)
        stamp = _event_time(timestamp)
        with self._lock:
            self._get_position(instrument_id)
            signature = (instrument_id, value, code, kind.strip().lower(), stamp)
            previous = self._signature("cost_ids", cost_id)
            if previous is not None:
                if previous != signature:
                    raise ValueError("cost_id was reused with different cost data")
                return False
            self._apply_cash_delta(code, -value)
            self._costs[code] = self._costs.get(code, 0.0) + value
            self._remember("cost_ids", cost_id, signature)
            return True

    def accrue_borrow_cost(self, cost_id: str, instrument_id: str,
                           elapsed_years: float, timestamp: float) -> float:
        """Accrue configured annual borrow cost for an open short position.

        For equity/crypto shorts, cost is short notional × annual rate × year
        fraction in quote currency. For deliverable FX, it is borrowed base
        units × annual rate × year fraction in base currency. A missing rate or
        borrow limit is an error, never interpreted as free borrowing.
        """
        years = _number(elapsed_years, "elapsed_years", positive=True)
        stamp = _event_time(timestamp)
        with self._lock:
            state = self._get_position(instrument_id)
            spec = state.spec
            if state.quantity >= 0:
                raise ValueError("borrow cost requires a short position")
            if spec.borrow_limit is None or spec.borrow_fee_rate is None:
                raise ValueError("short position has no explicit borrow configuration")
            mark = self._required_position_mark(state, stamp)
            if spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE:
                amount = (abs(state.quantity) * spec.contract_multiplier
                          * spec.borrow_fee_rate * years)
                code = spec.base_currency
            else:
                amount = abs(state.quantity) * abs(mark) * spec.contract_multiplier * spec.borrow_fee_rate * years
                code = spec.quote_currency
        applied = self.record_cost(cost_id, instrument_id, amount, code,
                                   kind="borrow", timestamp=stamp)
        return amount if applied else 0.0

    def apply_equity_split(self, action_id: str, instrument_id: str,
                           ratio: float, timestamp: float) -> bool:
        """Apply an explicit split ratio while preserving marked equity.

        ``ratio`` is new shares per old share (2.0 for a 2-for-1 split).
        The caller must use the position effective at the corporate-action
        time and supply a stable action ID so replay/retries stay idempotent.
        """
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id must be a non-empty string")
        factor = _number(ratio, "split ratio", positive=True)
        stamp = _event_time(timestamp)
        with self._lock:
            state = self._get_position(instrument_id)
            spec = state.spec
            if spec.asset_class != AssetClass.EQUITY:
                raise ValueError("stock splits can only be applied to equity instruments")
            signature = (instrument_id, factor, "split")
            prior = self._signature("corporate_action_ids", action_id)
            if prior is not None:
                legacy_match = (len(prior) == 4 and prior[0] == instrument_id
                                and prior[1] == factor and prior[3] == "split")
                if prior != signature and not legacy_match:
                    raise ValueError("corporate action ID was reused with different data")
                return False
            if state.mark_timestamp is not None and stamp < state.mark_timestamp:
                raise ValueError("split timestamp precedes the latest instrument mark")
            new_quantity = state.quantity * factor
            if abs(new_quantity) > 1e-12 and not spec.is_valid_quantity(abs(new_quantity)):
                raise ValueError("split result does not align to the instrument lot size")
            state.quantity = 0.0 if abs(new_quantity) <= 1e-12 else new_quantity
            state.average_entry_price /= factor
            if state.mark_price is not None:
                state.mark_price /= factor
            state.mark_timestamp = stamp if state.mark_price is not None else state.mark_timestamp
            self._remember("corporate_action_ids", action_id, signature)
            return True

    def apply_equity_dividend(self, action_id: str, instrument_id: str,
                              amount_per_share: float, eligible_quantity: float,
                              timestamp: float) -> Optional[float]:
        """Credit/debit a declared per-share dividend for record-date holdings.

        Positive declared dividends credit longs and debit shorts. The
        Caller supplies the corporate-action amount and signed eligible
        quantity held on the record date. The ledger does not infer eligibility
        or source corporate actions, so late postings remain correct after the
        current position has changed.
        """
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id must be a non-empty string")
        amount = _number(amount_per_share, "dividend per share", non_negative=True)
        eligible = _number(eligible_quantity, "eligible quantity")
        stamp = _event_time(timestamp)
        with self._lock:
            state = self._get_position(instrument_id)
            spec = state.spec
            if spec.asset_class != AssetClass.EQUITY:
                raise ValueError("cash dividends can only be applied to equity instruments")
            if abs(eligible) > 1e-12 and not spec.is_valid_quantity(abs(eligible)):
                raise ValueError("eligible dividend quantity is not aligned to the equity lot size")
            if state.mark_timestamp is not None and stamp < state.mark_timestamp:
                raise ValueError("dividend timestamp precedes the latest instrument mark")
            signed_cash = eligible * spec.contract_multiplier * amount
            signature = (instrument_id, amount, eligible, "dividend", signed_cash)
            prior = self._signature("corporate_action_ids", action_id)
            if prior is not None:
                legacy_match = (len(prior) == 6 and prior[0] == instrument_id
                                and prior[1] == amount and prior[2] == eligible
                                and prior[4] == "dividend" and prior[5] == signed_cash)
                if prior != signature and not legacy_match:
                    raise ValueError("corporate action ID was reused with different data")
                return None
            self._apply_cash_delta(spec.quote_currency, signed_cash)
            if signed_cash >= 0:
                if signed_cash:
                    self._income[spec.quote_currency] = (
                        self._income.get(spec.quote_currency, 0.0) + signed_cash)
            else:
                self._costs[spec.quote_currency] = (
                    self._costs.get(spec.quote_currency, 0.0) - signed_cash)
            self._remember("corporate_action_ids", action_id, signature)
            return signed_cash

    def record_financing(self, adjustment_id: str, instrument_id: str,
                         amount: float, currency: str, *, timestamp: float,
                         kind: str = "fx_financing") -> bool:
        """Post explicit signed financing for rolling FX exposure.

        Positive values are credits; negative values are debits. Rates and
        accrual conventions vary by provider, so the caller must supply an
        explicit amount rather than letting the ledger invent one.
        """
        if not isinstance(adjustment_id, str) or not adjustment_id.strip():
            raise ValueError("adjustment_id must be a non-empty string")
        value = _number(amount, "financing amount")
        code = _currency(currency)
        stamp = _event_time(timestamp)
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("financing kind must be a non-empty string")
        with self._lock:
            state = self._get_position(instrument_id)
            spec = state.spec
            if spec.asset_class != AssetClass.FX or spec.fx_accounting_mode != FXAccountingMode.ROLLING:
                raise ValueError("financing postings require rolling FX")
            if code != spec.settlement_currency:
                raise ValueError("financing currency must match the FX settlement currency")
            signature = (instrument_id, value, code, kind.strip().lower())
            prior = self._signature("cash_adjustment_ids", adjustment_id)
            if prior is not None:
                legacy_match = (len(prior) == 5 and prior[0] == instrument_id
                                and prior[1] == value and prior[2] == code
                                and prior[3] == kind.strip().lower())
                if prior != signature and not legacy_match:
                    raise ValueError("financing ID was reused with different data")
                return False
            self._apply_cash_delta(code, value)
            if value >= 0:
                if value:
                    self._income[code] = self._income.get(code, 0.0) + value
            else:
                self._costs[code] = self._costs.get(code, 0.0) - value
            self._remember("cash_adjustment_ids", adjustment_id, signature)
            return True

    def settle_future(self, instrument_id: str, settlement_price: float, timestamp: float) -> float:
        """Post futures variation margin and reset open basis to settlement.

        Returns the settlement cash flow in the instrument's settlement
        currency. This method is invalid for non-futures instruments.
        """
        stamp = _event_time(timestamp)
        with self._lock:
            state = self._get_position(instrument_id)
            if state.spec.asset_class != AssetClass.FUTURE:
                raise ValueError("variation settlement is only valid for futures")
            price = _number(settlement_price, "settlement_price")
            if not state.spec.is_valid_price(price):
                raise ValueError("settlement price is not aligned to the instrument tick")
            if state.mark_timestamp is not None and stamp < state.mark_timestamp:
                raise ValueError("settlement timestamp precedes the latest mark")
            return self._post_future_settlement(state, price, stamp)

    def record_sourced_settlement(self, action_id: str, instrument_id: str,
                                  settlement_price: float, effective_ts: float) -> float:
        """Post sourced variation once without crossing unreconciled fill history."""
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("settlement requires a stable action ID")
        stamp = _event_time(effective_ts)
        price = _number(settlement_price, "settlement_price")
        with self._lock:
            state = self._get_position(instrument_id)
            if state.spec.asset_class != AssetClass.FUTURE:
                raise ValueError("variation settlement is only valid for futures")
            if not state.spec.is_valid_price(price):
                raise ValueError("settlement price is not aligned to the instrument tick")
            signature = ("future_settlement", instrument_id, price, stamp)
            prior = self._signature("corporate_action_ids", action_id)
            if prior is not None:
                if prior != signature:
                    raise ValueError("settlement action ID was reused with different inputs")
                return 0.0
            if not state.fill_history_verified and stamp < _event_time(self.clock()):
                raise ValueError("backdated settlement requires verified fill history or reconciliation")
            variation = self._post_future_settlement(state, price, stamp)
            state.sourced_settlement_price = price
            self._remember("corporate_action_ids", action_id, signature)
            return variation

    def _post_future_settlement(self, state: _Position, price: float, stamp: float) -> float:
        if state.settlement_timestamp is not None and stamp < state.settlement_timestamp:
            raise ValueError("settlement timestamp precedes the latest economic settlement")
        if state.last_fill_timestamp is not None and stamp < state.last_fill_timestamp:
            raise ValueError("settlement timestamp precedes the latest fill; reconciliation is required")
        variation = (state.quantity * state.spec.contract_multiplier
                     * (price - state.average_entry_price))
        if state.quantity != 0:
            self._apply_cash_delta(state.spec.settlement_currency, variation)
            state.realized_pnl += variation
            state.average_entry_price = price
        state.settlement_timestamp = stamp
        state.sourced_settlement_price = None
        # A basis reset cannot refresh or replace a newer observed valuation.
        if state.mark_timestamp is None or stamp >= state.mark_timestamp:
            state.mark_price, state.mark_timestamp = price, stamp
        return variation

    def record_sourced_adjustment(self, adjustment_id: str, instrument_id: str,
                                  amount: float, currency: str, *, timestamp: float,
                                  kind: str) -> bool:
        """Post a signed sourced cost correction; positive amounts are credits."""
        if not isinstance(adjustment_id, str) or not adjustment_id.strip():
            raise ValueError("adjustment requires a stable ID")
        value = _number(amount, "adjustment amount")
        code = _currency(currency)
        _event_time(timestamp)
        with self._lock:
            state = self._get_position(instrument_id)
            if code not in {state.spec.quote_currency, state.spec.settlement_currency,
                            state.spec.base_currency, self.reporting_currency}:
                raise ValueError("adjustment currency does not belong to the instrument/account")
            signature = (instrument_id, value, code, kind)
            prior = self._signature("cash_adjustment_ids", adjustment_id)
            if prior is not None:
                if prior != signature:
                    raise ValueError("adjustment ID was reused with different inputs")
                return False
            self._apply_cash_delta(code, value)
            if value >= 0:
                self._income[code] = self._income.get(code, 0.0) + value
            else:
                self._costs[code] = self._costs.get(code, 0.0) - value
            self._remember("cash_adjustment_ids", adjustment_id, signature)
            return True

    def reserve_cash(self, reservation_id: str, currency: str, amount: float) -> None:
        """Reserve spendable cash in a currency, failing on over-reservation."""
        self._reserve(reservation_id, "cash", _currency(currency), amount, None)

    def reserve_margin(self, reservation_id: str, instrument_id: str,
                       quantity: float, price: float) -> float:
        """Reserve initial margin for a pending rolling-FX or futures order."""
        qty = _number(quantity, "quantity", positive=True)
        px = _number(price, "price")
        with self._lock:
            state = self._get_position(instrument_id)
            if state.spec.asset_class != AssetClass.FUTURE and not (
                    state.spec.asset_class == AssetClass.FX
                    and state.spec.fx_accounting_mode == FXAccountingMode.ROLLING):
                raise ValueError("margin reservations require futures or rolling FX")
            if state.spec.asset_class != AssetClass.FUTURE and px <= 0:
                raise ValueError("rolling FX price must be positive")
            amount, _ = self._margin_amounts(state.spec, qty, px)
            currency = state.spec.settlement_currency
            self._reserve(reservation_id, "margin", currency, amount, instrument_id)
            return amount

    def set_margin_reservation(self, reservation_id: str, instrument_id: str,
                               amount: float) -> float:
        """Atomically set an instrument's aggregate incremental margin hold.

        The target is additional margin beyond what is already committed to
        the open position. A zero target releases the hold, allowing an order
        that only reduces an open position to remain placeable.
        """
        value = _number(amount, "amount", non_negative=True)
        with self._lock:
            state = self._get_position(instrument_id)
            derivative = (state.spec.asset_class == AssetClass.FUTURE or
                          (state.spec.asset_class == AssetClass.FX and
                           state.spec.fx_accounting_mode == FXAccountingMode.ROLLING))
            if not derivative:
                raise ValueError("incremental margin reservations require futures or rolling FX")
            currency = state.spec.settlement_currency
            existing = self._reservations.get(reservation_id)
            if existing is not None and (existing.kind != "margin"
                    or existing.currency != currency or existing.instrument_id != instrument_id):
                raise ValueError("reservation_id already belongs to a different margin hold")
            if value <= 1e-12:
                self._reservations.pop(reservation_id, None)
                return 0.0
            available = self._available_cash(currency, exclude_reservation=reservation_id)
            if value > available + 1e-9:
                raise ValueError("reservation exceeds available cash or margin capacity")
            self._reservations[reservation_id] = _Reservation(
                reservation_id, "margin", currency, value, instrument_id)
            return value

    def resize_reservation(self, reservation_id: str, amount: float) -> None:
        """Resize an existing hold after a partial fill; zero releases it."""
        value = _number(amount, "amount", non_negative=True)
        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                raise KeyError("unknown reservation_id")
            if value == 0:
                del self._reservations[reservation_id]
                return
            without = self._available_cash(reservation.currency, exclude_reservation=reservation_id)
            if value > without + 1e-9:
                raise ValueError("reservation exceeds available cash or margin capacity")
            self._reservations[reservation_id] = _Reservation(
                reservation_id, reservation.kind, reservation.currency, value,
                reservation.instrument_id)

    def release_reservation(self, reservation_id: str) -> bool:
        """Release a pending cash or margin hold; return whether it existed."""
        with self._lock:
            return self._reservations.pop(reservation_id, None) is not None

    def get_position(self, instrument_id: str, as_of: Optional[float] = None) -> PositionSnapshot:
        """Return one instrument position using its latest fresh mark."""
        with self._lock:
            state = self._get_position(instrument_id)
            stamp = float(self.clock()) if as_of is None else _event_time(as_of, "as_of")
            return self._position_snapshot(state, stamp)

    def snapshot(self, as_of: Optional[float] = None) -> PortfolioSnapshot:
        """Build an account snapshot or fail closed on any missing/stale mark."""
        stamp = float(self.clock()) if as_of is None else _event_time(as_of, "as_of")
        with self._lock:
            # Keep closed instruments in the snapshot so realized P&L remains
            # reportable after the final unit is sold/settled.
            positions = {key: self._position_snapshot(value, stamp)
                         for key, value in self._positions.items()}
            needed_currencies = {ccy for ccy, amount in self._cash.items() if abs(amount) > 1e-12}
            for position in positions.values():
                if abs(position.quantity) > 1e-12 or abs(position.realized_pnl) > 1e-12:
                    needed_currencies.add(position.pnl_currency)
            needed_currencies.add(self.reporting_currency)
            accrued_by_currency = self._accrued_borrow_by_currency(stamp)
            needed_currencies.update(ccy for ccy, amount in accrued_by_currency.items() if amount)
            rates = {ccy: self._fx_rate(ccy, stamp) for ccy in needed_currencies}

            equity = sum(amount * rates[ccy] for ccy, amount in self._cash.items()
                          if abs(amount) > 1e-12)
            realized = 0.0
            unrealized = 0.0
            initial_margin = 0.0
            maintenance_margin = 0.0
            for instrument_id, pos in positions.items():
                state = self._positions[instrument_id]
                if abs(pos.quantity) <= 1e-12 and abs(pos.realized_pnl) <= 1e-12:
                    continue
                if abs(pos.quantity) > 1e-12 or abs(pos.realized_pnl) > 1e-12:
                    rate = rates[pos.pnl_currency]
                    realized += pos.realized_pnl * rate
                    unrealized += pos.unrealized_pnl * rate
                asset = state.spec.asset_class
                if asset in {AssetClass.EQUITY, AssetClass.CRYPTO}:
                    equity += float(pos.market_value) * self._fx_rate(state.spec.quote_currency, stamp)
                elif asset == AssetClass.FX and state.spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE:
                    # Delivered base and quote balances are already in cash.
                    pass
                else:
                    equity += pos.unrealized_pnl * rate
                if pos.initial_margin_required:
                    initial_margin += pos.initial_margin_required * self._fx_rate(
                        state.spec.settlement_currency, stamp)
                if pos.maintenance_margin_required:
                    maintenance_margin += pos.maintenance_margin_required * self._fx_rate(
                        state.spec.settlement_currency, stamp)

            fees_by_currency = dict(self._fees)
            costs_by_currency = dict(self._costs)
            income_by_currency = dict(self._income)
            fee_total = sum(value * self._fx_rate(ccy, stamp) for ccy, value in fees_by_currency.items())
            cost_total = sum(value * self._fx_rate(ccy, stamp) for ccy, value in costs_by_currency.items())
            income_total = sum(value * self._fx_rate(ccy, stamp)
                               for ccy, value in income_by_currency.items())
            accrued_total = sum(value * self._fx_rate(ccy, stamp)
                                for ccy, value in accrued_by_currency.items() if value)
            equity -= accrued_total
            total_pnl = equity - self._starting_equity - self._contributions
            translation = (total_pnl - realized - unrealized - income_total
                           + fee_total + cost_total + accrued_total)
            cash_reservations = self._reservation_totals("cash")
            margin_reservations = self._reservation_totals("margin")
            currencies = set(self._cash) | set(cash_reservations) | set(margin_reservations)
            available = {ccy: self._available_cash(ccy, as_of=stamp) for ccy in currencies}
            return PortfolioSnapshot(
                as_of=stamp, reporting_currency=self.reporting_currency,
                cash_by_currency=dict(self._cash), available_cash_by_currency=available,
                reserved_cash_by_currency=cash_reservations,
                reserved_margin_by_currency=margin_reservations,
                fees_paid_by_currency=fees_by_currency, costs_paid_by_currency=costs_by_currency,
                income_by_currency=income_by_currency,
                positions=positions, starting_equity=self._starting_equity,
                contributions=self._contributions, equity=equity, total_pnl=total_pnl,
                realized_pnl=realized, unrealized_pnl=unrealized,
                translation_pnl=translation, fees_paid=fee_total, costs_paid=cost_total,
                income=income_total,
                initial_margin_required=initial_margin,
                maintenance_margin_required=maintenance_margin,
                accrued_costs_by_currency=accrued_by_currency,
                accrued_costs=accrued_total,
            )

    def export_state(self) -> Dict:
        """Return a JSON-safe account checkpoint without live reservations.

        Working orders are intentionally not restored after a process restart;
        the engine cancels them and recreates quotes only from fresh feeds.
        """
        with self._lock:
            return {
                "schema_version": 3,
                "reporting_currency": self.reporting_currency,
                "starting_equity": self._starting_equity,
                "contributions": self._contributions,
                "cash": dict(self._cash),
                "opening_balances": (None if self._opening_balances is None
                                     else dict(self._opening_balances)),
                "fx_marks": {key: list(value) for key, value in self._fx_marks.items()},
                "fees": dict(self._fees),
                "costs": dict(self._costs),
                "income": dict(self._income),
                "borrow_state": {key: dict(value) for key, value in self._borrow_state.items()},
                "valuation_marks": {key: dict(value) for key, value in self._valuation_marks.items()},
                "positions": {
                    key: {
                        "quantity": state.quantity,
                        "average_entry_price": state.average_entry_price,
                        "realized_pnl": state.realized_pnl,
                        "mark_price": state.mark_price,
                        "mark_timestamp": state.mark_timestamp,
                        "settlement_timestamp": state.settlement_timestamp,
                        "sourced_settlement_price": state.sourced_settlement_price,
                        "last_fill_timestamp": state.last_fill_timestamp,
                        "fill_history_verified": state.fill_history_verified,
                        "allow_short": state.allow_short,
                        "grandfathered_short": state.grandfathered_short,
                    }
                    for key, state in self._positions.items()
                },
                "instrument_specs": {
                    key: _instrument_fingerprint(state.spec)
                    for key, state in self._positions.items()
                },
                "fill_signatures": {key: list(value) for key, value in self._fill_signatures.items()},
                "cost_ids": {key: list(value) for key, value in self._cost_ids.items()},
                "cash_adjustment_ids": {
                    key: list(value) for key, value in self._cash_adjustment_ids.items()},
                "corporate_action_ids": {
                    key: list(value) for key, value in self._corporate_action_ids.items()},
            }

    def restore_state(self, data: Dict, *, allow_policy_changes: bool = False) -> None:
        """Restore a compatible checkpoint after instruments are registered.

        It refuses changed accounting metadata and missing registered
        instruments rather than silently reinterpreting stored positions.
        """
        if not isinstance(data, dict) or data.get("schema_version") not in (2, 3):
            raise ValueError("unsupported portfolio checkpoint schema")
        if data.get("reporting_currency") != self.reporting_currency:
            raise ValueError("checkpoint reporting currency does not match configuration")
        starting_equity = _number(data.get("starting_equity"), "starting_equity", positive=True)
        if not math.isclose(starting_equity, self._starting_equity, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("checkpoint starting capital does not match configuration")
        raw_positions = data.get("positions")
        if not isinstance(raw_positions, dict):
            raise ValueError("checkpoint positions must be an object")
        saved_specs = data.get("instrument_specs")
        if not isinstance(saved_specs, dict) or set(saved_specs) != set(raw_positions):
            raise ValueError("checkpoint is missing instrument accounting fingerprints")
        missing = set(raw_positions) - set(self._positions)
        if missing:
            raise ValueError("checkpoint contains instruments absent from configuration: {}".format(
                sorted(missing)))

        def numeric_map(raw, name, *, non_negative=False):
            if not isinstance(raw, dict):
                raise ValueError("checkpoint {} must be an object".format(name))
            result = {}
            for key, value in raw.items():
                result[_currency(key)] = _number(value, "checkpoint {} value".format(name),
                                                 non_negative=non_negative)
            return result

        cash = numeric_map(data.get("cash"), "cash")
        opening_raw = data.get("opening_balances")
        opening_balances = (None if opening_raw is None else
                            numeric_map(opening_raw, "opening_balances", non_negative=True))
        fees = numeric_map(data.get("fees"), "fees", non_negative=True)
        costs = numeric_map(data.get("costs"), "costs", non_negative=True)
        income = numeric_map(data.get("income", {}), "income", non_negative=True)
        contributions = _number(data.get("contributions"), "contributions")
        fx_marks = {}
        for currency, mark in data.get("fx_marks", {}).items():
            if not isinstance(mark, (list, tuple)) or len(mark) not in (2, 3):
                raise ValueError("checkpoint FX marks must contain a rate, timestamp, and optional age")
            fx_marks[_currency(currency)] = (
                _number(mark[0], "checkpoint FX rate", positive=True),
                _event_time(mark[1], "checkpoint FX timestamp"),
                self.max_mark_age_seconds if len(mark) == 2 else
                _number(mark[2], "checkpoint FX max age", positive=True),
            )

        restored_positions = {}
        for instrument_id, raw in raw_positions.items():
            if not isinstance(raw, dict) or instrument_id not in self._positions:
                raise ValueError("checkpoint position is malformed or unregistered")
            current = self._positions[instrument_id]
            saved_spec = saved_specs.get(instrument_id)
            current_spec = _instrument_fingerprint(current.spec)
            mutable = {"initial_margin_rate", "maintenance_margin_rate", "initial_margin_per_contract",
                       "maintenance_margin_per_contract", "borrow_fee_rate"}
            compatible = (isinstance(saved_spec, dict)
                          and {key: value for key, value in saved_spec.items() if key not in mutable}
                          == {key: value for key, value in current_spec.items() if key not in mutable})
            if saved_spec != current_spec and not (allow_policy_changes and compatible):
                raise ValueError("checkpoint accounting metadata changed for {}".format(instrument_id))
            saved_allow_short = raw.get("allow_short")
            if not isinstance(saved_allow_short, bool):
                raise ValueError("checkpoint short policy is malformed for {}".format(instrument_id))
            saved_grandfathered_short = raw.get("grandfathered_short", False)
            if not isinstance(saved_grandfathered_short, bool):
                raise ValueError("checkpoint grandfathered-short flag is malformed for {}".format(
                    instrument_id))
            quantity = _number(raw.get("quantity"), "checkpoint position quantity")
            if abs(quantity) > 1e-12:
                if quantity < 0 and current.spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}:
                    if not (saved_allow_short or saved_grandfathered_short):
                        raise ValueError("checkpoint contains a short without a prior borrow policy")
                    # Preserve an existing short so it can be covered after a
                    # borrow policy is tightened. New risk is still blocked
                    # by the engine's current configuration.
                else:
                    self._validate_short(current, quantity)
            avg = _number(raw.get("average_entry_price"), "checkpoint average entry")
            realized = _number(raw.get("realized_pnl"), "checkpoint realized P&L")
            mark_price = raw.get("mark_price")
            mark_timestamp = raw.get("mark_timestamp")
            if mark_price is not None:
                mark_price = _number(mark_price, "checkpoint mark price")
                if current.spec.asset_class != AssetClass.FUTURE and mark_price <= 0:
                    raise ValueError("checkpoint non-futures mark must be positive")
                # Midpoint valuation marks can fall between tradable ticks.
            if mark_timestamp is not None:
                mark_timestamp = _event_time(mark_timestamp, "checkpoint mark timestamp")
            settlement_timestamp = raw.get("settlement_timestamp")
            if settlement_timestamp is not None:
                settlement_timestamp = _event_time(settlement_timestamp, "checkpoint settlement timestamp")
                if (current.spec.asset_class != AssetClass.FUTURE or mark_timestamp is None
                        or settlement_timestamp > mark_timestamp):
                    raise ValueError("checkpoint settlement requires a future and an equal or later mark")
            sourced_settlement_price = raw.get("sourced_settlement_price")
            if sourced_settlement_price is not None:
                sourced_settlement_price = _number(sourced_settlement_price, "checkpoint sourced settlement price")
                if settlement_timestamp is None or not current.spec.is_valid_price(sourced_settlement_price):
                    raise ValueError("checkpoint sourced settlement requires its timestamp and a tick-valid price")
            last_fill_timestamp = raw.get("last_fill_timestamp")
            if last_fill_timestamp is not None:
                last_fill_timestamp = _event_time(last_fill_timestamp, "checkpoint last fill timestamp")
                if mark_timestamp is None or last_fill_timestamp > mark_timestamp:
                    raise ValueError("checkpoint latest fill requires an equal or later mark")
            fill_history_verified = raw.get("fill_history_verified", False)
            if not isinstance(fill_history_verified, bool):
                raise ValueError("checkpoint fill-history verification must be boolean")
            if fill_history_verified and quantity != 0 and last_fill_timestamp is None:
                raise ValueError("verified open-position fill history requires a latest fill timestamp")
            if (quantity != 0 or realized != 0) and (mark_price is None or mark_timestamp is None):
                raise ValueError("open or realized position {} is missing its last mark".format(instrument_id))
            restored_grandfathered_short = (
                quantity < -1e-12
                and current.spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}
                and not current.allow_short)
            restored_positions[instrument_id] = (
                quantity, avg, realized, mark_price, mark_timestamp,
                restored_grandfathered_short, settlement_timestamp,
                sourced_settlement_price, last_fill_timestamp, fill_history_verified)

        fill_signatures = {}
        for fill_id, signature in data.get("fill_signatures", {}).items():
            if not isinstance(signature, list) or len(signature) != 7:
                raise ValueError("checkpoint fill signature is malformed")
            fill_signatures[str(fill_id)] = tuple(signature)
        cost_ids = {}
        for cost_id, signature in data.get("cost_ids", {}).items():
            if not isinstance(signature, list) or len(signature) != 5:
                raise ValueError("checkpoint cost signature is malformed")
            cost_ids[str(cost_id)] = tuple(signature)
        cash_adjustment_ids = {}
        for adjustment_id, signature in data.get("cash_adjustment_ids", {}).items():
            if not isinstance(signature, list) or len(signature) not in (4, 5):
                raise ValueError("checkpoint cash-adjustment signature is malformed")
            cash_adjustment_ids[str(adjustment_id)] = tuple(signature)
        corporate_action_ids = {}
        for action_id, signature in data.get("corporate_action_ids", {}).items():
            if not isinstance(signature, list) or len(signature) not in (3, 4, 5, 6):
                raise ValueError("checkpoint corporate-action signature is malformed")
            corporate_action_ids[str(action_id)] = tuple(signature)

        with self._lock:
            self._cash = cash
            self._opening_balances = opening_balances
            self._fees = fees
            self._costs = costs
            self._income = income
            self._contributions = contributions
            self._fx_marks = fx_marks
            self._fill_signatures = fill_signatures
            self._cost_ids = cost_ids
            self._cash_adjustment_ids = cash_adjustment_ids
            self._corporate_action_ids = corporate_action_ids
            self._borrow_state = {str(key): dict(value) for key, value in data.get("borrow_state", {}).items()}
            self._valuation_marks = {str(key): dict(value) for key, value in data.get("valuation_marks", {}).items()}
            self._reservations = {}
            for instrument_id, values in restored_positions.items():
                state = self._positions[instrument_id]
                (state.quantity, state.average_entry_price, state.realized_pnl,
                 state.mark_price, state.mark_timestamp,
                 state.grandfathered_short, state.settlement_timestamp,
                 state.sourced_settlement_price, state.last_fill_timestamp, state.fill_history_verified) = values

    def _reserve(self, reservation_id: str, kind: str, currency: str,
                 amount: float, instrument_id: Optional[str]) -> None:
        if not isinstance(reservation_id, str) or not reservation_id.strip():
            raise ValueError("reservation_id must be a non-empty string")
        value = _number(amount, "amount", positive=True)
        with self._lock:
            existing = self._reservations.get(reservation_id)
            reservation = _Reservation(reservation_id, kind, currency, value, instrument_id)
            if existing is not None:
                if existing == reservation:
                    return
                raise ValueError("reservation_id already exists with different details")
            if value > self._available_cash(currency) + 1e-9:
                raise ValueError("reservation exceeds available cash or margin capacity")
            self._reservations[reservation_id] = reservation

    def _available_cash(self, currency: str, as_of: Optional[float] = None,
                        exclude_reservation: Optional[str] = None) -> float:
        balance = self._cash.get(currency, 0.0)
        reserved = sum(r.amount for r in self._reservations.values()
                       if r.currency == currency and r.reservation_id != exclude_reservation)
        margin_in_use = 0.0
        stamp = float(self.clock()) if as_of is None else as_of
        balance -= self._accrued_borrow_by_currency(stamp).get(currency, 0.0)
        for state in self._positions.values():
            if state.spec.settlement_currency != currency or state.quantity == 0:
                continue
            derivative = (state.spec.asset_class == AssetClass.FUTURE or
                          (state.spec.asset_class == AssetClass.FX and
                           state.spec.fx_accounting_mode == FXAccountingMode.ROLLING))
            if derivative:
                mark = self._required_position_mark(state, stamp)
                initial, _ = self._margin_amounts(state.spec, state.quantity, mark)
                margin_in_use += initial
        return balance - reserved - margin_in_use

    def _reservation_totals(self, kind: str) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for reservation in self._reservations.values():
            if reservation.kind == kind:
                totals[reservation.currency] = totals.get(reservation.currency, 0.0) + reservation.amount
        return totals

    def _apply_cash_delta(self, currency: str, amount: float) -> None:
        self._cash[currency] = self._cash.get(currency, 0.0) + amount
        if abs(self._cash[currency]) < 1e-12:
            self._cash[currency] = 0.0

    def _apply_fee(self, currency: str, amount: float) -> None:
        if amount:
            self._apply_cash_delta(currency, -amount)
            self._fees[currency] = self._fees.get(currency, 0.0) + amount

    def _update_position(self, state: _Position, delta: float, price: float) -> None:
        old = state.quantity
        new = old + delta
        multiplier = state.spec.contract_multiplier
        if old == 0 or old * delta > 0:
            old_abs = abs(old)
            delta_abs = abs(delta)
            state.average_entry_price = ((old_abs * state.average_entry_price + delta_abs * price)
                                         / (old_abs + delta_abs))
        else:
            closed = min(abs(old), abs(delta))
            state.realized_pnl += (closed * multiplier * (price - state.average_entry_price)
                                   * (1.0 if old > 0 else -1.0))
            if abs(new) <= 1e-12:
                state.average_entry_price = 0.0
                new = 0.0
            elif old * new > 0:
                # A partial close retains the remaining old lot's average cost.
                pass
            else:
                state.average_entry_price = price
        state.quantity = new
        if abs(new) <= 1e-12:
            state.grandfathered_short = False

    def _apply_derivative_realized(self, state: _Position, delta: float, price: float) -> None:
        old = state.quantity
        if old * delta < 0:
            closed = min(abs(old), abs(delta))
            realized = (closed * state.spec.contract_multiplier * (price - state.average_entry_price)
                        * (1.0 if old > 0 else -1.0))
            self._apply_cash_delta(state.spec.settlement_currency, realized)

    def _validate_short(self, state: _Position, new_quantity: float,
                        *, allow_risk_reduction: bool = False) -> None:
        if new_quantity >= -1e-12:
            return
        spec = state.spec
        if (allow_risk_reduction and state.quantity < -1e-12
                and abs(new_quantity) < abs(state.quantity) - 1e-12):
            return
        if spec.asset_class == AssetClass.FUTURE:
            # Both directions are native to a margined futures contract.
            self._validate_margin(spec)
            return
        if (spec.asset_class == AssetClass.FX
                and spec.fx_accounting_mode == FXAccountingMode.ROLLING):
            # Rolling FX is leveraged long/short exposure, governed by margin.
            self._require_margin(spec)
            return
        if not state.allow_short:
            raise ValueError("short position requires explicit allow_short configuration")
        limit = spec.borrow_limit
        rate = spec.borrow_fee_rate
        if limit is None or limit <= 0 or rate is None:
            raise ValueError("short position requires explicit borrow_limit and borrow_fee_rate")
        if abs(new_quantity) > limit + 1e-12:
            raise ValueError("short position exceeds borrow_limit")

    @staticmethod
    def _side(side) -> str:
        if isinstance(side, Side):
            return "buy" if side == Side.BUY else "sell"
        if isinstance(side, str) and side.strip().lower() in {"buy", "sell"}:
            return side.strip().lower()
        raise ValueError("side must be Side.BUY/SELL or 'buy'/'sell'")

    def _get_position(self, instrument_id: str) -> _Position:
        if not isinstance(instrument_id, str) or not instrument_id.strip():
            raise ValueError("instrument_id must be a non-empty string")
        try:
            return self._positions[instrument_id]
        except KeyError:
            raise KeyError("unregistered instrument_id: {}".format(instrument_id))

    def _fx_rate(self, currency: str, as_of: float) -> float:
        if currency == self.reporting_currency:
            return 1.0
        mark = self._fx_marks.get(currency)
        if mark is None:
            raise ValueError("missing {} to {} conversion mark".format(currency, self.reporting_currency))
        rate, timestamp, max_age = mark
        age = as_of - timestamp
        if age < 0 or age > max_age:
            raise ValueError("stale {} to {} conversion mark".format(currency, self.reporting_currency))
        return rate

    def _matching_valuation_mark(self, state: _Position) -> dict | None:
        mark = self._valuation_marks.get(state.spec.instrument_id)
        if (mark is not None and mark["timestamp"] == state.mark_timestamp
                and mark["price"] == state.mark_price):
            return mark
        return None

    def _required_position_mark(self, state: _Position, as_of: float) -> float:
        if state.mark_price is None or state.mark_timestamp is None:
            raise ValueError("missing mark for {}".format(state.spec.instrument_id))
        age = as_of - state.mark_timestamp
        if age < 0 or age > self.max_mark_age_seconds:
            closure = self._matching_valuation_mark(state)
            if not (closure and closure["timestamp"] <= as_of <= closure["valid_until"]):
                raise ValueError("stale mark for {}".format(state.spec.instrument_id))
        return state.mark_price

    def _position_snapshot(self, state: _Position, as_of: float) -> PositionSnapshot:
        spec = state.spec
        closure = self._matching_valuation_mark(state)
        mark = self._required_position_mark(state, as_of) if state.quantity != 0 else state.mark_price
        pnl_currency = (spec.quote_currency if spec.asset_class != AssetClass.FUTURE
                        and not (spec.asset_class == AssetClass.FX
                                 and spec.fx_accounting_mode == FXAccountingMode.ROLLING)
                        else spec.settlement_currency)
        unrealized = (state.quantity * spec.contract_multiplier * (mark - state.average_entry_price)
                      if state.quantity != 0 and mark is not None else 0.0)
        market_value: Optional[float]
        if spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}:
            market_value = state.quantity * mark * spec.contract_multiplier if mark is not None else None
        elif spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE:
            market_value = state.quantity * mark * spec.contract_multiplier if mark is not None else None
        else:
            market_value = None
        initial = maintenance = 0.0
        derivative = (spec.asset_class == AssetClass.FUTURE or
                      (spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.ROLLING))
        if derivative and state.quantity != 0:
            initial, maintenance = self._margin_amounts(spec, state.quantity, mark)
        basis = closure["basis"] if closure else "observed_quote"
        if (not closure and state.sourced_settlement_price is not None
                and state.mark_timestamp == state.settlement_timestamp
                and state.mark_price == state.sourced_settlement_price):
            basis = "official_settlement"
        return PositionSnapshot(
            instrument_id=spec.instrument_id, asset_class=_enum_value(spec.asset_class),
            quantity=state.quantity, quantity_unit=spec.quantity_unit,
            average_entry_price=state.average_entry_price if state.quantity != 0 else None,
            mark_price=mark, mark_timestamp=state.mark_timestamp, pnl_currency=pnl_currency,
            realized_pnl=state.realized_pnl, unrealized_pnl=unrealized,
            market_value=market_value, initial_margin_required=initial,
            maintenance_margin_required=maintenance,
            mark_basis=basis,
            valuation_valid_until=closure["valid_until"] if closure else None,
            last_fill_timestamp=state.last_fill_timestamp,
            fill_history_verified=state.fill_history_verified,
        )
