"""Canonical, unit-aware instrument metadata for multi-asset paper trading.

Prices and quantities remain numeric at the API boundary, but their meaning is
defined by an :class:`InstrumentSpec` (for example, USD per share or contracts
with a multiplier).  Provider adapters should resolve symbols to this immutable
specification before publishing market events or accepting paper orders.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, ROUND_UP
from enum import Enum
import math
from typing import FrozenSet, Optional, Union

from ..utils.types import Side


class AssetClass(str, Enum):
    """Supported instrument families."""

    CRYPTO = "crypto"
    EQUITY = "equity"
    FX = "fx"
    FUTURE = "future"


class FXAccountingMode(str, Enum):
    """How a spot FX position is settled and carried in the paper ledger."""

    DELIVERABLE = "deliverable"
    ROLLING = "rolling"


_DecimalInput = Union[int, float, Decimal]
_QUANTITY_ROUNDING_MODES = frozenset((ROUND_DOWN, ROUND_UP, ROUND_HALF_UP))


def _finite_decimal(value: _DecimalInput, field_name: str) -> Decimal:
    """Convert a numeric input without importing its binary float error."""
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number".format(field_name))
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("{} must be a finite number".format(field_name))
    if not result.is_finite():
        raise ValueError("{} must be a finite number".format(field_name))
    return result


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(field_name))
    return value.strip()


@dataclass(frozen=True)
class InstrumentSpec:
    """Immutable trading and accounting conventions for one listed instrument.

    ``tick_size`` is in price units; ``quantity_step`` and ``min_quantity`` are
    in ``quantity_unit``. ``contract_multiplier`` converts one quantity unit
    into the underlying contract exposure.  Expiry timestamps must be timezone
    aware; they are normalized to UTC.  Prices may be negative for contracts
    whose market convention permits it, so admissible price sign remains a
    product-level decision rather than a global assumption.
    """

    instrument_id: str
    asset_class: AssetClass
    symbol: str
    venue: str
    base_currency: str
    quote_currency: str
    settlement_currency: str
    quantity_unit: str
    tick_size: float
    quantity_step: float
    min_quantity: float
    contract_multiplier: float = 1.0
    session_timezone: str = "UTC"
    expiry: Optional[datetime] = None
    fx_accounting_mode: Optional[FXAccountingMode] = None
    initial_margin_rate: Optional[float] = None
    maintenance_margin_rate: Optional[float] = None
    initial_margin_per_contract: Optional[float] = None
    maintenance_margin_per_contract: Optional[float] = None
    borrow_limit: Optional[float] = None
    borrow_fee_rate: Optional[float] = None
    borrow_rate_source: Optional[str] = None
    borrow_rate_effective_from: Optional[float] = None
    borrow_rate_effective_to: Optional[float] = None
    borrow_day_count: str = "ACT/365F"
    capabilities: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in ("instrument_id", "symbol", "venue", "quantity_unit", "session_timezone"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))

        for name in ("base_currency", "quote_currency", "settlement_currency"):
            value = _required_text(getattr(self, name), name).upper()
            object.__setattr__(self, name, value)

        if not isinstance(self.asset_class, AssetClass):
            raise ValueError("asset_class must be an AssetClass")

        for name in ("tick_size", "quantity_step", "min_quantity", "contract_multiplier"):
            value = _finite_decimal(getattr(self, name), name)
            if value <= 0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, float(value))

        min_quantity = Decimal(str(self.min_quantity))
        quantity_step = Decimal(str(self.quantity_step))
        if min_quantity % quantity_step != 0:
            raise ValueError("min_quantity must be aligned to quantity_step")

        for name in ("initial_margin_rate", "maintenance_margin_rate",
                     "initial_margin_per_contract", "maintenance_margin_per_contract",
                     "borrow_limit", "borrow_fee_rate"):
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            value = _finite_decimal(raw_value, name)
            if value < 0:
                raise ValueError("{} must be non-negative".format(name))
            object.__setattr__(self, name, float(value))

        if self.initial_margin_rate is not None and self.maintenance_margin_rate is not None:
            if self.maintenance_margin_rate > self.initial_margin_rate:
                raise ValueError("maintenance_margin_rate cannot exceed initial_margin_rate")
        if (self.initial_margin_per_contract is not None
                and self.maintenance_margin_per_contract is not None
                and self.maintenance_margin_per_contract > self.initial_margin_per_contract):
            raise ValueError("maintenance_margin_per_contract cannot exceed initial_margin_per_contract")
        if ((self.initial_margin_per_contract is None)
                != (self.maintenance_margin_per_contract is None)):
            raise ValueError("per-contract initial and maintenance margins must be configured together")

        if self.asset_class == AssetClass.FUTURE:
            if self.expiry is None:
                raise ValueError("future instruments require an expiry")
            has_contract_margin = (self.initial_margin_per_contract is not None
                                   and self.initial_margin_per_contract > 0
                                   and self.maintenance_margin_per_contract is not None
                                   and self.maintenance_margin_per_contract > 0)
            if not has_contract_margin:
                raise ValueError("future instruments require explicit positive per-contract margin amounts")

        if self.expiry is not None:
            if not isinstance(self.expiry, datetime):
                raise ValueError("expiry must be a timezone-aware datetime")
            if self.expiry.tzinfo is None or self.expiry.utcoffset() is None:
                raise ValueError("expiry must be a timezone-aware datetime")
            object.__setattr__(self, "expiry", self.expiry.astimezone(timezone.utc))

        if self.asset_class == AssetClass.FX:
            if not isinstance(self.fx_accounting_mode, FXAccountingMode):
                raise ValueError("FX instruments require an FXAccountingMode")
            if (self.fx_accounting_mode == FXAccountingMode.ROLLING
                    and self.settlement_currency != self.quote_currency):
                raise ValueError("rolling FX settlement_currency must match quote_currency")
        elif self.fx_accounting_mode is not None:
            raise ValueError("fx_accounting_mode is only valid for FX instruments")

        if isinstance(self.capabilities, str):
            raise ValueError("capabilities must be an iterable of capability names")
        try:
            capabilities = frozenset(_required_text(value, "capability").lower() for value in self.capabilities)
        except TypeError:
            raise ValueError("capabilities must be an iterable of capability names")
        object.__setattr__(self, "capabilities", capabilities)
        if self.borrow_rate_source is not None:
            object.__setattr__(self, "borrow_rate_source", _required_text(self.borrow_rate_source, "borrow_rate_source"))
        if self.borrow_day_count != "ACT/365F":
            raise ValueError("only explicit ACT/365F estimated borrow accrual is supported")
        for name in ("borrow_rate_effective_from", "borrow_rate_effective_to"):
            value = getattr(self, name)
            if value is not None:
                value = float(_finite_decimal(value, name))
                if value < 0:
                    raise ValueError("borrow effective timestamps must be non-negative")
                object.__setattr__(self, name, value)
        if (self.borrow_rate_effective_from is not None and self.borrow_rate_effective_to is not None
                and self.borrow_rate_effective_to <= self.borrow_rate_effective_from):
            raise ValueError("borrow effective end must follow its start")

    def round_price(self, price: _DecimalInput, side: Optional[Side] = None) -> float:
        """Round a price to a legal tick, protecting a submitted limit.

        Buy limits round down and sell limits round up so rounding never makes
        the order more aggressive than requested. With no side, round to the
        nearest tick using half-up ties.
        """
        value = _finite_decimal(price, "price")
        if side is not None and not isinstance(side, Side):
            raise ValueError("side must be Side.BUY, Side.SELL, or None")
        if side == Side.BUY:
            rounding = ROUND_FLOOR
        elif side == Side.SELL:
            rounding = ROUND_CEILING
        else:
            rounding = ROUND_HALF_UP
        tick = Decimal(str(self.tick_size))
        ticks = (value / tick).to_integral_value(rounding=rounding)
        return float(ticks * tick)

    def round_quantity(self, quantity: _DecimalInput, rounding: str = ROUND_DOWN) -> float:
        """Round a positive order size to the instrument quantity step.

        The default rounds down to avoid exceeding requested exposure. The
        result must still meet the instrument's minimum order quantity.
        """
        value = _finite_decimal(quantity, "quantity")
        if value <= 0:
            raise ValueError("quantity must be positive")
        if rounding not in _QUANTITY_ROUNDING_MODES:
            raise ValueError("rounding must be ROUND_DOWN, ROUND_UP, or ROUND_HALF_UP")
        step = Decimal(str(self.quantity_step))
        units = (value / step).to_integral_value(rounding=rounding)
        rounded = units * step
        if rounded < Decimal(str(self.min_quantity)):
            raise ValueError("rounded quantity is below min_quantity")
        return float(rounded)

    def is_valid_price(self, price: _DecimalInput) -> bool:
        """Return whether a finite price lies exactly on a tick."""
        try:
            value = _finite_decimal(price, "price")
        except ValueError:
            return False
        tick = Decimal(str(self.tick_size))
        return value % tick == 0

    def is_valid_quantity(self, quantity: _DecimalInput) -> bool:
        """Return whether a finite positive size meets minimum and step rules."""
        try:
            value = _finite_decimal(quantity, "quantity")
        except ValueError:
            return False
        if value < Decimal(str(self.min_quantity)) or value <= 0:
            return False
        step = Decimal(str(self.quantity_step))
        return value % step == 0

