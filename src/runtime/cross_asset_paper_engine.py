"""Local paper engine for independent instruments and read-only market feeds.

Each instrument has its own quote readiness and strategy state. A single
portfolio ledger and risk gate aggregate exposures and P&L across currencies.
No provider method can submit, cancel, or query an exchange order.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from ..config.cross_asset_config import CrossAssetConfig, InstrumentTradingConfig
from ..connectors.market_data import MarketDataProvider, ProviderCapabilities
from ..market.events import MarketDataEvent, MarketEventKind
from ..market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from ..orderbook.order import Fill, Order
from ..runtime.portfolio_ledger import PortfolioLedger, PortfolioSnapshot
from .portfolio_risk import PortfolioProposal, PortfolioRiskEvaluator, RiskContext
from .lifecycle_accounting import AccountActionSchedule
from .recording import EXECUTION_SEMANTICS_VERSION
from ..strategies.instrument_aware_market_maker import InstrumentAwareMarketMaker
from ..utils.types import OrderStatus, OrderType, Price, Quantity, Side, Timestamp


@dataclass
class _WorkingOrder:
    order: Order
    accepted_event_cutoffs: Dict[str, float]
    accepted_receive_ts: float
    remaining_quantity: float
    reservation_id: str
    fee_reservation_id: str | None = None


def _price_time_priority(candidate: tuple[str, _WorkingOrder]) -> tuple[float, float, str]:
    order_id, working = candidate
    price = working.order.price
    if price is None:
        raise ValueError("A working paper limit order requires a price")
    return (-float(price) if working.order.side == Side.BUY else float(price),
            working.accepted_receive_ts, order_id)


@dataclass
class _InstrumentRuntime:
    config: InstrumentTradingConfig
    strategy: InstrumentAwareMarketMaker
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    bids: Tuple[Tuple[float, float], ...] = ()
    asks: Tuple[Tuple[float, float], ...] = ()
    last_quote_event_ts: Optional[float] = None
    last_quote_receive_ts: Optional[float] = None
    book_fresh_after_ts: Optional[float] = None
    last_trade_event_ts: Optional[float] = None
    last_trade_receive_ts: Optional[float] = None
    last_event_by_kind: Dict[Tuple[str, str], Tuple[float, float, str]] = field(default_factory=dict)
    last_sequence_by_kind: Dict[Tuple[str, str], int] = field(default_factory=dict)
    book_synchronized: bool = True
    market_open: bool = False
    market_status: str = "waiting_for_session_status"
    active_orders: Dict[str, _WorkingOrder] = field(default_factory=dict)
    last_strategy_receive_ts: float = 0.0
    total_orders: int = 0
    total_fills: int = 0
    total_rejections: int = 0
    last_rejection: Optional[Dict[str, Any]] = None
    total_traded_notional: float = 0.0
    recent_public_trades: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=20))
    roll_liquidity: dict[Side, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def spec(self) -> InstrumentSpec:
        return self.config.spec

    @property
    def mid_price(self) -> Optional[float]:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0


class CrossAssetPaperEngine:
    """Multi-instrument paper engine with a shared currency-aware portfolio."""

    OPEN_STATES = frozenset({"open", "trading", "active", "tradeable", "continuous", "regular"})
    CLOSED_STATES = frozenset({"closed", "halted", "paused", "non_tradeable", "unknown", "error"})

    def __init__(
        self,
        config: CrossAssetConfig,
        *,
        clock: Callable[[], float] = time.time,
        quote_refresh_seconds: float = 1.0,
        max_clock_skew_seconds: Optional[float] = None,
        queue_fraction: Optional[float] = None,
        state_manager: Any = None,
        id_factory: Optional[Callable[[str], str]] = None,
        strategy_factory: Optional[Callable[[InstrumentTradingConfig], Any]] = None,
        allow_config_migration: bool = False,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if max_clock_skew_seconds is None:
            max_clock_skew_seconds = config.max_clock_skew_seconds
        for name, value in (("quote_refresh_seconds", quote_refresh_seconds),
                            ("max_clock_skew_seconds", max_clock_skew_seconds)):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if quote_refresh_seconds <= 0:
            raise ValueError("quote_refresh_seconds must be positive")
        self.config = config
        self.defer_dashboard_updates = False
        self._transition_depth = 0
        self.id_factory = id_factory or (lambda prefix: "{}:{}".format(prefix, uuid.uuid4().hex))
        self.allow_config_migration = bool(allow_config_migration)
        self.config_epoch = 1
        self.risk_evaluator = PortfolioRiskEvaluator()
        self._provider_generations: Dict[str, int] = {}
        self.account_actions = AccountActionSchedule()
        self._accounting_blocks: Dict[str, str] = {}
        self.state_manager = state_manager
        self.account_id = config.account_id
        self.clock = clock
        self.quote_refresh_seconds = float(quote_refresh_seconds)
        self.max_clock_skew_seconds = float(max_clock_skew_seconds)
        self.queue_fraction_override = queue_fraction
        if queue_fraction is not None and (not math.isfinite(queue_fraction) or not 0 <= queue_fraction <= 1):
            raise ValueError("queue_fraction must be in [0, 1]")
        self.start_time = float(clock())
        self._risk_day = datetime.fromtimestamp(self.start_time, timezone.utc).date()
        self._risk_day_start_equity = config.starting_capital
        self._last_account_equity = config.starting_capital
        self._last_account_timestamp = self.start_time
        self.is_running = False
        self._lock = threading.RLock()
        self._risk_lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable[[Any], None]]] = {}
        self._providers: Dict[str, MarketDataProvider] = {}
        self._provider_instruments: Dict[str, Tuple[str, ...]] = {}
        self._runtime: Dict[str, _InstrumentRuntime] = {}
        self._rolled_futures = set()
        self._orders_per_minute: Deque[float] = deque()
        self._seen_event_ids: Dict[str, set] = {}
        self._seen_event_order: Dict[str, Deque[str]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.total_orders_sent = 0
        self.total_fills = 0
        self.total_rejections = 0
        self.total_traded_notional = 0.0
        self.circuit_breaker_active = False
        self.circuit_breaker_reason: Optional[str] = None
        self._assumptions_unverified = False

        self.ledger = PortfolioLedger(
            initial_cash=config.starting_capital,
            reporting_currency=config.reporting_currency,
            max_mark_age_seconds=config.max_mark_age_seconds,
            clock=clock,
        )
        for instrument_id, item in config.instruments.items():
            self.ledger.register_instrument(item.spec, allow_short=item.allow_short)
            strategy = InstrumentAwareMarketMaker(
                item.spec,
                order_size=item.order_size,
                max_inventory=item.max_position_quantity,
                spread_bps=item.spread_bps,
                inventory_skew_bps=item.inventory_skew_bps,
                strategy_id="mm:{}".format(instrument_id),
                maker_fee=item.fees.maker_rate,
                taker_fee=item.fees.taker_rate,
                allow_short=item.allow_short,
            )
            if strategy_factory is not None:
                strategy = strategy_factory(item)
            self._runtime[instrument_id] = _InstrumentRuntime(
                config=item,
                strategy=strategy,
                market_open=item.spec.asset_class == AssetClass.CRYPTO,
                market_status="continuous_crypto" if item.spec.asset_class == AssetClass.CRYPTO
                else "waiting_for_session_status",
            )
            self._seen_event_ids[instrument_id] = set()
            self._seen_event_order[instrument_id] = deque(maxlen=20_000)

        self._fx_edges: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self._conversion_mark_sources: Dict[str, str] = {}
        for currency, mark in config.conversion_marks.items():
            rate, source, max_age = mark
            stamp = self.start_time
            self._fx_edges[(currency, config.reporting_currency)] = (rate, stamp, max_age)
            self.ledger.set_fx_mark(currency, rate, stamp, max_age_seconds=max_age)
            self._conversion_mark_sources[currency] = source
        if config.starting_balances:
            self.ledger.set_initial_balances(config.starting_balances, self.start_time)
        if state_manager is not None and callable(getattr(state_manager, "get_cross_asset_signature", None)):
            self.ledger.set_signature_lookup(lambda namespace, key:
                state_manager.get_cross_asset_signature(self.account_id, namespace, key))
        self._restore_checkpoint()

    def _restore_checkpoint(self) -> None:
        if self.state_manager is None:
            return
        loaded = self.state_manager.load_cross_asset_snapshot(self.account_id)
        if loaded is None:
            return
        saved = loaded["snapshot"]
        if not isinstance(saved, dict) or saved.get("schema_version") != 1:
            raise ValueError("unsupported cross-asset account checkpoint")
        state = saved.get("engine", {})
        if not isinstance(state, dict):
            raise ValueError("cross-asset engine checkpoint is malformed")
        self.account_actions.restore_state(state.get("account_actions", {}))
        for action in sorted(self.account_actions.actions.values(), key=lambda value: value["effective_ts"]):
            if action["status"] != "applied" or action["action_type"] != "borrow_terms":
                continue
            runtime = self._runtime.get(action["instrument_id"])
            if runtime is None:
                continue
            payload = action["payload"]
            spec = replace(runtime.spec, borrow_fee_rate=float(payload["annual_rate"]),
                borrow_limit=float(payload["borrow_limit"]), borrow_rate_source=action["source"],
                borrow_rate_effective_from=action["effective_ts"],
                borrow_rate_effective_to=float(payload["effective_to"]),
                borrow_day_count=payload.get("day_count", "ACT/365F"))
            runtime.config = replace(runtime.config, spec=spec)
            items = dict(self.config.instruments)
            items[spec.instrument_id] = runtime.config
            self.config = replace(self.config, instruments=items)
            self.ledger._positions[spec.instrument_id].spec = spec
        saved_strategy_parameters = state.get("strategy_parameters")
        if saved_strategy_parameters is None:
            self._assumptions_unverified = True
        else:
            if (not isinstance(saved_strategy_parameters, dict)
                    or not set(saved_strategy_parameters).issubset(self._runtime)):
                raise ValueError("checkpoint strategy parameters do not match configured instruments")
            for instrument_id, parameters in saved_strategy_parameters.items():
                if not isinstance(parameters, dict) or set(parameters) != {
                        "spread_bps", "inventory_skew_bps"}:
                    raise ValueError("checkpoint strategy parameters are malformed for {}".format(
                        instrument_id))
                for name, value in parameters.items():
                    parsed = float(value)
                    if not math.isfinite(parsed) or parsed < 0:
                        raise ValueError("checkpoint {} must be finite and non-negative".format(name))
                    setattr(self._runtime[instrument_id].strategy, name, parsed)
        saved_assumptions = state.get("execution_assumptions")
        saved_version = (saved_assumptions or {}).get("_portfolio", {}).get("execution_semantics_version", 1)
        if saved_version != EXECUTION_SEMANTICS_VERSION and not self.allow_config_migration:
            raise ValueError("checkpoint execution semantics changed; explicit account migration is required")
        if saved_assumptions is None:
            # Older checkpoints keep their fills and P&L; show that the prior
            # fee and fill-model settings were not captured for comparison.
            self._assumptions_unverified = True
        elif not self._assumptions_compatible(saved_assumptions, self._execution_assumptions()) and not self.allow_config_migration:
            raise ValueError(
                "checkpoint fee or fill-model assumptions changed; use a new account_id "
                "to keep paper performance histories comparable")
        self.ledger.restore_state(saved["ledger"], allow_policy_changes=self.allow_config_migration)
        floors = state.get("market_data_floors")
        if floors is None:
            # Legacy state cannot prove the unit epoch of an older observation.
            floors = {instrument_id: loaded["timestamp"] for instrument_id in self._runtime}
            self._assumptions_unverified = True
        if not isinstance(floors, dict) or not set(floors).issubset(self._runtime):
            raise ValueError("checkpoint market data floors do not match configured instruments")
        for instrument_id, value in floors.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= loaded["timestamp"]):
                raise ValueError("checkpoint market data floor must precede its processing time")
            self._runtime[instrument_id].book_fresh_after_ts = float(value)
        budgets = state.get("roll_liquidity", {})
        if not isinstance(budgets, dict) or not set(budgets).issubset(self._runtime):
            raise ValueError("checkpoint roll liquidity does not match configured instruments")
        for instrument_id, sides in budgets.items():
            runtime = self._runtime[instrument_id]
            if runtime.spec.asset_class != AssetClass.FUTURE or not isinstance(sides, dict):
                raise ValueError("checkpoint roll liquidity requires a futures instrument")
            for side, values in sides.items():
                if side not in {"BUY", "SELL"} or not isinstance(values, list) or len(values) != 3:
                    raise ValueError("checkpoint roll liquidity requires side, price, displayed and remaining size")
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in values):
                    raise ValueError("checkpoint roll liquidity must be finite")
                price, displayed, remaining = values
                if not runtime.spec.is_valid_price(price) or not 0 <= remaining <= displayed:
                    raise ValueError("checkpoint roll liquidity violates price or quantity bounds")
                runtime.roll_liquidity[Side[side]] = tuple(values)
        if self.ledger._opening_balances is None:
            self._assumptions_unverified = True
        self._accounting_blocks.update(state.get("accounting_blocks", {}))
        self.config_epoch = int(state.get("config_epoch", 1))
        if saved_assumptions is not None and saved_assumptions != self._execution_assumptions():
            self.config_epoch += 1
        self._risk_day = datetime.fromisoformat(state.get("risk_day", self._risk_day.isoformat())).date()
        self._risk_day_start_equity = float(state.get("risk_day_start_equity", self.config.starting_capital))
        self._last_account_equity = float(state.get("last_account_equity", self._risk_day_start_equity))
        self._last_account_timestamp = float(state.get("last_account_timestamp", loaded["timestamp"]))
        self.total_orders_sent = int(state.get("total_orders_sent", 0))
        self.total_fills = int(state.get("total_fills", 0))
        self.total_rejections = int(state.get("total_rejections", 0))
        self.total_traded_notional = float(state.get("total_traded_notional", 0.0))
        self.circuit_breaker_active = bool(state.get("circuit_breaker_active", False))
        self.circuit_breaker_reason = state.get("circuit_breaker_reason")
        self.trades = list(state.get("trades", []))[-2000:]
        for instrument_id, counts in state.get("instrument_totals", {}).items():
            runtime = self._runtime.get(instrument_id)
            if runtime is None:
                raise ValueError("checkpoint contains an unconfigured instrument {}".format(instrument_id))
            runtime.total_orders = int(counts.get("orders", 0))
            runtime.total_fills = int(counts.get("fills", 0))
            runtime.total_rejections = int(counts.get("rejections", 0))
            rejection = counts.get("last_rejection")
            runtime.last_rejection = dict(rejection) if rejection is not None else None
            runtime.total_traded_notional = float(counts.get("notional", 0.0))
        read_events = getattr(self.state_manager, "get_cross_asset_events", None)
        if callable(read_events) and any(runtime.total_rejections and runtime.last_rejection is None
                                         for runtime in self._runtime.values()):
            # Older checkpoints retain counts only. Bound the legacy lookup;
            # absent details remain unavailable rather than inventing a reason.
            for event in read_events(self.account_id, limit=1000):
                if event["event_type"] != "order_rejected":
                    continue
                payload = event["payload"]
                runtime = self._runtime.get(payload.get("instrument_id"))
                if runtime is not None and runtime.last_rejection is None:
                    runtime.last_rejection = {"reason": payload["reason"],
                        "order_id": payload["order_id"], "timestamp": event["timestamp"]}
        raw_edges = state.get("fx_edges", [])
        for edge in raw_edges:
            if not isinstance(edge, list) or len(edge) != 5:
                raise ValueError("checkpoint FX edge is malformed")
            base, quote, rate, timestamp, max_age = edge
            self._fx_edges[(str(base), str(quote))] = (
                float(rate), float(timestamp), float(max_age))
        self._conversion_mark_sources.update(state.get("conversion_mark_sources", {}))
        self._rolled_futures = set(state.get("rolled_futures", []))

    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "ledger": self.ledger.export_state(),
            "engine": {
                "execution_assumptions": self._execution_assumptions(),
                "config_epoch": self.config_epoch,
                "account_actions": self.account_actions.export_state(),
                "accounting_blocks": dict(self._accounting_blocks),
                "market_data_floors": {instrument_id: runtime.book_fresh_after_ts
                    for instrument_id, runtime in self._runtime.items()
                    if runtime.book_fresh_after_ts is not None},
                "roll_liquidity": {instrument_id: {side.name: list(values)
                    for side, values in runtime.roll_liquidity.items()}
                    for instrument_id, runtime in self._runtime.items()
                    if runtime.spec.asset_class == AssetClass.FUTURE and runtime.roll_liquidity},
                "strategy_parameters": self._strategy_parameters(),
                "risk_day": self._risk_day.isoformat(),
                "risk_day_start_equity": self._risk_day_start_equity,
                "last_account_equity": self._last_account_equity,
                "last_account_timestamp": self._last_account_timestamp,
                "total_orders_sent": self.total_orders_sent,
                "total_fills": self.total_fills,
                "total_rejections": self.total_rejections,
                "total_traded_notional": self.total_traded_notional,
                "circuit_breaker_active": self.circuit_breaker_active,
                "circuit_breaker_reason": self.circuit_breaker_reason,
                "trades": self.trades[-2000:],
                "instrument_totals": {
                    instrument_id: {
                        "orders": runtime.total_orders,
                        "fills": runtime.total_fills,
                        "rejections": runtime.total_rejections,
                        "last_rejection": runtime.last_rejection,
                        "notional": runtime.total_traded_notional,
                    }
                    for instrument_id, runtime in self._runtime.items()
                },
                "fx_edges": [
                    [base, quote, rate, timestamp, max_age]
                    for (base, quote), (rate, timestamp, max_age) in self._fx_edges.items()
                ],
                "conversion_mark_sources": self._conversion_mark_sources,
                "rolled_futures": sorted(self._rolled_futures),
            },
        }

    def _execution_assumptions(self) -> Dict[str, Any]:
        """Capture configuration that changes quote, risk, fill, or mark outcomes."""
        result = {
            "_portfolio": {
                "execution_semantics_version": EXECUTION_SEMANTICS_VERSION,
                "reporting_currency": self.config.reporting_currency,
                "max_daily_loss": self.config.max_daily_loss,
                "max_gross_exposure": self.config.max_gross_exposure,
                "max_orders_per_minute": self.config.max_orders_per_minute,
                "max_mark_age_seconds": self.config.max_mark_age_seconds,
                "market_data_stale_after": self.config.market_data_stale_after,
                "max_clock_skew_seconds": self.max_clock_skew_seconds,
                "conversion_marks": {
                    currency: list(mark)
                    for currency, mark in self.config.conversion_marks.items()
                },
            },
        }
        for instrument_id, item in self.config.instruments.items():
            provider = self.config.providers[item.provider_id]
            result[instrument_id] = {
                "provider": {
                    "type": provider.provider_type,
                    "dataset": provider.dataset,
                    "schemas": list(provider.schemas),
                    "stype_in": provider.stype_in,
                    "environment": provider.environment,
                    "symbol": item.provider_symbol,
                },
                "fill_model": item.fill_model,
                "order_size": item.order_size,
                "max_order_quantity": item.max_order_quantity,
                "max_position_quantity": item.max_position_quantity,
                "max_notional": item.max_notional,
                "allow_short": item.allow_short,
                "enabled": item.enabled,
                "queue_fraction": item.queue_fraction,
                "queue_fraction_override": self.queue_fraction_override,
                "maker_latency_ms": item.maker_latency_ms,
                "borrow_fee_rate": item.spec.borrow_fee_rate,
                "borrow_limit": item.spec.borrow_limit,
                "borrow_day_count": item.spec.borrow_day_count,
                "borrow_rate_source": item.spec.borrow_rate_source,
                "borrow_rate_effective_from": item.spec.borrow_rate_effective_from,
                "borrow_rate_effective_to": item.spec.borrow_rate_effective_to,
                "auto_roll": item.auto_roll,
                "roll_to": item.roll_to,
                "roll_before_expiry_seconds": item.roll_before_expiry_seconds,
                "fees": {
                    "maker_rate": item.fees.maker_rate,
                    "taker_rate": item.fees.taker_rate,
                    "maker_per_unit": item.fees.maker_per_unit,
                    "taker_per_unit": item.fees.taker_per_unit,
                    "currency": item.fees.currency,
                },
            }
        return result

    @staticmethod
    def _assumptions_compatible(saved: Any, current: Any) -> bool:
        """Old checkpoints may omit newly added optional assumptions."""
        if isinstance(saved, dict) and isinstance(current, dict):
            if ("execution_semantics_version" in current
                    and saved.get("execution_semantics_version", 1) != current["execution_semantics_version"]):
                return False
            return all(key in current and CrossAssetPaperEngine._assumptions_compatible(value, current[key])
                       for key, value in saved.items())
        return saved == current

    def _strategy_parameters(self) -> Dict[str, Dict[str, float]]:
        return {
            instrument_id: {
                "spread_bps": runtime.strategy.spread_bps,
                "inventory_skew_bps": runtime.strategy.inventory_skew_bps,
            }
            for instrument_id, runtime in self._runtime.items()
        }

    def _save_checkpoint(self, timestamp: float) -> None:
        if self.state_manager is not None and not self._transition_depth:
            try:
                self.state_manager.save_cross_asset_snapshot(
                    self.account_id, timestamp, self._checkpoint_payload())
            except Exception as exc:
                self._stop_for_persistence_failure(exc)

    def _stop_for_persistence_failure(self, exc: Exception) -> None:
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = "paper_history_persistence_failed"
        with self._lock:
            self.is_running = False
        self._cancel_all_quotes(reason=self.circuit_breaker_reason)
        self.emit("persistence_error", {"message": str(exc), "execution_mode": "local_paper"})
        raise RuntimeError("paper history could not be saved; strategy stopped") from exc

    def _restore_failed_ledger(self, original: PortfolioLedger) -> None:
        self.ledger = original
        if self.circuit_breaker_reason == "paper_history_persistence_failed":
            # The halt already removed working orders while the candidate was
            # active. Their holds in the original financial state must not be
            # resurrected when rolling the candidate back.
            with original._lock:
                original._reservations.clear()

    def _persist_event(self, event_type: str, event_id: str, payload: Dict[str, Any],
                       timestamp: float) -> None:
        if self.state_manager is None:
            return
        try:
            # Journal timestamps describe when the economic event took effect.
            # A late-arriving adjustment may be backdated, but its checkpoint
            # must still advance to processing time or recovery can load an
            # older ledger snapshot and lose the adjustment.
            checkpoint_timestamp = max(float(self.clock()), float(timestamp))
            inserted = self.state_manager.commit_cross_asset_event(
                self.account_id, event_id, event_type, timestamp,
                dict(payload, config_epoch=self.config_epoch),
                self._checkpoint_payload(), snapshot_timestamp=checkpoint_timestamp)
            if inserted is False:
                raise RuntimeError("durable event already exists; the candidate transition needs reconciliation")
        except Exception as exc:
            self._stop_for_persistence_failure(exc)

    @contextmanager
    def _candidate_ledger_transition(self):
        original = self.ledger
        self.ledger = original.clone()
        self._transition_depth += 1
        try:
            yield
        except Exception:
            self._restore_failed_ledger(original)
            raise
        finally:
            self._transition_depth -= 1
        self._emit_dashboard_state()

    def _current_daily_pnl(self) -> Optional[float]:
        try:
            snapshot = self.ledger.snapshot(as_of=float(self.clock()))
        except (ValueError, KeyError):
            return None
        return snapshot.equity - self._risk_day_start_equity

    def on(self, event: str, callback: Callable[[Any], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._callbacks.setdefault(event, []).append(callback)

    def emit(self, event: str, data: Any) -> None:
        with self._lock:
            callbacks = list(self._callbacks.get(event, ()))
        for callback in callbacks:
            try:
                callback(data)
            except Exception as exc:
                print("[PAPER] {} callback failed: {}".format(event, exc))

    def _emit_dashboard_state(self) -> None:
        """The application may coalesce UI snapshots without merging feed events."""
        if not self.defer_dashboard_updates and not self._transition_depth:
            self.emit("multi_asset_state", self.get_dashboard_state())

    def register_provider(self, provider_id: str, provider: MarketDataProvider,
                          instrument_ids: Iterable[str]) -> None:
        """Attach one read-only provider to its configured instruments."""
        if provider_id in self._providers:
            raise ValueError("provider_id is already registered: {}".format(provider_id))
        ids = tuple(instrument_ids)
        if not ids or any(instrument_id not in self._runtime for instrument_id in ids):
            raise ValueError("provider must reference configured instruments")
        for instrument_id in ids:
            item = self._runtime[instrument_id]
            capabilities = provider.capabilities
            if item.config.fill_model == "trade_print_queue" and not capabilities.trades:
                raise ValueError("{} requires a provider with trade data".format(instrument_id))
            if item.config.fill_model == "fx_quote_cross" and not capabilities.quote:
                raise ValueError("{} requires quote data for modeled FX fills".format(instrument_id))
            if not capabilities.quote:
                raise ValueError("{} provider cannot publish two-sided quote data".format(instrument_id))
            item.capabilities = capabilities
        self._providers[provider_id] = provider
        self._provider_instruments[provider_id] = ids

    def start_providers(self) -> None:
        """Start configured read-only feeds; partial provider failure is surfaced."""
        attempted = []
        try:
            for provider_id, provider in self._providers.items():
                # Include the currently starting provider in rollback: a start
                # call can allocate a worker or socket and then raise.
                attempted.append(provider_id)
                provider.start(self.on_market_event)
        except Exception:
            for provider_id in reversed(attempted):
                try:
                    self._providers[provider_id].stop()
                except Exception:
                    pass
            raise

    def stop_providers(self) -> None:
        for provider in self._providers.values():
            try:
                provider.stop()
            except Exception as exc:
                print("[PAPER] provider stop failed: {}".format(exc))

    def start_strategy(self) -> None:
        with self._risk_lock:
            with self._lock:
                if self.is_running:
                    return
                self.is_running = True
            self.emit("strategy_status", {"is_running": True, "execution_mode": "local_paper"})
            self._refresh_all_quotes()

    def market_data_is_fresh(self) -> bool:
        """Whether at least one configured instrument has fresh usable data.

        Readiness is evaluated per instrument, so a closed or disconnected
        market cannot suppress quoting on another healthy feed.
        """
        now = float(self.clock())
        return any(self._instrument_ready(runtime, now) for runtime in self._runtime.values())

    def stop_strategy(self) -> None:
        with self._risk_lock:
            with self._lock:
                self.is_running = False
            self._cancel_all_quotes(reason="global_stop")
            self.emit("strategy_status", {"is_running": False, "execution_mode": "local_paper"})
            self._save_checkpoint(float(self.clock()))

    def acknowledge_risk_circuit_breaker(self) -> Dict[str, Any]:
        """Clear a risk latch only after maintenance and daily-loss checks pass."""
        with self._risk_lock:
            if not self.circuit_breaker_active:
                return {"cleared": False, "message": "No active risk halt."}
            if self.circuit_breaker_reason == "paper_history_persistence_failed":
                raise ValueError("paper history must be repaired before risk halt acknowledgment")
            if self.circuit_breaker_reason == "command_outcome_reconciliation_required":
                raise ValueError(
                    "committed command effects require outcome reconciliation before risk can resume"
                )
            now = float(self.clock())
            snapshot = self.ledger.snapshot(as_of=now)
            if snapshot.equity + 1e-9 < snapshot.maintenance_margin_required:
                raise ValueError("maintenance margin is still breached; reduce exposure first")
            if self._daily_loss_breached(snapshot, now):
                raise ValueError("daily loss limit is still breached")
            previous_reason = self.circuit_breaker_reason
            self.circuit_breaker_active = False
            self.circuit_breaker_reason = None
            self._persist_event("risk_circuit_breaker_acknowledged",
                                "risk-ack:{}".format(int(now * 1_000_000)), {
                                    "previous_reason": previous_reason,
                                }, now)
            self._emit_dashboard_state()
            return {"cleared": True, "message": "Risk checks passed; paper strategy may be started."}

    def reset_strategy(self) -> None:
        raise RuntimeError("Cross-asset account reset is disabled to preserve paper history")

    def update_parameters(self, params: dict) -> None:
        with self._risk_lock:
            self._update_parameters(params)

    def _update_parameters(self, params: dict) -> None:
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        allowed = {"spread_bps", "inventory_skew_bps", "instrument_id"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError("unsupported cross-asset strategy parameters: {}".format(sorted(unknown)))
        instrument_id = params.get("instrument_id")
        if instrument_id is not None and instrument_id not in self._runtime:
            raise ValueError("instrument_id is not configured")
        values = {}
        for name, value in params.items():
            if name == "instrument_id":
                continue
            parsed = float(value)
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
            values[name] = parsed
        if not values:
            raise ValueError("at least one strategy parameter is required")
        targets = ([self._runtime[instrument_id]] if instrument_id is not None
                   else list(self._runtime.values()))
        with self._lock:
            for runtime in targets:
                for name, value in values.items():
                    setattr(runtime.strategy, name, value)
        for runtime in targets:
            self._cancel_instrument_quotes(runtime, "strategy_parameters_changed")
        if instrument_id is None:
            self._refresh_all_quotes()
        elif self.is_running:
            runtime = targets[0]
            if self._instrument_ready(runtime, float(self.clock())):
                self._generate_quotes(runtime, float(self.clock()))
        now = float(self.clock())
        event_id = "strategy-parameters:{}:{}".format(
            instrument_id or "all", self.id_factory("parameters"))
        self._persist_event("strategy_parameters_updated", event_id, {
            "instrument_id": instrument_id,
            "updated": values,
            "current": self._strategy_parameters(),
        }, now)

    def invalidate_provider(self, provider_id: str, reason: str,
                            generation: Optional[int] = None) -> None:
        """Invalidate only the affected stream; old connection callbacks cannot fill."""
        with self._risk_lock:
            if generation is not None:
                self._provider_generations[provider_id] = int(generation)
            configured = self.config.providers.get(provider_id)
            ids = self._provider_instruments.get(provider_id,
                configured.instrument_ids if configured is not None else ())
            for instrument_id in ids:
                runtime = self._runtime[instrument_id]
                self._cancel_instrument_quotes(runtime, "provider_unavailable:" + str(reason))
                runtime.bids = runtime.asks = ()
                runtime.book_synchronized = False
                runtime.last_quote_event_ts = runtime.last_quote_receive_ts = None
                runtime.last_trade_event_ts = runtime.last_trade_receive_ts = None
                runtime.last_event_by_kind.clear()
                runtime.roll_liquidity.clear()
                runtime.last_sequence_by_kind.clear()
                self._seen_event_ids[instrument_id].clear()
                self._seen_event_order[instrument_id].clear()
                runtime.market_open = runtime.spec.asset_class == AssetClass.CRYPTO
                runtime.market_status = "provider_unavailable: " + str(reason)

    def on_market_event(self, event: MarketDataEvent, provider_id: Optional[str] = None,
                        generation: Optional[int] = None) -> bool:
        """Serialize provider updates with risk checks and paper fills."""
        with self._risk_lock:
            if provider_id is not None and generation is not None:
                current = self._provider_generations.get(provider_id)
                if current is not None and int(generation) < current:
                    return False
                self._provider_generations[provider_id] = int(generation)
            return self._apply_market_event(event)

    def _apply_market_event(self, event: MarketDataEvent) -> bool:
        """Validate, apply, account, and route one normalized event."""
        if not isinstance(event, MarketDataEvent):
            raise TypeError("event must be MarketDataEvent")
        runtime = self._runtime.get(event.instrument_id)
        if runtime is None:
            self.emit("provider_error", {"source": event.source, "message": "unconfigured instrument"})
            return False
        now = float(self.clock())
        if event.receive_ts > now + self.max_clock_skew_seconds:
            return self._reject_event(runtime, "receive_timestamp_in_future")
        if now - event.receive_ts > self.config.market_data_stale_after:
            return self._reject_event(runtime, "stale_event_on_arrival")
        if event.event_ts > event.receive_ts + self.max_clock_skew_seconds + 1e-9:
            return self._reject_event(runtime, "source_clock_skew_exceeds_configured_bound")
        if event.event_ts > now + self.max_clock_skew_seconds:
            return self._reject_event(runtime, "source_timestamp_in_future")
        if now - event.event_ts > self.config.market_data_stale_after + self.max_clock_skew_seconds:
            return self._reject_event(runtime, "stale_source_event")
        if event.kind == MarketEventKind.TRADE:
            if event.side is None or event.price is None:
                return self._reject_event(runtime, "trade_missing_price_size_or_side")
            if runtime.spec.asset_class != AssetClass.FUTURE and event.price <= 0:
                return self._reject_event(runtime, "nonpositive_trade_price")
            if not runtime.spec.is_valid_price(event.price):
                return self._reject_event(runtime, "trade_off_tick")
        elif event.kind == MarketEventKind.BOOK_SNAPSHOT:
            if not self._valid_book(runtime, event.bids, event.asks):
                runtime.book_synchronized = False
                self._cancel_instrument_quotes(runtime, "invalid_book")
                return self._reject_event(runtime, "empty_crossed_or_off_tick_book")

        event_id = self._event_identity(event)
        with self._lock:
            if event_id in self._seen_event_ids[event.instrument_id]:
                return False
            stream_key = (event.source, self._sequence_stream(event.kind))
            prior = runtime.last_event_by_kind.get(stream_key)
            if prior and (event.event_ts < prior[0] or event.receive_ts < prior[1]):
                return False
            previous_seq = runtime.last_sequence_by_kind.get(stream_key)
            if event.sequence is not None and previous_seq is not None and event.sequence <= previous_seq:
                return False
            if (event.kind == MarketEventKind.BOOK_DELTA and event.sequence is not None
                    and previous_seq is not None and event.sequence != previous_seq + 1):
                runtime.book_synchronized = False
                self._cancel_instrument_quotes(runtime, "book_sequence_gap")
                return self._reject_event(runtime, "book_sequence_gap_requires_snapshot")

        if any(action["instrument_id"] == event.instrument_id and action["effective_ts"] > now
               for action in self.account_actions.due(event.event_ts)):
            return self._reject_event(runtime, "account_action_processing_boundary_not_reached")
        due = self.account_actions.due(now)
        market_event = event.kind in {
            MarketEventKind.BOOK_SNAPSHOT, MarketEventKind.BOOK_DELTA, MarketEventKind.TRADE}
        if market_event:
            mark_time = self.ledger._positions[event.instrument_id].mark_timestamp
            if (event.kind != MarketEventKind.TRADE
                    and mark_time is not None and event.receive_ts < mark_time):
                return self._reject_event(runtime, "market_event_precedes_valuation_epoch")
            for action in due:
                if (action["instrument_id"] == event.instrument_id
                        and action["action_type"] in {"split", "settlement", "valuation_mark"}
                        and (event.receive_ts < action["effective_ts"] or (
                            action["action_type"] == "split" and event.event_ts < action["effective_ts"]))):
                    return self._reject_event(runtime, "market_event_precedes_account_action_epoch")
        if due:
            self._process_account_actions_locked(now)
        if market_event and self._split_reconciliation_required(event.instrument_id):
            self._cancel_instrument_quotes(runtime, "split_reconciliation_required")
            return self._reject_event(runtime, "split_reconciliation_required")
        if (runtime.book_fresh_after_ts is not None
                and event.kind in {MarketEventKind.BOOK_SNAPSHOT, MarketEventKind.BOOK_DELTA, MarketEventKind.TRADE}
                and min(event.event_ts, event.receive_ts) < runtime.book_fresh_after_ts):
            return self._reject_event(runtime, "market_event_precedes_equity_split")
        with self._lock:
            self._remember_event(runtime, event_id)
            if event.sequence is not None:
                runtime.last_sequence_by_kind[stream_key] = event.sequence
            runtime.last_event_by_kind[stream_key] = (event.event_ts, event.receive_ts, event.source)

        accepted = False
        if event.kind == MarketEventKind.BOOK_SNAPSHOT:
            accepted = self._handle_book_snapshot(runtime, event)
        elif event.kind == MarketEventKind.BOOK_DELTA:
            accepted = self._handle_book_delta(runtime, event)
        elif event.kind == MarketEventKind.TRADE:
            accepted = self._handle_trade(runtime, event)
        elif event.kind == MarketEventKind.STATUS:
            accepted = self._handle_status(runtime, event)
        elif event.kind == MarketEventKind.SESSION:
            accepted = self._handle_session(runtime, event)

        if accepted:
            self._sync_strategy_inventory(runtime)
            self._refresh_quotes_for(runtime, event)
            self._emit_dashboard_state()
        return accepted

    def _event_identity(self, event: MarketDataEvent) -> str:
        if event.sequence is not None:
            material = "{}|{}|{}|{}".format(event.source, event.instrument_id, event.kind.value, event.sequence)
        else:
            material = (
                f"{event.source}|{event.instrument_id}|{event.kind.value}|{event.event_ts}|"
                f"{event.price}|{event.quantity}|{(event.bids, event.asks)}|{event.status}|"
                f"{event.session}|{event.side.name if event.side is not None else None}"
            )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _sequence_stream(kind: MarketEventKind) -> str:
        if kind in {MarketEventKind.BOOK_SNAPSHOT, MarketEventKind.BOOK_DELTA}:
            return "book"
        return kind.value

    def _remember_event(self, runtime: _InstrumentRuntime, event_id: str) -> None:
        seen = self._seen_event_ids[runtime.spec.instrument_id]
        order = self._seen_event_order[runtime.spec.instrument_id]
        if len(order) == order.maxlen:
            expired = order.popleft()
            seen.discard(expired)
        order.append(event_id)
        seen.add(event_id)

    def _reject_event(self, runtime: _InstrumentRuntime, reason: str) -> bool:
        self.emit("market_data_rejected", {"instrument_id": runtime.spec.instrument_id, "reason": reason})
        return False

    @staticmethod
    def _valid_book(runtime: _InstrumentRuntime, bids, asks) -> bool:
        if not bids or not asks:
            return False
        if any(not runtime.spec.is_valid_price(price) or size <= 0 for price, size in (*bids, *asks)):
            return False
        return float(bids[0][0]) <= float(asks[0][0])

    def _handle_book_snapshot(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> bool:
        if (runtime.book_fresh_after_ts is not None
                and event.event_ts < runtime.book_fresh_after_ts):
            runtime.book_synchronized = False
            self._cancel_instrument_quotes(runtime, "post_split_book_required")
            return self._reject_event(runtime, "book_snapshot_precedes_equity_split")
        bids = tuple(sorted(((float(p), float(q)) for p, q in event.bids), key=lambda row: row[0], reverse=True))
        asks = tuple(sorted(((float(p), float(q)) for p, q in event.asks), key=lambda row: row[0]))
        if not self._valid_book(runtime, bids, asks):
            runtime.book_synchronized = False
            self._cancel_instrument_quotes(runtime, "invalid_book")
            return self._reject_event(runtime, "empty_crossed_or_off_tick_book")
        with self._lock:
            runtime.bids, runtime.asks = bids, asks
            runtime.book_synchronized = True
            runtime.last_quote_event_ts = event.event_ts
            runtime.last_quote_receive_ts = event.receive_ts
            self._refresh_roll_liquidity(runtime)
        self._mark_instrument_and_fx(runtime, event)
        self._fill_from_fx_quote(runtime, event)
        self.emit("instrument_market", self._market_payload(runtime))
        return True

    def _handle_book_delta(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> bool:
        with self._lock:
            # A delta cannot re-establish a book after a split or a sequence
            # gap. It needs a fresh post-action snapshot first.
            if (not runtime.book_synchronized or not runtime.bids or not runtime.asks
                    or (runtime.book_fresh_after_ts is not None
                        and (runtime.last_quote_event_ts is None
                              or runtime.last_quote_event_ts < runtime.book_fresh_after_ts))):
                runtime.book_synchronized = False
                self._cancel_instrument_quotes(runtime, "book_snapshot_required")
                return self._reject_event(runtime, "book_delta_without_snapshot")
            bid_map, ask_map = dict(runtime.bids), dict(runtime.asks)
            for side_map, levels in ((bid_map, event.bids), (ask_map, event.asks)):
                for price, quantity in levels:
                    if not runtime.spec.is_valid_price(price):
                        runtime.book_synchronized = False
                        self._cancel_instrument_quotes(runtime, "invalid_book_delta")
                        return self._reject_event(runtime, "book_delta_off_tick")
                    if quantity == 0:
                        side_map.pop(price, None)
                    else:
                        side_map[price] = quantity
            bids = tuple(sorted(bid_map.items(), key=lambda row: row[0], reverse=True))
            asks = tuple(sorted(ask_map.items(), key=lambda row: row[0]))
            if not self._valid_book(runtime, bids, asks):
                runtime.book_synchronized = False
                self._cancel_instrument_quotes(runtime, "invalid_book_delta")
                return self._reject_event(runtime, "book_delta_crossed_or_empty")
            runtime.bids, runtime.asks = bids, asks
            runtime.last_quote_event_ts = event.event_ts
            runtime.last_quote_receive_ts = event.receive_ts
            self._refresh_roll_liquidity(runtime)
        self._mark_instrument_and_fx(runtime, event)
        self._fill_from_fx_quote(runtime, event)
        self.emit("instrument_market", self._market_payload(runtime))
        return True

    def _refresh_roll_liquidity(self, runtime: _InstrumentRuntime) -> None:
        """Credit observed depth increases, never repeated unchanged touch sizes."""
        for side, levels in ((Side.BUY, runtime.asks), (Side.SELL, runtime.bids)):
            if not levels:
                runtime.roll_liquidity.pop(side, None)
                continue
            price, displayed = levels[0]
            previous = runtime.roll_liquidity.get(side)
            remaining = displayed
            if previous is not None and previous[0] == price:
                remaining = min(displayed, previous[2] + max(0.0, displayed - previous[1]))
            runtime.roll_liquidity[side] = (price, displayed, remaining)

    def _handle_trade(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> bool:
        if event.price is None or event.quantity is None or event.side is None:
            return self._reject_event(runtime, "trade_missing_price_size_or_side")
        if runtime.spec.asset_class != AssetClass.FUTURE and event.price <= 0:
            return self._reject_event(runtime, "nonpositive_trade_price")
        if not runtime.spec.is_valid_price(event.price):
            return self._reject_event(runtime, "trade_off_tick")
        runtime.last_trade_event_ts = event.event_ts
        runtime.last_trade_receive_ts = event.receive_ts
        runtime.recent_public_trades.append({
            "instrument_id": event.instrument_id, "source": event.source,
            "timestamp": event.event_ts, "receive_timestamp": event.receive_ts,
            "price": event.price, "quantity": event.quantity,
            "side": event.side.name.lower(), "sequence": event.sequence,
        })
        if self.is_running and self._instrument_ready(runtime, float(self.clock())):
            self._fill_from_trade(runtime, event)
        self.emit("public_trade", {
            "instrument_id": event.instrument_id, "source": event.source,
            "price": event.price, "quantity": event.quantity, "side": event.side.name.lower(),
            "event_timestamp": event.event_ts, "sequence": event.sequence,
        })
        return True

    def _handle_status(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> bool:
        status = str(event.status or "unknown").strip().lower()
        runtime.market_status = status
        if status in self.OPEN_STATES:
            runtime.market_open = True
        elif status in self.CLOSED_STATES:
            runtime.market_open = False
        else:
            runtime.market_open = False
        if not runtime.market_open:
            self._cancel_instrument_quotes(runtime, "market_{}".format(status))
        return True

    def _handle_session(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> bool:
        session = str(event.session or "unknown").strip().lower()
        runtime.market_status = session
        runtime.market_open = session in self.OPEN_STATES
        if not runtime.market_open:
            self._cancel_instrument_quotes(runtime, "session_{}".format(session))
        return True

    def _mark_instrument_and_fx(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> None:
        mid = runtime.mid_price
        if mid is None:
            return
        self.ledger.mark_instrument(runtime.spec.instrument_id, mid, event.receive_ts,
                                    processing_timestamp=float(self.clock()))
        observe_source = getattr(runtime.strategy, "observe_market_with_source", None)
        if callable(observe_source):
            observe_source(mid, event.receive_ts, event.event_ts)
        else:
            observe = getattr(runtime.strategy, "observe_market", None)
            if callable(observe):
                observe(mid, event.receive_ts)
        if runtime.spec.asset_class == AssetClass.FX and mid > 0:
            self._fx_edges[(runtime.spec.base_currency, runtime.spec.quote_currency)] = (
                mid, event.receive_ts, self.config.max_mark_age_seconds)
            self._conversion_mark_sources[runtime.spec.base_currency] = event.source
            self._conversion_mark_sources[runtime.spec.quote_currency] = event.source
            self._refresh_fx_marks(event.receive_ts)

    def _refresh_fx_marks(self, timestamp: float) -> None:
        reporting = self.config.reporting_currency
        graph: Dict[str, List[Tuple[str, float, float]]] = {}
        for (base, quote), (mid, mark_ts, max_age) in self._fx_edges.items():
            age = timestamp - mark_ts
            if age < 0 or age > max_age or mid <= 0:
                continue
            graph.setdefault(base, []).append((quote, mid, mark_ts))
            graph.setdefault(quote, []).append((base, 1.0 / mid, mark_ts))
        currencies = set(graph) - {reporting}
        for currency in currencies:
            queue = deque([(currency, 1.0, timestamp)])
            seen = {currency}
            rate = None
            rate_ts = None
            while queue:
                current, value, oldest = queue.popleft()
                if current == reporting:
                    rate, rate_ts = value, oldest
                    break
                for neighbor, edge_rate, edge_ts in graph.get(current, ()):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, value * edge_rate, min(oldest, edge_ts)))
            if rate is not None and currency != reporting:
                try:
                    max_age = min((edge[2] for pair, edge in self._fx_edges.items()
                                   if currency in pair), default=self.config.max_mark_age_seconds)
                    self.ledger.set_fx_mark(currency, rate, rate_ts, max_age_seconds=max_age)
                except ValueError:
                    # Ignore a stale/out-of-order conversion mark; portfolio
                    # snapshots will fail closed until an admissible mark arrives.
                    pass

    def _instrument_ready(self, runtime: _InstrumentRuntime, now: float) -> bool:
        return self._instrument_readiness_code(runtime, now) == "ready"

    def _instrument_readiness_code(self, runtime: _InstrumentRuntime, now: float) -> str:
        """Use the execution predicate for both feed gating and UI explanations."""
        if not runtime.config.enabled:
            runtime.market_status = "quoting_disabled"
            return "quoting_disabled"
        if runtime.spec.instrument_id in self._rolled_futures:
            runtime.market_open = False
            runtime.market_status = "rolled"
            return "rolled"
        if runtime.spec.expiry is not None and now >= runtime.spec.expiry.timestamp():
            runtime.market_open = False
            runtime.market_status = "expired"
            return "expired"
        if not runtime.market_open:
            return "market_halted" if runtime.market_status == "halted" else "session_not_open"
        if not runtime.book_synchronized:
            return "book_unsynchronized"
        if runtime.mid_price is None:
            return "quote_missing"
        quote_receive = runtime.last_quote_receive_ts
        if quote_receive is None:
            return "quote_missing"
        quote_age = now - quote_receive
        if not 0 <= quote_age <= self.config.market_data_stale_after:
            return "quote_clock_skew" if quote_age < 0 else "quote_stale"
        if runtime.config.fill_model == "trade_print_queue":
            trade_receive = runtime.last_trade_receive_ts
            if trade_receive is None:
                return "trades_missing"
            trade_age = now - trade_receive
            if not 0 <= trade_age <= self.config.market_data_stale_after:
                return "trade_clock_skew" if trade_age < 0 else "trades_stale"
        return "ready"

    def _instrument_readiness(self, runtime: _InstrumentRuntime, now: float) -> Dict[str, Any]:
        code = self._instrument_readiness_code(runtime, now)
        messages = {
            "ready": "Required market feeds are fresh; accounting and risk checks apply separately.",
            "quoting_disabled": "Quoting is disabled for this instrument in the configuration.",
            "rolled": "This contract has been rolled; select the replacement contract.",
            "expired": "This contract has expired.",
            "market_halted": "Trading is halted for this instrument.",
            "session_not_open": "Market session is not open ({}).".format(runtime.market_status),
            "book_unsynchronized": "The quote book needs a fresh synchronized snapshot.",
            "quote_missing": "Waiting for a two-sided quote and its receive timestamp.",
            "quote_clock_skew": "The latest quote receive timestamp is ahead of the account clock.",
            "quote_stale": "The quote feed is older than the configured freshness limit.",
            "trades_missing": "This fill model requires public trade prints; none have arrived yet.",
            "trade_clock_skew": "The latest trade receive timestamp is ahead of the account clock.",
            "trades_stale": "The public trade feed is older than the configured freshness limit.",
        }
        return {"code": code, "message": messages[code],
                "requires_trades": runtime.config.fill_model == "trade_print_queue"}

    def _refresh_all_quotes(self) -> None:
        if not self.is_running:
            return
        now = float(self.clock())
        for runtime in self._runtime.values():
            if self._instrument_ready(runtime, now):
                self._generate_quotes(runtime, now)

    def _refresh_quotes_for(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> None:
        now = float(self.clock())
        if not self.is_running:
            return
        if not self._instrument_ready(runtime, now):
            self._cancel_instrument_quotes(runtime, "feed_or_session_not_ready")
            return
        if event.kind in {MarketEventKind.STATUS, MarketEventKind.SESSION}:
            if (runtime.market_open
                    and now - runtime.last_strategy_receive_ts >= self.quote_refresh_seconds
                    and not self._quote_within_eligibility_horizon(runtime, now)):
                self._generate_quotes(runtime, now)
            return
        if now - runtime.last_strategy_receive_ts >= self.quote_refresh_seconds:
            # Keep a quote alive through the configured provider-clock
            # uncertainty window. Replacing it sooner would prevent a
            # conservative post-order event-time gate from ever becoming
            # eligible on feeds with source clock skew and delivery latency.
            # The observed receive-minus-event delay is included because a
            # valid post-order source event may arrive after the local order
            # age has exceeded the configured clock-skew allowance.
            if self._quote_within_eligibility_horizon(runtime, now):
                return
            self._generate_quotes(runtime, now)

    def _quote_within_eligibility_horizon(self, runtime: _InstrumentRuntime,
                                          now: float) -> bool:
        eligibility_horizon = (self.max_clock_skew_seconds
                               + self._observed_source_lag(runtime)
                               + runtime.config.maker_latency_ms / 1000.0)
        return any(now - working.accepted_receive_ts < eligibility_horizon
                   for working in runtime.active_orders.values())

    @staticmethod
    def _observed_source_lag(runtime: _InstrumentRuntime) -> float:
        """Return the largest observed positive source-to-receive delay."""
        return max((max(0.0, receive_ts - event_ts)
                    for event_ts, receive_ts, _source in runtime.last_event_by_kind.values()),
                   default=0.0)

    def _generate_quotes(self, runtime: _InstrumentRuntime, now: float) -> None:
        mid = runtime.mid_price
        if mid is None or not self._instrument_ready(runtime, now):
            return
        self._cancel_instrument_quotes(runtime, "requote")
        position = self.ledger.get_position(runtime.spec.instrument_id, as_of=now).quantity
        runtime.strategy.sync_inventory(position)
        orders = runtime.strategy.on_market_event({
            "instrument_id": runtime.spec.instrument_id,
            "time": now,
            "mid_price": mid,
            "best_bid": runtime.bids[0][0],
            "best_ask": runtime.asks[0][0],
            "market_data_ready": True,
            "market_open": runtime.market_open,
        })
        runtime.last_strategy_receive_ts = now
        for order in orders:
            order = replace(order, order_id=self.id_factory("order"))
            # This engine's ordinary fill model is resting/maker only. Keep
            # generated inventory skews from becoming unmodeled taker orders.
            if order.side == Side.BUY:
                passive = min(float(order.price), runtime.asks[0][0] - runtime.spec.tick_size)
            else:
                passive = max(float(order.price), runtime.bids[0][0] + runtime.spec.tick_size)
            order = replace(order, price=Price(runtime.spec.round_price(passive, order.side)))
            closing_side = (Side.SELL if position > 1e-12 else
                            Side.BUY if position < -1e-12 else None)
            is_reducing_side = order.side == closing_side
            if self.circuit_breaker_active and not is_reducing_side:
                continue
            if is_reducing_side:
                requested = min(float(order.quantity), abs(position),
                                 runtime.config.max_order_quantity)
                units = math.floor((requested + 1e-12) / runtime.spec.quantity_step)
                reduced_quantity = units * runtime.spec.quantity_step
                if reduced_quantity + 1e-12 < runtime.spec.min_quantity:
                    continue
                if reduced_quantity + 1e-12 < float(order.quantity):
                    order = replace(order, quantity=Quantity(reduced_quantity))
            self._accept_order(runtime, order, now)

    def _accept_order(self, runtime: _InstrumentRuntime, order: Order, now: float) -> bool:
        with self._risk_lock:
            return self._accept_order_unlocked(runtime, order, now)

    def _accept_order_unlocked(self, runtime: _InstrumentRuntime, order: Order,
                               now: float) -> bool:
        spec, item = runtime.spec, runtime.config
        quantity = float(order.quantity)
        price = float(order.price) if order.price is not None else float("nan")
        if order.instrument_id != spec.instrument_id or order.order_type != OrderType.LIMIT:
            return self._reject_order(runtime, order, "invalid_instrument_or_order_type")
        if not spec.is_valid_quantity(quantity) or quantity > item.max_order_quantity + 1e-12:
            return self._reject_order(runtime, order, "quantity_limit_or_lot_violation")
        if not spec.is_valid_price(price):
            return self._reject_order(runtime, order, "price_tick_violation")
        if spec.asset_class != AssetClass.FUTURE and price <= 0:
            return self._reject_order(runtime, order, "nonpositive_limit_price")
        if self._split_reconciliation_required(spec.instrument_id):
            return self._reject_order(runtime, order, "split_reconciliation_required")
        cutoff = now - 60.0
        while self._orders_per_minute and self._orders_per_minute[0] <= cutoff:
            self._orders_per_minute.popleft()
        try:
            position = self.ledger.get_position(spec.instrument_id, as_of=now).quantity
        except (ValueError, KeyError) as exc:
            return self._reject_order(runtime, order, "instrument_mark_unavailable: {}".format(exc))
        risk_reducing = self._is_strictly_risk_reducing(runtime, order, position)
        if spec.instrument_id in self._accounting_blocks and not risk_reducing:
            return self._reject_order(runtime, order, self._accounting_blocks[spec.instrument_id])
        try:
            snapshot = self.ledger.snapshot(as_of=now)
        except ValueError as exc:
            if not risk_reducing:
                return self._reject_order(runtime, order, "account_mark_unavailable: {}".format(exc))
            # A stale mark elsewhere in the portfolio must not trap an
            # operator's strictly reducing close order. All risk-increasing
            # orders still require a fresh, reconciled account snapshot.
            snapshot = None
        if (len(self._orders_per_minute) >= self.config.max_orders_per_minute
                and not risk_reducing):
            return self._reject_order(runtime, order, "portfolio_order_rate_limit")
        if snapshot is not None:
            self._enforce_account_risk(snapshot, now)
        if self.circuit_breaker_active and (
                not risk_reducing or self.circuit_breaker_reason == "paper_history_persistence_failed"):
            return self._reject_order(runtime, order, "circuit_breaker_active")

        buys_pending = sum(row.remaining_quantity for row in runtime.active_orders.values()
                           if row.order.side == Side.BUY)
        sells_pending = sum(row.remaining_quantity for row in runtime.active_orders.values()
                            if row.order.side == Side.SELL)
        proposed_long = position + buys_pending + (quantity if order.side == Side.BUY else 0.0)
        proposed_short = position - sells_pending - (quantity if order.side == Side.SELL else 0.0)
        if (max(abs(proposed_long), abs(proposed_short)) > item.max_position_quantity + 1e-12
                and not risk_reducing):
            return self._reject_order(runtime, order, "instrument_position_limit")
        if min(proposed_long, proposed_short) < -1e-12 and not item.allow_short and not risk_reducing:
            return self._reject_order(runtime, order, "shorting_not_configured")
        cash_security_short = spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}
        deliverable_fx_short = (spec.asset_class == AssetClass.FX
                                and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE)
        if (min(proposed_long, proposed_short) < -1e-12
                and (cash_security_short or deliverable_fx_short) and not risk_reducing):
            if spec.borrow_limit is None or abs(min(proposed_long, proposed_short)) > spec.borrow_limit + 1e-12:
                return self._reject_order(runtime, order, "borrow_limit")

        if not risk_reducing:
            if snapshot is None:
                return self._reject_order(runtime, order, "account_mark_unavailable")
            notional = quantity * abs(price) * spec.contract_multiplier
            risk_currency = (spec.settlement_currency if spec.asset_class == AssetClass.FUTURE
                             else spec.quote_currency)
            conversion = self._currency_rate(risk_currency, now)
            if conversion is None:
                return self._reject_order(runtime, order, "missing_or_stale_currency_conversion")
            notional_reporting = notional * conversion
            if notional_reporting > item.max_notional + 1e-12:
                return self._reject_order(runtime, order, "instrument_notional_limit")
            try:
                projected_gross = self._projected_gross_exposure(
                    snapshot, now, extra_order=(runtime, order))
            except ValueError as exc:
                return self._reject_order(
                    runtime, order, "portfolio_conversion_unavailable: {}".format(exc))
            if projected_gross > self.config.max_gross_exposure + 1e-12:
                return self._reject_order(runtime, order, "portfolio_gross_exposure_limit")
            try:
                projected_margin = self._projected_initial_margin_reporting(
                    snapshot, now, extra_order=(runtime, order))
            except ValueError as exc:
                return self._reject_order(
                    runtime, order, "projected_margin_unavailable: {}".format(exc))
            if snapshot.equity + 1e-9 < projected_margin:
                return self._reject_order(runtime, order, "projected_initial_margin_exceeds_equity")
            try:
                proposal = self._portfolio_proposal(snapshot, now, extra_order=(runtime, order))
            except ValueError as exc:
                return self._reject_order(runtime, order, "portfolio_risk_input_unavailable: " + str(exc))
            fee = item.fees.calculate(quantity, price, spec.contract_multiplier, maker=True)
            fee_rate = self._currency_rate(item.fees.currency, now)
            if fee_rate is None:
                return self._reject_order(runtime, order, "missing_or_stale_fee_conversion")
            proposal = replace(proposal, fees_reporting=proposal.fees_reporting + fee * fee_rate)
            decision = self.risk_evaluator.validate(proposal, RiskContext(
                self.config, snapshot.equity, snapshot.available_cash_by_currency,
                self.circuit_breaker_active, now))
            if not decision.accepted:
                return self._reject_order(runtime, order, decision.reason)

        if runtime.bids and runtime.asks and (
                (order.side == Side.BUY and price >= runtime.asks[0][0])
                or (order.side == Side.SELL and price <= runtime.bids[0][0])):
            return self._reject_order(runtime, order, "marketable_order_requires_taker_model")

        margin_order = (spec.asset_class == AssetClass.FUTURE or
                        (spec.asset_class == AssetClass.FX
                         and spec.fx_accounting_mode == FXAccountingMode.ROLLING))
        reservation_id = (self._margin_reservation_id(spec.instrument_id) if margin_order
                          else "order:{}".format(order.order_id))
        fee_reservation_id = None
        try:
            if (margin_order and not risk_reducing) or (not margin_order and (
                    item.fees.currency != spec.quote_currency or order.side == Side.SELL)):
                # Fees can use a different currency from margin. A strict
                # close can instead use the collateral it releases.
                fee = item.fees.calculate(quantity, price, spec.contract_multiplier, maker=True)
                if risk_reducing and order.side == Side.SELL and item.fees.currency == spec.quote_currency:
                    fee = max(0.0, fee - quantity * price * spec.contract_multiplier)
                if fee:
                    fee_reservation_id = f"order-fee:{order.order_id}"
                    self.ledger.reserve_cash(fee_reservation_id, item.fees.currency, fee)
            self._reserve_for_order(runtime, order, reservation_id)
        except (ValueError, KeyError) as exc:
            if fee_reservation_id is not None:
                self.ledger.release_reservation(fee_reservation_id)
            return self._reject_order(runtime, order, "reservation_failed: {}".format(exc))

        accepted = _WorkingOrder(
            order=order,
            # Translate the latest relevant source event into local time using
            # its observed clock lag. Never move eligibility into the future:
            # that starves orders when the quote refresh interval is shorter
            # than the allowed provider clock skew.
            accepted_event_cutoffs=self._acceptance_event_cutoffs(runtime, now),
            accepted_receive_ts=now,
            remaining_quantity=quantity,
            reservation_id=reservation_id,
            fee_reservation_id=fee_reservation_id,
        )
        with self._lock:
            runtime.active_orders[order.order_id] = accepted
            runtime.total_orders += 1
            self.total_orders_sent += 1
            self._orders_per_minute.append(now)
        try:
            self._persist_event("order_accepted", order.order_id, {
                "instrument_id": spec.instrument_id, "order_id": order.order_id,
                "side": order.side.name, "quantity": quantity, "price": price,
            }, now)
        except Exception:
            runtime.total_orders -= 1
            self.total_orders_sent -= 1
            if self._orders_per_minute and self._orders_per_minute[-1] == now:
                self._orders_per_minute.pop()
            raise
        self.emit("order_accepted", {
            "instrument_id": spec.instrument_id, "order_id": order.order_id,
            "side": order.side.name, "quantity": quantity, "price": price,
            "execution_mode": "local_paper",
        })
        return True

    @staticmethod
    def _margin_reservation_id(instrument_id: str) -> str:
        return "pending-margin:{}".format(instrument_id)

    def _is_strictly_risk_reducing(self, runtime: _InstrumentRuntime,
                                   order: Order, position: float) -> bool:
        working = runtime.active_orders.get(order.order_id)
        quantity = working.remaining_quantity if working is not None else float(order.quantity)
        pending = [row for row in runtime.active_orders.values() if row is not working]
        if position > 1e-12 and order.side == Side.SELL and quantity <= position + 1e-12:
            same_direction = sum(row.remaining_quantity for row in pending
                                 if row.order.side == Side.SELL)
            has_increasing_order = any(row.order.side == Side.BUY for row in pending)
            return not has_increasing_order and same_direction + quantity <= position + 1e-12
        if position < -1e-12 and order.side == Side.BUY and quantity <= abs(position) + 1e-12:
            same_direction = sum(row.remaining_quantity for row in pending
                                 if row.order.side == Side.BUY)
            has_increasing_order = any(row.order.side == Side.SELL for row in pending)
            return not has_increasing_order and same_direction + quantity <= abs(position) + 1e-12
        return False

    def _enforce_account_risk(self, snapshot: PortfolioSnapshot, now: float) -> bool:
        if snapshot.equity + 1e-9 < snapshot.maintenance_margin_required:
            self._trip_circuit_breaker("maintenance_margin_breach", now)
            return True
        if self._daily_loss_breached(snapshot, now):
            self._trip_circuit_breaker("portfolio_loss_limit", now)
            return True
        return False

    def submit_paper_order(self, instrument_id: str, side: Any, quantity: float,
                           price: float, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit a local manual limit order through the normal risk evaluator.

        The application service durably caches all request outcomes. An
        accepted order also has a stable event ID for direct library retries.
        """
        with self._risk_lock:
            runtime = self._runtime.get(instrument_id)
            if runtime is None:
                raise ValueError("instrument is not configured")
            try:
                normalized = side if isinstance(side, Side) else Side[str(side).upper()]
            except KeyError as exc:
                raise ValueError("paper order side must be BUY or SELL") from exc
            if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
                raise ValueError("manual paper request ID must be a non-empty string")
            order_id = "manual:" + request_id if request_id else self.id_factory("manual")
            expected = {"instrument_id": instrument_id, "order_id": order_id,
                        "side": normalized.name, "quantity": float(quantity), "price": float(price)}
            lookup = getattr(self.state_manager, "get_cross_asset_event", None)
            prior = lookup(self.account_id, order_id) if callable(lookup) else None
            if prior is not None:
                payload = prior.get("payload", prior)
                if any(payload.get(key) != value for key, value in expected.items()):
                    raise ValueError("manual request ID was reused with different order inputs")
                return {"accepted": True, "order_id": order_id, "duplicate": True}
            if order_id in runtime.active_orders:
                existing = runtime.active_orders[order_id].order
                if existing.side != normalized or float(existing.quantity) != float(quantity) or float(existing.price) != float(price):
                    raise ValueError("manual request ID was reused with different order inputs")
                return {"accepted": True, "order_id": order_id, "duplicate": True}
            now = float(self.clock())
            if not self._instrument_ready(runtime, now):
                raise ValueError("manual paper order requires fresh open market data")
            order = Order(order_id=order_id, side=normalized, order_type=OrderType.LIMIT,
                quantity=Quantity(quantity), price=Price(price), timestamp=Timestamp(now),
                trader_id="manual_paper", instrument_id=instrument_id)
            accepted = self._accept_order(runtime, order, now)
            return {"accepted": accepted, "order_id": order_id, "duplicate": False}

    def _trip_circuit_breaker(self, reason: str, now: float) -> None:
        already_active = self.circuit_breaker_active and self.circuit_breaker_reason == reason
        self.circuit_breaker_active = True
        self.circuit_breaker_reason = reason
        with self._lock:
            self.is_running = False
        self._cancel_all_quotes(reason=reason)
        self._save_checkpoint(now)
        if not already_active:
            payload = {"is_running": False, "circuit_breaker_active": True,
                       "circuit_breaker_reason": reason, "execution_mode": "local_paper"}
            self.emit("strategy_status", payload)
            self._emit_dashboard_state()

    def _acceptance_event_cutoffs(self, runtime: _InstrumentRuntime,
                                  now: float) -> Dict[str, float]:
        """Return a conservative provider-time boundary for order acceptance.

        Event and receive timestamps cannot distinguish source clock offset
        from variable network delay. Treat the configured skew allowance as
        the maximum uncertainty and require a print to be newer than that
        bound. Instruments with no source observed yet fail closed for fills.
        """
        sources = {source for _event_ts, _receive_ts, source
                   in runtime.last_event_by_kind.values()}
        cutoff = now + self.max_clock_skew_seconds + runtime.config.maker_latency_ms / 1000.0
        return {source: cutoff for source in sources}

    def _reserve_for_order(self, runtime: _InstrumentRuntime, order: Order, reservation_id: str) -> None:
        spec, item = runtime.spec, runtime.config
        quantity, price = float(order.quantity), float(order.price)
        if spec.asset_class == AssetClass.FUTURE or (
                spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.ROLLING):
            self._refresh_margin_reservation(runtime, additional_order=order, now=float(self.clock()))
            return
        if order.side == Side.BUY:
            notional = quantity * price * spec.contract_multiplier
            fee = item.fees.calculate(quantity, price, spec.contract_multiplier, maker=True)
            self.ledger.reserve_cash(reservation_id, spec.quote_currency,
                                     notional + (fee if item.fees.currency == spec.quote_currency else 0.0))
        elif spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE:
            if item.allow_short:
                return
            self.ledger.reserve_cash(
                reservation_id, spec.base_currency, quantity * spec.contract_multiplier)
        elif order.side == Side.SELL:
            # Inventory is bounded above in unit space; no cash is consumed by
            # a sell until a local fill is posted.
            return

    def _refresh_margin_reservation(self, runtime: _InstrumentRuntime,
                                    additional_order: Optional[Order] = None,
                                    now: Optional[float] = None) -> float:
        """Reserve the worst-case margin increase across this instrument's quotes."""
        spec = runtime.spec
        stamp = float(self.clock()) if now is None else float(now)
        position = self.ledger.get_position(spec.instrument_id, as_of=stamp)
        buys = sum(row.remaining_quantity for row in runtime.active_orders.values()
                   if row.order.side == Side.BUY)
        sells = sum(row.remaining_quantity for row in runtime.active_orders.values()
                    if row.order.side == Side.SELL)
        prices = [abs(float(row.order.price)) for row in runtime.active_orders.values()]
        if additional_order is not None:
            if additional_order.side == Side.BUY:
                buys += float(additional_order.quantity)
            else:
                sells += float(additional_order.quantity)
            prices.append(abs(float(additional_order.price)))

        worst_quantity = max(abs(position.quantity + buys), abs(position.quantity - sells))
        if spec.asset_class == AssetClass.FUTURE:
            projected_margin = worst_quantity * float(spec.initial_margin_per_contract)
        elif worst_quantity <= 1e-12:
            projected_margin = 0.0
        else:
            mark = position.mark_price
            if mark is not None:
                prices.append(abs(float(mark)))
            elif runtime.mid_price is not None:
                prices.append(abs(float(runtime.mid_price)))
            if not prices:
                raise ValueError("rolling FX margin needs a current mark or order price")
            initial_rate = spec.initial_margin_rate
            if initial_rate is None or initial_rate <= 0:
                raise ValueError("rolling FX requires explicit initial margin")
            projected_margin = worst_quantity * max(prices) * spec.contract_multiplier * initial_rate

        additional_margin = max(0.0, projected_margin - position.initial_margin_required)
        return self.ledger.set_margin_reservation(
            self._margin_reservation_id(spec.instrument_id), spec.instrument_id, additional_margin)

    def _reject_order(self, runtime: _InstrumentRuntime, order: Order, reason: str) -> bool:
        runtime.total_rejections += 1
        self.total_rejections += 1
        timestamp = float(self.clock())
        previous_rejection = runtime.last_rejection
        runtime.last_rejection = {"reason": reason, "order_id": order.order_id, "timestamp": timestamp}
        try:
            self._persist_event("order_rejected", "rejection:{}:{}".format(
                order.order_id, self.total_rejections), {
                    "instrument_id": runtime.spec.instrument_id, "order_id": order.order_id,
                    "reason": reason,
                }, timestamp)
        except RuntimeError:
            runtime.last_rejection = previous_rejection
            raise
        self.emit("order_rejected", {
            "instrument_id": runtime.spec.instrument_id, "order_id": order.order_id,
            "side": order.side.name, "quantity": float(order.quantity),
            "price": float(order.price) if order.price is not None else None,
            "reason": reason,
        })
        return False

    def _fill_from_trade(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> None:
        if not runtime.market_open or not self._instrument_ready(runtime, float(self.clock())):
            return
        self._cancel_accounting_blocked_quotes(runtime, float(self.clock()))
        available = float(event.quantity) * (self.queue_fraction_override
                                             if self.queue_fraction_override is not None
                                             else runtime.config.queue_fraction)
        if available <= 0:
            return
        candidates = []
        for order_id, working in runtime.active_orders.items():
            order = working.order
            accepted_cutoff = working.accepted_event_cutoffs.get(event.source)
            if accepted_cutoff is None or event.event_ts <= accepted_cutoff:
                continue
            if event.receive_ts < working.accepted_receive_ts + runtime.config.maker_latency_ms / 1000.0:
                continue
            crosses = ((order.side == Side.BUY and event.side == Side.SELL and event.price <= float(order.price))
                       or (order.side == Side.SELL and event.side == Side.BUY and event.price >= float(order.price)))
            if crosses:
                candidates.append((order_id, working))
        candidates.sort(key=_price_time_priority)
        for order_id, working in candidates:
            if available <= 1e-12:
                break
            if runtime.active_orders.get(order_id) is not working:
                continue
            quantity = self._round_fill_quantity(runtime.spec, min(working.remaining_quantity, available))
            if quantity <= 0:
                continue
            # Treat the queue interaction as a paper estimate and price it at
            # our resting limit; never credit a maker quote with impossible
            # price improvement from a more aggressive public print.
            self._record_fill(runtime, working, event, quantity, float(working.order.price),
                              estimated=True, model="trade_print_queue_estimate")
            available -= quantity

    def _fill_from_fx_quote(self, runtime: _InstrumentRuntime, event: MarketDataEvent) -> None:
        if runtime.config.fill_model != "fx_quote_cross" or not self.is_running:
            return
        if not runtime.market_open or not self._instrument_ready(runtime, float(self.clock())):
            return
        self._cancel_accounting_blocked_quotes(runtime, float(self.clock()))
        if not event.asks or not event.bids:
            return
        best_bid, bid_size = event.bids[0]
        best_ask, ask_size = event.asks[0]
        tick = runtime.spec.tick_size
        candidates = []
        for order_id, working in runtime.active_orders.items():
            accepted_cutoff = working.accepted_event_cutoffs.get(event.source)
            if accepted_cutoff is None or event.event_ts <= accepted_cutoff:
                continue
            if event.receive_ts < working.accepted_receive_ts + runtime.config.maker_latency_ms / 1000.0:
                continue
            order = working.order
            # Price arithmetic uses binary floats; include a tiny tolerance so
            # an exact one-tick move is not lost to representation noise.
            tolerance = max(abs(tick) * 1e-9, 1e-12)
            if order.side == Side.BUY and best_ask <= float(order.price) - tick + tolerance:
                candidates.append((order_id, working))
            elif order.side == Side.SELL and best_bid >= float(order.price) + tick - tolerance:
                candidates.append((order_id, working))
        candidates.sort(key=_price_time_priority)
        queue_fraction = (self.queue_fraction_override if self.queue_fraction_override is not None
                          else runtime.config.queue_fraction)
        available = {Side.BUY: float(ask_size) * queue_fraction,
                     Side.SELL: float(bid_size) * queue_fraction}
        for order_id, working in candidates:
            if runtime.active_orders.get(order_id) is not working:
                continue
            side = working.order.side
            quantity = self._round_fill_quantity(
                runtime.spec, min(working.remaining_quantity, available[side]))
            if quantity <= 0:
                continue
            self._record_fill(runtime, working, event, quantity, float(working.order.price), estimated=True,
                              model="fx_quote_cross_estimate")
            available[side] -= quantity

    @staticmethod
    def _round_fill_quantity(spec: InstrumentSpec, requested: float) -> float:
        if not math.isfinite(requested) or requested < spec.min_quantity:
            return 0.0
        units = math.floor((requested + 1e-12) / spec.quantity_step)
        rounded = units * spec.quantity_step
        return rounded if rounded >= spec.min_quantity else 0.0

    def _record_fill(self, runtime: _InstrumentRuntime, working: _WorkingOrder,
                     event: MarketDataEvent, quantity: float, price: float,
                     *, estimated: bool, model: str) -> None:
        original_ledger = self.ledger
        original_trades = list(self.trades)
        totals = (self.total_fills, self.total_traded_notional,
                  runtime.total_fills, runtime.total_traded_notional)
        self.ledger = original_ledger.clone()
        self._transition_depth += 1
        try:
            self._apply_fill(runtime, working, event, quantity, price, estimated=estimated, model=model)
        except Exception:
            self._restore_failed_ledger(original_ledger)
            self.trades = original_trades
            (self.total_fills, self.total_traded_notional,
             runtime.total_fills, runtime.total_traded_notional) = totals
            self._cancel_all_quotes("fill_transition_failed")
            with self.ledger._lock:
                self.ledger._reservations.clear()
            raise
        finally:
            self._transition_depth -= 1
        self._emit_dashboard_state()

    def _apply_fill(self, runtime: _InstrumentRuntime, working: _WorkingOrder,
                     event: MarketDataEvent, quantity: float, price: float,
                     *, estimated: bool, model: str) -> None:
        spec, item = runtime.spec, runtime.config
        quantity = min(quantity, working.remaining_quantity)
        if quantity <= 0:
            return
        fee = item.fees.calculate(quantity, price, spec.contract_multiplier, maker=True)
        position_before = self.ledger.get_position(spec.instrument_id, as_of=event.receive_ts)
        fill = Fill(
            fill_id=self.id_factory("fill"),
            order_id=working.order.order_id,
            side=working.order.side,
            quantity=Quantity(quantity),
            price=Price(price),
            timestamp=Timestamp(event.receive_ts - self.start_time),
            trader_id=working.order.trader_id,
            is_aggressor=False,
            instrument_id=spec.instrument_id,
            source_event_id=self._event_identity(event),
            execution_model=model,
            estimated=estimated,
        )
        self.ledger.record_fill(
            fill.fill_id,
            spec.instrument_id,
            fill.side,
            quantity,
            price,
            fee=fee,
            fee_currency=item.fees.currency,
            timestamp=event.receive_ts,
            fill_instrument_id=fill.instrument_id,
            allow_risk_reduction=True,
            processing_timestamp=float(self.clock()),
        )
        position_after = self.ledger.get_position(spec.instrument_id, as_of=event.receive_ts)
        try:
            account_pnl_at_fill = self.ledger.snapshot(as_of=event.receive_ts).total_pnl
        except (ValueError, KeyError):
            account_pnl_at_fill = None
        margin_order = (spec.asset_class == AssetClass.FUTURE or
                        (spec.asset_class == AssetClass.FX
                         and spec.fx_accounting_mode == FXAccountingMode.ROLLING))
        with self._lock:
            working.remaining_quantity -= quantity
            if working.fee_reservation_id is not None:
                self.ledger.release_reservation(working.fee_reservation_id)
            if not margin_order:
                self.ledger.release_reservation(working.reservation_id)
            if working.remaining_quantity <= 1e-12:
                runtime.active_orders.pop(working.order.order_id, None)
            elif not margin_order:
                replacement = self._remaining_reservation(runtime, working)
                if replacement > 0:
                    self._resize_or_recreate_reservation(runtime, working, replacement)
            runtime.total_fills += 1
            self.total_fills += 1
            notional = quantity * abs(price) * spec.contract_multiplier
            runtime.total_traded_notional += notional
            self.total_traded_notional += notional
            trade = {
                "timestamp": event.receive_ts,
                "instrument_id": spec.instrument_id,
                "asset_class": spec.asset_class.value,
                "order_id": fill.order_id,
                "fill_id": fill.fill_id,
                "side": fill.side.name,
                "quantity": quantity,
                "quantity_unit": spec.quantity_unit,
                "quote_currency": spec.quote_currency,
                "settlement_currency": spec.settlement_currency,
                "price": price,
                "fee": fee,
                "fee_currency": item.fees.currency,
                "realized_pnl": position_after.realized_pnl - position_before.realized_pnl,
                "realized_pnl_currency": (spec.settlement_currency
                                           if spec.asset_class == AssetClass.FUTURE
                                           else spec.quote_currency),
                "total_pnl": account_pnl_at_fill,
                "estimated": estimated,
                "execution_model": model,
                "source": event.source,
                "source_event_id": fill.source_event_id,
            }
            self.trades.append(trade)
            self.trades = self.trades[-2000:]
        if margin_order or working.fee_reservation_id is not None:
            try:
                if margin_order:
                    self._refresh_margin_reservation(runtime, now=event.receive_ts)
                if (working.fee_reservation_id is not None and working.remaining_quantity > 1e-12
                        and runtime.active_orders.get(working.order.order_id) is working):
                    remaining_fee = item.fees.calculate(working.remaining_quantity, price,
                                                        spec.contract_multiplier, maker=True)
                    self.ledger.reserve_cash(working.fee_reservation_id, item.fees.currency,
                                             remaining_fee)
            except (ValueError, KeyError):
                if margin_order:
                    self._cancel_instrument_quotes(runtime, "remaining_margin_reservation_failed")
                else:
                    self._cancel_single_order(runtime, working.order.order_id, "remaining_fee_reservation_failed")
        self._enforce_after_fill(event.receive_ts)
        self._persist_event("fill", fill.fill_id, trade, event.receive_ts)
        runtime.strategy.sync_inventory(self.ledger.get_position(spec.instrument_id, as_of=event.receive_ts).quantity)
        observe_fill = getattr(runtime.strategy, "observe_fill", None)
        if callable(observe_fill):
            observe_fill(trade)
        self.emit("strategy_fill", trade)
        self.emit("position", self.get_status())
        self._emit_dashboard_state()

    def _remaining_reservation(self, runtime: _InstrumentRuntime, working: _WorkingOrder) -> float:
        item, spec = runtime.config, runtime.spec
        if spec.asset_class == AssetClass.FUTURE or (
                spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.ROLLING):
            # The caller recreates the hold with the exact remaining quantity.
            return working.remaining_quantity
        if working.order.side == Side.BUY:
            fee = item.fees.calculate(working.remaining_quantity, float(working.order.price),
                                       spec.contract_multiplier, maker=True)
            return (working.remaining_quantity * float(working.order.price) * spec.contract_multiplier
                    + (fee if item.fees.currency == spec.quote_currency else 0.0))
        if spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE:
            if not item.allow_short:
                return working.remaining_quantity * spec.contract_multiplier
            return 0.0
        return 0.0

    def _resize_or_recreate_reservation(self, runtime: _InstrumentRuntime,
                                        working: _WorkingOrder, amount: float) -> None:
        spec = runtime.spec
        try:
            if spec.asset_class == AssetClass.FUTURE or (
                    spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.ROLLING):
                self.ledger.reserve_margin(working.reservation_id, spec.instrument_id,
                                           working.remaining_quantity, float(working.order.price))
            else:
                currency = spec.quote_currency if working.order.side == Side.BUY else spec.base_currency
                self.ledger.reserve_cash(working.reservation_id, currency, amount)
        except (ValueError, KeyError):
            self._cancel_single_order(runtime, working.order.order_id, "remaining_reservation_failed")

    def _projected_gross_exposure(self, snapshot: PortfolioSnapshot, now: float,
                                  extra_order: Optional[Tuple[_InstrumentRuntime, Order]] = None) -> float:
        """Sum worst-case per-instrument exposure after pending and candidate fills.

        Buy and sell quotes are mutually exclusive directional inventory
        scenarios for the same instrument; summing their notionals separately
        double-counted closing orders and blocked risk reduction at the cap.
        """
        total = 0.0
        for runtime in self._runtime.values():
            spec = runtime.spec
            position = snapshot.positions[spec.instrument_id]
            buy_quantity = sum(row.remaining_quantity for row in runtime.active_orders.values()
                               if row.order.side == Side.BUY)
            sell_quantity = sum(row.remaining_quantity for row in runtime.active_orders.values()
                                if row.order.side == Side.SELL)
            prices = [abs(float(row.order.price)) for row in runtime.active_orders.values()]
            if extra_order is not None and extra_order[0] is runtime:
                candidate = extra_order[1]
                if candidate.side == Side.BUY:
                    buy_quantity += float(candidate.quantity)
                else:
                    sell_quantity += float(candidate.quantity)
                prices.append(abs(float(candidate.price)))

            worst_quantity = max(abs(position.quantity + buy_quantity),
                                 abs(position.quantity - sell_quantity))
            if worst_quantity <= 1e-12:
                continue
            if position.mark_price is not None:
                prices.append(abs(float(position.mark_price)))
            elif runtime.mid_price is not None:
                prices.append(abs(float(runtime.mid_price)))
            if not prices:
                raise ValueError("missing price mark for projected exposure {}".format(spec.instrument_id))
            reference_price = max(prices)
            notional = worst_quantity * reference_price * spec.contract_multiplier
            margin = 0.0
            if spec.asset_class == AssetClass.FUTURE:
                margin = worst_quantity * float(spec.initial_margin_per_contract)
            elif spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.ROLLING:
                if spec.initial_margin_rate is None or spec.initial_margin_rate <= 0:
                    raise ValueError("rolling FX exposure requires explicit initial margin")
                margin = notional * spec.initial_margin_rate
            currency = spec.settlement_currency if spec.asset_class == AssetClass.FUTURE else spec.quote_currency
            rate = self._currency_rate(currency, now)
            if rate is None:
                raise ValueError("missing/stale exposure conversion for {}".format(spec.instrument_id))
            total += max(notional, margin) * rate
        return total

    def _portfolio_proposal(self, snapshot: PortfolioSnapshot, now: float, *,
                            extra_order=None, excluded_instruments=()) -> PortfolioProposal:
        bounds, prices, rates = {}, {}, {}
        native_cash: dict[str, float] = {}
        gross = margin = pending_fees = 0.0
        for runtime in self._runtime.values():
            spec = runtime.spec
            rows = (() if spec.instrument_id in excluded_instruments
                    else tuple(runtime.active_orders.values()))
            for row in rows:
                fee = runtime.config.fees.calculate(row.remaining_quantity, float(row.order.price),
                                                   spec.contract_multiplier, maker=True)
                fee_rate = self._currency_rate(runtime.config.fees.currency, now)
                if fee and fee_rate is None:
                    raise ValueError("missing or stale pending fee conversion")
                pending_fees += fee * (fee_rate or 0.0)
            buys = sum(row.remaining_quantity for row in rows if row.order.side == Side.BUY)
            sells = sum(row.remaining_quantity for row in rows if row.order.side == Side.SELL)
            reference_prices = [abs(float(row.order.price)) for row in rows]
            if extra_order is not None and extra_order[0] is runtime:
                order = extra_order[1]
                quantity, price = float(order.quantity), float(order.price)
                fee_currency = runtime.config.fees.currency
                fee = runtime.config.fees.calculate(quantity, price, spec.contract_multiplier, maker=True)
                native_cash[fee_currency] = native_cash.get(fee_currency, 0.0) + fee
                derivative = (spec.asset_class == AssetClass.FUTURE or (
                    spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.ROLLING))
                if not derivative:
                    if order.side == Side.BUY:
                        native_cash[spec.quote_currency] = (native_cash.get(spec.quote_currency, 0.0)
                                                           + quantity * price * spec.contract_multiplier)
                    elif spec.asset_class == AssetClass.FX and not runtime.config.allow_short:
                        native_cash[spec.base_currency] = (native_cash.get(spec.base_currency, 0.0)
                                                          + quantity * spec.contract_multiplier)
                if order.side == Side.BUY:
                    buys += float(order.quantity)
                else:
                    sells += float(order.quantity)
                reference_prices.append(abs(float(order.price)))
            position = snapshot.positions[spec.instrument_id]
            low, high = position.quantity - sells, position.quantity + buys
            bounds[spec.instrument_id] = (low, high)
            maximum = max(abs(low), abs(high))
            if position.mark_price is not None:
                reference_prices.append(abs(position.mark_price))
            if runtime.mid_price is not None:
                reference_prices.append(abs(runtime.mid_price))
            prices[spec.instrument_id] = max(reference_prices, default=0.0)
            currency = spec.settlement_currency if spec.asset_class == AssetClass.FUTURE else spec.quote_currency
            rate = self._currency_rate(currency, now)
            if maximum <= 1e-12:
                continue
            if rate is None:
                raise ValueError("missing or stale exposure conversion")
            rates[currency] = rate
            notional = maximum * prices[spec.instrument_id] * spec.contract_multiplier
            initial = 0.0
            if spec.asset_class == AssetClass.FUTURE:
                initial = maximum * float(spec.initial_margin_per_contract)
            elif spec.asset_class == AssetClass.FX and spec.fx_accounting_mode == FXAccountingMode.ROLLING:
                initial = notional * float(spec.initial_margin_rate)
            gross += max(notional, initial) * rate
            settlement_rate = self._currency_rate(spec.settlement_currency, now)
            if initial and settlement_rate is None:
                raise ValueError("missing or stale margin conversion")
            margin += initial * (settlement_rate or 0.0)
        return PortfolioProposal(bounds, prices, rates, gross, margin,
                                 fees_reporting=self.ledger.closed_market_stress(now) + pending_fees,
                                 native_cash_required=native_cash)

    def _projected_initial_margin_reporting(
            self, snapshot: PortfolioSnapshot, now: float,
            extra_order: Optional[Tuple[_InstrumentRuntime, Order]] = None) -> float:
        """Net-liquidation equity required for worst-case projected positions."""
        total = 0.0
        for runtime in self._runtime.values():
            spec = runtime.spec
            position = snapshot.positions[spec.instrument_id].quantity
            buys = sum(row.remaining_quantity for row in runtime.active_orders.values()
                       if row.order.side == Side.BUY)
            sells = sum(row.remaining_quantity for row in runtime.active_orders.values()
                        if row.order.side == Side.SELL)
            prices = [abs(float(row.order.price)) for row in runtime.active_orders.values()]
            if extra_order is not None and extra_order[0] is runtime:
                candidate = extra_order[1]
                if candidate.side == Side.BUY:
                    buys += float(candidate.quantity)
                else:
                    sells += float(candidate.quantity)
                prices.append(abs(float(candidate.price)))
            worst_quantity = max(abs(position + buys), abs(position - sells))
            is_margin_product = (
                spec.asset_class == AssetClass.FUTURE or
                (spec.asset_class == AssetClass.FX
                 and spec.fx_accounting_mode == FXAccountingMode.ROLLING))
            if not is_margin_product or worst_quantity <= 1e-12:
                continue
            current = self.ledger.get_position(spec.instrument_id, as_of=now)
            if current.mark_price is not None:
                prices.append(abs(float(current.mark_price)))
            if runtime.mid_price is not None:
                prices.append(abs(float(runtime.mid_price)))
            if spec.asset_class == AssetClass.FUTURE:
                required = worst_quantity * float(spec.initial_margin_per_contract)
            else:
                if spec.initial_margin_rate is None or spec.initial_margin_rate <= 0:
                    raise ValueError("rolling FX requires explicit initial margin")
                if not prices:
                    raise ValueError("rolling-FX margin needs a current price")
                required = (worst_quantity * max(prices) * spec.contract_multiplier
                            * spec.initial_margin_rate)
            conversion = self._currency_rate(spec.settlement_currency, now)
            if conversion is None:
                raise ValueError("missing or stale settlement currency conversion")
            total += required * conversion
        return total

    def _currency_rate(self, currency: str, now: float) -> Optional[float]:
        code = currency.upper()
        if code == self.config.reporting_currency:
            return 1.0
        # The ledger also validates these marks when producing portfolio P&L.
        # This resolver uses only timestamped provider FX edges and never invents
        # a par rate or falls back to a cached stale value.
        return self._fx_rate_from_edges(code, now)

    def _fx_rate_from_edges(self, currency: str, now: float) -> Optional[float]:
        if currency == self.config.reporting_currency:
            return 1.0
        graph: Dict[str, List[Tuple[str, float, float]]] = {}
        for (base, quote), (mid, ts, max_age) in self._fx_edges.items():
            if now - ts < 0 or now - ts > max_age or mid <= 0:
                continue
            graph.setdefault(base, []).append((quote, mid, ts))
            graph.setdefault(quote, []).append((base, 1.0 / mid, ts))
        queue = deque([(currency, 1.0, now)])
        visited = {currency}
        while queue:
            current, rate, _oldest = queue.popleft()
            if current == self.config.reporting_currency:
                return rate
            for neighbor, edge, _ts in graph.get(current, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, rate * edge, _ts))
        return None

    def _cancel_single_order(self, runtime: _InstrumentRuntime, order_id: str, reason: str) -> None:
        working = runtime.active_orders.pop(order_id, None)
        if working is None:
            return
        if working.fee_reservation_id is not None:
            self.ledger.release_reservation(working.fee_reservation_id)
        if runtime.spec.asset_class == AssetClass.FUTURE or (
                runtime.spec.asset_class == AssetClass.FX
                and runtime.spec.fx_accounting_mode == FXAccountingMode.ROLLING):
            try:
                self._refresh_margin_reservation(runtime)
            except (ValueError, KeyError):
                self._cancel_instrument_quotes(runtime, "remaining_margin_reservation_failed")
        else:
            self.ledger.release_reservation(working.reservation_id)
        self.emit("orders_cancelled", {"instrument_id": runtime.spec.instrument_id,
                                        "order_ids": [order_id], "reason": reason})

    def _split_reconciliation_required(self, instrument_id: str) -> bool:
        return any(action["instrument_id"] == instrument_id and action["action_type"] == "split"
                   and action["status"] == "failed" for action in self.account_actions.actions.values())

    def _cancel_accounting_blocked_quotes(self, runtime: _InstrumentRuntime, now: float) -> None:
        """Block new exposure immediately while keeping bounded close orders."""
        spec = runtime.spec
        if self._split_reconciliation_required(spec.instrument_id):
            self._cancel_instrument_quotes(runtime, "split_reconciliation_required")
            return
        for order_id, working in list(runtime.active_orders.items()):
            with self.ledger._lock:
                currencies = set()
                for reservation_id in (working.reservation_id, working.fee_reservation_id):
                    if reservation_id is None:
                        continue
                    reservation = self.ledger._reservations.get(reservation_id)
                    if reservation is not None and reservation.kind == "cash":
                        currencies.add(reservation.currency)
                # Mandatory debits can invalidate a previously funded hold.
                unfunded = False
                for currency in currencies:
                    held = sum(row.amount for row in self.ledger._reservations.values()
                               if row.currency == currency)
                    if self.ledger._cash.get(currency, 0.0) < held - 1e-9:
                        unfunded = True
                        break
                position = self.ledger._positions[spec.instrument_id].quantity
                reducing = self._is_strictly_risk_reducing(runtime, working.order, position)
                if not unfunded and not reducing:
                    try:
                        unfunded = any(self.ledger._available_cash(currency, as_of=now) < -1e-9
                                       for currency in currencies)
                    except (ValueError, KeyError):
                        unfunded = True
            if unfunded:
                self._cancel_single_order(runtime, order_id, "native_cash_reservation_no_longer_funded")
        borrowed = (spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}
                    or (spec.asset_class == AssetClass.FX
                        and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE))
        if (runtime.config.allow_short and borrowed
                and (not spec.borrow_rate_source or spec.borrow_rate_effective_from is None
                     or spec.borrow_rate_effective_to is None
                     or not spec.borrow_rate_effective_from <= now < spec.borrow_rate_effective_to)):
            self._accounting_blocks[spec.instrument_id] = "current_sourced_borrow_terms_required"
        reason = self._accounting_blocks.get(spec.instrument_id)
        if reason is None:
            return
        position = self.ledger._positions[spec.instrument_id].quantity
        close_side = Side.SELL if position > 1e-12 else Side.BUY if position < -1e-12 else None
        remaining_close = abs(position)
        # Account for remaining quantities, not original quantities, and count
        # each working quote once. Retained closes together cannot cross zero.
        rows = sorted(list(runtime.active_orders.items()),
                      key=lambda item: (item[1].accepted_receive_ts, item[0]))
        for order_id, working in rows:
            if (working.order.side == close_side
                    and working.remaining_quantity <= remaining_close + 1e-12):
                remaining_close -= working.remaining_quantity
            else:
                self._cancel_single_order(runtime, order_id, reason)

    def _cancel_instrument_quotes(self, runtime: _InstrumentRuntime, reason: str) -> None:
        margin_order = (runtime.spec.asset_class == AssetClass.FUTURE or
                        (runtime.spec.asset_class == AssetClass.FX
                         and runtime.spec.fx_accounting_mode == FXAccountingMode.ROLLING))
        with self._lock:
            ids = list(runtime.active_orders)
            for order_id in ids:
                working = runtime.active_orders.pop(order_id)
                if working.fee_reservation_id is not None:
                    self.ledger.release_reservation(working.fee_reservation_id)
                if not margin_order:
                    self.ledger.release_reservation(working.reservation_id)
            if margin_order:
                self.ledger.set_margin_reservation(
                    self._margin_reservation_id(runtime.spec.instrument_id),
                    runtime.spec.instrument_id, 0.0)
        if ids:
            self.emit("orders_cancelled", {"instrument_id": runtime.spec.instrument_id,
                                            "order_ids": ids, "reason": reason})

    def _cancel_all_quotes(self, reason: str = "global_stop") -> None:
        for runtime in self._runtime.values():
            self._cancel_instrument_quotes(runtime, reason)

    def expire_stale_feeds(self, now: Optional[float] = None) -> None:
        with self._risk_lock:
            self._expire_stale_feeds(now)

    def _expire_stale_feeds(self, now: Optional[float] = None) -> None:
        current = float(self.clock()) if now is None else float(now)
        self.process_account_actions(current)
        if self.is_running and not self.circuit_breaker_active:
            self._roll_due_futures(current)
        for runtime in self._runtime.values():
            if runtime.spec.expiry is not None and current >= runtime.spec.expiry.timestamp():
                runtime.market_open = False
                runtime.market_status = "expired"
                self._cancel_instrument_quotes(runtime, "instrument_expired_manual_settlement_required")
                continue
            if not self._instrument_ready(runtime, current):
                self._cancel_instrument_quotes(runtime, "stale_or_unready_feed")
        try:
            snapshot = self.ledger.snapshot(as_of=current)
            self._enforce_account_risk(snapshot, current)
        except ValueError:
            # Existing positions remain marked stale and are surfaced as such;
            # no risk-increasing quote can pass the mark gate.
            return

    def _daily_loss_breached(self, snapshot: PortfolioSnapshot, now: float) -> bool:
        """Compare equity with the last known valuation at the UTC day boundary.

        If the process or a feed misses midnight, carry forward the last known
        equity instead of resetting the loss baseline to the first later mark.
        This conservative gap policy prevents overnight losses disappearing.
        """
        risk_day = datetime.fromtimestamp(now, timezone.utc).date()
        if risk_day != self._risk_day:
            self._risk_day = risk_day
            self._risk_day_start_equity = self._last_account_equity
        breached = snapshot.equity - self._risk_day_start_equity <= -self.config.max_daily_loss
        self._last_account_equity = snapshot.equity
        self._last_account_timestamp = now
        return breached

    def _enforce_after_fill(self, now: float) -> None:
        try:
            snapshot = self.ledger.snapshot(as_of=now)
        except ValueError:
            return
        self._enforce_account_risk(snapshot, now)

    def schedule_account_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        with self._risk_lock:
            if not isinstance(action, dict) or action.get("instrument_id") not in self._runtime:
                raise ValueError("account action requires a configured instrument")
            previous = self.account_actions.export_state()
            result = self.account_actions.schedule(action)
            if result["action_id"] in previous:
                return result
            try:
                self._persist_event("account_action_scheduled", "schedule:" + result["action_id"],
                                    result, float(self.clock()))
            except Exception:
                self.account_actions.actions = previous
                raise
            return result

    def import_account_actions(self, actions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.schedule_account_action(action) for action in actions]

    def get_account_actions(self) -> List[Dict[str, Any]]:
        with self._risk_lock:
            return list(self.account_actions.export_state().values())

    def record_funding(self, funding_id: str, currency: str, amount: float, source: str,
                        *, fx_rate_to_reporting: Optional[float] = None,
                        timestamp: Optional[float] = None) -> bool:
        """Record a sourced paper deposit; never rewrite original capital."""
        if not isinstance(funding_id, str) or not funding_id.strip() or not isinstance(source, str) or not source.strip():
            raise ValueError("paper funding requires a stable ID and source")
        code = str(currency).upper()
        value = float(amount)
        rate = 1.0 if code == self.config.reporting_currency else float(fx_rate_to_reporting or 0.0)
        if not math.isfinite(value) or value <= 0 or not math.isfinite(rate) or rate <= 0:
            raise ValueError("paper deposits and their explicit conversion rates must be positive")
        stamp = float(self.clock()) if timestamp is None else float(timestamp)
        signature = ("funding", code, value, rate, source.strip())
        with self._risk_lock, self._candidate_ledger_transition():
            prior = self.ledger._signature("cash_adjustment_ids", funding_id)
            if prior is not None:
                if prior != signature:
                    raise ValueError("funding ID was reused with different inputs")
                return False
            self.ledger.fund_currency(code, value, stamp, fx_rate_to_reporting=rate)
            self.ledger._remember("cash_adjustment_ids", funding_id, signature)
            prior_risk = (self._risk_day, self._risk_day_start_equity,
                          self._last_account_equity, self._last_account_timestamp)
            processing_ts = float(self.clock())
            processing_day = datetime.fromtimestamp(processing_ts, timezone.utc).date()
            if processing_day != self._risk_day:
                self._risk_day = processing_day
                self._risk_day_start_equity = self._last_account_equity
            external_flow = value * rate
            self._risk_day_start_equity += external_flow
            self._last_account_equity += external_flow
            pair = (code, self.config.reporting_currency)
            previous_edge = self._fx_edges.get(pair)
            previous_source = self._conversion_mark_sources.get(code)
            if code != self.config.reporting_currency:
                self._fx_edges[pair] = (
                    rate, stamp, self.config.max_mark_age_seconds)
                self._conversion_mark_sources[code] = source.strip()
            try:
                self._persist_event("funding", "funding:" + funding_id, {
                    "funding_id": funding_id, "currency": code, "amount": value, "source": source.strip(),
                    "fx_rate_to_reporting": rate, "external_flow_reporting": external_flow,
                    "effective_ts": stamp, "processing_ts": processing_ts}, stamp)
            except Exception:
                (self._risk_day, self._risk_day_start_equity,
                 self._last_account_equity, self._last_account_timestamp) = prior_risk
                if previous_edge is None:
                    self._fx_edges.pop(pair, None)
                else:
                    self._fx_edges[pair] = previous_edge
                if previous_source is None:
                    self._conversion_mark_sources.pop(code, None)
                else:
                    self._conversion_mark_sources[code] = previous_source
                raise
            self.emit("funding", {"funding_id": funding_id, "currency": code,
                "amount": value, "source": source.strip(), "external_flow_reporting": external_flow,
                "effective_ts": stamp, "processing_ts": processing_ts})
            return True

    def process_account_actions(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Apply due sourced inputs once; mandatory costs may trigger a risk halt."""
        stamp = float(self.clock()) if now is None else float(now)
        with self._risk_lock:
            return self._process_account_actions_locked(stamp)

    def _process_account_actions_locked(self, stamp: float) -> list[dict[str, Any]]:
        completed = []
        for action in self.account_actions.due(stamp):
            instrument_id = action["instrument_id"]
            runtime = self._runtime[instrument_id]
            original = self.ledger
            candidate = original.clone()
            original_config = runtime.config
            original_global_config = self.config
            original_epoch = self.config_epoch
            original_unit_epoch = runtime.book_fresh_after_ts
            corrected = self.account_actions.actions.get(action.get("corrects"))
            corrected_status = corrected["status"] if corrected is not None else None
            payload = action["payload"]
            kind = action["action_type"]
            effective = action["effective_ts"]
            try:
                if kind == "financing":
                    candidate.record_financing(action["action_id"], instrument_id,
                        payload["amount"], payload["currency"], timestamp=effective)
                elif kind == "borrow_charge":
                    actual = float(payload["actual_amount"])
                    estimated = float(payload["estimated_amount"])
                    start, end = float(payload["interval_start"]), float(payload["interval_end"])
                    if (not all(math.isfinite(value) for value in (actual, estimated, start, end))
                            or min(actual, estimated, start) < 0 or not start < end <= effective):
                        raise ValueError("borrow reconciliation requires positive sourced amounts and an elapsed interval")
                    candidate.record_sourced_adjustment(action["action_id"], instrument_id,
                        estimated - actual, payload["currency"], timestamp=effective,
                        kind="sourced_borrow_reconciliation")
                elif kind == "dividend":
                    candidate.apply_equity_dividend(action["action_id"], instrument_id,
                        payload["amount_per_share"], payload["eligible_quantity"], stamp)
                elif kind == "split":
                    last = original._positions[instrument_id].mark_timestamp
                    if last is not None and last > effective:
                        raise ValueError("late split requires reconciliation against later market events and fills")
                    candidate.apply_equity_split(action["action_id"], instrument_id, payload["ratio"], effective)
                    runtime.book_fresh_after_ts = effective
                elif kind == "settlement":
                    if any(trade.get("instrument_id") == instrument_id
                           and trade.get("timestamp", 0) > effective for trade in self.trades):
                        raise ValueError("late settlement with intervening fills requires reconciliation")
                    candidate.record_sourced_settlement(action["action_id"], instrument_id,
                                                        payload["settlement_price"], effective)
                elif kind == "valuation_mark":
                    next_open = float(payload["next_open_ts"])
                    if not math.isfinite(next_open) or next_open <= effective:
                        raise ValueError("valuation next opening must follow effective close")
                    candidate.set_valuation_mark(instrument_id, payload["price"], effective,
                        valid_until=next_open + 60.0, source=action["source"],
                        basis=payload["basis"], stress_fraction=payload["stress_fraction"],
                        processing_timestamp=stamp)
                elif kind == "borrow_terms":
                    spec = replace(runtime.spec, borrow_fee_rate=float(payload["annual_rate"]),
                        borrow_limit=float(payload["borrow_limit"]), borrow_rate_source=action["source"],
                        borrow_rate_effective_from=effective,
                        borrow_rate_effective_to=float(payload["effective_to"]),
                        borrow_day_count=payload.get("day_count", "ACT/365F"))
                    candidate.update_instrument_policy(spec, effective)
                    candidate.accrue_borrow(stamp)
                    runtime.config = replace(runtime.config, spec=spec)
                    items = dict(self.config.instruments)
                    items[instrument_id] = runtime.config
                    self.config = replace(self.config, instruments=items)
                    self.config_epoch += 1
                self.ledger = candidate
                action.update(status="applied", processed_ts=stamp, error=None)
                if corrected is not None and corrected["status"] == "failed":
                    corrected["status"] = "superseded"
                self._accounting_blocks.pop(instrument_id, None)
                self._persist_event("account_action_applied", "apply:" + action["action_id"], action, effective)
            except (ValueError, KeyError, TypeError) as exc:
                self.ledger = original
                runtime.config, self.config = original_config, original_global_config
                self.config_epoch = original_epoch
                runtime.book_fresh_after_ts = original_unit_epoch
                if corrected is not None:
                    corrected["status"] = corrected_status
                action.update(status="failed", processed_ts=stamp, error=str(exc))
                self._accounting_blocks[instrument_id] = "account_input_required: " + str(exc)
                self._persist_event("account_action_failed", "failed:" + action["action_id"], action, stamp)
            except Exception:
                self._restore_failed_ledger(original)
                runtime.config, self.config = original_config, original_global_config
                self.config_epoch = original_epoch
                runtime.book_fresh_after_ts = original_unit_epoch
                if corrected is not None:
                    corrected["status"] = corrected_status
                action.update(status="pending", processed_ts=None, error=None)
                raise
            if action["status"] == "applied" and kind == "split":
                self._cancel_instrument_quotes(runtime, "equity_split")
                runtime.bids = runtime.asks = ()
                runtime.book_synchronized = False
                runtime.book_fresh_after_ts = effective
                runtime.last_quote_event_ts = runtime.last_quote_receive_ts = None
                runtime.last_trade_event_ts = runtime.last_trade_receive_ts = None
                runtime.recent_public_trades.clear()
            completed.append(dict(action))
        if self.ledger._borrow_state or any(
                state.quantity < -1e-12 and state.spec.borrow_rate_source
                for state in self.ledger._positions.values()):
            original_ledger = self.ledger
            candidate = original_ledger.clone()
            candidate.accrue_borrow(stamp)
            if original_ledger._borrow_state != candidate._borrow_state:
                self.ledger = candidate
                try:
                    self._save_checkpoint(stamp)
                except Exception:
                    self._restore_failed_ledger(original_ledger)
                    raise
        for runtime in self._runtime.values():
            spec = runtime.spec
            if (runtime.config.allow_short and spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}
                    and (not spec.borrow_rate_source or spec.borrow_rate_effective_from is None
                         or spec.borrow_rate_effective_to is None
                         or not spec.borrow_rate_effective_from <= stamp < spec.borrow_rate_effective_to)):
                self._accounting_blocks[spec.instrument_id] = "current_sourced_borrow_terms_required"
        for action in self.account_actions.actions.values():
            if action["status"] == "failed":
                self._accounting_blocks[action["instrument_id"]] = "account_input_required: " + str(action["error"])
        for runtime in self._runtime.values():
            self._cancel_accounting_blocked_quotes(runtime, stamp)
        if completed:
            self._enforce_after_fill(stamp)
        return completed

    def apply_equity_split(self, action_id: str, instrument_id: str,
                           ratio: float, timestamp: Optional[float] = None) -> bool:
        """Apply and persist an explicitly sourced paper equity split."""
        with self._risk_lock, self._candidate_ledger_transition():
            runtime = self._runtime.get(instrument_id)
            if runtime is None or runtime.spec.asset_class != AssetClass.EQUITY:
                raise ValueError("split requires a configured equity instrument")
            stamp = float(self.clock()) if timestamp is None else float(timestamp)
            original_unit_epoch = runtime.book_fresh_after_ts
            applied = self.ledger.apply_equity_split(action_id, instrument_id, ratio, stamp)
            if not applied:
                return False
            position = self.ledger.get_position(instrument_id, as_of=stamp)
            runtime.strategy.sync_inventory(position.quantity)
            payload = {"action_id": action_id, "instrument_id": instrument_id,
                       "ratio": float(ratio), "timestamp": stamp,
                       "position_after": position.quantity}
            self._cancel_instrument_quotes(runtime, "equity_split_requires_fresh_book")
            runtime.bids = ()
            runtime.asks = ()
            runtime.book_synchronized = False
            runtime.last_quote_event_ts = None
            runtime.last_quote_receive_ts = None
            runtime.last_trade_event_ts = None
            runtime.last_trade_receive_ts = None
            runtime.book_fresh_after_ts = stamp
            runtime.recent_public_trades.clear()
            try:
                self._persist_event("equity_split", action_id, payload, stamp)
            except Exception:
                runtime.book_fresh_after_ts = original_unit_epoch
                raise
            self.emit("corporate_action", dict(payload, action="split"))
            self._emit_dashboard_state()
            return True

    def apply_equity_dividend(self, action_id: str, instrument_id: str,
                              amount_per_share: float, eligible_quantity: float,
                              timestamp: Optional[float] = None) -> float:
        """Post an explicitly sourced per-share dividend to the paper ledger."""
        with self._risk_lock, self._candidate_ledger_transition():
            runtime = self._runtime.get(instrument_id)
            if runtime is None or runtime.spec.asset_class != AssetClass.EQUITY:
                raise ValueError("dividend requires a configured equity instrument")
            stamp = float(self.clock()) if timestamp is None else float(timestamp)
            amount = self.ledger.apply_equity_dividend(
                action_id, instrument_id, amount_per_share, eligible_quantity, stamp)
            if amount is None:
                return 0.0
            payload = {"action_id": action_id, "instrument_id": instrument_id,
                       "amount_per_share": float(amount_per_share), "cash_amount": amount,
                       "eligible_quantity": float(eligible_quantity),
                       "currency": runtime.spec.quote_currency, "timestamp": stamp}
            self._persist_event("equity_dividend", action_id, payload, stamp)
            self._enforce_after_fill(float(self.clock()))
            self.emit("corporate_action", dict(payload, action="dividend"))
            self._emit_dashboard_state()
            return amount

    def record_fx_financing(self, adjustment_id: str, instrument_id: str,
                            amount: float, currency: str, *,
                            timestamp: Optional[float] = None,
                            kind: str = "fx_financing") -> bool:
        """Post an explicit signed financing amount for rolling FX exposure."""
        with self._risk_lock, self._candidate_ledger_transition():
            runtime = self._runtime.get(instrument_id)
            if (runtime is None or runtime.spec.asset_class != AssetClass.FX
                    or runtime.spec.fx_accounting_mode != FXAccountingMode.ROLLING):
                raise ValueError("financing requires a configured rolling-FX instrument")
            stamp = float(self.clock()) if timestamp is None else float(timestamp)
            applied = self.ledger.record_financing(
                adjustment_id, instrument_id, amount, currency, timestamp=stamp, kind=kind)
            if not applied:
                return False
            payload = {"adjustment_id": adjustment_id, "instrument_id": instrument_id,
                       "amount": float(amount), "currency": currency.upper(),
                       "kind": kind, "timestamp": stamp}
            self._persist_event("fx_financing", adjustment_id, payload, stamp)
            self._enforce_after_fill(float(self.clock()))
            self.emit("fx_financing", payload)
            self._emit_dashboard_state()
            return True

    def settle_future(self, instrument_id: str, settlement_price: float,
                      timestamp: Optional[float] = None) -> float:
        with self._risk_lock, self._candidate_ledger_transition():
            return self._settle_future(instrument_id, settlement_price, timestamp)

    def _settle_future(self, instrument_id: str, settlement_price: float,
                       timestamp: Optional[float] = None) -> float:
        """Apply an explicitly supplied futures variation settlement locally.

        No settlement value is inferred from a last trade or a quote. Feed
        adapters or an operator must explicitly provide a valid settlement.
        """
        runtime = self._runtime.get(instrument_id)
        if runtime is None or runtime.spec.asset_class != AssetClass.FUTURE:
            raise ValueError("settlement requires a configured futures instrument")
        stamp = float(self.clock()) if timestamp is None else float(timestamp)
        self._cancel_instrument_quotes(runtime, "futures_variation_settlement")
        variation = self.ledger.settle_future(instrument_id, settlement_price, stamp)
        self._persist_event("futures_settlement", "settlement:{}:{}".format(instrument_id, stamp), {
            "instrument_id": instrument_id, "price": float(settlement_price),
            "variation": variation, "currency": runtime.spec.settlement_currency,
        }, stamp)
        self.emit("futures_settlement", {
            "instrument_id": instrument_id, "price": float(settlement_price),
            "variation": variation, "currency": runtime.spec.settlement_currency,
            "timestamp": stamp,
        })
        self._emit_dashboard_state()
        return variation

    def _roll_due_futures(self, now: float) -> None:
        for runtime in list(self._runtime.values()):
            item = runtime.config
            if (not item.auto_roll or not item.roll_to
                    or runtime.spec.instrument_id in self._rolled_futures
                    or runtime.spec.expiry is None):
                continue
            if now < runtime.spec.expiry.timestamp() - item.roll_before_expiry_seconds:
                continue
            target = self._runtime.get(item.roll_to)
            if target is None or not self._instrument_ready(runtime, now) or not self._instrument_ready(target, now):
                continue
            try:
                self.roll_future(runtime.spec.instrument_id, item.roll_to, timestamp=now)
            except (ValueError, KeyError) as exc:
                self.emit("provider_error", {
                    "instrument_id": runtime.spec.instrument_id,
                    "message": "Automatic paper futures roll skipped: {}".format(exc),
                })

    def roll_future(self, old_instrument_id: str, new_instrument_id: str,
                    timestamp: Optional[float] = None,
                    quantity: Optional[float] = None) -> Dict[str, Any]:
        with self._risk_lock:
            return self._roll_future(old_instrument_id, new_instrument_id, timestamp, quantity)

    def _roll_future(self, old_instrument_id: str, new_instrument_id: str,
                     timestamp: Optional[float] = None,
                     quantity: Optional[float] = None) -> Dict[str, Any]:
        """Simulate a two-leg local futures roll at current touch quotes.

        The old contract is closed at its executable top-of-book price and the
        replacement is opened at its executable top-of-book price. Both legs,
        fees, and margin checks are local paper accounting events.
        """
        old = self._runtime.get(old_instrument_id)
        new = self._runtime.get(new_instrument_id)
        if old is None or new is None:
            raise ValueError("both roll contracts must be configured")
        if old is new:
            raise ValueError("a futures roll requires two distinct contracts")
        if self.circuit_breaker_active:
            raise ValueError("a futures roll cannot reopen exposure during a risk halt")
        if old_instrument_id in self._rolled_futures:
            raise ValueError("this futures contract has already been rolled")
        if old.spec.asset_class != AssetClass.FUTURE or new.spec.asset_class != AssetClass.FUTURE:
            raise ValueError("futures rolls require two configured futures contracts")
        if (old.spec.venue != new.spec.venue or old.spec.base_currency != new.spec.base_currency
                or old.spec.settlement_currency != new.spec.settlement_currency
                or old.spec.contract_multiplier != new.spec.contract_multiplier):
            raise ValueError("futures roll contract metadata is incompatible")
        stamp = float(self.clock()) if timestamp is None else float(timestamp)
        if old.spec.expiry is not None and stamp >= old.spec.expiry.timestamp():
            raise ValueError("expired contracts need explicit settlement before rolling")
        if new.spec.expiry is not None and stamp >= new.spec.expiry.timestamp():
            raise ValueError("replacement futures contract is expired")
        if not self._instrument_ready(old, stamp) or not self._instrument_ready(new, stamp):
            raise ValueError("both contracts need fresh open market data before rolling")

        old_position = self.ledger.get_position(old_instrument_id, as_of=stamp).quantity
        if abs(old_position) <= 1e-12:
            self._rolled_futures.add(old_instrument_id)
            old.market_open = False
            old.market_status = "rolled"
            self._cancel_instrument_quotes(old, "futures_contract_rolled")
            self._save_checkpoint(stamp)
            return {"rolled": False, "reason": "no open position"}
        close_side = Side.BUY if old_position < 0 else Side.SELL
        open_side = Side.BUY if old_position > 0 else Side.SELL
        close_price = old.asks[0][0] if close_side == Side.BUY else old.bids[0][0]
        open_price = new.asks[0][0] if open_side == Side.BUY else new.bids[0][0]
        quantity = abs(old_position) if quantity is None else float(quantity)
        if (not old.spec.is_valid_quantity(quantity) or not new.spec.is_valid_quantity(quantity)
                or quantity > abs(old_position) + 1e-12
                or quantity > new.config.max_order_quantity + 1e-12):
            raise ValueError("roll quantity violates inventory, lot, or replacement order limits")
        self._refresh_roll_liquidity(old)
        self._refresh_roll_liquidity(new)
        close_liquidity = old.roll_liquidity[close_side][2]
        open_liquidity = new.roll_liquidity[open_side][2]
        if quantity > min(close_liquidity, open_liquidity) + 1e-12:
            raise ValueError("insufficient displayed liquidity for an atomic futures roll")
        if not old.spec.is_valid_price(close_price) or not new.spec.is_valid_price(open_price):
            raise ValueError("roll touch quotes are not aligned to contract ticks")

        original_ledger = self.ledger
        candidate = original_ledger.clone()
        original_trades = list(self.trades)
        old_fee = old.config.fees.calculate(quantity, close_price, old.spec.contract_multiplier, maker=False)
        new_fee = new.config.fees.calculate(quantity, open_price, new.spec.contract_multiplier, maker=False)
        roll_key = self.id_factory("roll")
        for runtime in (old, new):
            for working in runtime.active_orders.values():
                candidate.release_reservation(working.reservation_id)
                if working.fee_reservation_id is not None:
                    candidate.release_reservation(working.fee_reservation_id)
            candidate.release_reservation(self._margin_reservation_id(runtime.spec.instrument_id))
        candidate.record_fill(
                "roll-close:" + roll_key, old_instrument_id, close_side, quantity, close_price,
                fee=old_fee, fee_currency=old.config.fees.currency, timestamp=stamp,
                allow_risk_reduction=True)
        candidate.record_fill(
                "roll-open:" + roll_key, new_instrument_id, open_side, quantity, open_price,
                fee=new_fee, fee_currency=new.config.fees.currency, timestamp=stamp)
        after = candidate.snapshot(as_of=stamp)
        proposal = self._portfolio_proposal(after, stamp,
            excluded_instruments=(old_instrument_id, new_instrument_id))
        decision = self.risk_evaluator.validate(proposal, RiskContext(
            self.config, after.equity, after.available_cash_by_currency, self.circuit_breaker_active, stamp))
        if not decision.accepted:
            raise ValueError("futures roll rejected: " + decision.reason)
        if any(value < -1e-9 for value in after.available_cash_by_currency.values()):
            raise ValueError("insufficient native-currency cash for rolled margin and fees")

        legs = []
        for runtime, side, price, fee, action in (
            (old, close_side, close_price, old_fee, "close"),
            (new, open_side, open_price, new_fee, "open"),
        ):
            leg = {
                "timestamp": stamp, "instrument_id": runtime.spec.instrument_id,
                "asset_class": "future", "side": side.name, "quantity": quantity,
                "quantity_unit": runtime.spec.quantity_unit, "price": price,
                "fee": fee, "fee_currency": runtime.config.fees.currency,
                "estimated": True, "execution_model": "simulated_futures_roll_touch",
                "source": self._provider_for_instrument(runtime.spec.instrument_id),
                "roll_action": action, "roll_id": roll_key,
            }
            legs.append(leg)
            self.trades.append(leg)
            runtime.total_fills += 1
            runtime.total_traded_notional += quantity * abs(price) * runtime.spec.contract_multiplier
            self.total_fills += 1
            self.total_traded_notional += quantity * abs(price) * runtime.spec.contract_multiplier
        self.trades = self.trades[-2000:]
        completed = quantity >= abs(old_position) - 1e-12
        if completed:
            self._rolled_futures.add(old_instrument_id)
        self.ledger = candidate
        original_budgets = {old_instrument_id: dict(old.roll_liquidity),
                            new_instrument_id: dict(new.roll_liquidity)}
        for runtime, side in ((old, close_side), (new, open_side)):
            price, displayed, available = runtime.roll_liquidity[side]
            runtime.roll_liquidity[side] = (price, displayed, available - quantity)
        try:
            self._persist_event("futures_roll", "roll:" + roll_key, {
                "old_instrument_id": old_instrument_id, "new_instrument_id": new_instrument_id,
                "legs": legs,
            }, stamp)
        except Exception:
            self._restore_failed_ledger(original_ledger)
            old.roll_liquidity = original_budgets[old_instrument_id]
            new.roll_liquidity = original_budgets[new_instrument_id]
            if completed:
                self._rolled_futures.discard(old_instrument_id)
            for runtime, price in ((old, close_price), (new, open_price)):
                runtime.total_fills -= 1
                runtime.total_traded_notional -= quantity * abs(price) * runtime.spec.contract_multiplier
                self.total_fills -= 1
                self.total_traded_notional -= quantity * abs(price) * runtime.spec.contract_multiplier
            self.trades = original_trades
            raise
        for runtime in (old, new):
            self._cancel_instrument_quotes(runtime, "futures_roll_committed")
            runtime.strategy.sync_inventory(candidate.get_position(runtime.spec.instrument_id, as_of=stamp).quantity)
        if completed:
            old.market_open = False
            old.market_status = "rolled"
        for leg in legs:
            self.emit("strategy_fill", leg)
        result = {"rolled": True, "roll_id": roll_key, "legs": legs}
        self.emit("futures_roll", result)
        self._emit_dashboard_state()
        return result

    def get_dashboard_state(self) -> Dict[str, Any]:
        now = float(self.clock())
        instrument_rows = []
        for runtime in self._runtime.values():
            spec = runtime.spec
            readiness = self._instrument_readiness(runtime, now)
            position_error = None
            try:
                position = self.ledger.get_position(spec.instrument_id, as_of=now)
            except ValueError as exc:
                position_error = str(exc)
                raw = self.ledger._positions[spec.instrument_id]
                position = self.ledger.get_position(spec.instrument_id,
                    as_of=raw.mark_timestamp if raw.mark_timestamp is not None else now)
            instrument_rows.append({
                "instrument_id": spec.instrument_id,
                "symbol": spec.symbol,
                "asset_class": spec.asset_class.value,
                "venue": spec.venue,
                "quantity_unit": spec.quantity_unit,
                "quote_currency": spec.quote_currency,
                "settlement_currency": spec.settlement_currency,
                "fx_accounting_mode": (spec.fx_accounting_mode.value
                                        if spec.fx_accounting_mode else None),
                "contract_multiplier": spec.contract_multiplier,
                "expiry": spec.expiry.isoformat() if spec.expiry else None,
                "roll_to": runtime.config.roll_to,
                "auto_roll": runtime.config.auto_roll,
                "provider": self._provider_for_instrument(spec.instrument_id),
                "market_data_ready": readiness["code"] == "ready",
                "readiness": readiness,
                "market_open": runtime.market_open,
                "market_status": runtime.market_status,
                "accounting_status": self._accounting_blocks.get(spec.instrument_id, "ready"),
                "valuation_valid": position_error is None,
                "valuation_error": position_error,
                "mark_basis": position.mark_basis,
                "valuation_valid_until": position.valuation_valid_until,
                "quote_age_seconds": self._age(now, runtime.last_quote_receive_ts),
                "trade_age_seconds": self._age(now, runtime.last_trade_receive_ts),
                "best_bid": runtime.bids[0][0] if runtime.bids else None,
                "best_ask": runtime.asks[0][0] if runtime.asks else None,
                "bids": [{"price": price, "size": size} for price, size in runtime.bids[:10]],
                "asks": [{"price": price, "size": size} for price, size in runtime.asks[:10]],
                "mid_price": runtime.mid_price,
                "position": position.quantity,
                "total_traded_notional": runtime.total_traded_notional,
                "realized_pnl": position.realized_pnl,
                "unrealized_pnl": position.unrealized_pnl,
                "risk_reduction_status": self._risk_reduction_status(runtime, position.quantity),
                "active_orders": [self._order_payload(row) for row in runtime.active_orders.values()],
                "public_trades": list(runtime.recent_public_trades),
                "total_orders": runtime.total_orders,
                "total_fills": runtime.total_fills,
                "total_rejections": runtime.total_rejections,
                "last_rejection": dict(runtime.last_rejection) if runtime.last_rejection is not None else None,
                "last_rejection_available": runtime.total_rejections == 0 or runtime.last_rejection is not None,
                "spread_bps": runtime.strategy.spread_bps,
                "inventory_skew_bps": runtime.strategy.inventory_skew_bps,
            })
        portfolio, portfolio_error = self._portfolio_payload(now)
        return {
            "schema_version": 2,
            "execution_mode": "local_paper",
            "paper_mode": True,
            "is_running": self.is_running,
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "execution_assumptions_verified": not self._assumptions_unverified,
            "reporting_currency": self.config.reporting_currency,
            "risk_limits": {
                "reporting_currency": self.config.reporting_currency,
                "max_daily_loss": self.config.max_daily_loss,
                "max_gross_exposure": self.config.max_gross_exposure,
                "max_orders_per_minute": self.config.max_orders_per_minute,
                "instruments": {instrument_id: {
                    "max_order_quantity": runtime.config.max_order_quantity,
                    "max_position_quantity": runtime.config.max_position_quantity,
                    "max_notional": runtime.config.max_notional,
                    "allow_short": runtime.config.allow_short,
                    "enabled": runtime.config.enabled,
                } for instrument_id, runtime in self._runtime.items()},
            },
            "config_epoch": self.config_epoch,
            "account_actions": self.get_account_actions(),
            "accounting_complete": not bool(self._accounting_blocks),
            "portfolio": portfolio,
            "portfolio_error": portfolio_error,
            "providers": self._provider_status_payload(),
            "currency_conversion_marks": self._conversion_mark_payload(now),
            "instruments": instrument_rows,
            "recent_fills": self.trades[-100:],
            "totals": {
                "total_orders_sent": self.total_orders_sent,
                "total_fills": self.total_fills,
                "total_rejections": self.total_rejections,
                "total_traded_notional": self.total_traded_notional,
                "active_orders": sum(len(runtime.active_orders) for runtime in self._runtime.values()),
            },
        }

    def _portfolio_payload(self, now: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            snapshot = self.ledger.snapshot(as_of=now)
        except (ValueError, KeyError) as exc:
            return None, str(exc)
        return ({
            "reporting_currency": snapshot.reporting_currency,
            "equity": snapshot.equity,
            "starting_equity": snapshot.starting_equity,
            "total_pnl": snapshot.total_pnl,
            "realized_pnl": snapshot.realized_pnl,
            "unrealized_pnl": snapshot.unrealized_pnl,
            "translation_pnl": snapshot.translation_pnl,
            "fees_paid": snapshot.fees_paid,
            "costs_paid": snapshot.costs_paid,
            "accrued_costs": snapshot.accrued_costs,
            "accrued_costs_by_currency": dict(snapshot.accrued_costs_by_currency),
            "income": snapshot.income,
            "cash_by_currency": dict(snapshot.cash_by_currency),
            "available_cash_by_currency": dict(snapshot.available_cash_by_currency),
            "reserved_cash_by_currency": dict(snapshot.reserved_cash_by_currency),
            "reserved_margin_by_currency": dict(snapshot.reserved_margin_by_currency),
            "initial_margin_required": snapshot.initial_margin_required,
            "maintenance_margin_required": snapshot.maintenance_margin_required,
        }, None)

    def _risk_reduction_status(self, runtime: _InstrumentRuntime, position: float) -> str:
        if not self.circuit_breaker_active or abs(position) <= 1e-12:
            return "normal"
        units = math.floor((abs(position) + 1e-12) / runtime.spec.quantity_step)
        reducible = units * runtime.spec.quantity_step
        if reducible + 1e-12 < runtime.spec.min_quantity:
            return "dust_below_minimum_lot"
        return "reduce_only"

    def _conversion_mark_payload(self, now: float) -> List[Dict[str, Any]]:
        reporting = self.config.reporting_currency
        marks = []
        for (base, quote), (rate, timestamp, max_age) in self._fx_edges.items():
            if quote == reporting:
                currency, direct_rate = base, rate
            elif base == reporting:
                currency, direct_rate = quote, 1.0 / rate if rate else 0.0
            else:
                continue
            age = now - timestamp
            marks.append({
                "currency": currency,
                "reporting_currency": reporting,
                "rate": direct_rate,
                "source": self._conversion_mark_sources.get(currency, "provider_fx_quote"),
                "age_seconds": max(0.0, age),
                "max_age_seconds": max_age,
                "stale": age < 0 or age > max_age,
            })
        return marks

    def _provider_status_payload(self) -> List[Dict[str, Any]]:
        result = []
        for provider_id, provider in self._providers.items():
            status = provider.status
            result.append({
                "provider_id": provider_id,
                "provider": status.provider,
                "state": status.state.value,
                "message": status.message,
                "connected": status.connected,
                "events_received": status.events_received,
                "last_event_ts": status.last_event_ts,
                "last_receive_ts": status.last_receive_ts,
                "last_heartbeat_ts": status.last_heartbeat_ts,
                "instrument_ids": self._provider_instruments[provider_id],
                "capabilities": {
                    "quote": provider.capabilities.quote,
                    "order_book": provider.capabilities.order_book,
                    "trades": provider.capabilities.trades,
                    "sessions": provider.capabilities.sessions,
                    "status": provider.capabilities.status,
                },
            })
        return result

    def _provider_for_instrument(self, instrument_id: str) -> Optional[str]:
        for provider_id, ids in self._provider_instruments.items():
            if instrument_id in ids:
                return provider_id
        return None

    @staticmethod
    def _age(now: float, timestamp: Optional[float]) -> Optional[float]:
        return None if timestamp is None else max(0.0, now - timestamp)

    @staticmethod
    def _order_payload(working: _WorkingOrder) -> Dict[str, Any]:
        return {
            "instrument_id": working.order.instrument_id,
            "order_id": working.order.order_id,
            "side": working.order.side.name,
            "price": float(working.order.price),
            "quantity": float(working.order.quantity),
            "remaining_quantity": working.remaining_quantity,
            "status": "active" if working.remaining_quantity == float(working.order.quantity)
            else "partially_filled",
            "accepted_at": working.accepted_receive_ts,
        }

    def _market_payload(self, runtime: _InstrumentRuntime) -> Dict[str, Any]:
        return {
            "instrument_id": runtime.spec.instrument_id,
            "bids": [{"price": p, "size": q} for p, q in runtime.bids[:10]],
            "asks": [{"price": p, "size": q} for p, q in runtime.asks[:10]],
            "best_bid": runtime.bids[0][0] if runtime.bids else None,
            "best_ask": runtime.asks[0][0] if runtime.asks else None,
            "mid_price": runtime.mid_price,
            "quote_age_seconds": self._age(float(self.clock()), runtime.last_quote_receive_ts),
            "market_data_ready": self._instrument_ready(runtime, float(self.clock())),
        }

    def get_status(self) -> Dict[str, Any]:
        state = self.get_dashboard_state()
        portfolio = state["portfolio"] or {}
        instruments = state["instruments"]
        selected = instruments[0] if instruments else {}
        totals = state["totals"]
        return {
            "uptime_seconds": float(self.clock()) - self.start_time,
            "position": selected.get("position", 0.0),
            "cash": portfolio.get("cash_by_currency", {}).get(self.config.reporting_currency, 0.0),
            "equity": portfolio.get("equity", self.config.starting_capital),
            "mid_price": selected.get("mid_price", 0.0),
            "realized_pnl": portfolio.get("realized_pnl", 0.0),
            "unrealized_pnl": portfolio.get("unrealized_pnl", 0.0),
            "total_pnl": portfolio.get("total_pnl", 0.0),
            "total_fees": portfolio.get("fees_paid", 0.0),
            "total_orders_sent": totals["total_orders_sent"],
            "total_fills": totals["total_fills"],
            "total_rejections": totals["total_rejections"],
            "total_traded_notional": totals["total_traded_notional"],
            "active_orders": totals["active_orders"],
            "active_orders_detail": self.get_active_orders_detail(),
            "paper_mode": True,
            "execution_mode": "local_paper",
            "market_data_source": "read-only configured providers",
            "is_running": self.is_running,
            "market_data_ready": any(item["market_data_ready"] for item in instruments),
            "instrument_states": instruments,
            "providers": state["providers"],
            "portfolio_error": state["portfolio_error"],
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "daily_pnl": self._current_daily_pnl(),
            "risk_day": self._risk_day.isoformat(),
        }

    def get_full_status(self) -> Dict[str, Any]:
        return self.get_status()

    def get_active_orders_detail(self) -> List[Dict[str, Any]]:
        return [self._order_payload(working)
                for runtime in self._runtime.values()
                for working in runtime.active_orders.values()]

    def get_trades(self, limit: int = 50) -> Dict[str, Any]:
        count = max(0, int(limit))
        return {"trades": self.trades[-count:] if count else [], "total": len(self.trades)}

    def _sync_strategy_inventory(self, runtime: _InstrumentRuntime) -> None:
        try:
            position = self.ledger.get_position(runtime.spec.instrument_id, as_of=float(self.clock())).quantity
        except ValueError:
            return
        runtime.strategy.sync_inventory(position)

