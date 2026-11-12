"""Strict configuration for multi-instrument local paper trading.

This loader only accepts market-data providers and a local paper execution
mode. Provider secrets are read by adapters from environment variables and are
never accepted from the JSON configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ..market.instruments import AssetClass, FXAccountingMode, InstrumentSpec


class CrossAssetConfigurationError(ValueError):
    """Raised when a cross-asset paper configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class FeeSchedule:
    """Explicit paper fee model; rates apply to notional and unit fees to size."""

    maker_rate: float
    taker_rate: float
    maker_per_unit: float
    taker_per_unit: float
    currency: str

    def __post_init__(self) -> None:
        for name in ("maker_rate", "taker_rate", "maker_per_unit", "taker_per_unit"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise CrossAssetConfigurationError("fees.{} must be finite and non-negative".format(name))
            object.__setattr__(self, name, value)
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise CrossAssetConfigurationError("fees.currency must be a non-empty currency code")
        object.__setattr__(self, "currency", self.currency.strip().upper())

    def calculate(self, quantity: float, price: float, multiplier: float, *, maker: bool) -> float:
        rate = self.maker_rate if maker else self.taker_rate
        per_unit = self.maker_per_unit if maker else self.taker_per_unit
        amount = float(quantity) * abs(float(price)) * float(multiplier) * rate + float(quantity) * per_unit
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("calculated fee must be finite and non-negative")
        return amount


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    provider_type: str
    instrument_ids: Tuple[str, ...]
    dataset: Optional[str] = None
    schemas: Tuple[str, ...] = ()
    stype_in: str = "raw_symbol"
    environment: str = "practice"
    account_id_env: str = "OANDA_ACCOUNT_ID"
    symbols: Tuple[str, ...] = ()


@dataclass(frozen=True)
class InstrumentTradingConfig:
    spec: InstrumentSpec
    provider_id: str
    provider_symbol: str
    order_size: float
    max_order_quantity: float
    max_position_quantity: float
    max_notional: float
    allow_short: bool
    queue_fraction: float
    maker_latency_ms: float
    spread_bps: float
    inventory_skew_bps: float
    fees: FeeSchedule
    fill_model: str
    auto_roll: bool = False
    roll_to: Optional[str] = None
    roll_before_expiry_seconds: float = 86400.0
    enabled: bool = True


@dataclass(frozen=True)
class CrossAssetConfig:
    starting_capital: float
    reporting_currency: str
    max_daily_loss: float
    max_gross_exposure: float
    max_orders_per_minute: int
    max_mark_age_seconds: float
    market_data_stale_after: float
    providers: Mapping[str, ProviderConfig]
    instruments: Mapping[str, InstrumentTradingConfig]
    account_id: str = "cross_asset_paper"
    state_db_path: str = "data/trading_state.db"
    conversion_marks: Mapping[str, Tuple[float, str, float]] = field(default_factory=dict)
    starting_balances: Mapping[str, float] = field(default_factory=dict)
    max_clock_skew_seconds: float = 3.0


_FORBIDDEN_SECRET_KEYS = frozenset({
    "api_key", "api_secret", "access_token", "authorization", "bearer_token", "password",
})
_PROVIDER_TYPES = frozenset({"binance", "databento", "oanda", "coinbase"})
_FILL_MODELS = frozenset({"trade_print_queue", "fx_quote_cross"})


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrossAssetConfigurationError("{} must be numeric".format(name)) from exc
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CrossAssetConfigurationError("{} must be finite and {}".format(name, qualifier))
    return number


def _reject_embedded_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_SECRET_KEYS:
                raise CrossAssetConfigurationError(
                    "{} contains a secret field; configure provider credentials in environment variables".format(
                        path + "." + str(key)))
            _reject_embedded_secrets(child, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_embedded_secrets(child, "{}[{}]".format(path, index))


def _parse_expiry(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CrossAssetConfigurationError("instrument expiry must be an ISO-8601 string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CrossAssetConfigurationError("instrument expiry must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise CrossAssetConfigurationError("instrument expiry must include a timezone")
    return result


def _parse_instrument(raw: Mapping[str, Any]) -> InstrumentTradingConfig:
    try:
        metadata = dict(raw["instrument"])
        metadata["asset_class"] = AssetClass(metadata["asset_class"])
        if metadata.get("fx_accounting_mode") is not None:
            metadata["fx_accounting_mode"] = FXAccountingMode(metadata["fx_accounting_mode"])
        metadata["expiry"] = _parse_expiry(metadata.get("expiry"))
        metadata["capabilities"] = frozenset(metadata.get("capabilities", ()))
        spec = InstrumentSpec(**metadata)
        strategy = raw["strategy"]
        risk = raw["risk"]
        raw_fees = raw["fees"]
        roll = raw.get("roll", {})
        if not isinstance(roll, Mapping):
            raise CrossAssetConfigurationError("roll must be an object")
        fees = FeeSchedule(
            maker_rate=raw_fees["maker_rate"],
            taker_rate=raw_fees["taker_rate"],
            maker_per_unit=raw_fees["maker_per_unit"],
            taker_per_unit=raw_fees["taker_per_unit"],
            currency=raw_fees["currency"],
        )
        result = InstrumentTradingConfig(
            spec=spec,
            provider_id=str(raw["provider_id"]).strip(),
            provider_symbol=str(raw["provider_symbol"]).strip(),
            order_size=_positive_number(strategy["order_size"], "strategy.order_size"),
            max_order_quantity=_positive_number(risk["max_order_quantity"], "risk.max_order_quantity"),
            max_position_quantity=_positive_number(risk["max_position_quantity"], "risk.max_position_quantity"),
            max_notional=_positive_number(risk["max_notional"], "risk.max_notional"),
            allow_short=risk.get("allow_short", False),
            queue_fraction=_positive_number(raw.get("queue_fraction", 0.5), "queue_fraction", allow_zero=True),
            maker_latency_ms=_positive_number(raw.get("maker_latency_ms", 100), "maker_latency_ms", allow_zero=True),
            spread_bps=_positive_number(strategy.get("spread_bps", 8), "strategy.spread_bps", allow_zero=True),
            inventory_skew_bps=_positive_number(strategy.get("inventory_skew_bps", 8), "strategy.inventory_skew_bps", allow_zero=True),
            fees=fees,
            fill_model=str(raw["fill_model"]).strip().lower(),
            auto_roll=roll.get("auto_roll", False),
            roll_to=(str(roll["roll_to"]).strip() if roll.get("roll_to") else None),
            roll_before_expiry_seconds=_positive_number(
                roll.get("roll_before_expiry_seconds", 86400),
                "roll.roll_before_expiry_seconds"),
            enabled=raw.get("enabled", True),
        )
    except KeyError as exc:
        raise CrossAssetConfigurationError("missing required instrument configuration field: {}".format(exc)) from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CrossAssetConfigurationError):
            raise
        raise CrossAssetConfigurationError("invalid instrument configuration: {}".format(exc)) from exc

    if not result.provider_id or not result.provider_symbol:
        raise CrossAssetConfigurationError("provider_id and provider_symbol must be non-empty")
    if not isinstance(result.allow_short, bool):
        raise CrossAssetConfigurationError("risk.allow_short must be a JSON boolean")
    if not isinstance(result.auto_roll, bool):
        raise CrossAssetConfigurationError("roll.auto_roll must be a JSON boolean")
    if not isinstance(result.enabled, bool):
        raise CrossAssetConfigurationError("enabled must be a JSON boolean")
    if result.fees.currency not in {result.spec.quote_currency, result.spec.settlement_currency}:
        raise CrossAssetConfigurationError(
            "fees.currency must match the instrument quote or settlement currency"
        )
    if result.fill_model not in _FILL_MODELS:
        raise CrossAssetConfigurationError("fill_model must be one of {}".format(sorted(_FILL_MODELS)))
    if result.queue_fraction > 1:
        raise CrossAssetConfigurationError("queue_fraction must be in [0, 1]")
    if result.order_size > result.max_order_quantity:
        raise CrossAssetConfigurationError("strategy.order_size cannot exceed risk.max_order_quantity")
    if result.max_order_quantity > result.max_position_quantity:
        raise CrossAssetConfigurationError("max_order_quantity cannot exceed max_position_quantity")
    if result.allow_short and result.spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}:
        if result.spec.borrow_limit is None or result.spec.borrow_fee_rate is None:
            raise CrossAssetConfigurationError("cash-security shorting requires explicit borrow limit and fee")
        if (not result.spec.borrow_rate_source or result.spec.borrow_rate_effective_from is None
                or result.spec.borrow_rate_effective_to is None):
            raise CrossAssetConfigurationError("cash-security shorting requires sourced effective-dated borrow terms")
    if (result.allow_short and result.spec.asset_class == AssetClass.FX
            and result.spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE):
        if result.spec.borrow_limit is None or result.spec.borrow_fee_rate is None:
            raise CrossAssetConfigurationError("deliverable FX shorting requires explicit borrow limit and fee")
        if (not result.spec.borrow_rate_source or result.spec.borrow_rate_effective_from is None
                or result.spec.borrow_rate_effective_to is None):
            raise CrossAssetConfigurationError("deliverable FX borrowing requires sourced effective-dated terms")
    if result.spec.asset_class == AssetClass.FX and result.fill_model != "fx_quote_cross":
        raise CrossAssetConfigurationError("spot FX must use the explicit fx_quote_cross estimate model")
    if result.spec.asset_class != AssetClass.FX and result.fill_model != "trade_print_queue":
        raise CrossAssetConfigurationError("non-FX instruments must use trade_print_queue")
    return result


def load_cross_asset_config(path: Union[str, Path]) -> CrossAssetConfig:
    """Load and validate a local-paper multi-asset JSON configuration."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CrossAssetConfigurationError("cannot read configuration {}: {}".format(config_path, exc)) from exc
    except json.JSONDecodeError as exc:
        raise CrossAssetConfigurationError("invalid JSON in {}: {}".format(config_path, exc)) from exc
    return parse_cross_asset_config(raw)


def parse_cross_asset_config(raw: Dict[str, Any]) -> CrossAssetConfig:
    """Validate an in-memory public configuration using the file-loader rules."""
    if not isinstance(raw, dict):
        raise CrossAssetConfigurationError("configuration root must be an object")
    _reject_embedded_secrets(raw)
    if raw.get("mode") != "paper" or raw.get("execution_mode") != "local_paper":
        raise CrossAssetConfigurationError("only mode=paper and execution_mode=local_paper are supported")

    providers: Dict[str, ProviderConfig] = {}
    for item in raw.get("providers", []):
        try:
            provider_id = str(item["provider_id"]).strip()
            provider_type = str(item["type"]).strip().lower()
            instrument_ids = tuple(str(value).strip() for value in item["instrument_ids"])
        except (KeyError, TypeError) as exc:
            raise CrossAssetConfigurationError("each provider needs provider_id, type, instrument_ids") from exc
        if not provider_id or provider_id in providers:
            raise CrossAssetConfigurationError("provider_id must be unique and non-empty")
        if provider_type not in _PROVIDER_TYPES:
            raise CrossAssetConfigurationError("unsupported market-data provider: {}".format(provider_type))
        if not instrument_ids or any(not value for value in instrument_ids):
            raise CrossAssetConfigurationError("provider {} needs non-empty instrument_ids".format(provider_id))
        if provider_type == "coinbase" and len(instrument_ids) != 1:
            raise CrossAssetConfigurationError("Coinbase valuation quotes require exactly one USDT/USD instrument")
        providers[provider_id] = ProviderConfig(
            provider_id=provider_id,
            provider_type=provider_type,
            instrument_ids=instrument_ids,
            dataset=item.get("dataset"),
            schemas=tuple(item.get("schemas", ())),
            stype_in=str(item.get("stype_in", "raw_symbol")),
            environment=str(item.get("environment", "practice")),
            account_id_env=str(item.get("account_id_env", "OANDA_ACCOUNT_ID")),
            symbols=tuple(item.get("symbols", ())),
        )

    instruments: Dict[str, InstrumentTradingConfig] = {}
    for item in raw.get("instruments", []):
        parsed = _parse_instrument(item)
        instrument_id = parsed.spec.instrument_id
        if instrument_id in instruments:
            raise CrossAssetConfigurationError("duplicate instrument_id: {}".format(instrument_id))
        if parsed.provider_id not in providers:
            raise CrossAssetConfigurationError("instrument {} references unknown provider {}".format(
                instrument_id, parsed.provider_id))
        provider = providers[parsed.provider_id]
        if instrument_id not in provider.instrument_ids:
            raise CrossAssetConfigurationError("provider {} does not declare instrument {}".format(
                provider.provider_id, instrument_id))
        if provider.provider_type == "databento" and parsed.spec.asset_class not in {AssetClass.EQUITY, AssetClass.FUTURE}:
            raise CrossAssetConfigurationError("Databento provider supports equity/futures in this app")
        if provider.provider_type == "oanda" and parsed.spec.asset_class != AssetClass.FX:
            raise CrossAssetConfigurationError("OANDA provider accepts FX instruments only")
        if provider.provider_type == "binance" and parsed.spec.asset_class != AssetClass.CRYPTO:
            raise CrossAssetConfigurationError("Binance provider accepts crypto instruments only")
        if provider.provider_type == "coinbase" and (
                parsed.enabled or parsed.spec.asset_class != AssetClass.FX
                or parsed.spec.fx_accounting_mode != FXAccountingMode.DELIVERABLE
                or parsed.spec.base_currency != "USDT" or parsed.spec.quote_currency != "USD"
                or parsed.spec.settlement_currency != "USD" or parsed.provider_symbol != "USDT-USD"
                or parsed.spec.quantity_unit != "USDT" or parsed.spec.venue != "COINBASE"
                or parsed.spec.contract_multiplier != 1
                or parsed.fill_model != "fx_quote_cross"):
            raise CrossAssetConfigurationError(
                "Coinbase valuation quotes require a disabled deliverable USDT/USD instrument")
        instruments[instrument_id] = parsed

    all_provider_ids = {item for provider in providers.values() for item in provider.instrument_ids}
    if not instruments or set(instruments) != all_provider_ids:
        raise CrossAssetConfigurationError("provider and instrument declarations must match exactly")

    for instrument in instruments.values():
        if instrument.auto_roll and not instrument.roll_to:
            raise CrossAssetConfigurationError("auto_roll requires an explicit roll_to instrument_id")
        if instrument.roll_to:
            target = instruments.get(instrument.roll_to)
            if instrument.spec.asset_class != AssetClass.FUTURE or target is None:
                raise CrossAssetConfigurationError("roll_to must name a configured futures instrument")
            if (target.spec.asset_class != AssetClass.FUTURE
                    or target.spec.venue != instrument.spec.venue
                    or target.spec.base_currency != instrument.spec.base_currency
                    or target.spec.settlement_currency != instrument.spec.settlement_currency
                    or target.spec.contract_multiplier != instrument.spec.contract_multiplier
                    or target.spec.expiry <= instrument.spec.expiry):
                raise CrossAssetConfigurationError("roll_to futures contract metadata is incompatible")

    try:
        reporting_currency = str(raw.get("reporting_currency", "USD")).strip().upper()
        risk = raw["risk"]
        account_id = str(raw.get("account_id", "cross_asset_paper")).strip()
        state_db_path = str(raw.get("state_db_path", "data/trading_state.db")).strip()
        if not account_id or not state_db_path:
            raise CrossAssetConfigurationError("account_id and state_db_path must be non-empty")
        conversion_marks = {}
        raw_marks = raw.get("conversion_marks", {})
        if not isinstance(raw_marks, Mapping):
            raise CrossAssetConfigurationError("conversion_marks must be an object")
        for currency, mark in raw_marks.items():
            code = str(currency).strip().upper()
            if not code or code == reporting_currency or not isinstance(mark, Mapping):
                raise CrossAssetConfigurationError("conversion mark must name a non-reporting currency object")
            rate = _positive_number(mark.get("rate_to_reporting"), "conversion_marks.{}.rate".format(code))
            source = str(mark.get("source", "")).strip()
            age = _positive_number(mark.get("max_age_seconds"),
                                   "conversion_marks.{}.max_age_seconds".format(code))
            if not source:
                raise CrossAssetConfigurationError("conversion mark {} needs an explicit source label".format(code))
            conversion_marks[code] = (rate, source, age)
        starting_capital = _positive_number(raw["starting_capital"], "starting_capital")
        starting_balances = {}
        raw_balances = raw.get("starting_balances", {reporting_currency: starting_capital})
        if not isinstance(raw_balances, Mapping):
            raise CrossAssetConfigurationError("starting_balances must be an object")
        for currency, amount in raw_balances.items():
            code = str(currency).strip().upper()
            if not code:
                raise CrossAssetConfigurationError("starting balance currency must not be empty")
            starting_balances[code] = _positive_number(
                amount, "starting_balances.{}".format(code), allow_zero=True)
        if not starting_balances or not any(value > 0 for value in starting_balances.values()):
            raise CrossAssetConfigurationError("starting_balances must include a positive balance")
        opening_equity = 0.0
        for currency, amount in starting_balances.items():
            if currency == reporting_currency:
                rate = 1.0
            elif currency in conversion_marks:
                rate = conversion_marks[currency][0]
            else:
                raise CrossAssetConfigurationError(
                    "starting balance {} needs an explicit conversion mark to {}".format(
                        currency, reporting_currency))
            opening_equity += amount * rate
        if not math.isclose(opening_equity, starting_capital, rel_tol=1e-8, abs_tol=0.01):
            raise CrossAssetConfigurationError(
                "starting_balances must reconcile to starting_capital in the reporting currency")
        return CrossAssetConfig(
            starting_capital=starting_capital,
            reporting_currency=reporting_currency,
            max_daily_loss=_positive_number(risk["max_daily_loss"], "risk.max_daily_loss"),
            max_gross_exposure=_positive_number(risk["max_gross_exposure"], "risk.max_gross_exposure"),
            max_orders_per_minute=int(_positive_number(risk["max_orders_per_minute"], "risk.max_orders_per_minute")),
            max_mark_age_seconds=_positive_number(raw.get("max_mark_age_seconds", 30), "max_mark_age_seconds"),
            market_data_stale_after=_positive_number(raw.get("market_data_stale_after", 10), "market_data_stale_after"),
            providers=providers,
            instruments=instruments,
            account_id=account_id,
            state_db_path=state_db_path,
            conversion_marks=conversion_marks,
            starting_balances=starting_balances,
            max_clock_skew_seconds=_positive_number(
                raw.get("max_clock_skew_seconds", 3),
                "max_clock_skew_seconds", allow_zero=True),
        )
    except KeyError as exc:
        raise CrossAssetConfigurationError("missing required top-level configuration field: {}".format(exc)) from exc
