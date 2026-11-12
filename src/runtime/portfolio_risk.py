"""Pure portfolio transition checks shared by quotes and local futures rolls."""

from dataclasses import dataclass, field
import math
from typing import Mapping, Tuple

from ..config.cross_asset_config import CrossAssetConfig
from ..market.instruments import AssetClass, FXAccountingMode


@dataclass(frozen=True)
class PortfolioProposal:
    """Worst-case resulting exposure, including working orders and costs."""

    position_bounds: Mapping[str, Tuple[float, float]]
    prices: Mapping[str, float]
    rates: Mapping[str, float]
    gross_exposure: float
    initial_margin: float
    fees_reporting: float = 0.0
    native_cash_required: Mapping[str, float] = field(default_factory=dict)
    strictly_reducing: bool = False


@dataclass(frozen=True)
class RiskContext:
    config: CrossAssetConfig
    equity: float
    available_cash: Mapping[str, float]
    halted: bool = False
    as_of: float = 0.0


@dataclass(frozen=True)
class RiskDecision:
    accepted: bool
    reason: str = "accepted"


class PortfolioRiskEvaluator:
    """Validate an already priced transition without mutating the account.

    Callers validate order mechanics and prove the strict-close predicate.
    Reopening a rolled contract is always new risk and never uses that escape.
    """

    def validate(self, proposal: PortfolioProposal, context: RiskContext) -> RiskDecision:
        if proposal.strictly_reducing:
            return RiskDecision(True)
        numbers = (context.equity, proposal.gross_exposure,
                   proposal.initial_margin, proposal.fees_reporting)
        if not all(math.isfinite(float(value)) for value in numbers):
            return RiskDecision(False, "nonfinite_portfolio_risk_input")
        if context.halted:
            return RiskDecision(False, "circuit_breaker_active")
        for instrument_id, bounds in proposal.position_bounds.items():
            item = context.config.instruments[instrument_id]
            spec = item.spec
            low, high = min(bounds), max(bounds)
            maximum = max(abs(low), abs(high))
            if maximum > item.max_position_quantity + 1e-9:
                return RiskDecision(False, "instrument_position_limit")
            if low < -1e-9 and not item.allow_short:
                return RiskDecision(False, "shorting_not_configured")
            borrowed = (spec.asset_class in {AssetClass.EQUITY, AssetClass.CRYPTO}
                        or (spec.asset_class == AssetClass.FX
                            and spec.fx_accounting_mode == FXAccountingMode.DELIVERABLE))
            if low < -1e-9 and borrowed:
                if spec.borrow_limit is None or abs(low) > spec.borrow_limit + 1e-9:
                    return RiskDecision(False, "borrow_limit")
                if spec.borrow_fee_rate is None:
                    return RiskDecision(False, "borrow_fee_missing")
                if (not spec.borrow_rate_source or spec.borrow_rate_effective_from is None
                        or spec.borrow_rate_effective_to is None
                        or not spec.borrow_rate_effective_from <= context.as_of < spec.borrow_rate_effective_to):
                    return RiskDecision(False, "current_sourced_borrow_terms_required")
            currency = (spec.settlement_currency if spec.asset_class == AssetClass.FUTURE
                        else spec.quote_currency)
            rate = proposal.rates.get(currency)
            price = proposal.prices.get(instrument_id)
            if maximum > 1e-9:
                if rate is None or price is None or not math.isfinite(rate) or rate <= 0:
                    return RiskDecision(False, "missing_or_stale_currency_conversion")
                if maximum * abs(price) * spec.contract_multiplier * rate > item.max_notional + 1e-9:
                    return RiskDecision(False, "instrument_notional_limit")
        if proposal.gross_exposure > context.config.max_gross_exposure + 1e-9:
            return RiskDecision(False, "portfolio_gross_exposure_limit")
        if proposal.initial_margin > context.equity - proposal.fees_reporting + 1e-9:
            return RiskDecision(False, "projected_initial_margin_exceeds_equity")
        for currency, amount in proposal.native_cash_required.items():
            if amount > context.available_cash.get(currency, 0.0) + 1e-9:
                return RiskDecision(False, "insufficient_native_currency_cash")
        return RiskDecision(True)
