"""Assumption-based cash and Gaussian bounds; no market or account access."""

import math
from decimal import ROUND_FLOOR, Decimal
from statistics import NormalDist


def _finite(value: float | Decimal, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return number


def cash_round_trip_limit(
    capital: float | Decimal,
    notional: float | Decimal,
    entry_cost: float | Decimal,
    *,
    settled_fraction: float = 1.0,
) -> int:
    """Upper bound on fully paid tickets using distinct settled dollars once.

    Exit costs must fit the sale proceeds. This assumes a flat-price sale and
    no borrowing, investment gains, fill limitation or unsettled-proceeds reuse.
    Capital is the available USD after prefunding charges, not EUR valuation.
    """
    _finite(capital, "capital")
    _finite(notional, "notional", positive=True)
    _finite(entry_cost, "entry_cost")
    fraction = _finite(settled_fraction, "settled_fraction")
    if fraction > 1:
        raise ValueError("settled_fraction must be at most 1")
    funds = Decimal(str(capital)) * Decimal(str(settled_fraction))
    ticket = Decimal(str(notional)) + Decimal(str(entry_cost))
    return int((funds / ticket).to_integral_value(rounding=ROUND_FLOOR))


def cycles_per_slot(
    holding_minutes: float,
    *,
    entry_minutes: float = 375,
    flat_minutes: float = 385,
) -> int:
    """Perfect-fill cycles for a fixed assumed duration, not a time-stop bound.

    Starts lie in [open, entry_cutoff) and the final exit is no later than the
    flat deadline. Quote/cancel latency and execution delays only reduce this.
    """
    holding = _finite(holding_minutes, "holding_minutes", positive=True)
    entry = _finite(entry_minutes, "entry_minutes", positive=True)
    flat = _finite(flat_minutes, "flat_minutes", positive=True)
    if flat < entry:
        raise ValueError("flat_minutes must not precede entry_minutes")
    return min(math.ceil(entry / holding), math.floor(flat / holding))


def required_net_edge(
    target: float,
    sigma_bps: float,
    trades: int,
    *,
    correlation: float = 0.0,
    activity: float = 1.0,
) -> dict[str, float] | None:
    """Required active-day mean after execution costs, before fixed overhead.

    Equal-notional jointly Gaussian round trips have equicorrelation rho.
    Inactive sessions are non-green. The returned buffer excludes fixed
    overhead: add its actual dollars divided by n*notional separately.
    """
    probability = _finite(target, "target", positive=True)
    sigma = _finite(sigma_bps, "sigma_bps", positive=True)
    rho = _finite(correlation, "correlation")
    active = _finite(activity, "activity", positive=True)
    if not 0.5 < probability < 1 or rho > 1 or active > 1:
        raise ValueError("target must be in (0.5, 1); correlation and activity at most 1")
    if isinstance(trades, bool) or not isinstance(trades, int) or trades < 1:
        raise ValueError("trades must be a positive integer")
    conditional = probability / active
    if conditional >= 1:
        return None
    effective = trades / (1 + (trades - 1) * rho)
    return {
        "all_in_net_bps": NormalDist().inv_cdf(conditional) * sigma / math.sqrt(effective),
        "effective_count": effective,
        "traded_day_green_target": conditional,
    }


def session_overhead(monthly_usd: float, sessions_in_billing_month: int) -> float:
    """Allocate known recurring cost across every session in that billing month."""
    amount = _finite(monthly_usd, "monthly_usd")
    if (
        isinstance(sessions_in_billing_month, bool)
        or not isinstance(sessions_in_billing_month, int)
        or sessions_in_billing_month < 1
    ):
        raise ValueError("sessions_in_billing_month must be a positive integer")
    return amount / sessions_in_billing_month
