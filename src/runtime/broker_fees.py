"""Unrounded 11 Nov 2026 IBIE Pro USD stock costs for the offline research envelope.

US whole shares, direct individual cash account, regular-hours SmartRouting.
Tiered venue examples use a Nasdaq-listed, Nasdaq/Island-displayed execution.
Sources: IBIE /en/pricing/commissions-stocks.php and /en/accounts/fees/INETstkfee.php,
SEC Section 31 advisory 2026-2, FINRA filing SR-FINRA-2026-021.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, getcontext, localcontext
from types import MappingProxyType

AS_OF = date(2026, 11, 11)
ZERO = Decimal(0)
HALF_PERCENT = Decimal("0.005")
ONE_PERCENT = Decimal("0.01")
SEC_RATE = Decimal("0.0000206")


def _finite(value: Decimal, field: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{field} must be a finite Decimal, got {value!r}")


def _stamp_scenario(rate: Decimal) -> None:
    _finite(rate, "stamp_rate")
    if rate not in (ZERO, Decimal("0.04")):
        raise ValueError(f"stamp_rate must select the 0 or 0.04 commission scenario, got {rate}")


@dataclass(frozen=True)
class FeeAssumptions:
    """monthly_shares_before counts eligible volume already traded on Tiered this month."""

    pricing: str
    stamp_rate: Decimal
    finra_pass_through: bool
    tiered_cap: Decimal | None = None
    monthly_shares_before: int = 0

    def __post_init__(self) -> None:
        if self.pricing not in ("fixed", "tiered"):
            raise ValueError(f"pricing must be fixed or tiered, got {self.pricing!r}")
        _stamp_scenario(self.stamp_rate)
        if type(self.finra_pass_through) is not bool:
            raise ValueError("finra_pass_through must explicitly be True or False")
        if type(self.monthly_shares_before) is not int or self.monthly_shares_before < 0:
            raise ValueError(
                f"monthly_shares_before must be a nonnegative integer, got {self.monthly_shares_before!r}"
            )
        if self.tiered_cap is not None:
            _finite(self.tiered_cap, "tiered_cap")
            if self.pricing != "tiered" or self.tiered_cap not in (HALF_PERCENT, ONE_PERCENT):
                raise ValueError(
                    f"tiered_cap requires Tiered pricing and 0.005 or 0.01, got {self.tiered_cap}"
                )


@dataclass(frozen=True)
class FeeBreakdown:
    components: Mapping[str, Decimal]
    eligible_tiered_shares: int

    @property
    def total(self) -> Decimal:
        return sum(self.components.values(), ZERO)


def _tiered_commission(quantity: int, before: int) -> Decimal:
    after = before + quantity
    lower = 0
    commission = ZERO
    for upper, rate in (
        (300_000, "0.0035"),
        (3_000_000, "0.0020"),
        (20_000_000, "0.0015"),
        (100_000_000, "0.0010"),
    ):
        shares = max(0, min(after, upper) - max(before, lower))
        commission += shares * Decimal(rate)
        lower = upper
    commission += max(0, after - max(before, lower)) * Decimal("0.0005")
    return max(commission, Decimal("0.35"))


def order_fees(
    quantity: int,
    price: Decimal,
    side: str,
    assumptions: FeeAssumptions,
    *,
    liquidity: str = "remove",
) -> FeeBreakdown:
    """Price cumulative fills at their share-weighted average within one fee band.

    The fills must share venue, add/remove status, and price band for every
    execution-based charge. In particular, fills straddling USD 1 on Nasdaq
    require separate venue calculations while retaining the order commission minimum.
    """
    if type(quantity) is not int or quantity < 1:
        raise ValueError(f"quantity must be a positive whole number of shares, got {quantity!r}")
    _finite(price, "price")
    if price <= 0 or side not in ("buy", "sell") or liquidity not in ("add", "remove"):
        raise ValueError(
            f"expected positive price, buy/sell side and add/remove liquidity, got {price}, {side!r}, {liquidity!r}"
        )
    value = price * quantity
    clearing = venue = nyse_pass = finra_pass = ZERO
    eligible = 0
    if assumptions.pricing == "fixed":
        commission = min(max(Decimal(1), quantity * Decimal("0.005")), value * ONE_PERCENT)
    else:
        uncapped = _tiered_commission(quantity, assumptions.monthly_shares_before)
        cap = assumptions.tiered_cap
        if cap is None:
            if min(uncapped, value * HALF_PERCENT) != min(uncapped, value * ONE_PERCENT):
                raise ValueError(
                    "conflicting published Tiered caps affect this order; select tiered_cap 0.005 or 0.01"
                )
            cap = ONE_PERCENT
        commission = min(uncapped, value * cap)
        if commission == uncapped:
            eligible = quantity
        clearing = min(Decimal("0.00020") * quantity, HALF_PERCENT * value)
        nyse_pass = commission * Decimal("0.000175")
        if assumptions.finra_pass_through:
            finra_pass = commission * Decimal("0.00056")
        if price >= 1:
            venue = quantity * (Decimal("0.003") if liquidity == "remove" else Decimal("-0.0013"))
        elif liquidity == "remove":
            venue = value * Decimal("0.003")
    return FeeBreakdown(
        MappingProxyType(
            {
                "commission": commission,
                "sec_section31": value * SEC_RATE if side == "sell" else ZERO,
                "finra_taf": ZERO,
                "cat": Decimal("0.000003") * quantity,
                "clearing": clearing,
                "nyse_pass_through": nyse_pass,
                "finra_pass_through": finra_pass,
                "venue": venue,
                "stamp": assumptions.stamp_rate * commission,
            }
        ),
        eligible,
    )


def round_trip_fees(
    quantity: int,
    entry_price: Decimal,
    exit_price: Decimal,
    assumptions: FeeAssumptions,
    *,
    entry_liquidity: str = "add",
    exit_liquidity: str = "remove",
) -> FeeBreakdown:
    """Two same-month orders, with the entry's eligible volume included before the sale."""
    entry = order_fees(quantity, entry_price, "buy", assumptions, liquidity=entry_liquidity)
    exit_assumptions = replace(
        assumptions,
        monthly_shares_before=assumptions.monthly_shares_before + entry.eligible_tiered_shares,
    )
    exit_fees = order_fees(quantity, exit_price, "sell", exit_assumptions, liquidity=exit_liquidity)
    return FeeBreakdown(
        MappingProxyType(
            {key: val + exit_fees.components[key] for key, val in entry.components.items()}
        ),
        entry.eligible_tiered_shares + exit_fees.eligible_tiered_shares,
    )


def break_even_gain(
    quantity: int,
    entry_price: Decimal,
    assumptions: FeeAssumptions,
    *,
    extra_usd: Decimal = ZERO,
    entry_liquidity: str = "add",
    exit_liquidity: str = "remove",
) -> Decimal:
    """Minimum nonnegative total USD gain covering costs within a continuous sale-fee band."""
    _finite(extra_usd, "extra_usd")
    if extra_usd < 0:
        raise ValueError(f"extra_usd must be nonnegative, got {extra_usd}")
    _finite(entry_price, "entry_price")
    if assumptions.pricing == "tiered" and entry_price < 1:
        raise ValueError(
            "Tiered break-even requires an entry price of at least USD 1 (Nasdaq fee band)"
        )
    entry = order_fees(quantity, entry_price, "buy", assumptions, liquidity=entry_liquidity)
    exit_assumptions = replace(
        assumptions,
        monthly_shares_before=assumptions.monthly_shares_before + entry.eligible_tiered_shares,
    )
    output_precision = getcontext().copy()
    with localcontext() as work:
        work.prec += 12
        gain = ZERO
        # At the root, total gain equals fees repriced at entry + gain / shares.
        # Percent-of-notional slopes are small; guard digits protect the final reported decimal.
        for _ in range(work.prec + 10):
            exit_price = entry_price + gain / quantity
            sale = order_fees(
                quantity, exit_price, "sell", exit_assumptions, liquidity=exit_liquidity
            )
            required = entry.total + extra_usd + sale.total
            if gain == ZERO and required <= ZERO:
                return ZERO  # Assumed earned rebates already cover the round trip.
            if output_precision.plus(gain) == output_precision.plus(required):
                return output_precision.plus(required)
            gain = required
    raise ValueError(
        "no continuous break-even solution within the supported fee bands at this precision"
    )


def manual_fx_fee(
    notional_usd: Decimal, eur_per_usd: Decimal, *, stamp_rate: Decimal
) -> Mapping[str, Decimal]:
    """One executed manual conversion, first FX band, USD equivalent charged in EUR."""
    _stamp_scenario(stamp_rate)
    _finite(notional_usd, "notional_usd")
    _finite(eur_per_usd, "eur_per_usd")
    if notional_usd <= 0 or eur_per_usd <= 0:
        raise ValueError(
            f"expected positive USD notional and EUR per USD, got {notional_usd}, {eur_per_usd}"
        )
    if notional_usd > 1_000_000_000:
        raise ValueError(
            f"conversion exceeds the first USD 1 billion monthly FX band: {notional_usd}"
        )
    commission = max(Decimal(2), notional_usd * Decimal("0.000020"))
    stamp = commission * stamp_rate
    return MappingProxyType(
        {
            "commission_usd_equivalent": commission,
            "commission_eur": commission * eur_per_usd,
            "stamp_usd_equivalent": stamp,
            "stamp_eur": stamp * eur_per_usd,
            "total_usd_equivalent": commission + stamp,
        }
    )
