from dataclasses import replace
from decimal import Decimal

import pytest

from src.runtime.broker_fees import (
    FeeAssumptions,
    break_even_gain,
    manual_fx_fee,
    order_fees,
    round_trip_fees,
)

FIXED = FeeAssumptions("fixed", Decimal(0), False)
TIERED = FeeAssumptions("tiered", Decimal(0), True)


@pytest.mark.parametrize(
    ("assumptions", "expected"),
    [
        (FIXED, "2.004126"),
        (TIERED, ".7110405"),
        (replace(TIERED, finra_pass_through=False), ".7106485"),
    ],
)
def test_one_share_two_nasdaq_removals(assumptions, expected):
    fees = round_trip_fees(1, Decimal(200), Decimal(200), assumptions, entry_liquidity="remove")
    assert fees.total == Decimal(expected)
    assert fees.components["finra_taf"] == 0
    assert fees.components["cat"] == Decimal(".000006")
    assert fees.components["sec_section31"] == Decimal(".004120")


def test_tiered_cap_conflict_requires_explicit_scenario():
    with pytest.raises(ValueError, match="cap"):
        order_fees(1, Decimal(45), "buy", TIERED)
    for cap, expected in [(".005", ".225"), (".01", ".35")]:
        fees = order_fees(1, Decimal(45), "buy", replace(TIERED, tiered_cap=Decimal(cap)))
        assert fees.components["commission"] == Decimal(expected)
    assert order_fees(10, Decimal(".2"), "buy", FIXED).components["commission"] == Decimal(".02")


def test_same_order_partials_share_minimum():
    first = order_fees(10, Decimal(100), "buy", FIXED)
    cumulative = order_fees(20, Decimal(100), "buy", FIXED)
    assert cumulative.total - first.total == Decimal(".000030")
    assert (first.total * 2) - cumulative.total == Decimal(1)


def test_venue_cashflows_and_portuguese_commission_sensitivity():
    fees = order_fees(
        2, Decimal(150), "buy", replace(TIERED, stamp_rate=Decimal(".04")), liquidity="add"
    )
    assert fees.components["venue"] == Decimal("-.0026")
    assert fees.components["stamp"] == Decimal(".014")
    assert fees.components["sec_section31"] == 0
    low_price = order_fees(20, Decimal(".02"), "sell", replace(TIERED, tiered_cap=Decimal(".005")))
    assert low_price.components["clearing"] == Decimal(".002")
    assert low_price.components["venue"] == Decimal(".0012")


def test_progressive_volume_accumulates_across_same_month_orders():
    assumptions = replace(TIERED, monthly_shares_before=299_900)
    assert order_fees(300, Decimal(100), "buy", assumptions).components["commission"] == Decimal(
        ".75"
    )
    assert round_trip_fees(300, Decimal(100), Decimal(100), assumptions).components[
        "commission"
    ] == Decimal("1.35")
    # A capped entry is ineligible for the monthly tier counter; the exit is a new order.
    tiny = replace(assumptions, tiered_cap=Decimal(".005"))
    assert round_trip_fees(300, Decimal(".1"), Decimal(100), tiny).components[
        "commission"
    ] == Decimal(".9")


@pytest.mark.parametrize(("value", "expected"), [("10000", "2"), ("200000", "4")])
def test_manual_fx_charges_reporting_currency(value, expected):
    fee = Decimal(expected)
    assert manual_fx_fee(Decimal(value), Decimal(".9"), stamp_rate=Decimal(".04")) == {
        "commission_usd_equivalent": fee,
        "commission_eur": fee * Decimal(".9"),
        "stamp_usd_equivalent": fee * Decimal(".04"),
        "stamp_eur": fee * Decimal(".9") * Decimal(".04"),
        "total_usd_equivalent": fee * Decimal("1.04"),
    }


def test_break_even_gain_reprices_sec_and_includes_allocated_cost():
    flat = round_trip_fees(10, Decimal(100), Decimal(100), FIXED)
    assert flat.total == Decimal("2.020660")
    for extra in (Decimal(0), Decimal(".72")):
        gain = break_even_gain(10, Decimal(100), FIXED, extra_usd=extra)
        assert gain == (flat.total + extra) / (1 - Decimal(".0000206"))
        exit_price = Decimal(100) + gain / 10
        assert abs(
            gain - round_trip_fees(10, Decimal(100), exit_price, FIXED).total - extra
        ) < Decimal("1e-25")


def test_break_even_crosses_sale_commission_cap():
    gain = break_even_gain(1, Decimal("99.9"), FIXED, extra_usd=Decimal(1))
    sale = order_fees(1, Decimal("99.9") + gain, "sell", FIXED)
    assert sale.components["commission"] == Decimal(1)
    assert abs(
        gain - (order_fees(1, Decimal("99.9"), "buy", FIXED).total + sale.total + 1)
    ) < Decimal("1e-25")


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: order_fees(-1, Decimal(2), "buy", FIXED),
        lambda: order_fees(True, Decimal(2), "buy", FIXED),
        lambda: order_fees(1, Decimal("NaN"), "buy", FIXED),
        lambda: order_fees(1, Decimal(-2), "buy", FIXED),
        lambda: order_fees(1, Decimal(2), "short", FIXED),
        lambda: order_fees(1, Decimal(2), "buy", FIXED, liquidity="maybe"),
        lambda: replace(FIXED, stamp_rate=Decimal("NaN")),
        lambda: replace(TIERED, tiered_cap=Decimal(".1")),
        lambda: replace(TIERED, monthly_shares_before=-1),
        lambda: replace(TIERED, finra_pass_through=1),
        lambda: manual_fx_fee(Decimal(1), Decimal("Infinity"), stamp_rate=Decimal(0)),
        lambda: manual_fx_fee(Decimal("1000000001"), Decimal(1), stamp_rate=Decimal(0)),
        lambda: manual_fx_fee(Decimal(1), Decimal(1), stamp_rate=Decimal(".5")),
        lambda: break_even_gain(1, Decimal(200), FIXED, extra_usd=Decimal(-1)),
    ],
)
def test_invalid_fee_inputs_rejected(invoke):
    with pytest.raises(ValueError):
        invoke()


def test_tiered_break_even_rejects_subdollar_entry_before_discontinuous_venue_rebate():
    assumptions = replace(TIERED, tiered_cap=Decimal(".005"))
    entry = Decimal(".9897")
    with pytest.raises(
        ValueError, match="Tiered break-even requires an entry price of at least USD 1"
    ):
        break_even_gain(1, entry, assumptions, exit_liquidity="add")
    assert order_fees(1, entry, "buy", assumptions, liquidity="add").components["venue"] == 0
    assert break_even_gain(1, entry, FIXED) > 0


def test_earned_maker_surplus_requires_no_positive_price_gain():
    high_volume = replace(TIERED, monthly_shares_before=150_000_000)
    fees = round_trip_fees(5000, Decimal(10), Decimal(10), high_volume, exit_liquidity="add")
    assert fees.total < 0
    assert break_even_gain(5000, Decimal(10), high_volume, exit_liquidity="add") == 0
