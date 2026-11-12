import pytest

from src.metrics.pnl import PnLTracker
from src.orderbook.order import Fill
from src.utils.types import Price, Quantity, Side, Timestamp


def fill(order_id, side, qty, price, aggressor=False):
    return Fill(order_id=order_id, side=side, quantity=Quantity(qty), price=Price(price),
                timestamp=Timestamp(1.0), trader_id="acct", is_aggressor=aggressor)


def test_average_cost_partial_close_and_fee_reconciliation():
    pnl = PnLTracker("acct", maker_fee=0.001, taker_fee=0.002, starting_capital=10_000.0)
    pnl.record_fill(fill("b1", Side.BUY, 1.0, 100.0), 101.0, Timestamp(1.0), 101.0)
    pnl.record_fill(fill("b2", Side.BUY, 1.0, 120.0), 121.0, Timestamp(2.0), 121.0)
    pnl.record_fill(fill("s1", Side.SELL, 1.0, 130.0), 129.0, Timestamp(3.0), 129.0)
    current = pnl.get_current_pnl(130.0)
    assert pnl.inventory == 1.0
    assert pnl.cost_basis == 110.0
    assert pnl.fees_paid == 0.35
    assert current["total_pnl"] == pytest.approx(current["realized_pnl"] + current["unrealized_pnl"])
    assert current["unrealized_pnl"] == 20.0


def test_reversal_creates_new_short_cost_basis():
    pnl = PnLTracker("acct", maker_fee=0.0, taker_fee=0.0)
    pnl.record_fill(fill("b", Side.BUY, 1.0, 100.0), 100.0, Timestamp(1.0), 100.0)
    pnl.record_fill(fill("s", Side.SELL, 3.0, 90.0), 90.0, Timestamp(2.0), 90.0)
    assert pnl.inventory == -2.0
    assert pnl.cost_basis == 180.0
    assert pnl.get_current_pnl(80.0)["unrealized_pnl"] == 20.0


def test_reset_clears_position_fees_and_pnl():
    pnl = PnLTracker("acct")
    pnl.record_fill(fill("b", Side.BUY, 1.0, 100.0), 100.0, Timestamp(1.0), 100.0)
    pnl.reset()
    assert pnl.inventory == 0.0
    assert pnl.fees_paid == 0.0
    assert pnl.get_current_pnl(100.0)["total_pnl"] == 0.0


def test_pnl_tracker_rejects_invalid_fee_rates():
    for fee in (float("nan"), float("inf"), -0.001):
        with pytest.raises(ValueError, match="maker_fee"):
            PnLTracker(maker_fee=fee)
        with pytest.raises(ValueError, match="taker_fee"):
            PnLTracker(taker_fee=fee)
