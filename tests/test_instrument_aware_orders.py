import pytest

from src.orderbook.order import Fill, Order
from src.utils.types import OrderStatus, OrderType, Price, Quantity, Side, Timestamp


def test_order_instrument_id_survives_status_transition():
    order = Order(
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=Quantity(2),
        price=Price(100),
        timestamp=Timestamp(1),
        trader_id="cross-asset-test",
        instrument_id="NASDAQ:AAPL",
    )

    updated = order.with_status(OrderStatus.ACTIVE)

    assert updated.instrument_id == "NASDAQ:AAPL"
    assert updated.status == OrderStatus.ACTIVE


def test_fill_retains_execution_evidence_metadata():
    fill = Fill(
        order_id="order-1",
        side=Side.BUY,
        quantity=Quantity(1),
        price=Price(100),
        timestamp=Timestamp(2),
        trader_id="cross-asset-test",
        is_aggressor=False,
        instrument_id="OANDA:EUR_USD",
        source_event_id="quote-9",
        execution_model="fx_quote_cross",
        estimated=True,
    )

    assert fill.instrument_id == "OANDA:EUR_USD"
    assert fill.source_event_id == "quote-9"
    assert fill.execution_model == "fx_quote_cross"
    assert fill.estimated is True


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), -1.0])
def test_order_rejects_nonfinite_or_nonpositive_quantity(quantity):
    with pytest.raises(ValueError, match="quantity"):
        Order(
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(quantity),
            price=Price(100),
            timestamp=Timestamp(1),
            trader_id="cross-asset-test",
        )


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -1.0])
def test_fill_rejects_nonfinite_or_nonpositive_price(price):
    with pytest.raises(ValueError, match="price"):
        Fill(
            order_id="order-1",
            side=Side.BUY,
            quantity=Quantity(1),
            price=Price(price),
            timestamp=Timestamp(1),
            trader_id="cross-asset-test",
            is_aggressor=False,
        )
