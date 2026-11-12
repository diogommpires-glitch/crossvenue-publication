"""Unit tests for the transient self-impact model."""

import math

import pytest

from src.market.self_impact import SelfImpactModel
from src.orderbook.order import Fill
from src.utils.types import Price, Quantity, Side, Timestamp


def make_fill(order_id="order-1", side=Side.BUY, quantity=50.0):
    return Fill(
        order_id=order_id,
        side=side,
        quantity=Quantity(quantity),
        price=Price(100.0),
        timestamp=Timestamp(0.0),
        trader_id="mm_strategy",
        is_aggressor=True,
    )


def test_impact_is_normalized_by_reference_liquidity():
    model = SelfImpactModel(impact_coefficient_bps=10.0, reference_quantity=100.0)

    model.record_fill(make_fill(), Timestamp(0.0), reference_quantity=200.0)

    assert model.get_relative_impact(Price(100.0)) == pytest.approx(0.0002500312526, rel=1e-8)


def test_decay_half_life_can_be_advanced_without_private_methods():
    model = SelfImpactModel(
        impact_coefficient_bps=10.0,
        reference_quantity=100.0,
        decay_halflife=8.0,
    )
    model.record_fill(make_fill(), Timestamp(0.0))
    initial = model.get_cumulative_impact()

    model.advance_time(Timestamp(8.0))

    assert model.get_cumulative_impact() == pytest.approx(initial * 0.5, rel=1e-9)


def test_reset_clears_time_fill_and_impact_state():
    model = SelfImpactModel()
    model.record_fill(make_fill(), Timestamp(0.0))

    model.reset()

    assert model.get_cumulative_impact() == 0.0
    assert model.get_strategy_fill_count() == 0
    assert model.get_total_fill_count() == 0


def test_opposite_flow_moves_price_immediately_after_impact_saturates():
    model = SelfImpactModel(impact_coefficient_bps=5000.0, reference_quantity=1.0)
    model.record_fill(make_fill("large-buy", Side.BUY, 100.0), Timestamp(0.0), 1.0)
    saturated = model.get_cumulative_impact()
    assert saturated == pytest.approx(math.log1p(0.1))
    model.record_fill(make_fill("small-sell", Side.SELL, 0.01), Timestamp(0.0), 1.0)
    assert model.get_cumulative_impact() < saturated
    assert model.get_relative_impact(Price(100.0)) < 0.1
