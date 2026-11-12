import math

import numpy as np
import pytest

from src.market.market_simulator import Event, EventType, MarketSimulator
from src.orderbook.order import Order
from src.utils.types import OrderType, Quantity, Side, Timestamp


def test_log_gbm_uses_full_elapsed_time_without_capping():
    sim = MarketSimulator(initial_price=100.0, volatility=0.5, drift=0.0, random_seed=3)
    sim.last_price_update = Timestamp(0.0)
    sim.current_time = Timestamp(25.0)
    rng = np.random.RandomState(3)
    sigma = sim.vol_per_second
    expected = 100.0 * math.exp(-0.5 * sigma * sigma * 25.0 + sigma * math.sqrt(25.0) * rng.randn())
    sim._update_mid_price(Event(Timestamp(25.0), EventType.PRICE_UPDATE, {}))
    assert float(sim.base_mid_price) == pytest.approx(expected)


def test_crypto_calendar_converts_annual_volatility_to_per_second_consistently():
    seconds_per_year = 365.25 * 24 * 3600
    sim = MarketSimulator(volatility=0.5, seconds_per_year=seconds_per_year)
    assert sim.vol_per_second * math.sqrt(seconds_per_year) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="seconds_per_year"):
        MarketSimulator(seconds_per_year=0.0)


def test_default_strategy_id_is_classified_as_self_impact():
    sim = MarketSimulator(initial_price=100.0, impact_enabled=True, random_seed=2)
    sim.initialize_book(levels=1, orders_per_level=1)
    order = Order(side=Side.BUY, order_type=OrderType.MARKET, quantity=Quantity(100),
                  timestamp=Timestamp(0.0), trader_id="as_mm")
    _, fills = sim.submit_strategy_order(order)
    assert fills
    assert sim.self_impact.strategy_id == "as_mm"
    assert sim.self_impact.get_strategy_fill_count() == sum(
        fill.trader_id == "as_mm" and fill.is_aggressor for fill in fills
    )


def test_impact_stats_distinguish_raw_and_applied_cap():
    sim = MarketSimulator(initial_price=100.0, impact_enabled=True, impact_coefficient_bps=5000.0,
                          impact_reference_quantity=1.0, random_seed=1)
    sim.initialize_book(levels=1, orders_per_level=1)
    for index in range(4):
        order = Order(side=Side.BUY, order_type=OrderType.MARKET, quantity=Quantity(10_000),
                      timestamp=Timestamp(index), trader_id="as_mm")
        sim.submit_strategy_order(order)
    stats = sim.self_impact.get_statistics()
    assert abs(stats["cumulative_impact_bps"]) <= 1_000.0 + 1e-6
    assert abs(stats["raw_cumulative_impact_bps"]) >= abs(stats["cumulative_impact_bps"])
    assert stats["max_applied_relative_impact"] == pytest.approx(0.1)


def test_impact_moves_reference_mid_without_repricing_resting_quotes():
    sim = MarketSimulator(initial_price=100.0, impact_enabled=True,
                          impact_coefficient_bps=5000.0, impact_reference_quantity=1.0,
                          random_seed=1)
    sim.initialize_book(levels=3, orders_per_level=2)
    resting_prices = {order_id: float(order.price) for order_id, order in sim.book._orders.items()}
    # Trigger a reference-price impulse while leaving untouched passive levels in the book.
    order = Order(side=Side.BUY, order_type=OrderType.MARKET, quantity=Quantity(1),
                  timestamp=Timestamp(0.0), trader_id="as_mm")
    sim.submit_strategy_order(order)
    state = sim.get_market_state()
    assert float(state["mid_price"]) > float(state["base_mid_price"])
    assert any(order_id in sim.book._orders for order_id in resting_prices)
    for order_id, price in resting_prices.items():
        if order_id in sim.book._orders:
            assert float(sim.book._orders[order_id].price) == pytest.approx(price)


def test_strategy_can_share_the_simulator_calendar():
    from src.strategies.avellaneda_stoikov import AvellanedaStoikov

    year = 3600.0
    sim = MarketSimulator(volatility=0.5, seconds_per_year=year)
    strategy = AvellanedaStoikov(volatility=0.5, seconds_per_year=year)
    assert sim.seconds_per_year == strategy.seconds_per_year
    assert sim.vol_per_second == pytest.approx(strategy.vol_per_second)


def test_reset_replays_the_same_seeded_event_schedule():
    sim = MarketSimulator(random_seed=88)
    sim.schedule_organic_orders(20.0)
    first = [(float(event.timestamp), event.event_type) for event in sim.event_queue]
    sim.reset()
    sim.schedule_organic_orders(20.0)
    second = [(float(event.timestamp), event.event_type) for event in sim.event_queue]
    assert first == second


@pytest.mark.parametrize("kwargs", [
    {"initial_price": 0.0}, {"volatility": float("nan")},
    {"market_order_rate": 0.0}, {"typical_order_size": 0},
])
def test_invalid_simulator_market_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MarketSimulator(**kwargs)
