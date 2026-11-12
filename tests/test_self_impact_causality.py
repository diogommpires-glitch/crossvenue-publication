"""Regression tests for impact timing, attribution, and price composition."""

import pytest

from src.market.market_simulator import Event, EventType, MarketSimulator
from src.market.self_impact import SelfImpactModel
from src.orderbook.order import Fill
from src.utils.types import Price, Quantity, Side, Timestamp


def make_fill(
    order_id="order-1",
    side=Side.BUY,
    quantity=100.0,
    timestamp=0.0,
    trader_id="mm_strategy",
    is_aggressor=True,
):
    return Fill(
        order_id=order_id,
        side=side,
        quantity=Quantity(quantity),
        price=Price(100.0),
        timestamp=Timestamp(timestamp),
        trader_id=trader_id,
        is_aggressor=is_aggressor,
    )


def make_simulator(impact_enabled=True, seed=7):
    simulator = MarketSimulator(
        initial_price=100.0,
        volatility=0.0,
        drift=0.0,
        typical_order_size=100,
        spread_mean=0.10,
        impact_enabled=impact_enabled,
        random_seed=seed,
        impact_coefficient_bps=10.0,
        impact_reference_quantity=100.0,
        impact_decay_halflife=10.0,
    )
    simulator.last_price_update = Timestamp(0.0)
    return simulator


def update_at(simulator, timestamp):
    simulator.current_time = Timestamp(timestamp)
    simulator._update_mid_price(
        Event(Timestamp(timestamp), EventType.PRICE_UPDATE, {})
    )


def test_aggressive_buy_impulse_is_composed_once_and_not_double_counted():
    simulator = make_simulator()
    simulator.self_impact.record_fill(make_fill(), Timestamp(0.0), 100.0)

    state = simulator.get_market_state()

    assert state["base_mid_price"] == pytest.approx(100.0)
    assert state["mid_price"] > state["base_mid_price"]
    assert state["impact_adjusted_mid"] == pytest.approx(state["mid_price"])
    assert state["mid_price"] == pytest.approx(100.0 * 1.001, rel=1e-6)


def test_impact_decays_by_half_life_without_new_fills_through_simulator():
    simulator = make_simulator()
    simulator.self_impact.record_fill(make_fill(), Timestamp(0.0), 100.0)
    initial_impact = simulator.self_impact.get_cumulative_impact()

    update_at(simulator, 10.0)

    assert simulator.self_impact.get_cumulative_impact() == pytest.approx(
        initial_impact * 0.5, rel=1e-9
    )
    assert simulator.base_mid_price == pytest.approx(100.0)
    assert 100.0 < simulator.mid_price < 100.1


def test_no_fill_event_frequency_does_not_change_final_impact_price():
    one_event = make_simulator()
    many_events = make_simulator()
    for simulator in (one_event, many_events):
        simulator.self_impact.record_fill(make_fill(), Timestamp(0.0), 100.0)

    update_at(one_event, 10.0)
    for timestamp in range(1, 11):
        update_at(many_events, float(timestamp))

    assert one_event.mid_price == pytest.approx(many_events.mid_price, rel=1e-12)
    assert one_event.self_impact.get_cumulative_impact() == pytest.approx(
        many_events.self_impact.get_cumulative_impact(), rel=1e-12
    )


def test_sell_aggressor_moves_price_down_and_passive_fill_adds_no_impulse():
    model = SelfImpactModel(
        impact_coefficient_bps=10.0,
        reference_quantity=100.0,
        strategy_id="mm_strategy",
    )
    model.record_fill(make_fill(side=Side.SELL), Timestamp(0.0), 100.0)
    assert model.get_cumulative_impact() < 0.0

    passive_model = SelfImpactModel(strategy_id="mm_strategy")
    passive_model.record_fill(
        make_fill(is_aggressor=False), Timestamp(0.0), 100.0
    )
    assert passive_model.get_strategy_fill_count() == 1
    assert passive_model.get_cumulative_impact() == 0.0


def test_other_aggressor_flow_is_in_price_and_separately_attributed():
    simulator = make_simulator()
    external_fill = make_fill(
        side=Side.SELL,
        trader_id="organic_market_1",
    )
    simulator.self_impact.record_fill(external_fill, Timestamp(0.0), 100.0)

    state = simulator.get_market_state()

    assert state["market_impact"] < 0.0
    assert state["strategy_impact"] == 0.0
    assert state["mid_price"] < state["base_mid_price"]


def test_partial_fills_of_one_order_are_linear_in_total_size():
    one_fill = SelfImpactModel(impact_coefficient_bps=10.0, reference_quantity=100.0)
    split_fills = SelfImpactModel(impact_coefficient_bps=10.0, reference_quantity=100.0)

    one_fill.record_fill(make_fill(quantity=100.0), Timestamp(0.0), 100.0)
    split_fills.record_fill(make_fill(quantity=40.0), Timestamp(0.0), 100.0)
    split_fills.record_fill(make_fill(quantity=60.0), Timestamp(0.0), 100.0)

    assert split_fills.get_cumulative_impact() == pytest.approx(
        one_fill.get_cumulative_impact(), rel=1e-12
    )


def test_opposite_same_time_aggressors_cancel_the_temporary_state():
    model = SelfImpactModel(impact_coefficient_bps=10.0, reference_quantity=100.0)
    model.record_fill(make_fill(order_id="buy", side=Side.BUY), Timestamp(0.0), 100.0)
    model.record_fill(make_fill(order_id="sell", side=Side.SELL), Timestamp(0.0), 100.0)

    assert model.get_cumulative_impact() == pytest.approx(0.0, abs=1e-12)


def test_relative_impact_is_bounded_and_handles_invalid_price():
    model = SelfImpactModel(impact_coefficient_bps=5000.0, reference_quantity=100.0)
    model.record_fill(make_fill(), Timestamp(0.0), 100.0)

    assert model.get_relative_impact(Price(100.0)) == pytest.approx(0.10)
    assert model.get_relative_impact(Price(0.0)) == 0.0
    assert model.get_relative_impact(Price(-1.0)) == 0.0


def test_impact_and_no_impact_share_the_same_exogenous_price_path():
    without_impact = MarketSimulator(
        initial_price=100.0, volatility=0.2, impact_enabled=False, random_seed=42
    )
    with_impact = MarketSimulator(
        initial_price=100.0, volatility=0.2, impact_enabled=True, random_seed=42
    )
    without_impact.initialize_book(levels=5, orders_per_level=2)
    with_impact.initialize_book(levels=5, orders_per_level=2)

    without_impact.run(5.0)
    with_impact.run(5.0)

    assert with_impact.base_mid_price == pytest.approx(without_impact.base_mid_price)
