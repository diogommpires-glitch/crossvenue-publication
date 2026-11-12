from datetime import datetime, timezone

import pytest

from src.market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from src.strategies.instrument_aware_market_maker import InstrumentAwareMarketMaker


def spec(asset_class=AssetClass.EQUITY, **overrides):
    values = {
        "instrument_id": "NASDAQ:AAPL",
        "asset_class": AssetClass.EQUITY,
        "symbol": "AAPL",
        "venue": "NASDAQ",
        "base_currency": "USD",
        "quote_currency": "USD",
        "settlement_currency": "USD",
        "quantity_unit": "shares",
        "tick_size": 0.01,
        "quantity_step": 1,
        "min_quantity": 1,
    }
    if asset_class == AssetClass.FX:
        values.update({
            "instrument_id": "OANDA:EUR_USD",
            "asset_class": AssetClass.FX,
            "symbol": "EUR_USD",
            "venue": "OANDA",
            "base_currency": "EUR",
            "quote_currency": "USD",
            "settlement_currency": "USD",
            "quantity_unit": "base_units",
            "tick_size": 0.00001,
            "quantity_step": 1000,
            "min_quantity": 1000,
            "fx_accounting_mode": FXAccountingMode.DELIVERABLE,
        })
    elif asset_class == AssetClass.FUTURE:
        values.update({
            "instrument_id": "XCME:ESZ26",
            "asset_class": AssetClass.FUTURE,
            "symbol": "ESZ26",
            "venue": "XCME",
            "base_currency": "ES",
            "quote_currency": "USD",
            "settlement_currency": "USD",
            "quantity_unit": "contracts",
            "tick_size": 0.25,
            "quantity_step": 1,
            "min_quantity": 1,
            "contract_multiplier": 50,
            "expiry": datetime(2026, 12, 18, tzinfo=timezone.utc),
            "initial_margin_per_contract": 20_000,
            "maintenance_margin_per_contract": 18_000,
        })
    values.update(overrides)
    return InstrumentSpec(**values)


@pytest.mark.parametrize(
    "instrument,mid,size",
    [
        (spec(), 200.003, 2),
        (spec(AssetClass.FX), 1.085001, 1000),
        (spec(AssetClass.FUTURE), 6000.125, 1),
    ],
)
def test_quotes_use_instrument_tick_and_lot(instrument, mid, size):
    strategy = InstrumentAwareMarketMaker(
        instrument, order_size=size, max_inventory=size * 2, spread_bps=8
    )
    orders = strategy.on_market_event({
        "instrument_id": instrument.instrument_id,
        "time": 10,
        "mid_price": mid,
        "market_data_ready": True,
        "market_open": True,
    })

    assert len(orders) == 2
    assert all(order.instrument_id == instrument.instrument_id for order in orders)
    assert all(instrument.is_valid_price(order.price) for order in orders)
    assert all(instrument.is_valid_quantity(order.quantity) for order in orders)
    assert orders[0].price < mid < orders[1].price


def test_inventory_skew_moves_reservation_away_from_existing_position():
    instrument = spec()
    strategy = InstrumentAwareMarketMaker(
        instrument, order_size=1, max_inventory=2, spread_bps=10, inventory_skew_bps=20
    )
    flat = strategy.on_market_event({"time": 1, "mid_price": 100, "market_data_ready": True})
    strategy.sync_inventory(1)
    long = strategy.on_market_event({"time": 2, "mid_price": 100, "market_data_ready": True})

    assert long[0].price < flat[0].price
    assert long[1].price < flat[1].price


def test_inventory_limit_suppresses_only_the_exposure_increasing_side():
    strategy = InstrumentAwareMarketMaker(spec(), order_size=1, max_inventory=1)
    strategy.sync_inventory(1)

    orders = strategy.on_market_event({"time": 1, "mid_price": 100, "market_data_ready": True})

    assert [order.side.name for order in orders] == ["SELL"]


def test_futures_quotes_support_negative_prices_and_keep_bid_below_ask():
    instrument = spec(AssetClass.FUTURE)
    strategy = InstrumentAwareMarketMaker(instrument, order_size=1, max_inventory=2, spread_bps=8)
    orders = strategy.on_market_event({
        "instrument_id": instrument.instrument_id,
        "time": 1,
        "mid_price": -12.0,
        "market_open": True,
        "market_data_ready": True,
    })
    assert len(orders) == 2
    assert orders[0].price < -12 < orders[1].price


@pytest.mark.parametrize("market_state", [
    {"time": 1, "mid_price": 100, "market_open": False},
    {"time": 1, "mid_price": 100, "market_data_ready": False},
])
def test_closed_or_unready_market_produces_no_quotes(market_state):
    strategy = InstrumentAwareMarketMaker(spec(), order_size=1, max_inventory=2)
    assert strategy.on_market_event(market_state) == []


def test_market_event_for_another_instrument_is_rejected():
    strategy = InstrumentAwareMarketMaker(spec(), order_size=1, max_inventory=2)
    with pytest.raises(ValueError, match="does not match"):
        strategy.on_market_event({
            "instrument_id": "NYSE:MSFT", "time": 1, "mid_price": 100
        })
