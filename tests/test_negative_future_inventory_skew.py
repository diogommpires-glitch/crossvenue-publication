from pathlib import Path

from src.config.cross_asset_config import load_cross_asset_config
from src.strategies.instrument_aware_market_maker import InstrumentAwareMarketMaker


def test_negative_future_prices_keep_inventory_skew_in_the_risk_reducing_direction():
    config = load_cross_asset_config(Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json")
    spec = config.instruments["CME:ESZ6"].spec
    maker = InstrumentAwareMarketMaker(spec, order_size=1, max_inventory=10,
                                       inventory_skew_bps=100)
    market = {"mid_price": -100, "time": 1000}
    flat = [float(order.price) for order in maker.on_market_event(market)]
    maker.sync_inventory(5)
    long = [float(order.price) for order in maker.on_market_event(market)]
    maker.sync_inventory(-5)
    short = [float(order.price) for order in maker.on_market_event(market)]
    assert all(left < right for left, right in zip(long, flat))
    assert all(left > right for left, right in zip(short, flat))
