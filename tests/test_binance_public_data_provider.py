from src.connectors.binance_provider import BinancePublicDataProvider
from src.connectors.market_data import ProviderState
from src.market.events import MarketEventKind
from src.market.instruments import AssetClass, InstrumentSpec
from src.utils.types import Side


class FakeBinanceWebSocket:
    def __init__(self, **kwargs):
        self.callbacks = kwargs
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_stats(self):
        return {"depth_connected": self.started, "trades_connected": self.started}


def crypto_spec():
    return InstrumentSpec(
        instrument_id="BINANCE:BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        symbol="BTCUSDT",
        venue="BINANCE",
        base_currency="BTC",
        quote_currency="USDT",
        settlement_currency="USDT",
        quantity_unit="BTC",
        tick_size=0.01,
        quantity_step=0.00001,
        min_quantity=0.00001,
    )


def test_binance_provider_normalizes_books_and_trades_without_order_methods():
    connector = FakeBinanceWebSocket()
    provider = BinancePublicDataProvider(crypto_spec(), connector=connector, receive_clock=lambda: 100.0)
    events = []
    provider.start(events.append)
    connector.callbacks["on_orderbook"](
        [(100.0, 2.0), (99.0, 1.0)], [(101.0, 3.0)], {"event_timestamp": 99.9, "sequence": 42}
    )
    connector.callbacks["on_trade"](100.0, 0.25, "buy", 99.95, 501)

    assert [event.kind for event in events] == [MarketEventKind.BOOK_SNAPSHOT, MarketEventKind.TRADE]
    assert events[0].event_ts == 99.9 and events[0].receive_ts == 100.0
    assert events[0].sequence == 42
    assert events[1].sequence == 501 and events[1].side == Side.BUY
    assert provider.status.state == ProviderState.CONNECTED
    assert provider.capabilities.trades and provider.capabilities.order_book
    assert not hasattr(provider, "place_order")
    provider.stop()
    assert connector.stopped


def test_binance_provider_uses_receive_time_when_depth_has_no_exchange_timestamp():
    connector = FakeBinanceWebSocket()
    provider = BinancePublicDataProvider(crypto_spec(), connector=connector, receive_clock=lambda: 12.0)
    events = []
    provider.start(events.append)
    connector.callbacks["on_orderbook"]([(100.0, 1.0)], [(101.0, 1.0)])
    assert events[0].event_ts == 12.0
    assert events[0].receive_ts == 12.0
