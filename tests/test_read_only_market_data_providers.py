"""Offline contract and normalization tests for the optional read-only feeds."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.connectors import databento_provider as databento_module
from src.connectors.provider_factory import build_market_data_providers
from src.connectors.databento_provider import DatabentoMarketDataProvider
from src.connectors.market_data import (
    MarketDataProvider,
    ProviderCapabilities,
    ProviderState,
)
from src.connectors.oanda_provider import OandaPricingProvider
from src.market.events import MarketEventKind
from src.market.instruments import AssetClass, FXAccountingMode, InstrumentSpec
from src.utils.types import Side
from src.config.cross_asset_config import load_cross_asset_config


def equity_spec():
    return InstrumentSpec(
        instrument_id="NASDAQ:AAPL",
        asset_class=AssetClass.EQUITY,
        symbol="AAPL",
        venue="NASDAQ",
        base_currency="USD",
        quote_currency="USD",
        settlement_currency="USD",
        quantity_unit="shares",
        tick_size=0.01,
        quantity_step=1,
        min_quantity=1,
    )


def future_spec():
    return InstrumentSpec(
        instrument_id="XCME:ESZ26",
        asset_class=AssetClass.FUTURE,
        symbol="ESZ26",
        venue="XCME",
        base_currency="ES",
        quote_currency="USD",
        settlement_currency="USD",
        quantity_unit="contracts",
        tick_size=0.25,
        quantity_step=1,
        min_quantity=1,
        expiry=datetime(2026, 12, 18, tzinfo=timezone.utc),
        initial_margin_per_contract=20_000,
        maintenance_margin_per_contract=18_000,
    )


def fx_spec():
    return InstrumentSpec(
        instrument_id="OANDA:EUR_USD",
        asset_class=AssetClass.FX,
        symbol="EUR_USD",
        venue="OANDA",
        base_currency="EUR",
        quote_currency="USD",
        settlement_currency="USD",
        quantity_unit="base_units",
        tick_size=0.00001,
        quantity_step=1,
        min_quantity=1,
        fx_accounting_mode=FXAccountingMode.ROLLING,
    )


class FakeDatabentoClient:
    def __init__(self, on_start=None):
        self.subscriptions = []
        self.callback = None
        self.started = False
        self.stopped = False
        self.on_start = on_start

    def subscribe(self, **kwargs):
        self.subscriptions.append(kwargs)

    def add_callback(self, callback):
        self.callback = callback

    def start(self):
        self.started = True
        if self.on_start:
            self.on_start(self.callback)

    def stop(self):
        self.stopped = True


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_provider_protocol_and_capabilities_are_read_only_market_data_contracts():
    databento = DatabentoMarketDataProvider(
        "XNAS.ITCH", {"AAPL": equity_spec()}, client=FakeDatabentoClient()
    )
    oanda = OandaPricingProvider("acct", {"EUR_USD": fx_spec()}, access_token="token")

    assert isinstance(databento, MarketDataProvider)
    assert isinstance(oanda, MarketDataProvider)
    assert databento.capabilities == ProviderCapabilities(
        quote=True, order_book=True, trades=True, sessions=False, status=True
    )
    assert oanda.capabilities.quote is True
    assert oanda.capabilities.order_book is False
    assert oanda.capabilities.trades is False
    assert not hasattr(oanda, "submit_order")
    assert not hasattr(oanda, "cancel_order")


def test_example_watchlist_constructs_read_only_providers_without_connecting():
    from pathlib import Path

    config_path = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"
    config = load_cross_asset_config(config_path)
    bindings = build_market_data_providers(config, environ={"OANDA_ACCOUNT_ID": "paper-test-account"})

    assert {instrument_id for _, _, ids in bindings for instrument_id in ids} == {
        "BINANCE:BTCUSDT", "NASDAQ:AAPL", "OANDA:EUR_USD", "CME:ESZ6"
    }
    assert len(bindings) == 4
    for _, provider, _ in bindings:
        assert isinstance(provider, MarketDataProvider)
        assert provider.status.state == ProviderState.STOPPED
        for name in ("submit_order", "place_order", "cancel_order", "get_account"):
            assert not hasattr(provider, name)


def test_oanda_provider_factory_rejects_duplicate_normalized_symbols():
    config_path = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"
    config = load_cross_asset_config(config_path)
    instruments = dict(config.instruments)
    eur = instruments["OANDA:EUR_USD"]
    gbp_spec = replace(
        eur.spec, instrument_id="OANDA:GBP_USD", symbol="GBP_USD", base_currency="GBP")
    instruments[gbp_spec.instrument_id] = replace(
        eur, spec=gbp_spec, provider_symbol="EUR_USD")
    providers = dict(config.providers)
    oanda = providers["oanda_fx"]
    providers["oanda_fx"] = replace(
        oanda, instrument_ids=oanda.instrument_ids + (gbp_spec.instrument_id,))
    duplicate_config = replace(config, instruments=instruments, providers=providers)

    with pytest.raises(ValueError, match="maps symbol EUR_USD more than once"):
        build_market_data_providers(
            duplicate_config, environ={"OANDA_ACCOUNT_ID": "paper-test-account"})


def test_databento_maps_definition_and_normalizes_book_and_trade_records():
    provider = DatabentoMarketDataProvider(
        "XNAS.ITCH",
        {"AAPL": equity_spec()},
        schemas=("definition", "mbp-1", "trades"),
        receive_clock=lambda: 1234.0,
    )
    assert provider.normalize_record(
        {"schema": "definition", "raw_symbol": "AAPL", "instrument_id": 77}, receive_ts=1234.0
    ) == ()

    book_events = provider.normalize_record(
        {
            "schema": "mbp-1",
            "instrument_id": 77,
            "ts_event": 1_700_000_000_000_000_000,
            "ts_recv": 1_700_000_000_100_000_000,
            "sequence": 42,
            "bid_px_00": 100_120_000_000,
            "bid_sz_00": 12,
            "ask_px_00": 100_130_000_000,
            "ask_sz_00": 8,
        },
        receive_ts=1234.5,
    )
    assert len(book_events) == 1
    book = book_events[0]
    assert book.kind == MarketEventKind.BOOK_SNAPSHOT
    assert book.instrument_id == "NASDAQ:AAPL"
    assert book.event_ts == pytest.approx(1_700_000_000.0)
    assert book.receive_ts == 1234.5
    assert book.sequence == 42
    assert book.bids == ((100.12, 12.0),)
    assert book.asks == ((100.13, 8.0),)

    trades = provider.normalize_record(
        {
            "schema": "trades",
            "instrument_id": 77,
            "ts_event": 1_700_000_001_000_000_000,
            "price": 100_140_000_000,
            "size": 3,
            "side": "B",
        },
        receive_ts=1235.0,
    )
    assert len(trades) == 1
    assert trades[0].kind == MarketEventKind.TRADE
    assert trades[0].price == pytest.approx(100.14)
    assert trades[0].quantity == 3
    assert trades[0].side == Side.BUY


def test_databento_accepts_normalized_float_prices_and_preserves_synchronous_start_events():
    events = []

    def emit_first_record(callback):
        callback({
            "schema": "trades",
            "instrument_id": 8,
            "price": 101.25,
            "size": 2,
            "side": "A",
            "ts_event": 1_700_000_000_000_000_000,
        })

    client = FakeDatabentoClient(on_start=emit_first_record)
    provider = DatabentoMarketDataProvider(
        "GLBX.MDP3",
        {"ESZ26": future_spec()},
        schemas=("trades",),
        instrument_id_map={8: "ESZ26"},
        client=client,
        receive_clock=lambda: 1000.0,
    )
    provider.start(events.append)

    assert client.started
    assert provider.status.state == ProviderState.CONNECTED
    assert len(events) == 1
    assert events[0].price == pytest.approx(101.25)
    assert events[0].side == Side.SELL
    assert client.subscriptions == [{
        "dataset": "GLBX.MDP3",
        "schema": "trades",
        "stype_in": "raw_symbol",
        "symbols": ["ESZ26"],
    }]
    provider.stop()
    assert client.stopped
    assert provider.status.state == ProviderState.STOPPED


def test_databento_missing_optional_sdk_has_actionable_error(monkeypatch):
    original_import_module = databento_module.importlib.import_module

    def fail_databento(name, *args, **kwargs):
        if name == "databento":
            raise ImportError("not installed")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(databento_module.importlib, "import_module", fail_databento)
    provider = DatabentoMarketDataProvider("XNAS.ITCH", {"AAPL": equity_spec()})

    with pytest.raises(ImportError, match="pip install databento.*DATABENTO_API_KEY"):
        provider.start(lambda event: None)
    assert provider.status.state == ProviderState.ERROR
    assert "inject a compatible client" in provider.status.message


def test_databento_unknown_id_is_reported_without_emitting_fake_data():
    provider = DatabentoMarketDataProvider("XNAS.ITCH", {"AAPL": equity_spec()})
    assert provider.normalize_record({
        "schema": "trades", "instrument_id": 999, "price": 100, "size": 1
    }) == ()
    assert provider.status.state == ProviderState.ERROR
    assert "instrument_id_map" in provider.status.message


def test_oanda_price_message_is_quote_only_and_never_becomes_a_trade():
    provider = OandaPricingProvider(
        "account-123", {"EUR_USD": fx_spec()}, access_token="token", receive_clock=lambda: 500.0
    )
    events = provider.normalize_message({
        "type": "PRICE",
        "instrument": "EUR_USD",
        "time": "2026-10-25T10:00:00.000000000Z",
        "bids": [
            {"price": "1.09990", "liquidity": "1000000"},
            {"price": "1.09995", "liquidity": "2000000"},
        ],
        "asks": [{"price": "1.10010", "liquidity": "1500000"}],
        "tradeable": True,
    })

    assert [event.kind for event in events] == [MarketEventKind.STATUS, MarketEventKind.BOOK_SNAPSHOT]
    assert events[1].bids == ((1.09995, 2_000_000.0), (1.0999, 1_000_000.0))
    assert events[1].asks == ((1.1001, 1_500_000.0),)
    assert events[0].status == "tradeable"
    assert all(event.kind != MarketEventKind.TRADE for event in events)
    assert all(event.source == "oanda" for event in events)


def test_oanda_heartbeat_uses_local_liveness_time_and_emits_no_event():
    provider = OandaPricingProvider(
        "account-123", {"EUR_USD": fx_spec()}, access_token="token", receive_clock=lambda: 500.0
    )
    assert provider.normalize_message({
        "type": "HEARTBEAT", "time": "2026-10-25T10:00:05Z"
    }) == ()
    assert provider.status.last_heartbeat_ts == 500.0
    assert provider.status.events_received == 0


def test_oanda_rejects_ambiguous_tradeable_and_unmapped_symbols():
    provider = OandaPricingProvider("account-123", {"EUR_USD": fx_spec()}, access_token="token")
    with pytest.raises(ValueError, match="tradeable field must be a boolean"):
        provider.normalize_message({
            "type": "PRICE", "instrument": "EUR_USD", "time": "2026-10-25T10:00:00Z",
            "bids": [], "asks": [], "tradeable": "no",
        })
    with pytest.raises(ValueError, match="unmapped OANDA instrument"):
        provider.normalize_message({
            "type": "PRICE", "instrument": "GBP_USD", "time": "2026-10-25T10:00:00Z",
            "bids": [], "asks": [],
        })


def test_oanda_get_snapshot_uses_only_pricing_route_and_closes_response():
    payload = {"prices": [{
        "type": "PRICE",
        "instrument": "EUR_USD",
        "time": "2026-10-25T10:00:00Z",
        "bids": [{"price": "1.1", "liquidity": "100"}],
        "asks": [{"price": "1.2", "liquidity": "200"}],
        "tradeable": False,
    }]}
    response = FakeResponse(payload)
    session = FakeSession(response)
    provider = OandaPricingProvider(
        "account/123", {"EUR_USD": fx_spec()}, access_token="secret", session=session,
        receive_clock=lambda: 600.0,
    )
    published = []
    provider._set_event_handler(published.append)

    events = provider.fetch_prices()

    assert len(events) == 2
    assert events[1].bids == ((1.1, 100.0),)
    assert events[0].status == "non_tradeable"
    assert published == list(events)
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == "https://api-fxpractice.oanda.com/v3/accounts/account%2F123/pricing"
    assert kwargs["params"] == {"instruments": "EUR_USD"}
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert response.closed


def test_oanda_requires_credentials_before_start_and_read_timeout_exceeds_heartbeat():
    provider = OandaPricingProvider("account-123", {"EUR_USD": fx_spec()})
    with pytest.raises(RuntimeError, match="OANDA_ACCESS_TOKEN"):
        provider.start(lambda event: None)
    with pytest.raises(ValueError, match="five-second pricing heartbeat"):
        OandaPricingProvider(
            "account-123", {"EUR_USD": fx_spec()}, access_token="token", read_timeout=5
        )
