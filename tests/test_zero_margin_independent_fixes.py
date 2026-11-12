"""Regressions for quote-side permissions and Databento trading status."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.connectors.databento_provider import DatabentoMarketDataProvider
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.causal_analysis import CausalVenueScout
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.strategies.instrument_aware_market_maker import InstrumentAwareMarketMaker
from src.utils.types import Side

NOW = 1_792_497_600.0  # 2026-10-20 UTC, before the example future's expiry.
CONFIG_PATH = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"


@pytest.mark.parametrize(
    "inventory,sell_quantity",
    [
        (0, None),
        (0.000005, None),
        (0.000009999999999, None),
        (0.0005, 0.0005),
        (0.000505, 0.0005),
        (0.001, 0.001),
    ],
)
def test_engine_long_only_strategy_does_not_propose_short_quotes(inventory, sell_quantity):
    engine = CrossAssetPaperEngine(load_cross_asset_config(CONFIG_PATH), clock=lambda: NOW)
    strategy = engine._runtime["BINANCE:BTCUSDT"].strategy
    strategy.sync_inventory(inventory)

    orders = strategy.on_market_event({"time": NOW, "mid_price": 80_000})

    expected = [(Side.BUY, 0.001)]
    if sell_quantity is not None:
        expected.append((Side.SELL, sell_quantity))
    assert [(order.side, float(order.quantity)) for order in orders] == expected


@pytest.mark.parametrize(
    "minimum,inventory,sell_quantity",
    [
        (1e-12, 0, None),
        (1e-12, 0.5e-12, None),
        (1e-12, 0.99999999e-12, None),
        (2e-12, 1e-12, None),
        (1e-12, 1e-12, 1e-12),
    ],
)
def test_long_only_tiny_lots_require_positive_minimum_sell(minimum, inventory, sell_quantity):
    spec = load_cross_asset_config(CONFIG_PATH).instruments["BINANCE:BTCUSDT"].spec
    spec = replace(spec, quantity_step=1e-12, min_quantity=minimum)
    strategy = InstrumentAwareMarketMaker(
        spec,
        order_size=1e-9,
        max_inventory=1,
        allow_short=False,
    )
    strategy.sync_inventory(inventory)

    orders = strategy.on_market_event({"time": NOW, "mid_price": 80_000})

    expected = [] if sell_quantity is None else [sell_quantity]
    assert [float(order.quantity) for order in orders if order.side == Side.SELL] == expected


@pytest.mark.parametrize("bought,sold", [(0.00003, 0.00002), (0.00499, 0.00498)])
def test_engine_retains_one_lot_reducing_close_after_float_subtraction(bought, sold):
    engine = CrossAssetPaperEngine(load_cross_asset_config(CONFIG_PATH), clock=lambda: NOW)
    instrument_id = "BINANCE:BTCUSDT"
    for fill_id, side, quantity in (("open", Side.BUY, bought), ("reduce", Side.SELL, sold)):
        engine.ledger.record_fill(
            fill_id,
            instrument_id,
            side,
            quantity,
            80_000,
            fee=0,
            fee_currency="USDT",
            timestamp=NOW,
        )
    position = engine.ledger.get_position(instrument_id, NOW).quantity
    assert 0 < position < 0.00001
    assert engine.on_market_event(
        MarketDataEvent(
            instrument_id,
            "fixture",
            MarketEventKind.BOOK_SNAPSHOT,
            NOW,
            NOW,
            bids=((79_999.99, 1),),
            asks=((80_000.01, 1),),
        )
    )
    assert engine.on_market_event(
        MarketDataEvent(
            instrument_id,
            "fixture",
            MarketEventKind.TRADE,
            NOW,
            NOW,
            price=80_000,
            quantity=0.01,
            side=Side.BUY,
        )
    )
    engine._trip_circuit_breaker("test_halt", NOW)

    engine.start_strategy()

    active = [
        order
        for order in engine.get_active_orders_detail()
        if order["instrument_id"] == instrument_id
    ]
    assert [(order["side"], order["quantity"]) for order in active] == [("SELL", 0.00001)]
    assert engine.ledger.get_position(instrument_id, NOW).quantity == position


def test_short_enabled_engine_and_legacy_strategy_keep_both_quote_prices():
    engine = CrossAssetPaperEngine(load_cross_asset_config(CONFIG_PATH), clock=lambda: NOW)
    future = engine._runtime["CME:ESZ6"].strategy
    orders = future.on_market_event({"time": NOW, "mid_price": 5_000})
    assert [(order.side, float(order.price)) for order in orders] == [
        (Side.BUY, 4_999),
        (Side.SELL, 5_001),
    ]

    equity = engine.config.instruments["NASDAQ:AAPL"].spec
    legacy = InstrumentAwareMarketMaker(equity, order_size=1, max_inventory=2)
    orders = legacy.on_market_event({"time": NOW, "mid_price": 100})
    assert [(order.side, float(order.price)) for order in orders] == [
        (Side.BUY, 99.96),
        (Side.SELL, 100.04),
    ]


@pytest.mark.parametrize(
    "is_trading,expected_status",
    [
        ("Y", "trading"),
        (b"Y", "trading"),
        (True, "trading"),
        ("N", "closed"),
        (b"N", "closed"),
        (False, "closed"),
        ("~", "unknown"),
        (b"~", "unknown"),
        (None, "unknown"),
        ("unrecognised", "unknown"),
    ],
)
def test_databento_distinguishes_unknown_trading_flag(is_trading, expected_status):
    spec = load_cross_asset_config(CONFIG_PATH).instruments["NASDAQ:AAPL"].spec
    provider = DatabentoMarketDataProvider(
        "XNAS.ITCH",
        {"AAPL": spec},
        instrument_id_map={1: "AAPL"},
        receive_clock=lambda: NOW,
    )

    (event,) = provider.normalize_record(
        {
            "schema": "status",
            "instrument_id": 1,
            "is_trading": is_trading,
        }
    )

    assert event.status == expected_status


@pytest.mark.parametrize(
    "member,expected_status",
    [
        ("YES", "trading"),
        ("NO", "closed"),
        ("NOT_AVAILABLE", "unknown"),
    ],
)
def test_databento_real_sdk_tristate_status(member, expected_status):
    dbn = pytest.importorskip("databento_dbn")
    spec = load_cross_asset_config(CONFIG_PATH).instruments["NASDAQ:AAPL"].spec
    provider = DatabentoMarketDataProvider(
        "XNAS.ITCH",
        {"AAPL": spec},
        instrument_id_map={1: "AAPL"},
        receive_clock=lambda: NOW,
    )
    record = dbn.StatusMsg(
        publisher_id=1,
        instrument_id=1,
        ts_event=int(NOW * 1e9),
        ts_recv=int(NOW * 1e9),
        is_trading=getattr(dbn.TriState, member),
    )

    (event,) = provider.normalize_record(record)

    assert event.status == expected_status


@pytest.mark.parametrize(
    "status_fields,expected_status",
    [
        ({"is_trading": "~", "action": 7}, "unknown"),
        ({"is_trading": None, "action": 7}, "unknown"),
        ({"action": 7}, "trading"),
    ],
)
def test_unknown_trading_flag_is_not_overridden_by_action(status_fields, expected_status):
    spec = load_cross_asset_config(CONFIG_PATH).instruments["NASDAQ:AAPL"].spec
    provider = DatabentoMarketDataProvider(
        "XNAS.ITCH",
        {"AAPL": spec},
        instrument_id_map={1: "AAPL"},
        receive_clock=lambda: NOW,
    )

    (event,) = provider.normalize_record(
        {
            "schema": "status",
            "instrument_id": 1,
            **status_fields,
        }
    )

    assert event.status == expected_status


@pytest.mark.parametrize(
    "kind,status,session,expected_samples",
    [
        (MarketEventKind.STATUS, "trading", None, 1),
        (MarketEventKind.STATUS, "unknown", None, 0),
        (MarketEventKind.STATUS, "closed", None, 0),
        (MarketEventKind.STATUS, "halted", None, 0),
        (MarketEventKind.STATUS, "unknown", "open", 0),
        (MarketEventKind.SESSION, "trading", "closed", 0),
        (MarketEventKind.SESSION, "trading", "unknown", 0),
        (MarketEventKind.SESSION, "closed", "open", 1),
    ],
)
def test_scout_preserves_history_only_for_known_healthy_status(
    kind, status, session, expected_samples
):
    scout = CausalVenueScout()
    for stamp, price in ((1, 100), (2, 101)):
        scout.observe(
            MarketDataEvent(
                "NASDAQ:AAPL",
                "databento",
                MarketEventKind.TRADE,
                stamp,
                stamp,
                price=price,
                quantity=1,
            ),
            stamp,
        )
    scout.observe(
        MarketDataEvent(
            "NASDAQ:AAPL",
            "databento",
            kind,
            2,
            2,
            status=status,
            session=session,
        ),
        2,
    )
    scout.observe(
        MarketDataEvent(
            "NASDAQ:AAPL",
            "databento",
            MarketEventKind.TRADE,
            3,
            3,
            price=102,
            quantity=1,
        ),
        3,
    )

    assert scout.summary(3)["streams"][0]["samples"] == expected_samples
