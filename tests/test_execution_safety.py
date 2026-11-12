"""Synthetic regressions for execution budgets, cancelled orders and native fees."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.utils.types import Side

CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"
NOW = datetime(2026, 10, 29, 12, tzinfo=UTC).timestamp()


class Clock:
    value = NOW

    def __call__(self):
        return self.value


def engine_for(
    instrument_id, *, max_daily_loss=500, balances=None, conversion_marks=None, **changes
):
    config = load_cross_asset_config(CONFIG)
    item = replace(config.instruments[instrument_id], **changes)
    config = replace(
        config,
        instruments={instrument_id: item},
        providers={},
        starting_balances=balances or {"USD": 40000, "USDT": 10000},
        max_daily_loss=max_daily_loss,
    )
    if conversion_marks is not None:
        config = replace(config, conversion_marks=conversion_marks)
    clock = Clock()
    engine = CrossAssetPaperEngine(
        config, clock=clock, max_clock_skew_seconds=0, quote_refresh_seconds=100
    )
    bid, ask = {
        "OANDA:EUR_USD": (1.10000, 1.10010),
        "CME:ESZ6": (5000, 5000.25),
        "BINANCE:BTCUSDT": (79999.99, 80000.01),
    }[instrument_id]
    for kind, data in (
        (MarketEventKind.BOOK_SNAPSHOT, {"bids": ((bid, 20000),), "asks": ((ask, 20000),)}),
        (MarketEventKind.STATUS, {"status": "tradeable"}),
    ):
        assert engine.on_market_event(
            MarketDataEvent(instrument_id, "synthetic", kind, NOW, NOW, **data)
        )
    if item.fill_model == "trade_print_queue":
        assert engine.on_market_event(
            MarketDataEvent(
                instrument_id,
                "synthetic",
                MarketEventKind.TRADE,
                NOW,
                NOW,
                sequence=1,
                price=bid,
                quantity=100,
                side=Side.BUY,
            )
        )
    return engine, clock


def fx_orders(*, max_daily_loss=500, side=Side.BUY):
    engine, clock = engine_for(
        "OANDA:EUR_USD", max_daily_loss=max_daily_loss, allow_short=side == Side.SELL
    )
    engine.start_strategy()
    original = next(
        order for order in engine.get_active_orders_detail() if order["side"] == side.name
    )
    improvement = 0.00002 if side == Side.BUY else -0.00002
    second = engine.submit_paper_order(
        "OANDA:EUR_USD", side, 1000, round(original["price"] + improvement, 5), request_id="second"
    )
    assert second["accepted"]
    return engine, clock, original, second


def moved_fx_quote(engine, clock, original, size):
    clock.value += 1
    if original["side"] == "BUY":
        ask = round(original["price"] - 0.00001, 5)
        bid = round(ask - 0.0001, 5)
    else:
        bid = round(original["price"] + 0.00001, 5)
        ask = round(bid + 0.0001, 5)
    assert engine.on_market_event(
        MarketDataEvent(
            "OANDA:EUR_USD",
            "synthetic",
            MarketEventKind.BOOK_SNAPSHOT,
            clock.value,
            clock.value,
            bids=((bid, size),),
            asks=((ask, size),),
        )
    )


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_fx_orders_share_the_quote_liquidity_budget(side):
    engine, clock, original, _second = fx_orders(side=side)
    moved_fx_quote(engine, clock, original, 10000)
    assert sum(trade["quantity"] for trade in engine.trades) == pytest.approx(1000)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_fx_orders_use_price_priority(side):
    engine, clock, original, second = fx_orders(side=side)
    moved_fx_quote(engine, clock, original, 10000)
    assert engine.trades[0]["order_id"] == second["order_id"]


def test_fx_orders_at_the_same_price_use_acceptance_time():
    engine, clock, original, earlier = fx_orders()
    clock.value += 0.01
    later = engine.submit_paper_order(
        "OANDA:EUR_USD", Side.BUY, 1000, round(original["price"] + 0.00002, 5), request_id="a_later"
    )
    assert later["accepted"]
    moved_fx_quote(engine, clock, original, 10000)
    assert [trade["order_id"] for trade in engine.trades] == [earlier["order_id"]]


def test_cancelled_fx_candidate_does_not_consume_remaining_liquidity():
    engine, clock, original, second = fx_orders()
    third = engine.submit_paper_order(
        "OANDA:EUR_USD", Side.BUY, 1000, round(original["price"] + 0.00001, 5), request_id="third"
    )
    assert third["accepted"]

    def cancel_middle(fill):
        if fill["order_id"] == second["order_id"]:
            engine._cancel_single_order(
                engine._runtime["OANDA:EUR_USD"], third["order_id"], "synthetic_cancel"
            )

    engine.on("strategy_fill", cancel_middle)
    moved_fx_quote(engine, clock, original, 20000)
    assert [trade["order_id"] for trade in engine.trades] == [
        second["order_id"],
        original["order_id"],
    ]


@pytest.mark.parametrize("instrument_id", ["OANDA:EUR_USD", "BINANCE:BTCUSDT"])
def test_fill_triggered_halt_stops_precomputed_candidates(instrument_id):
    engine, clock = engine_for(instrument_id, max_daily_loss=0.005)
    engine.start_strategy()
    original = next(order for order in engine.get_active_orders_detail() if order["side"] == "BUY")
    quantity = original["quantity"]
    assert engine.submit_paper_order(
        instrument_id, Side.BUY, quantity, original["price"], request_id="second"
    )["accepted"]
    if instrument_id == "OANDA:EUR_USD":
        moved_fx_quote(engine, clock, original, 30000)
    else:
        clock.value += 1
        assert engine.on_market_event(
            MarketDataEvent(
                instrument_id,
                "synthetic",
                MarketEventKind.TRADE,
                clock.value,
                clock.value,
                sequence=2,
                price=original["price"],
                quantity=0.01,
                side=Side.SELL,
            )
        )
    assert engine.circuit_breaker_reason == "portfolio_loss_limit"
    assert engine.total_fills == 1


def test_futures_order_requires_native_margin_and_fee_cash():
    engine, _clock = engine_for("CME:ESZ6", balances={"USD": 20000, "USDT": 30000})
    response = engine.submit_paper_order("CME:ESZ6", Side.BUY, 1, 4999.75, request_id="boundary")
    assert not response["accepted"]
    assert engine.ledger.snapshot(NOW).reserved_cash_by_currency == {}
    assert engine.ledger.snapshot(NOW).reserved_margin_by_currency == {}


def test_futures_fee_hold_resizes_after_partial_fill_and_releases_on_cancel():
    engine, clock = engine_for(
        "CME:ESZ6",
        balances={"USD": 50000},
        max_position_quantity=3,
        max_order_quantity=3,
        max_notional=1000000,
    )
    engine.start_strategy()
    engine._cancel_instrument_quotes(engine._runtime["CME:ESZ6"], "synthetic_manual_order")
    response = engine.submit_paper_order("CME:ESZ6", Side.BUY, 2, 4999.75, request_id="partial")
    assert response["accepted"]
    assert engine.ledger.snapshot(NOW).reserved_cash_by_currency == {"USD": 2.5}
    clock.value += 1
    assert engine.on_market_event(
        MarketDataEvent(
            "CME:ESZ6",
            "synthetic",
            MarketEventKind.TRADE,
            clock.value,
            clock.value,
            sequence=2,
            price=4999.75,
            quantity=20,
            side=Side.SELL,
        )
    )
    assert engine.ledger.snapshot(clock.value).reserved_cash_by_currency == {"USD": 1.25}
    engine.stop_strategy()
    snapshot = engine.ledger.snapshot(clock.value)
    assert snapshot.reserved_cash_by_currency == {}
    assert snapshot.reserved_margin_by_currency == {}


def test_opposing_futures_orders_hold_both_fees_and_release_independently():
    engine, _clock = engine_for("CME:ESZ6", balances={"USD": 20002.5, "USDT": 29997.5})
    buy = engine.submit_paper_order("CME:ESZ6", Side.BUY, 1, 4999.75, request_id="buy")
    sell = engine.submit_paper_order("CME:ESZ6", Side.SELL, 1, 5000.5, request_id="sell")
    assert buy["accepted"] and sell["accepted"]
    snapshot = engine.ledger.snapshot(NOW)
    assert snapshot.reserved_cash_by_currency == {"USD": 2.5}
    assert snapshot.reserved_margin_by_currency == {"USD": 20000}
    assert snapshot.available_cash_by_currency["USD"] == 0
    engine._cancel_single_order(engine._runtime["CME:ESZ6"], buy["order_id"], "synthetic_cancel")
    assert engine.ledger.snapshot(NOW).reserved_cash_by_currency == {"USD": 1.25}
    engine.stop_strategy()
    assert engine.ledger.snapshot(NOW).reserved_cash_by_currency == {}


@pytest.mark.parametrize("fee_cash, accepted", [(0, False), (1.25, True)])
def test_futures_fee_currency_is_reserved_separately_from_margin(fee_cash, accepted):
    config = load_cross_asset_config(CONFIG)
    item = config.instruments["CME:ESZ6"]
    marks = dict(config.conversion_marks, EUR=(1.2, "synthetic_fee_conversion", 86400))
    engine, _clock = engine_for(
        "CME:ESZ6",
        conversion_marks=marks,
        balances={"USD": 20000, "USDT": 30000 - fee_cash * 1.2, "EUR": fee_cash},
        spec=replace(item.spec, quote_currency="EUR"),
        fees=replace(item.fees, currency="EUR"),
    )
    response = engine.submit_paper_order("CME:ESZ6", Side.BUY, 1, 4999.75, request_id="currencies")
    assert response["accepted"] is accepted
    snapshot = engine.ledger.snapshot(NOW)
    assert snapshot.reserved_cash_by_currency == ({"EUR": 1.25} if accepted else {})
    assert snapshot.reserved_margin_by_currency == ({"USD": 20000} if accepted else {})


def test_fully_margined_futures_position_can_place_a_reducing_close():
    engine, _clock = engine_for("CME:ESZ6", balances={"USD": 20000, "USDT": 30000})
    engine.ledger.record_fill(
        "synthetic-open", "CME:ESZ6", Side.BUY, 1, 5000, fee=0, fee_currency="USD", timestamp=NOW
    )
    assert engine.ledger.snapshot(NOW).available_cash_by_currency["USD"] == 0
    assert engine.submit_paper_order("CME:ESZ6", Side.SELL, 1, 5000.25, request_id="close")[
        "accepted"
    ]


def test_failed_futures_fill_restores_cash_and_releases_pending_fee_holds():
    class FailingStore:
        def commit_cross_asset_event(self, *args, **kwargs):
            raise OSError("synthetic disk failure")

    engine, clock = engine_for("CME:ESZ6")
    engine.start_strategy()
    buy = next(order for order in engine.get_active_orders_detail() if order["side"] == "BUY")
    before = engine.ledger.export_state()
    assert engine.ledger.snapshot(NOW).reserved_cash_by_currency == {"USD": 2.5}
    engine.state_manager = FailingStore()
    clock.value += 1
    with pytest.raises(RuntimeError, match="paper history"):
        engine.on_market_event(
            MarketDataEvent(
                "CME:ESZ6",
                "synthetic",
                MarketEventKind.TRADE,
                clock.value,
                clock.value,
                sequence=2,
                price=buy["price"],
                quantity=20,
                side=Side.SELL,
            )
        )
    assert engine.ledger.export_state() == before
    assert engine.total_fills == 0
    assert engine.get_active_orders_detail() == []
    snapshot = engine.ledger.snapshot(clock.value)
    assert snapshot.reserved_cash_by_currency == {}
    assert snapshot.reserved_margin_by_currency == {}


def test_futures_roll_releases_pending_fee_holds_for_both_contracts():
    from tests.test_platform_accounting import roll_engine

    engine = roll_engine()
    engine.start_strategy()
    assert engine.ledger.snapshot(engine.clock()).reserved_cash_by_currency
    engine.roll_future("CME:ESZ6", "CME:ESH7")
    snapshot = engine.ledger.snapshot(engine.clock())
    assert snapshot.reserved_cash_by_currency == {}
    assert snapshot.reserved_margin_by_currency == {}
    assert engine.get_active_orders_detail() == []
