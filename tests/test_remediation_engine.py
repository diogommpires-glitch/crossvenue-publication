"""Regressions for effective actions, native fee holds and execution liquidity."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.state_manager import StateManager
from src.utils.types import Side
from tests.test_execution_safety import engine_for
from tests.test_platform_accounting import NOW, Clock, roll_engine
from tests.test_review_accounting import equity_engine as borrowed_equity_engine

CONFIG = Path(__file__).parents[1] / "config/cross_asset_paper.example.json"
EQUITY = "NASDAQ:AAPL"


def equity_engine(*, euro_cash=0, foreign_fee=False, manager=None):
    config = load_cross_asset_config(CONFIG)
    item = config.instruments[EQUITY]
    if foreign_fee:
        item = replace(
            item,
            spec=replace(item.spec, settlement_currency="EUR"),
            fees=replace(item.fees, maker_rate=0, maker_per_unit=1, currency="EUR"),
        )
    item = replace(item, maker_latency_ms=0, queue_fraction=1, allow_short=False)
    provider = replace(config.providers[item.provider_id], instrument_ids=(EQUITY,))
    config = replace(
        config,
        instruments={EQUITY: item},
        providers={item.provider_id: provider},
        starting_capital=10000 + euro_cash * 1.1,
        starting_balances={"USD": 10000, "EUR": euro_cash},
        conversion_marks={"EUR": (1.1, "synthetic fee fixture", 86400)},
    )
    clock = Clock()
    engine = CrossAssetPaperEngine(
        config,
        clock=clock,
        state_manager=manager,
        max_clock_skew_seconds=0,
        quote_refresh_seconds=100,
    )
    for kind, data in (
        (MarketEventKind.BOOK_SNAPSHOT, {"bids": ((199.99, 10),), "asks": ((200.01, 10),)}),
        (MarketEventKind.STATUS, {"status": "open"}),
        (MarketEventKind.TRADE, {"price": 200, "quantity": 10, "side": Side.BUY, "sequence": 1}),
    ):
        assert engine.on_market_event(MarketDataEvent(EQUITY, "fixture", kind, NOW, NOW, **data))
    return engine, clock


def schedule_split(engine, effective):
    engine.schedule_account_action(
        {
            "action_id": "scheduled-split",
            "action_type": "split",
            "instrument_id": EQUITY,
            "effective_ts": effective,
            "source": "synthetic sourced split",
            "payload": {"ratio": 2},
        }
    )


def test_due_split_cancels_old_order_before_post_split_print():
    engine, clock = equity_engine()
    engine.start_strategy()
    assert engine.get_active_orders_detail()
    schedule_split(engine, NOW + 0.05)
    clock.value = NOW + 0.1
    assert engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "fixture",
            MarketEventKind.TRADE,
            clock.value,
            clock.value,
            sequence=2,
            price=100,
            quantity=10,
            side=Side.SELL,
        )
    )
    assert engine.total_fills == 0
    assert engine.account_actions.actions["scheduled-split"]["status"] == "applied"
    assert engine.get_active_orders_detail() == []


def test_split_uses_effective_epoch_and_accepts_equal_time_snapshot():
    engine, clock = equity_engine()
    effective = NOW + 0.05
    schedule_split(engine, effective)
    clock.value = NOW + 0.1
    assert engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "fixture",
            MarketEventKind.BOOK_SNAPSHOT,
            effective,
            clock.value,
            sequence=2,
            bids=((99.99, 10),),
            asks=((100.01, 10),),
        )
    )
    assert engine.account_actions.actions["scheduled-split"]["status"] == "applied"
    assert engine._runtime[EQUITY].book_fresh_after_ts == effective
    assert engine._runtime[EQUITY].book_synchronized
    assert not engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "late-before",
            MarketEventKind.TRADE,
            effective - 0.01,
            clock.value,
            price=200,
            quantity=10,
            side=Side.SELL,
        )
    )
    assert engine._runtime[EQUITY].last_trade_event_ts is None


def test_skewed_future_split_event_cannot_advance_action_clock():
    engine, _clock = equity_engine()
    engine.max_clock_skew_seconds = 1
    schedule_split(engine, NOW + 0.5)
    assert not engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "fixture",
            MarketEventKind.TRADE,
            NOW + 0.6,
            NOW,
            sequence=2,
            price=100,
            quantity=10,
            side=Side.SELL,
        )
    )
    assert engine.account_actions.actions["scheduled-split"]["status"] == "pending"


def test_action_commit_failure_does_not_consume_triggering_market_event(tmp_path):
    manager = StateManager(str(tmp_path / "actions.db"))
    engine, clock = equity_engine(manager=manager)
    schedule_split(engine, NOW + 0.05)
    with manager._get_connection() as connection:
        connection.execute(
            "CREATE TRIGGER fail_action BEFORE INSERT ON paper_account_journal "
            "WHEN NEW.event_type='account_action_applied' "
            "BEGIN SELECT RAISE(ABORT, 'synthetic action failure'); END"
        )
    clock.value = NOW + 0.1
    event = MarketDataEvent(
        EQUITY,
        "fixture",
        MarketEventKind.BOOK_SNAPSHOT,
        clock.value,
        clock.value,
        sequence=2,
        bids=((99.99, 10),),
        asks=((100.01, 10),),
    )
    with pytest.raises(RuntimeError, match="paper history"):
        engine.on_market_event(event)
    assert engine.account_actions.actions["scheduled-split"]["status"] == "pending"
    assert engine._event_identity(event) not in engine._seen_event_ids[EQUITY]
    assert engine.total_fills == 0
    assert engine._runtime[EQUITY].book_fresh_after_ts is None


@pytest.mark.parametrize("scheduled", [False, True])
def test_split_epoch_survives_restart_for_both_split_entry_points(tmp_path, scheduled):
    manager = StateManager(str(tmp_path / "split.db"))
    engine, clock = equity_engine(manager=manager)
    engine.ledger.record_fill(
        "initial-long", EQUITY, Side.BUY, 1, 200, fee=0, fee_currency="USD", timestamp=NOW
    )
    effective = NOW + 0.05
    clock.value = NOW + 0.1
    if scheduled:
        schedule_split(engine, effective)
        engine.process_account_actions(clock.value)
    else:
        engine.apply_equity_split("direct-split", EQUITY, 2, timestamp=effective)
    restored = CrossAssetPaperEngine(
        engine.config, clock=clock, state_manager=manager, max_clock_skew_seconds=0
    )
    old_units = MarketDataEvent(
        EQUITY,
        "after-restart",
        MarketEventKind.BOOK_SNAPSHOT,
        NOW,
        clock.value,
        bids=((199.99, 10),),
        asks=((200.01, 10),),
    )
    assert not restored.on_market_event(old_units)
    assert restored._runtime[EQUITY].book_fresh_after_ts == effective
    assert restored.ledger.snapshot(clock.value).equity == 10000
    assert restored.ledger.get_position(EQUITY, clock.value).quantity == 2


def test_split_does_not_mix_future_source_epoch_with_older_local_arrival():
    engine, clock = equity_engine()
    engine.max_clock_skew_seconds = 1
    effective = NOW + 0.05
    schedule_split(engine, effective)
    clock.value = NOW + 0.1
    event = MarketDataEvent(
        EQUITY,
        "fixture",
        MarketEventKind.BOOK_SNAPSHOT,
        effective,
        NOW,
        sequence=2,
        bids=((99.99, 10),),
        asks=((100.01, 10),),
    )
    assert not engine.on_market_event(event)
    assert engine.account_actions.actions["scheduled-split"]["status"] == "pending"
    assert engine._event_identity(event) not in engine._seen_event_ids[EQUITY]
    assert engine._runtime[EQUITY].mid_price == 200


def test_unsequenced_status_identity_distinguishes_halt_but_rejects_duplicate():
    engine, clock = engine_for("BINANCE:BTCUSDT")
    halted = MarketDataEvent(
        "BINANCE:BTCUSDT",
        "synthetic",
        MarketEventKind.STATUS,
        clock.value,
        clock.value,
        status="halted",
    )
    assert engine.on_market_event(halted)
    assert not engine._runtime["BINANCE:BTCUSDT"].market_open
    assert not engine.on_market_event(halted)


def test_failed_late_split_blocks_old_reducing_quotes_and_post_split_marks():
    engine, clock = borrowed_equity_engine()
    engine.ledger.record_fill(
        "existing-short", EQUITY, Side.SELL, 1, 200, fee=0, fee_currency="USD", timestamp=NOW
    )
    engine.start_strategy()
    clock.value = NOW + 0.1
    assert engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "review",
            MarketEventKind.BOOK_SNAPSHOT,
            clock.value,
            clock.value,
            bids=((199.99, 10),),
            asks=((200.01, 10),),
        )
    )
    schedule_split(engine, NOW + 0.05)
    clock.value = NOW + 0.2
    assert not engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "review",
            MarketEventKind.BOOK_SNAPSHOT,
            clock.value,
            clock.value,
            bids=((99.99, 10),),
            asks=((100.01, 10),),
        )
    )
    assert engine.account_actions.actions["scheduled-split"]["status"] == "failed"
    assert engine.get_active_orders_detail() == []
    assert engine.ledger.get_position(EQUITY, clock.value).mark_price == 200
    assert not engine.submit_paper_order(EQUITY, Side.BUY, 1, 199.99)["accepted"]
    assert engine.total_fills == 0


@pytest.mark.parametrize("instrument", ["BINANCE:BTCUSDT", "OANDA:EUR_USD"])
def test_source_time_must_pass_maker_activation_boundary(instrument):
    engine, clock = engine_for(instrument, maker_latency_ms=50)
    engine.start_strategy()
    buy = next(row for row in engine.get_active_orders_detail() if row["side"] == "BUY")
    clock.value += 0.06
    if instrument.startswith("OANDA"):
        ask = round(buy["price"] - 0.00002, 5)
        event = MarketDataEvent(
            instrument,
            "synthetic",
            MarketEventKind.BOOK_SNAPSHOT,
            clock.value - 0.059,
            clock.value,
            bids=((round(ask - 0.0001, 5), 10000),),
            asks=((ask, 10000),),
        )
    else:
        event = MarketDataEvent(
            instrument,
            "synthetic",
            MarketEventKind.TRADE,
            clock.value - 0.059,
            clock.value,
            sequence=2,
            price=buy["price"],
            quantity=1,
            side=Side.SELL,
        )
    assert engine.on_market_event(event)
    assert engine.total_fills == 0


def test_cash_security_rejects_unfunded_foreign_fee_without_dangling_holds():
    engine, _clock = equity_engine(foreign_fee=True)
    assert not engine.submit_paper_order(EQUITY, Side.BUY, 1, 199.99)["accepted"]
    assert engine.ledger.snapshot(NOW).reserved_cash_by_currency == {}


def test_foreign_cash_fee_is_resized_on_partial_fill_and_released_on_cancel():
    engine, clock = equity_engine(foreign_fee=True, euro_cash=2)
    engine.is_running = True
    engine._runtime[EQUITY].last_strategy_receive_ts = NOW
    accepted = engine.submit_paper_order(EQUITY, Side.BUY, 2, 199.99, request_id="fee-partial")
    assert accepted["accepted"]
    assert engine.ledger.snapshot(NOW).reserved_cash_by_currency == {"USD": 399.98, "EUR": 2}
    proposal = engine._portfolio_proposal(engine.ledger.snapshot(NOW), NOW)
    assert proposal.native_cash_required == {}
    clock.value += 0.1
    assert engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "fixture",
            MarketEventKind.TRADE,
            clock.value,
            clock.value,
            sequence=2,
            price=199.99,
            quantity=1,
            side=Side.SELL,
        )
    )
    snapshot = engine.ledger.snapshot(clock.value)
    assert snapshot.cash_by_currency["EUR"] == 1
    assert snapshot.reserved_cash_by_currency == {"USD": 199.99, "EUR": 1}
    assert not engine.submit_paper_order(EQUITY, Side.BUY, 1, 199.99)["accepted"]
    engine._cancel_single_order(engine._runtime[EQUITY], accepted["order_id"], "test_cancel")
    assert engine.ledger.snapshot(clock.value).reserved_cash_by_currency == {}


def test_mandatory_cash_cost_cancels_a_fee_hold_that_is_no_longer_funded():
    engine, clock = equity_engine(foreign_fee=True, euro_cash=2)
    engine.is_running = True
    engine._runtime[EQUITY].last_strategy_receive_ts = NOW
    assert engine.submit_paper_order(EQUITY, Side.BUY, 2, 199.99)["accepted"]
    engine.schedule_account_action(
        {
            "action_id": "known-cash-cost",
            "action_type": "borrow_charge",
            "instrument_id": EQUITY,
            "source": "synthetic mandatory cost",
            "effective_ts": NOW + 0.05,
            "payload": {
                "actual_amount": 2,
                "estimated_amount": 0,
                "currency": "EUR",
                "interval_start": NOW - 1,
                "interval_end": NOW + 0.05,
            },
        }
    )
    clock.value = NOW + 0.1
    assert engine.on_market_event(
        MarketDataEvent(
            EQUITY,
            "fixture",
            MarketEventKind.TRADE,
            clock.value,
            clock.value,
            sequence=2,
            price=199.99,
            quantity=1,
            side=Side.SELL,
        )
    )
    assert engine.total_fills == 0
    assert engine.ledger.snapshot(clock.value).cash_by_currency.get("EUR", 0) == 0
    assert engine.ledger.snapshot(clock.value).reserved_cash_by_currency == {}


def test_accrued_borrow_on_another_instrument_cannot_fund_an_increasing_order():
    base, _ = borrowed_equity_engine()
    old = base.config.instruments[EQUITY]
    old = replace(old, fees=replace(old.fees, maker_rate=0, maker_per_unit=0))
    other_id = "NASDAQ:MSFT"
    other = replace(
        old,
        spec=replace(old.spec, instrument_id=other_id, symbol="MSFT", base_currency="MSFT"),
        provider_symbol="MSFT",
    )
    provider = replace(base.config.providers[old.provider_id], instrument_ids=(EQUITY, other_id))
    config = replace(
        base.config,
        instruments={EQUITY: old, other_id: other},
        providers={old.provider_id: provider},
        starting_capital=199.98,
        starting_balances={"USD": 199.98},
    )
    clock = Clock()
    engine = CrossAssetPaperEngine(
        config, clock=clock, max_clock_skew_seconds=0, quote_refresh_seconds=100
    )
    for identity in (EQUITY, other_id):
        for kind, data in (
            (MarketEventKind.BOOK_SNAPSHOT, {"bids": ((199.99, 10),), "asks": ((200.01, 10),)}),
            (MarketEventKind.STATUS, {"status": "open"}),
            (
                MarketEventKind.TRADE,
                {"price": 200, "quantity": 10, "side": Side.BUY, "sequence": 1},
            ),
        ):
            assert engine.on_market_event(
                MarketDataEvent(identity, "fixture", kind, NOW, NOW, **data)
            )
    engine.ledger.record_fill(
        "short-for-borrow", EQUITY, Side.SELL, 1, 200, fee=0, fee_currency="USD", timestamp=NOW
    )
    engine.is_running = True
    engine._runtime[other_id].last_strategy_receive_ts = NOW
    assert engine.submit_paper_order(other_id, Side.BUY, 2, 199.99)["accepted"]
    clock.value += 0.5
    assert engine.ledger.snapshot(clock.value).available_cash_by_currency["USD"] < 0
    assert engine.on_market_event(
        MarketDataEvent(
            other_id,
            "fixture",
            MarketEventKind.TRADE,
            clock.value,
            clock.value,
            sequence=2,
            price=199.99,
            quantity=2,
            side=Side.SELL,
        )
    )
    assert engine.total_fills == 0
    assert engine.ledger.get_position(other_id, clock.value).quantity == 0


def test_repeated_partial_rolls_cannot_reuse_unchanged_displayed_depth():
    engine = roll_engine(old_quantity=3, replacement_limit=4)
    instruments = {
        key: replace(item, max_notional=1000000, max_position_quantity=4)
        for key, item in engine.config.instruments.items()
    }
    engine.config = replace(engine.config, instruments=instruments)
    for runtime in engine._runtime.values():
        runtime.config = instruments[runtime.spec.instrument_id]
        runtime.bids = ((5000, 1),)
        runtime.asks = ((5000.25, 1),)
    assert engine.roll_future("CME:ESZ6", "CME:ESH7", quantity=1)["rolled"]
    with pytest.raises(ValueError, match="displayed liquidity"):
        engine.roll_future("CME:ESZ6", "CME:ESH7", quantity=1)
    for identity in instruments:
        assert engine.on_market_event(
            MarketDataEvent(
                identity,
                "fixture",
                MarketEventKind.BOOK_SNAPSHOT,
                NOW,
                NOW,
                sequence=1,
                bids=((5000, 1),),
                asks=((5000.25, 1),),
            )
        )
    with pytest.raises(ValueError, match="displayed liquidity"):
        engine.roll_future("CME:ESZ6", "CME:ESH7", quantity=1)
    for identity in instruments:
        assert engine.on_market_event(
            MarketDataEvent(
                identity,
                "fixture",
                MarketEventKind.BOOK_SNAPSHOT,
                NOW,
                NOW,
                sequence=2,
                bids=((5000, 2),),
                asks=((5000.25, 2),),
            )
        )
    assert engine.roll_future("CME:ESZ6", "CME:ESH7", quantity=1)["rolled"]
    assert engine.ledger.get_position("CME:ESH7", NOW).quantity == 2


def test_partial_roll_liquidity_survives_financial_checkpoint_recovery(tmp_path):
    manager = StateManager(str(tmp_path / "roll.db"))
    engine = roll_engine(manager=manager, old_quantity=3, replacement_limit=4)
    instruments = {
        key: replace(item, max_notional=1000000, max_position_quantity=4)
        for key, item in engine.config.instruments.items()
    }
    engine.config = replace(engine.config, instruments=instruments)
    for runtime in engine._runtime.values():
        runtime.config = instruments[runtime.spec.instrument_id]
        runtime.bids, runtime.asks = ((5000, 1),), ((5000.25, 1),)
    assert engine.roll_future("CME:ESZ6", "CME:ESH7", quantity=1)["rolled"]
    restored = CrossAssetPaperEngine(
        engine.config, clock=Clock(), state_manager=manager, max_clock_skew_seconds=0
    )
    for identity in instruments:
        for kind, data in (
            (MarketEventKind.BOOK_SNAPSHOT, {"bids": ((5000, 1),), "asks": ((5000.25, 1),)}),
            (MarketEventKind.STATUS, {"status": "open"}),
            (MarketEventKind.TRADE, {"price": 5000, "quantity": 1, "side": Side.BUY}),
        ):
            assert restored.on_market_event(
                MarketDataEvent(identity, "after-restart", kind, NOW, NOW, **data)
            )
    with pytest.raises(ValueError, match="displayed liquidity"):
        restored.roll_future("CME:ESZ6", "CME:ESH7", quantity=1)
