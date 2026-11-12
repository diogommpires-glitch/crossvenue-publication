"""Regressions found by the independent accounting review."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.portfolio_ledger import PortfolioLedger
from src.utils.types import Side


CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"
NOW = datetime(2026, 10, 25, 12, tzinfo=timezone.utc).timestamp()


class Clock:
    value = NOW

    def __call__(self):
        return self.value


def equity_engine():
    config = load_cross_asset_config(CONFIG)
    item = config.instruments["NASDAQ:AAPL"]
    spec = replace(item.spec, borrow_limit=20, borrow_fee_rate=.365,
                   borrow_rate_source="review sourced estimate",
                   borrow_rate_effective_from=NOW - 1,
                   borrow_rate_effective_to=NOW + 1)
    item = replace(item, spec=spec, allow_short=True, maker_latency_ms=0, queue_fraction=1)
    provider = replace(config.providers[item.provider_id], instrument_ids=(spec.instrument_id,))
    config = replace(config, starting_capital=10000, starting_balances={"USD": 10000},
                     conversion_marks={}, instruments={spec.instrument_id: item},
                     providers={item.provider_id: provider})
    clock = Clock()
    engine = CrossAssetPaperEngine(config, clock=clock, quote_refresh_seconds=10,
                                  max_clock_skew_seconds=0)
    identity = spec.instrument_id
    for kind, data in (
        (MarketEventKind.BOOK_SNAPSHOT, {"bids": ((199.99, 10),), "asks": ((200.01, 10),)}),
        (MarketEventKind.STATUS, {"status": "open"}),
        (MarketEventKind.TRADE, {"price": 200, "quantity": 10, "side": Side.BUY, "sequence": 1}),
    ):
        assert engine.on_market_event(MarketDataEvent(identity, "review", kind, NOW, NOW, **data))
    return engine, clock


def test_late_official_close_retains_source_time_without_rewinding_borrow_cursor():
    engine, _clock = equity_engine()
    ledger = engine.ledger
    ledger.record_fill("review-short", "NASDAQ:AAPL", Side.SELL, 1, 200,
                       fee=0, fee_currency="USD", timestamp=NOW)
    ledger.accrue_borrow(NOW + .5)
    old_cursor = ledger.export_state()["borrow_state"]["NASDAQ:AAPL"]["last_ts"]
    ledger.set_valuation_mark("NASDAQ:AAPL", 201, NOW, valid_until=NOW + 86460,
                             source="official close fixture", basis="official_close",
                             stress_fraction=.1, processing_timestamp=NOW + .75)
    position = ledger.get_position("NASDAQ:AAPL", as_of=NOW + .75)
    assert position.mark_timestamp == NOW
    assert position.mark_price == 201
    assert ledger.export_state()["borrow_state"]["NASDAQ:AAPL"]["last_ts"] >= old_cursor
    # The prior observed price accrues through posting; the sourced close is
    # used prospectively without erasing costs already accrued by the timer.
    expected = 200 * .365 * .75 / (365 * 86400)
    assert ledger.snapshot(NOW + .75).accrued_costs == pytest.approx(expected)


def test_source_terms_expiry_cancels_existing_increasing_quotes_before_next_timer():
    engine, clock = equity_engine()
    engine.start_strategy()
    sell = next(row for row in engine.get_active_orders_detail() if row["side"] == "SELL")
    clock.value = NOW + 2
    assert engine.on_market_event(MarketDataEvent(
        "NASDAQ:AAPL", "review", MarketEventKind.TRADE, clock.value, clock.value,
        sequence=2, price=sell["price"], quantity=10, side=Side.BUY))
    assert engine.total_fills == 0
    assert engine.get_active_orders_detail() == []


def test_expired_borrow_terms_keep_a_proven_close_quote_and_it_can_fill():
    engine, clock = equity_engine()
    engine.ledger.record_fill("review-existing-short", "NASDAQ:AAPL", Side.SELL, 1, 200,
                             fee=0, fee_currency="USD", timestamp=NOW)
    engine.start_strategy()
    buy = next(row for row in engine.get_active_orders_detail() if row["side"] == "BUY")
    clock.value = NOW + 2
    engine.process_account_actions(clock.value)
    retained = engine.get_active_orders_detail()
    assert [row["side"] for row in retained] == ["BUY"]
    assert engine.on_market_event(MarketDataEvent(
        "NASDAQ:AAPL", "review", MarketEventKind.TRADE, clock.value, clock.value,
        sequence=2, price=buy["price"], quantity=10, side=Side.SELL))
    assert engine.total_fills == 1
    assert engine.ledger.get_position("NASDAQ:AAPL", as_of=clock.value).quantity == 0


def test_failed_mandatory_action_cancels_preexisting_increasing_quotes():
    engine, _clock = equity_engine()
    engine.start_strategy()
    assert engine.get_active_orders_detail()
    engine.schedule_account_action({
        "action_id": "bad-input", "action_type": "financing", "instrument_id": "NASDAQ:AAPL",
        "source": "invalid imported fixture", "effective_ts": NOW,
        "payload": {"amount": -1, "currency": "USD"},
    })
    result = engine.process_account_actions(NOW)
    assert result[0]["status"] == "failed"
    assert engine.get_active_orders_detail() == []


def test_duplicate_durable_event_rejects_and_restores_candidate_financing():
    class DuplicateStore:
        def load_cross_asset_snapshot(self, account_id):
            return None

        def commit_cross_asset_event(self, *args, **kwargs):
            return False

    engine = CrossAssetPaperEngine(load_cross_asset_config(CONFIG), clock=Clock(),
                                  state_manager=DuplicateStore())
    before = engine.ledger.export_state()
    with pytest.raises(RuntimeError, match="paper history"):
        engine.record_fx_financing("duplicate-event", "OANDA:EUR_USD", -2, "USD", timestamp=NOW)
    assert engine.ledger.export_state() == before
    assert engine.circuit_breaker_reason == "paper_history_persistence_failed"


def test_borrow_timer_checkpoint_failure_rolls_back_cost_and_cancels_reservations():
    class FailingCheckpointStore:
        def save_cross_asset_snapshot(self, *args, **kwargs):
            raise OSError("simulated midnight write failure")

    engine, clock = equity_engine()
    engine.ledger.record_fill("review-existing-short", "NASDAQ:AAPL", Side.SELL, 1, 200,
                             fee=0, fee_currency="USD", timestamp=NOW)
    engine.start_strategy()
    assert engine.ledger._reservations
    before = engine.ledger.export_state()
    engine.state_manager = FailingCheckpointStore()
    clock.value = NOW + 12 * 3600
    with pytest.raises(RuntimeError, match="paper history"):
        engine.process_account_actions(clock.value)
    assert engine.ledger.export_state() == before
    assert not engine.is_running
    assert engine.circuit_breaker_reason == "paper_history_persistence_failed"
    assert engine.get_active_orders_detail() == []
    assert engine.ledger._reservations == {}


def test_failed_roll_restores_full_recent_history_even_at_retention_boundary():
    from tests.test_platform_accounting import roll_engine

    class FailingStore:
        def load_cross_asset_snapshot(self, account_id):
            return None

        def commit_cross_asset_event(self, *args, **kwargs):
            raise OSError("simulated roll write failure")

    engine = roll_engine(manager=FailingStore())
    previous = [{"fill_id": "history-{}".format(index)} for index in range(2000)]
    engine.trades = list(previous)
    with pytest.raises(RuntimeError, match="paper history"):
        engine.roll_future("CME:ESZ6", "CME:ESH7")
    assert engine.trades == previous


def test_deposit_does_not_mask_daily_loss_and_risk_baseline_survives_restart(tmp_path):
    from src.runtime.state_manager import StateManager

    manager = StateManager(str(tmp_path / "funding-risk.db"))
    config = load_cross_asset_config(CONFIG)
    engine = CrossAssetPaperEngine(config, clock=Clock(), state_manager=manager)
    engine.record_fx_financing("loss", "OANDA:EUR_USD", -600, "USD", timestamp=NOW)
    assert engine._current_daily_pnl() == pytest.approx(-600)
    assert engine.record_funding("deposit", "USD", 1000, "sourced paper contribution", timestamp=NOW)
    assert engine._current_daily_pnl() == pytest.approx(-600)
    restarted = CrossAssetPaperEngine(config, clock=Clock(), state_manager=manager)
    assert restarted._current_daily_pnl() == pytest.approx(-600)


def test_failed_deposit_restores_risk_baseline_and_opening_allocation_is_immutable():
    class DuplicateStore:
        def commit_cross_asset_event(self, *args, **kwargs):
            return False

    engine = CrossAssetPaperEngine(load_cross_asset_config(CONFIG), clock=Clock())
    previous = (engine._risk_day, engine._risk_day_start_equity,
                engine._last_account_equity, engine._last_account_timestamp)
    opening = engine.ledger.export_state()["opening_balances"]
    engine.state_manager = DuplicateStore()
    with pytest.raises(RuntimeError, match="paper history"):
        engine.record_funding("rejected-deposit", "USD", 1000, "source")
    assert (engine._risk_day, engine._risk_day_start_equity,
            engine._last_account_equity, engine._last_account_timestamp) == previous
    assert engine.ledger.export_state()["opening_balances"] == opening


def test_legacy_missing_opening_allocation_is_kept_unverified_instead_of_using_current_cash():
    ledger = PortfolioLedger(100)
    ledger.fund_currency("USD", 50, NOW)
    state = ledger.export_state()
    assert state["opening_balances"] == {"USD": 100}
    state.pop("opening_balances")
    restored = PortfolioLedger(100)
    restored.restore_state(state)
    assert restored.export_state()["cash"] == {"USD": 150}
    assert restored.export_state()["opening_balances"] is None


def test_queued_market_event_after_timer_keeps_receive_time_and_forward_borrow_cursor():
    engine, clock = equity_engine()
    engine.ledger.record_fill("queued-existing-short", "NASDAQ:AAPL", Side.SELL, 1, 200,
                             fee=0, fee_currency="USD", timestamp=NOW)
    clock.value = NOW + .5
    engine.process_account_actions(clock.value)
    clock.value = NOW + .75
    # This quote arrived before the timer but was processed afterward. Its
    # source and receive timestamps remain intact; costs accrue using the
    # old position and price until the actual local processing boundary.
    received = NOW + .25
    assert engine.on_market_event(MarketDataEvent(
        "NASDAQ:AAPL", "review", MarketEventKind.BOOK_SNAPSHOT,
        received, received, bids=((200.99, 10),), asks=((201.01, 10),)))
    state = engine.ledger.export_state()
    assert state["positions"]["NASDAQ:AAPL"]["mark_timestamp"] == received
    assert state["borrow_state"]["NASDAQ:AAPL"]["last_ts"] == clock.value
    expected = 200 * .365 * .75 / (365 * 86400)
    assert engine.ledger.snapshot(clock.value).accrued_costs == pytest.approx(expected)


def test_queued_close_fill_after_timer_accrues_cost_until_processing_not_receipt():
    engine, clock = equity_engine()
    ledger = engine.ledger
    ledger.record_fill("queued-existing-short", "NASDAQ:AAPL", Side.SELL, 1, 200,
                       fee=0, fee_currency="USD", timestamp=NOW)
    ledger.accrue_borrow(NOW + .5)
    ledger.record_fill("queued-cover", "NASDAQ:AAPL", Side.BUY, 1, 200,
                       fee=0, fee_currency="USD", timestamp=NOW + .25,
                       processing_timestamp=NOW + .75, allow_risk_reduction=True)
    expected = 200 * .365 * .75 / (365 * 86400)
    assert ledger.snapshot(NOW + .75).accrued_costs == pytest.approx(expected)
    assert ledger.get_position("NASDAQ:AAPL", as_of=NOW + .75).quantity == 0
