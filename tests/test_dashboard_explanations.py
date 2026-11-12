"""Dashboard explanations share the engine's actual paper eligibility checks."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.state_manager import StateManager
from src.utils.types import Side


CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"
NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc).timestamp()
BTC = "BINANCE:BTCUSDT"


def make_engine(manager=None):
    return CrossAssetPaperEngine(load_cross_asset_config(CONFIG), clock=lambda: NOW,
                                state_manager=manager)


def instrument_row(engine, instrument_id=BTC):
    return next(row for row in engine.get_dashboard_state()["instruments"]
                if row["instrument_id"] == instrument_id)


def feed_btc(engine):
    assert engine.on_market_event(MarketDataEvent(instrument_id=BTC, source="dashboard-fixture",
        kind=MarketEventKind.BOOK_SNAPSHOT, event_ts=NOW, receive_ts=NOW,
        bids=((79999, 1),), asks=((80001, 1),)))
    assert engine.on_market_event(MarketDataEvent(instrument_id=BTC, source="dashboard-fixture",
        kind=MarketEventKind.TRADE, event_ts=NOW, receive_ts=NOW, sequence=1,
        price=80000, quantity=0.01, side=Side.BUY))


@pytest.mark.parametrize("condition,code,ready", [
    ("ready", "ready", True),
    ("freshness_boundary", "ready", True),
    ("disabled", "quoting_disabled", False),
    ("rolled", "rolled", False),
    ("expired", "expired", False),
    ("closed", "session_not_open", False),
    ("halted", "market_halted", False),
    ("unsynchronized", "book_unsynchronized", False),
    ("missing_book", "quote_missing", False),
    ("missing_quote_time", "quote_missing", False),
    ("stale_quote", "quote_stale", False),
    ("future_quote", "quote_clock_skew", False),
    ("nonfinite_quote", "quote_stale", False),
    ("missing_trades", "trades_missing", False),
    ("stale_trades", "trades_stale", False),
    ("future_trade", "trade_clock_skew", False),
    ("nonfinite_trade", "trades_stale", False),
])
def test_readiness_reasons_agree_with_execution_gate(condition, code, ready):
    engine = make_engine()
    runtime = engine._runtime[BTC]
    runtime.bids, runtime.asks = ((79999, 1),), ((80001, 1),)
    runtime.last_quote_receive_ts = runtime.last_trade_receive_ts = NOW
    stale_after = engine.config.market_data_stale_after
    if condition == "freshness_boundary":
        runtime.last_quote_receive_ts = runtime.last_trade_receive_ts = NOW - stale_after
    elif condition == "disabled":
        runtime.config = replace(runtime.config, enabled=False)
    elif condition == "rolled":
        engine._rolled_futures.add(BTC)
    elif condition == "expired":
        runtime.config = replace(runtime.config, spec=replace(runtime.spec,
            expiry=datetime.fromtimestamp(NOW - 1, timezone.utc)))
    elif condition in {"closed", "halted"}:
        runtime.market_open = False
        runtime.market_status = condition
    elif condition == "unsynchronized":
        runtime.book_synchronized = False
    elif condition == "missing_book":
        runtime.asks = ()
    elif condition == "missing_quote_time":
        runtime.last_quote_receive_ts = None
    elif condition == "stale_quote":
        runtime.last_quote_receive_ts = NOW - stale_after - 0.001
    elif condition == "future_quote":
        runtime.last_quote_receive_ts = NOW + 0.001
    elif condition == "nonfinite_quote":
        runtime.last_quote_receive_ts = float("nan")
    elif condition == "missing_trades":
        runtime.last_trade_receive_ts = None
    elif condition == "stale_trades":
        runtime.last_trade_receive_ts = NOW - stale_after - 0.001
    elif condition == "future_trade":
        runtime.last_trade_receive_ts = NOW + 0.001
    elif condition == "nonfinite_trade":
        runtime.last_trade_receive_ts = float("nan")

    row = instrument_row(engine)
    assert row["market_data_ready"] is ready
    assert engine._instrument_ready(runtime, NOW) is ready
    assert engine.market_data_is_fresh() is ready
    assert row["readiness"]["code"] == code
    assert row["readiness"]["message"]
    assert row["readiness"]["requires_trades"] is True
    if condition in {"disabled", "rolled", "expired"}:
        assert row["market_status"] == code
    if condition in {"rolled", "expired"}:
        assert row["market_open"] is False


def test_quote_only_fx_readiness_does_not_require_public_trades_or_promise_risk_acceptance():
    engine = make_engine()
    instrument_id = "OANDA:EUR_USD"
    runtime = engine._runtime[instrument_id]
    runtime.bids, runtime.asks = ((1.1, 1000),), ((1.1001, 1000),)
    runtime.market_open = True
    runtime.last_quote_receive_ts = NOW
    engine.circuit_breaker_active = True
    engine.circuit_breaker_reason = "daily_loss_limit"
    engine._accounting_blocks[instrument_id] = "missing_financing_inputs"

    row = instrument_row(engine, instrument_id)
    assert row["market_data_ready"] is True
    assert row["readiness"]["requires_trades"] is False
    assert row["accounting_status"] == "missing_financing_inputs"
    assert "risk checks apply separately" in row["readiness"]["message"]


def test_latest_rejection_is_committed_and_survives_recovery_without_changing_equity(tmp_path):
    manager = StateManager(str(tmp_path / "paper.db"))
    engine = make_engine(manager)
    feed_btc(engine)
    equity = engine.ledger.snapshot(as_of=NOW).equity
    result = engine.submit_paper_order(BTC, Side.BUY, 1000000, 80000)
    assert result["accepted"] is False
    row = instrument_row(engine)
    expected = dict(row["last_rejection"])
    assert expected["reason"] == "quantity_limit_or_lot_violation"
    assert expected["order_id"] == result["order_id"]
    assert expected["timestamp"] == NOW
    assert row["last_rejection_available"] is True
    row["last_rejection"]["reason"] = "browser mutation"
    assert instrument_row(engine)["last_rejection"] == expected
    restarted = make_engine(StateManager(str(tmp_path / "paper.db")))
    assert instrument_row(restarted)["last_rejection"] == expected
    assert restarted.ledger.snapshot(as_of=NOW).equity == equity
    assert len(manager.get_cross_asset_events(engine.account_id)) == 1


def test_old_checkpoint_recovers_actual_recent_rejection_and_reports_missing_details(tmp_path):
    manager = StateManager(str(tmp_path / "old.db"))
    engine = make_engine()
    state = engine._checkpoint_payload()
    state["engine"]["total_rejections"] = 2
    for instrument_id in (BTC, "NASDAQ:AAPL"):
        counts = state["engine"]["instrument_totals"][instrument_id]
        counts["rejections"] = 1
        counts.pop("last_rejection")
    manager.commit_cross_asset_event(engine.account_id, "old-rejection", "order_rejected", NOW,
        {"instrument_id": BTC, "order_id": "old-order", "reason": "instrument_position_limit"}, state)

    restarted = make_engine(manager)
    assert instrument_row(restarted)["last_rejection"] == {
        "reason": "instrument_position_limit", "order_id": "old-order", "timestamp": NOW}
    unknown = instrument_row(restarted, "NASDAQ:AAPL")
    assert unknown["total_rejections"] == 1
    assert unknown["last_rejection"] is None
    assert unknown["last_rejection_available"] is False
    assert len(manager.get_cross_asset_events(engine.account_id)) == 1


def test_failed_rejection_persistence_does_not_report_uncommitted_details(tmp_path):
    path = tmp_path / "failed.db"
    manager = StateManager(str(path))
    engine = make_engine(manager)
    feed_btc(engine)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TRIGGER reject_event BEFORE INSERT ON paper_account_journal "
            "BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END")
    with pytest.raises(RuntimeError, match="paper history could not be saved"):
        engine.submit_paper_order(BTC, Side.BUY, 1000000, 80000)
    assert instrument_row(engine)["last_rejection"] is None
    assert manager.get_cross_asset_events(engine.account_id) == []
    assert engine.circuit_breaker_active is True


def test_dashboard_limits_are_configured_values_and_exclude_provider_access():
    engine = make_engine()
    limits = engine.get_dashboard_state()["risk_limits"]
    assert set(limits) == {"reporting_currency", "max_daily_loss", "max_gross_exposure",
                           "max_orders_per_minute", "instruments"}
    assert limits["max_gross_exposure"] == engine.config.max_gross_exposure
    assert limits["max_daily_loss"] == engine.config.max_daily_loss
    assert limits["max_orders_per_minute"] == engine.config.max_orders_per_minute
    for instrument_id, values in limits["instruments"].items():
        assert set(values) == {"max_order_quantity", "max_position_quantity", "max_notional",
                               "allow_short", "enabled"}
        config = engine.config.instruments[instrument_id]
        assert values == {name: getattr(config, name) for name in values}
