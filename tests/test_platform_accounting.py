"""Regression cases for rolls, borrow liabilities, and sourced lifecycle inputs."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.market.instruments import AssetClass, InstrumentSpec
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.portfolio_ledger import PortfolioLedger
from src.runtime.state_manager import StateManager
from src.utils.types import Side


CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"
NOW = datetime(2026, 10, 25, 12, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value


def roll_engine(*, manager=None, replacement_limit=1, old_quantity=1, target_quantity=0):
    config = load_cross_asset_config(CONFIG)
    old = config.instruments["CME:ESZ6"]
    new_spec = replace(old.spec, instrument_id="CME:ESH7", symbol="ESH7",
                       expiry=datetime(2027, 3, 19, 14, 30, tzinfo=timezone.utc))
    new = replace(old, spec=new_spec, provider_symbol="ESH7", max_position_quantity=replacement_limit)
    instruments = {old.spec.instrument_id: old, new_spec.instrument_id: new}
    provider = replace(config.providers[old.provider_id], instrument_ids=tuple(instruments))
    config = replace(config, starting_capital=200000, starting_balances={"USD": 200000},
                     conversion_marks={}, instruments=instruments,
                     providers={old.provider_id: provider}, max_daily_loss=100000)
    clock = Clock()
    engine = CrossAssetPaperEngine(config, clock=clock, state_manager=manager, max_clock_skew_seconds=0)
    for key, runtime in engine._runtime.items():
        runtime.market_open = runtime.book_synchronized = True
        runtime.bids = ((5000.0, 10),)
        runtime.asks = ((5000.25, 10),)
        runtime.last_quote_receive_ts = runtime.last_trade_receive_ts = NOW
        engine.ledger.mark_instrument(key, 5000, NOW)
    engine.ledger.record_fill("old-open", old.spec.instrument_id, Side.BUY, old_quantity, 5000,
                             fee=0, fee_currency="USD", timestamp=NOW)
    if target_quantity:
        engine.ledger.record_fill("new-open", new_spec.instrument_id, Side.BUY, target_quantity, 5000,
                                 fee=0, fee_currency="USD", timestamp=NOW)
    return engine


def test_futures_roll_cannot_bypass_replacement_position_limit():
    engine = roll_engine(target_quantity=1)
    original = engine.ledger.export_state()
    with pytest.raises(ValueError, match="instrument_position_limit"):
        engine.roll_future("CME:ESZ6", "CME:ESH7")
    assert engine.ledger.export_state() == original
    assert engine.total_fills == 0


def test_futures_roll_checks_displayed_liquidity_and_halt_before_changing_ledger():
    engine = roll_engine()
    original = engine.ledger.export_state()
    engine._runtime["CME:ESH7"].asks = ((5000.25, 0.5),)
    with pytest.raises(ValueError, match="displayed liquidity"):
        engine.roll_future("CME:ESZ6", "CME:ESH7")
    assert engine.ledger.export_state() == original
    engine.circuit_breaker_active = True
    with pytest.raises(ValueError, match="risk halt"):
        engine.roll_future("CME:ESZ6", "CME:ESH7")


def test_futures_roll_persists_both_legs_before_emitting_and_recovers(tmp_path):
    manager = StateManager(str(tmp_path / "roll.db"))
    engine = roll_engine(manager=manager)
    emitted = []
    engine.on("strategy_fill", lambda leg: emitted.append(
        (leg, manager.load_cross_asset_snapshot(engine.account_id))))
    result = engine.roll_future("CME:ESZ6", "CME:ESH7")
    assert result["rolled"] is True
    assert len(emitted) == 2
    assert all(item[1]["snapshot"]["ledger"]["positions"]["CME:ESZ6"]["quantity"] == 0
               for item in emitted)
    assert engine.ledger.get_position("CME:ESH7", as_of=NOW).quantity == 1
    assert engine.ledger.snapshot(NOW).equity == pytest.approx(199997.5)


def test_futures_roll_persistence_failure_restores_financial_state():
    class FailingStore:
        def load_cross_asset_snapshot(self, account):
            return None

        def commit_cross_asset_event(self, *args, **kwargs):
            raise OSError("simulated disk full")

    engine = roll_engine(manager=FailingStore())
    original = engine.ledger.export_state()
    emitted = []
    engine.on("strategy_fill", emitted.append)
    with pytest.raises(RuntimeError, match="paper history"):
        engine.roll_future("CME:ESZ6", "CME:ESH7")
    assert engine.ledger.export_state() == original
    assert engine.total_fills == 0
    assert engine.trades == []
    assert not emitted
    assert engine.circuit_breaker_reason == "paper_history_persistence_failed"


def equity_spec():
    return InstrumentSpec("AAPL", AssetClass.EQUITY, "AAPL", "NASDAQ", "AAPL", "USD", "USD",
        "shares", .01, 1, 1, borrow_limit=100, borrow_fee_rate=.365,
        borrow_rate_source="fixture sourced estimate", borrow_rate_effective_from=NOW - 1,
        borrow_rate_effective_to=NOW + 7 * 86400)


def test_borrow_liability_uses_actual_short_intervals_and_survives_restore():
    ledger = PortfolioLedger(10000, max_mark_age_seconds=7 * 86400)
    ledger.register_instrument(equity_spec(), allow_short=True)
    ledger.record_fill("short", "AAPL", Side.SELL, 10, 100, fee=0, fee_currency="USD", timestamp=NOW)
    ledger.record_fill("partial-cover", "AAPL", Side.BUY, 5, 100, fee=0, fee_currency="USD", timestamp=NOW + 3600)
    ledger.mark_instrument("AAPL", 200, NOW + 7200)
    value = ledger.snapshot(NOW + 10800)
    expected = (1000 + 500 + 1000) * .365 * 3600 / (365 * 86400)
    assert value.accrued_costs == pytest.approx(expected)
    assert value.equity == pytest.approx(9500 - expected)
    assert value.translation_pnl == pytest.approx(0)
    restored = PortfolioLedger(10000, max_mark_age_seconds=7 * 86400)
    restored.register_instrument(equity_spec(), allow_short=True)
    restored.restore_state(ledger.export_state())
    assert restored.snapshot(NOW + 10800).equity == pytest.approx(value.equity)


def test_midnight_borrow_posting_does_not_double_count_costs():
    ledger = PortfolioLedger(10000, max_mark_age_seconds=7 * 86400)
    ledger.register_instrument(equity_spec(), allow_short=True)
    ledger.record_fill("short", "AAPL", Side.SELL, 10, 100, fee=0, fee_currency="USD", timestamp=NOW)
    midnight = NOW + 12 * 3600
    expected = 1000 * .365 / 365 / 2
    assert ledger.snapshot(midnight).accrued_costs == pytest.approx(expected)
    ledger.accrue_borrow(midnight)
    after = ledger.snapshot(midnight)
    assert after.costs_paid == pytest.approx(expected)
    assert after.accrued_costs == 0
    assert after.equity == pytest.approx(10000 - expected)
    ledger.accrue_borrow(midnight)
    assert ledger.snapshot(midnight).costs_paid == pytest.approx(expected)


def test_official_closure_valuation_keeps_source_time_and_expires():
    ledger = PortfolioLedger(10000, max_mark_age_seconds=30)
    ledger.register_instrument(equity_spec())
    ledger.record_fill("long", "AAPL", Side.BUY, 1, 100, fee=0, fee_currency="USD", timestamp=NOW)
    with pytest.raises(ValueError, match="stale mark"):
        ledger.snapshot(NOW + 100)
    ledger.set_valuation_mark("AAPL", 101, NOW, valid_until=NOW + 86460,
                             source="official fixture", basis="official_close", stress_fraction=.1)
    value = ledger.snapshot(NOW + 86400)
    assert value.equity == 10001
    assert value.positions["AAPL"].mark_timestamp == NOW
    assert value.positions["AAPL"].mark_basis == "official_close"
    assert ledger.closed_market_stress(NOW + 86400) == pytest.approx(10.1)
    with pytest.raises(ValueError, match="stale mark"):
        ledger.snapshot(NOW + 86461)


def test_scheduled_financing_is_idempotent_after_restart(tmp_path):
    config = load_cross_asset_config(CONFIG)
    clock = Clock()
    manager = StateManager(str(tmp_path / "actions.db"))
    engine = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    action = {"action_id": "carry", "action_type": "financing", "instrument_id": "OANDA:EUR_USD",
              "source": "sourced fixture", "effective_ts": NOW - 5,
              "payload": {"amount": -2, "currency": "USD"}}
    engine.schedule_account_action(action)
    assert engine.process_account_actions()[0]["status"] == "applied"
    restarted = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    restarted.schedule_account_action(action)
    assert restarted.process_account_actions() == []
    assert restarted.ledger.snapshot(NOW).cash_by_currency["USD"] == 39998


def test_additive_watchlist_preserves_existing_cash_history(tmp_path):
    config = load_cross_asset_config(CONFIG)
    manager = StateManager(str(tmp_path / "additive.db"))
    engine = CrossAssetPaperEngine(config, clock=Clock(), state_manager=manager)
    engine._save_checkpoint(NOW)
    added = replace(config.instruments["NASDAQ:AAPL"],
                    spec=replace(config.instruments["NASDAQ:AAPL"].spec,
                                 instrument_id="NASDAQ:MSFT", symbol="MSFT", base_currency="MSFT"),
                    provider_symbol="MSFT")
    instruments = dict(config.instruments, **{"NASDAQ:MSFT": added})
    providers = dict(config.providers)
    provider = providers[added.provider_id]
    providers[added.provider_id] = replace(provider, instrument_ids=provider.instrument_ids + ("NASDAQ:MSFT",))
    changed = replace(config, instruments=instruments, providers=providers)
    restored = CrossAssetPaperEngine(changed, clock=Clock(), state_manager=manager)
    assert restored.ledger.snapshot(NOW).equity == 50000
    assert restored.ledger.get_position("NASDAQ:MSFT", as_of=NOW).quantity == 0


def test_stale_instrument_does_not_prevent_dashboard_snapshot():
    engine = CrossAssetPaperEngine(load_cross_asset_config(CONFIG), clock=Clock(NOW + 100))
    engine.ledger.record_fill("long", "NASDAQ:AAPL", Side.BUY, 1, 100,
                             fee=0, fee_currency="USD", timestamp=NOW)
    data = engine.get_dashboard_state()
    assert len(data["instruments"]) == 4
    row = next(item for item in data["instruments"] if item["instrument_id"] == "NASDAQ:AAPL")
    assert row["valuation_valid"] is False
    assert data["portfolio_error"]
