import gzip
import json
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.paper_cli import diagnostics
from src.runtime.portfolio_analytics import PortfolioAnalytics
from src.runtime.recording import SessionRecorder, event_from_dict, event_to_dict, iter_records, load_manifest
from src.runtime.replay import FaultSpec, ReplayRunner, VirtualClock
from src.runtime.research_evaluation import evaluate_recording
from src.strategies.adaptive_market_maker import AdaptiveMarketMaker
from src.utils.types import Side


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config/cross_asset_paper.example.json"
FIXTURE = ROOT / "tests/fixtures/representative_recording"


def quote(stamp, *, source_stamp=None, mid=100.0, instrument="NASDAQ:AAPL"):
    return MarketDataEvent(instrument, "fixture", MarketEventKind.BOOK_SNAPSHOT,
        stamp if source_stamp is None else source_stamp, stamp,
        bids=((mid - 0.01, 100),), asks=((mid + 0.01, 100),))


def test_recording_preserves_arrival_order_and_redacts_credentials(tmp_path):
    directory = tmp_path / "capture"
    with SessionRecorder(directory, config={"api_key": "never-write-this", "safe": 1}) as recorder:
        recorder.record_market(quote(20, source_stamp=19), processing_ts=20)
        recorder.record_market(quote(21, source_stamp=18), processing_ts=21)
        recorder.append("command", {"session_token": "also-secret", "command": "stop_strategy"}, 21)
    rows = list(iter_records(directory))
    assert [row["ingest_seq"] for row in rows] == [1, 2, 3]
    assert [row["payload"]["event_ts"] for row in rows[:2]] == [19, 18]
    assert rows[-1]["payload"]["session_token"] == "[redacted]"
    assert "never-write-this" not in (directory / "manifest.json").read_text()
    assert event_from_dict(rows[0]["payload"]) == quote(20, source_stamp=19)
    trade = MarketDataEvent("NASDAQ:AAPL", "fixture", MarketEventKind.TRADE, 20, 20,
                            price=100, quantity=1, side=Side.BUY)
    assert event_from_dict(event_to_dict(trade)) == trade


def test_incomplete_and_corrupted_recordings_cannot_certify_replay(tmp_path):
    directory = tmp_path / "capture"
    recorder = SessionRecorder(directory, config={})
    recorder.record_market(quote(20))
    recorder.close(complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        list(iter_records(directory))
    assert len(list(iter_records(directory, allow_incomplete=True))) == 1
    manifest = load_manifest(directory, allow_incomplete=True)
    chunk = directory / manifest["chunks"][0]["file"]
    chunk.write_bytes(chunk.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        list(iter_records(directory, allow_incomplete=True))


def test_replay_uses_manifest_seed_and_is_deterministic_across_four_markets():
    first = ReplayRunner(CONFIG, research_assumption_override=True).run(FIXTURE)
    second = ReplayRunner(CONFIG, research_assumption_override=True).run(FIXTURE)
    assert first["seed"] == 7
    assert first["digest"] == second["digest"]
    assert first["financial"] == second["financial"]
    assert first["fills"]
    assert all(fill["estimated"] for fill in first["fills"])
    assert {fill["instrument_id"] for fill in first["fills"]} == {
        "BINANCE:BTCUSDT", "NASDAQ:AAPL", "OANDA:EUR_USD", "CME:ESZ6"}


def test_duplicate_fault_does_not_allocate_liquidity_twice():
    original = ReplayRunner(CONFIG, research_assumption_override=True).run(FIXTURE)
    duplicated = ReplayRunner(CONFIG, research_assumption_override=True).run(FIXTURE, faults=FaultSpec(duplicate_every=1))
    assert duplicated["duplicate_records"] > 0
    assert duplicated["financial"] == original["financial"]
    assert duplicated["fills"] == original["fills"]


def test_replay_restores_initial_account_without_touching_original_database(tmp_path):
    from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
    config = load_cross_asset_config(CONFIG)
    engine = CrossAssetPaperEngine(config, clock=VirtualClock(1792497600))
    engine.record_fx_financing("initial-carry", "OANDA:EUR_USD", -2, "USD")
    directory = tmp_path / "capture"
    with SessionRecorder(directory, config=json.loads(CONFIG.read_text())) as recorder:
        recorder.set_initial_state(engine._checkpoint_payload())
        recorder.append("timer", {}, 1792497601)
    result = ReplayRunner(config).run(directory)
    assert result["financial"]["cash"]["USD"] == pytest.approx(39998)
    assert result["financial"]["costs_paid"] == pytest.approx(2)
    assert result["financial"]["total_pnl"] == pytest.approx(-2)


def test_complete_minute_returns_remove_external_funding_and_do_not_annualize():
    analytics = PortfolioAnalytics()
    equity = 1000.0
    analytics.observe(0, {"equity": equity, "reporting_currency": "USD"})
    for stamp in range(1, 61):
        flow = 100 if stamp == 30 else 0
        equity = equity * 1.001 + flow
        analytics.observe(stamp, {"equity": equity, "reporting_currency": "USD"}, external_flow=flow)
    result = analytics.summary()
    assert result["return_observations"] == 1
    assert analytics._returns[0]["return"] == pytest.approx(1.001 ** 60 - 1)
    assert result["sharpe_ratio"] is None
    assert result["annualized"] is False


def test_invalid_or_missing_observation_breaks_return_interval():
    analytics = PortfolioAnalytics()
    for stamp in range(61):
        if stamp == 30:
            continue
        analytics.observe(stamp, {"equity": 1000 + stamp})
    assert analytics.summary()["return_observations"] == 0


def test_persisted_analytics_restart_matches_continuous_minute_phase_and_drawdown(tmp_path):
    from src.runtime.state_manager import StateManager
    manager = StateManager(str(tmp_path / "analytics.db"))
    continuous = PortfolioAnalytics()
    restored = PortfolioAnalytics(manager, "analytics-restart")
    equity = 1000.0
    for stamp in range(181):
        if stamp == 78:
            restored = PortfolioAnalytics(manager, "analytics-restart")
        flow = 100 if stamp == 30 else 0
        if stamp:
            equity = equity * (1.001 if stamp < 100 else 0.999) + flow
        payload = {"equity": equity, "reporting_currency": "USD"}
        continuous.observe(stamp, payload, external_flow=flow)
        restored.observe(stamp, payload, external_flow=flow)
    first, second = continuous.summary(), restored.summary()
    assert first["return_observations"] == second["return_observations"] == 3
    assert first["max_drawdown"] == pytest.approx(second["max_drawdown"])
    assert first["sharpe_ratio"] == pytest.approx(second["sharpe_ratio"])
    assert len(manager.get_account_history("analytics-restart", limit=1000)) == 181


def test_funding_event_is_applied_once_to_the_next_observation():
    analytics = PortfolioAnalytics()
    analytics.observe(0, {"equity": 1000})
    payload = {"funding_id": "deposit-1", "external_flow_reporting": 100}
    analytics.on_funding(payload)
    analytics.on_funding(payload)
    analytics.observe(1, {"equity": 1100})
    assert analytics.history(limit=1)[0]["external_flow"] == 100
    assert analytics._wealth_index == pytest.approx(1)


def test_markouts_require_post_fill_fresh_quotes_at_the_requested_horizon():
    analytics = PortfolioAnalytics(max_mark_age=1)
    analytics.on_fill({"fill_id": "f1", "instrument_id": "NASDAQ:AAPL", "timestamp": 10,
                       "price": 100, "side": "BUY"})
    analytics.on_market(quote(11, source_stamp=9, mid=102))
    assert analytics.summary()["markouts"]["1"]["count"] == 0
    analytics.on_market(quote(11, mid=101))
    analytics.on_market(quote(15, mid=99))
    result = analytics.summary()["markouts"]
    assert result["1"]["mean_bps"] == pytest.approx(100)
    assert result["5"]["mean_bps"] == pytest.approx(-100)
    assert result["30"]["count"] == 0


def test_mixed_currency_exposure_is_not_summed_without_conversion_marks():
    analytics = PortfolioAnalytics()
    instruments = [{"instrument_id": "BTC", "asset_class": "crypto", "position": 1,
                    "mid_price": 100, "quote_currency": "USDT", "contract_multiplier": 1}]
    result = analytics.observe(1, {"equity": 1000, "reporting_currency": "USD"}, instruments)
    assert result["gross_exposure"] is None
    assert result["gross_exposure_by_currency"] == {"USDT": 100}
    result = analytics.observe(2, {"equity": 1000, "reporting_currency": "USD", "fx_rates": {"USDT": 0.9}}, instruments)
    assert result["gross_exposure"] == pytest.approx(90)


def test_adaptive_spread_covers_fees_and_rejects_an_unaffordable_bound():
    item = load_cross_asset_config(CONFIG).instruments["NASDAQ:AAPL"]
    strategy = AdaptiveMarketMaker(item.spec, order_size=1, max_inventory=20,
        spread_bps=8, maker_fee=0.001, taker_fee=0.002, max_spread_bps=100)
    orders = strategy.on_market_event({"mid_price": 100, "time": 1})
    assert len(orders) == 2
    assert strategy.last_policy_state["spread_bps"] >= 30
    bounded = AdaptiveMarketMaker(item.spec, order_size=1, max_inventory=20,
        spread_bps=8, maker_fee=0.001, taker_fee=0.002, max_spread_bps=20)
    assert bounded.on_market_event({"mid_price": 100, "time": 1}) == []
    assert bounded.last_policy_state["reason"] == "cost_floor_exceeds_spread_bound"


def test_adaptive_volatility_and_markouts_use_only_observed_past_data():
    item = load_cross_asset_config(CONFIG).instruments["NASDAQ:AAPL"]
    strategy = AdaptiveMarketMaker(item.spec, order_size=1, max_inventory=20)
    strategy.observe_market(100, 1)
    strategy.observe_fill({"timestamp": 1, "price": 100, "side": "BUY"})
    strategy.observe_market(99, 2)
    assert not strategy._adverse_markouts
    strategy.observe_market(98, 6)
    assert strategy._adverse_markouts[0] == pytest.approx(200)
    assert strategy._variance > 0


def test_research_freezes_training_candidate_and_reports_holdouts_without_promotion():
    result = evaluate_recording(FIXTURE, CONFIG, candidates=[{
        "volatility_multiplier": 1.0, "adverse_selection_multiplier": 0.0}])
    assert result["status"] == "offline_evaluation_complete"
    assert result["split"]["training"][1] == result["split"]["validation"][0]
    assert result["split"]["validation"][1] == result["split"]["test"][0]
    assert "candidate_test" in result["scenarios"]["configured"]
    assert result["promotion_eligible"] is False
    assert result["profitability_guaranteed"] is False


def test_diagnostics_show_presence_without_credentials(monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "never-expose-me")
    result = diagnostics(CONFIG)
    assert "never-expose-me" not in json.dumps(result)
    assert result["live_acceptance"] == "pending"
    assert result["credentials_included"] is False
