import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.runtime.application_service import PaperApplicationService
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.portfolio_analytics import PortfolioAnalytics
from src.runtime.recording import SessionRecorder, iter_records
from src.runtime.replay import DeterministicIds, ReplayRunner, VirtualClock
from src.runtime.state_manager import StateManager
from src.market.events import MarketDataEvent, MarketEventKind
from src.strategies.adaptive_market_maker import adaptive_strategy_factory


CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"


def command(manager, config, identity, action, payload, revision=None):
    return {"protocol_version": 1, "account_id": config.account_id, "request_id": identity,
            "expected_revision": manager.get_account_revision(config.account_id) if revision is None else revision,
            "command": action, "payload": payload}


def test_actual_service_capture_replays_success_rejection_retry_and_partial_import(tmp_path):
    config = load_cross_asset_config(CONFIG)
    clock = VirtualClock(1792497600)
    manager = StateManager(str(tmp_path / "capture.db"))
    engine = CrossAssetPaperEngine(config, clock=clock, state_manager=manager,
                                  id_factory=DeterministicIds(123))
    analytics = PortfolioAnalytics(manager, config.account_id)
    directory = tmp_path / "recording"
    recorder = SessionRecorder(directory, config=json.loads(CONFIG.read_text()), seed=123)
    recorder.set_initial_state(engine._checkpoint_payload())
    service = PaperApplicationService(engine, manager, recorder=recorder, analytics=analytics, clock=clock)
    service.start()
    try:
        service.tick()
        service.flush()
        update = command(manager, config, "update", "update_params", {
            "instrument_id": "BINANCE:BTCUSDT", "spread_bps": 17})
        assert service.submit_command(update).result(timeout=5)["success"]
        assert service.submit_command(update).result(timeout=5)["success"]
        stale = command(manager, config, "stale", "update_params", {
            "instrument_id": "BINANCE:BTCUSDT", "spread_bps": 99}, revision=0)
        assert service.submit_command(stale).result(timeout=5)["code"] == "revision_conflict"
        rejected = command(manager, config, "reject", "record_funding", {
            "funding_id": "invalid", "currency": "USD", "amount": -1, "source": "fixture"})
        assert service.submit_command(rejected).result(timeout=5)["code"] == "rejected"
        action = {"action_id": "first-imported", "action_type": "financing",
                  "instrument_id": "OANDA:EUR_USD", "source": "fixture",
                  "effective_ts": clock.value + 100, "payload": {"amount": -2, "currency": "USD"}}
        imported = command(manager, config, "partial", "import_account_actions", {
            "actions": [action, {"invalid": "action"}]})
        assert service.submit_command(imported).result(timeout=5)["code"] == "rejected"
        funding = command(manager, config, "fund", "record_funding", {
            "funding_id": "valid-funding", "currency": "USD", "amount": 100, "source": "fixture"})
        assert service.submit_command(funding).result(timeout=5)["success"]
        clock.advance_to(clock.value + 1)
        service.tick()
        service.flush()
        expected_cash = engine.ledger.export_state()["cash"]
        assert analytics._previous["external_flow"] == 100
    finally:
        service.stop()
    recorded_commands = [row["payload"] for row in iter_records(directory) if row["kind"] == "command"]
    assert any(item.get("duplicate") for item in recorded_commands)
    assert any(item.get("outcome", {}).get("code") == "revision_conflict" for item in recorded_commands)
    instances = []

    def factory(configuration, *, clock, id_factory, **kwargs):
        instance = CrossAssetPaperEngine(configuration, clock=clock, id_factory=id_factory, **kwargs)
        instances.append(instance)
        return instance

    replayed = ReplayRunner(config, engine_factory=factory).run(directory)
    assert replayed["financial"]["cash"] == expected_cash
    assert replayed["financial"]["total_pnl"] == pytest.approx(0)
    assert instances[-1]._runtime["BINANCE:BTCUSDT"].strategy.spread_bps == 17
    assert [item["action_id"] for item in instances[-1].get_account_actions()] == ["first-imported"]
    assert replayed["analytics"]["external_flow"] == 100


def test_backdated_funding_after_last_sample_is_recovered_as_flow_not_profit(tmp_path):
    config = load_cross_asset_config(CONFIG)
    clock = VirtualClock(1792497600)
    path = str(tmp_path / "funding.db")
    manager = StateManager(path)
    engine = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    analytics = PortfolioAnalytics(manager, config.account_id)
    service = PaperApplicationService(engine, manager, analytics=analytics, clock=clock)
    service._observe(clock.value)
    assert analytics._previous["account_revision"] == 0
    engine.record_funding("late-posted", "USD", 100, "fixture", timestamp=clock.value - 100)
    # Simulate a crash after financial commit and before the next observation.
    clock.advance_to(clock.value + 1)
    manager = StateManager(path)
    recovered = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    restored_analytics = PortfolioAnalytics(manager, config.account_id)
    restored_service = PaperApplicationService(recovered, manager, analytics=restored_analytics, clock=clock)
    restored_service._observe(clock.value)
    assert restored_analytics._previous["external_flow"] == 100
    assert restored_analytics._wealth_index == pytest.approx(1)
    assert PortfolioAnalytics(manager, config.account_id)._pending_flow == 0


def test_delayed_prefill_source_quote_cannot_teach_a_five_second_markout():
    config = load_cross_asset_config(CONFIG)
    clock = VirtualClock(1792497600)
    engine = CrossAssetPaperEngine(config, clock=clock,
        strategy_factory=adaptive_strategy_factory())
    strategy = engine._runtime["NASDAQ:AAPL"].strategy
    analytics = PortfolioAnalytics()

    def quote(mid, source_ts):
        return MarketDataEvent("NASDAQ:AAPL", "causality-fixture", MarketEventKind.BOOK_SNAPSHOT,
            source_ts, clock.value, bids=((mid - 0.01, 100),), asks=((mid + 0.01, 100),))

    assert engine.on_market_event(quote(100, clock.value - 2))
    fill = {"fill_id": "causal-fill", "instrument_id": "NASDAQ:AAPL", "timestamp": clock.value,
            "price": 100, "side": "BUY"}
    strategy.observe_fill(fill)
    analytics.on_fill(fill)
    clock.advance_to(clock.value + 5)
    delayed = quote(50, clock.value - 6)
    assert engine.on_market_event(delayed)
    analytics.on_market(delayed)
    assert not strategy._adverse_markouts
    assert analytics.summary()["markouts"]["5"]["count"] == 0
    clock.advance_to(clock.value + 0.5)
    causal = quote(98, clock.value)
    assert engine.on_market_event(causal)
    analytics.on_market(causal)
    assert list(strategy._adverse_markouts) == [pytest.approx(200)]
    assert analytics.summary()["markouts"]["5"]["mean_bps"] == pytest.approx(-200)


def test_full_phase_latency_histogram_preserves_early_spikes_and_resets():
    service = PaperApplicationService(SimpleNamespace(account_id="latency", state_manager=None))
    for _ in range(1000):
        service._record_queue_delay(250)
    for _ in range(10000):
        service._record_queue_delay(0.25)
    assert service._delay_percentile() == 0.25
    assert service._phase_delay_percentile() == 250
    assert service._latency_count == 11000
    assert service._latency_max_ms == 250
    assert len(service._latency_bins) == 60002
    service.reset_latency_metrics()
    assert service._phase_delay_percentile() == 0
    assert service._latency_count == 0
    service._record_queue_delay(60001.25)
    assert service._phase_delay_percentile() == 60001.25
