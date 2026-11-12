import json
import math

import pytest

from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime.causal_analysis import (
    CausalVenueScout,
    analyze_recording,
    cross_entropy,
    entropy,
    information_gain,
    kl_divergence,
)
from src.runtime.paper_cli import main
from src.runtime.recording import SessionRecorder


def trade(stamp, price=100, instrument="A", source="fixture"):
    return MarketDataEvent(
        instrument, source, MarketEventKind.TRADE, stamp, stamp, price=price, quantity=1
    )


def feed(scout, count=45):
    prices = [100.0, 100.0]
    for i in range(count):
        for j, instrument in enumerate(("A", "B")):
            change = math.sin((i - j) * 1.71) * 0.001
            prices[j] *= 1 + change
            scout.observe(trade(i, prices[j], instrument), i)


def test_lagged_alignment_and_read_only_summary():
    scout = CausalVenueScout()
    feed(scout)
    first = scout.summary(45)
    edge = next(e for e in first["edges"] if e["source"][1] == "A" and e["target"][1] == "B")
    assert edge["correlation"] == pytest.approx(1)
    assert first == scout.summary(45)
    assert all(0 <= r["research_priority"] <= 1 for r in first["streams"])
    assert first["advisory_only"] is True


def test_open_bucket_and_later_events_do_not_change_prior_snapshot():
    scout = CausalVenueScout()
    feed(scout, 15)
    snapshot = scout.summary(14)
    samples = snapshot["streams"][0]["samples"]
    scout.observe(trade(14, 999, "A"), 14)
    assert scout.summary(14) == snapshot
    assert scout.summary(15)["streams"][0]["samples"] == samples + 1


def test_gaps_constant_series_and_sparse_streams_do_not_invent_edges():
    scout = CausalVenueScout()
    for t in range(40):
        scout.observe(trade(t), t)
        if t % 2 == 0:
            scout.observe(trade(t, instrument="B"), t)
    result = scout.summary(40)
    assert result["edges"] == []
    b = next(r for r in result["streams"] if r["instrument_id"] == "B")
    assert b["samples"] == 0
    assert b["research_priority"] is None


def test_bounded_state_provider_identity_expiry_and_reset():
    scout = CausalVenueScout(max_buckets=20, max_streams=2)
    for i in range(100):
        scout.observe(trade(i, source="one"), i)
        scout.observe(trade(i, source="two"), i)
    scout.observe(trade(100, source="three"), 100)
    assert scout.skipped_capacity == 1
    assert len(scout.summary(101)["streams"]) == 2
    assert all(len(h) <= 20 for h in scout._prices.values())
    assert all(r["samples"] == 0 for r in scout.summary(1000)["streams"])
    scout.reset()
    assert scout.summary()["streams"] == []
    assert scout.skipped_capacity == 0


def test_restore_matches_continuation_and_detects_tamper(tmp_path):
    scout = CausalVenueScout()
    feed(scout)
    path = tmp_path / "research.json"
    scout.save(path)
    restored = CausalVenueScout.load(path)
    assert restored.summary() == scout.summary()
    for item in (scout, restored):
        item.observe(trade(45, 101), 45)
    assert restored.summary(46) == scout.summary(46)
    with pytest.raises(FileExistsError):
        scout.save(path)
    state = json.loads(path.read_text())
    state["state"]["clock"] = 123
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="checksum"):
        CausalVenueScout.load(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bucket_seconds": 0},
        {"bucket_seconds": float("nan")},
        {"forgetting": 0},
        {"min_samples": 2},
        {"max_streams": True},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        CausalVenueScout(**kwargs)


def test_bad_clock_and_probability_inputs():
    scout = CausalVenueScout()
    scout.observe(trade(10), 10)
    for stamp in (9, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            scout.observe(trade(10), stamp)
    with pytest.raises(ValueError):
        entropy([0, 0])
    with pytest.raises(ValueError):
        information_gain(1, 0, 0)
    assert entropy([0.5, 0.5]) == 1
    assert information_gain(1, 0.5, 2) == 0.25
    assert cross_entropy([0.5, 0.5], [0.5, 0.5]) == 1
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0
    assert math.isinf(kl_divergence([1, 0], [0, 1]))
    with pytest.raises(ValueError):
        cross_entropy([1], [0.5, 0.5])


def test_snapshot_prices_regime_decay_and_bayesian_direction():
    scout = CausalVenueScout()
    for stamp in range(30):
        price = 100 * (1.00001**stamp)
        scout.observe(
            MarketDataEvent(
                "A",
                "fixture",
                MarketEventKind.BOOK_SNAPSHOT,
                stamp,
                stamp,
                bids=((price - 0.01, 5),),
                asks=((price + 0.01, 5),),
            ),
            stamp,
        )
    row = scout.summary(30)["streams"][0]
    assert row["regime"] == "SLOW"
    assert row["positive_return_probability"] > 0.8
    assert scout.summary(34)["streams"][0]["decay_weight"] < row["decay_weight"]
    assert scout.summary(40)["streams"][0]["stale"] is True


def test_recording_provider_boundary_resets_research_history(tmp_path):
    directory = tmp_path / "recording"
    with SessionRecorder(directory, config={}) as recorder:
        for i in range(20):
            recorder.record_market(trade(i, 100 + i), processing_ts=i)
        recorder.append("provider", {"state": "invalidated"}, 21, provider_id="fixture")
        recorder.record_market(trade(22), processing_ts=22)
    result = analyze_recording(directory)
    assert result["streams"][0]["samples"] == 0
    assert result["edges"] == []


def test_status_breaks_history_and_ignores_delta_and_nonpositive_prices():
    scout = CausalVenueScout()
    feed(scout)
    scout.observe(
        MarketDataEvent("A", "fixture", MarketEventKind.STATUS, 45, 45, status="disconnected"), 45
    )
    scout.observe(trade(46, -1, "C"), 46)
    scout.observe(
        MarketDataEvent("D", "fixture", MarketEventKind.BOOK_DELTA, 46, 46, bids=((100, 1),)), 46
    )
    assert [r["instrument_id"] for r in scout.summary()["streams"]] == ["B"]


def test_recording_cli_is_deterministic_and_does_not_overwrite(tmp_path, capsys):
    directory = tmp_path / "recording"
    with SessionRecorder(directory, config={}) as recorder:
        for i in range(25):
            recorder.record_market(trade(i, 100 + math.sin(i)), processing_ts=i)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    first = analyze_recording(directory)
    assert first == analyze_recording(directory)
    output = tmp_path / "report.json"
    assert main(["causal", str(directory), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == first
    with pytest.raises(FileExistsError):
        main(["causal", str(directory), "--output", str(output)])
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    capsys.readouterr()


def test_price_policy_sequences_and_provider_scoped_generations():
    scout = CausalVenueScout()
    feed(scout)
    before = scout.summary(45)
    scout.observe(
        MarketDataEvent(
            "A",
            "fixture",
            MarketEventKind.BOOK_SNAPSHOT,
            45,
            45,
            bids=((900, 1),),
            asks=((902, 1),),
        ),
        45,
    )
    assert scout.summary(45)["streams"] == before["streams"]
    assert scout.diagnostics["price_policy"] == 1
    scout.observe(trade(45, source="other"), 45)
    scout.provider_status("fixture", 1, "connecting")
    assert [r["provider"] for r in scout.summary()["streams"]] == ["other"]
    scout.observe(trade(46), 46, generation=0)
    assert scout.diagnostics["old_generation"] == 1
    event = MarketDataEvent(
        "A", "fixture", MarketEventKind.TRADE, 47, 47, sequence=10, price=100, quantity=1
    )
    scout.observe(event, 47, generation=1)
    scout.observe(event, 48, generation=1)
    assert scout.diagnostics["duplicate_or_reordered_sequence"] == 1
    assert list(scout._prices[("fixture", "A")]) == [(47, 100)]


def test_healthy_provider_notifications_preserve_history(tmp_path):
    directory = tmp_path / "recording"
    with SessionRecorder(directory, config={}) as recorder:
        for i in range(20):
            recorder.record_market(trade(i, 100 + i), provider_id="fixture", processing_ts=i)
            recorder.append("provider", {"state": "healthy"}, i, provider_id="fixture")
    assert analyze_recording(directory)["streams"][0]["samples"] > 10


def test_research_failures_do_not_change_accepted_account_events():
    from src.runtime.application_service import PaperApplicationService
    from tests.test_application_runtime import MinimalEngine

    engine = MinimalEngine()
    scout = CausalVenueScout()
    service = PaperApplicationService(engine, research=scout)
    service.start()
    try:
        service.enqueue_market("fixture", "not-a-market-event", 1)
        service.flush()
        assert engine.events == [("not-a-market-event", 1)]
        assert service.get_dashboard_state()["research"]["status"] == "error"
        assert service.get_dashboard_state()["application"]["last_error"] is None
    finally:
        service.stop()


@pytest.mark.parametrize(
    "field,value",
    [
        ("kinds", []),
        ("generations", {"fixture": -1}),
        ("prices", [[["fixture", "A"], []]]),
        ("diagnostics", {"bad": True}),
    ],
)
def test_checksummed_but_invalid_state_is_rejected(tmp_path, field, value):
    import hashlib

    from src.runtime.recording import canonical_json

    scout = CausalVenueScout()
    feed(scout)
    path = tmp_path / "state.json"
    scout.save(path)
    envelope = json.loads(path.read_text())
    envelope["state"][field] = value
    envelope["sha256"] = hashlib.sha256(canonical_json(envelope["state"]).encode()).hexdigest()
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError):
        CausalVenueScout.load(path)


def test_recording_continuation_matches_single_run(tmp_path):
    scout = CausalVenueScout()
    first, second = tmp_path / "first", tmp_path / "second"
    for directory, times in ((first, range(20)), (second, range(20, 40))):
        with SessionRecorder(directory, config={}) as recorder:
            for stamp in times:
                event = trade(stamp, 100 + math.sin(stamp))
                recorder.record_market(event, processing_ts=stamp)
                scout.observe(event, stamp)
    state = tmp_path / "state.json"
    analyze_recording(first, state_out=state)
    assert analyze_recording(second, state_in=state) == scout.summary(39)
    with pytest.raises(ValueError, match="backwards"):
        analyze_recording(first, state_in=state)


def test_direction_evaluation_is_chronological_and_has_no_execution_claim(tmp_path):
    from src.runtime.causal_analysis import evaluate_direction_recording

    directory = tmp_path / "recording"
    with SessionRecorder(directory, config={}) as recorder:
        for stamp in range(100):
            recorder.record_market(trade(stamp, 100 * 1.001**stamp), processing_ts=stamp)
    result = evaluate_direction_recording(directory)
    assert result["train_end"] < result["validation_end"] < result["end"]
    assert result["metrics"]["test"]["samples"] > 10
    assert result["metrics"]["test"]["brier"] < result["metrics"]["test"]["baseline_brier"]
    assert result["execution_evaluated"] is False
    assert result == evaluate_direction_recording(directory)


def test_versioned_configuration_requires_complete_valid_parameters():
    scout = CausalVenueScout(price_policy="midpoint")
    assert CausalVenueScout.from_parameters(scout.parameters()).parameters() == scout.parameters()
    with pytest.raises(ValueError):
        CausalVenueScout.from_parameters({"schema_version": 1})


def test_advisory_replay_has_identical_financial_state_and_fills():
    from pathlib import Path

    from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
    from src.runtime.replay import ReplayRunner

    root = Path(__file__).parents[1]
    scout = CausalVenueScout()

    class ResearchEngine(CrossAssetPaperEngine):
        def on_market_event(self, event, **kwargs):
            accepted = super().on_market_event(event, **kwargs)
            if accepted:
                scout.observe(event, self.clock(), **kwargs)
            return accepted

    config = root / "config/cross_asset_paper.example.json"
    directory = root / "tests/fixtures/representative_recording"
    baseline = ReplayRunner(config, research_assumption_override=True).run(directory)
    advisory = ReplayRunner(config, engine_factory=ResearchEngine, research_assumption_override=True).run(directory)
    assert advisory["financial"] == baseline["financial"]
    assert advisory["fills"] == baseline["fills"]
    assert advisory["digest"] == baseline["digest"]
    assert scout.summary()["streams"]


def test_runtime_feeds_only_accepted_events_and_invalidates_one_provider():
    from src.runtime.application_service import PaperApplicationService
    from tests.test_application_runtime import MinimalEngine

    class AcceptingEngine(MinimalEngine):
        def on_market_event(self, event, **kwargs):
            super().on_market_event(event, **kwargs)
            return event.instrument_id != "rejected"

    scout = CausalVenueScout()
    service = PaperApplicationService(AcceptingEngine(), research=scout, clock=lambda: 100)
    service.start()
    try:
        service.enqueue_market("one", trade(100), 1)
        service.enqueue_market("two", trade(100), 1)
        service.enqueue_market("one", trade(100, instrument="rejected"), 1)
        service.flush()
        assert set(scout._prices) == {("one", "A"), ("two", "A")}
        service.invalidate_provider("one", "fixture disconnect", 1)
        service.flush()
        assert set(scout._prices) == {("two", "A")}
    finally:
        service.stop()
