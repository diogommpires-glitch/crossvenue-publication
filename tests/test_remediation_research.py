"""Regressions for provider discontinuities and honest advisory statistics."""

import hashlib
import json
import threading

import numpy as np
import pytest

from src.connectors.market_data import ProviderState, ProviderStatus
from src.connectors.oanda_provider import OandaPricingProvider
from src.runtime import causal_analysis
from src.runtime.causal_analysis import CausalVenueScout, analyze_recording
from src.runtime.provider_supervisor import ProviderSupervisor
from src.runtime.recording import SessionRecorder, canonical_json
from src.runtime.research_evaluation import _metrics
from tests.test_causal_analysis import feed, trade
from tests.test_desktop_integration import make_bridge, make_order
from tests.test_read_only_market_data_providers import fx_spec


@pytest.mark.parametrize("tradeable", [False, "false"])
def test_oanda_halt_precedes_crossing_quote(tradeable):
    provider = OandaPricingProvider("test", {"EUR_USD": fx_spec()}, access_token="test")
    events = provider.normalize_message(
        {
            "type": "PRICE",
            "instrument": "EUR_USD",
            "time": "2026-10-25T10:00:00Z",
            "bids": [{"price": "1.1", "liquidity": "1000"}],
            "asks": [{"price": "1.1001", "liquidity": "1000"}],
            "tradeable": tradeable,
        }
    )
    assert events[0].status == "non_tradeable"
    assert events[1].bids


def test_socket_loss_fences_old_callbacks_before_retry():
    ready, invalidated = threading.Event(), threading.Event()

    class Provider:
        status = ProviderStatus("fixture", ProviderState.CONNECTED)

        def __init__(self):
            self.flags = {"depth_connected": True, "trades_connected": True}

        def start(self, handler):
            self.handler = handler
            handler("initial")
            ready.set()

        def stop(self):
            pass

        def status_snapshot(self):
            return dict(self.flags)

    provider = Provider()
    events, invalidations = [], []

    def invalidate(*args):
        invalidations.append(args)
        invalidated.set()

    supervisor = ProviderSupervisor(
        "fixture",
        ("A",),
        lambda: provider,
        lambda *args: events.append(args),
        lambda *args: None,
        invalidate,
        poll_seconds=0.01,
    )
    supervisor.start()
    try:
        assert ready.wait(1)
        generation = supervisor.status["generation"]
        old_handler = provider.handler
        provider.flags["trades_connected"] = False
        old_handler("depth-after-trade-loss")
        assert invalidated.wait(1)
        old_handler("old")
        assert all(entry[1] == "initial" for entry in events)
        assert [entry[1] for entry in events if entry[2] == generation] == ["initial"]
        assert supervisor.status["generation"] > generation
        assert invalidations[0][2] == generation
        # The writer invalidates the revoked generation, while its immediate
        # successor is allowed through; reserving the successor must not revoke it.
        from src.runtime.application_service import PaperApplicationService
        from tests.test_application_runtime import MinimalEngine

        engine = MinimalEngine()
        service = PaperApplicationService(engine)
        service.start()
        try:
            service.invalidate_provider(*invalidations[0])
            service.enqueue_market("fixture", "revoked", generation)
            service.enqueue_market("fixture", "replacement", generation + 1)
            service.flush()
            assert engine.events == [("replacement", generation + 1)]
        finally:
            service.stop()
    finally:
        supervisor.stop()


@pytest.mark.parametrize("stamp_offset", [-0.5, 0.0, None])
def test_legacy_bridge_rejects_unproven_post_acceptance_print(stamp_offset):
    bridge = make_bridge()
    bridge.on_orderbook_update([(99, 1)], [(101, 1)])
    bridge._process_strategy_order(make_order("causal", qty=0.1, price=100), 100)
    accepted = bridge.order_timestamps["causal"]
    bridge.last_trade_time = accepted
    bridge.is_running = True
    try:
        stamp = accepted + stamp_offset if stamp_offset is not None else None
        bridge._check_paper_fills(99, "sell", 0.2, stamp)
        assert bridge.total_fills == 0
    finally:
        bridge.is_running = False


def test_legacy_maker_uses_resting_limit_price():
    bridge = make_bridge()
    bridge.on_orderbook_update([(99, 1)], [(101, 1)])
    bridge._process_strategy_order(make_order("limit", qty=0.1, price=100), 100)
    accepted = bridge.order_timestamps["limit"]
    bridge.last_trade_time = accepted
    bridge.is_running = True
    try:
        bridge._check_paper_fills(99, "sell", 0.2, accepted + 0.001)
        assert bridge.trades[-1]["price"] == 100
    finally:
        bridge.is_running = False


def test_cashflow_inside_invalid_mark_gap_has_exact_net_but_unavailable_risk():
    result = _metrics(
        {
            "fills": [],
            "observations": [
                {"timestamp": 0, "equity": 100, "valuation_valid": True},
                {"timestamp": 1, "equity": None, "valuation_valid": False, "external_flow": 100},
                {"timestamp": 2, "equity": 150, "valuation_valid": True},
            ],
        },
        0,
        2,
    )
    assert result["net_pnl"] == -50
    assert result["valid"] is False
    assert result["max_drawdown"] is None
    assert result["score"] is None
    assert result["return"] is None
    assert result["reason"] == "valuation_gap"


@pytest.mark.parametrize("kind", ["timer", "provider"])
def test_nonmarket_record_tail_advances_saved_continuation_clock(tmp_path, kind):
    directory = tmp_path / "recording"
    with SessionRecorder(directory, config={}) as recorder:
        recorder.record_market(trade(1000), processing_ts=1000)
        recorder.append(kind, {"state": "healthy"}, 1005, provider_id="fixture")
    state = tmp_path / "research.json"
    assert analyze_recording(directory, state_out=state)["processing_ts"] == 1005
    restored = CausalVenueScout.load(state)
    assert restored._clock == 1005
    with pytest.raises(ValueError, match="backwards"):
        restored.observe(trade(1001), 1001)


def test_timer_only_and_empty_continuation_preserve_clock_without_market_input(tmp_path):
    directory = tmp_path / "timer-only"
    with SessionRecorder(directory, config={}) as recorder:
        recorder.append("timer", {}, 1005)
    state = tmp_path / "timer-state.json"
    first = analyze_recording(directory, state_out=state)
    assert first["processing_ts"] == 1005
    assert first["streams"] == []
    empty = tmp_path / "empty"
    with SessionRecorder(empty, config={}):
        pass
    continued = tmp_path / "continued.json"
    assert analyze_recording(empty, state_in=state, state_out=continued) == first
    assert CausalVenueScout.load(continued)._clock == 1005


def test_open_bucket_input_cannot_refresh_closed_return():
    scout = CausalVenueScout(min_samples=5)
    feed(scout, 15)
    for identity in ("A", "B"):
        scout.observe(trade(44.25, 101, identity), 44.25)
    result = scout.summary(44.5)
    row = result["streams"][0]
    assert row["input_age_seconds"] == 0.25
    assert row["closed_return_age_seconds"] == 29.5
    assert row["stale"] is True
    assert not result["discoveries"]


def test_fresh_streams_do_not_promote_stale_aligned_pair():
    scout = CausalVenueScout(min_samples=5)
    feed(scout, 15)
    for stamp, identity in [(40, "A"), (41, "A"), (44, "B"), (45, "B")]:
        scout.observe(trade(stamp, 100 + stamp, identity), stamp)
    result = scout.summary(46)
    assert all(not row["stale"] for row in result["streams"])
    assert result["edges"]
    assert all(edge["paired_stale"] and not edge["discovery_eligible"] for edge in result["edges"])
    assert result["discoveries"] == []


def test_short_descriptive_correlations_are_not_discoveries():
    scout = CausalVenueScout(min_samples=5)
    feed(scout, 15)
    result = scout.summary(15)
    assert result["edges"]
    assert not any(edge["discovery_eligible"] for edge in result["edges"])
    assert result["discoveries"] == []
    assert result["discovery_method"]["family_tests"] == 2
    assert result["discovery_method"]["calibrated"] is False


def test_discovery_tail_and_exact_by_step_up():
    assert causal_analysis._pearson_p_value(0.5, 30) == pytest.approx(
        0.003568523308817683, rel=1e-10
    )
    assert causal_analysis._by_rejections([0.002, 0.003, 1.0]) == {0, 1}
    assert causal_analysis._by_rejections([1.0, 1.0]) == set()


def test_discovery_uses_effective_samples_and_reports_null_limitations():
    scout = CausalVenueScout(min_samples=5, forgetting=0.5)
    feed(scout, 100)
    result = scout.summary(100)
    assert all(edge["samples"] > 32 for edge in result["edges"])
    assert all(edge["effective_samples"] < 32 for edge in result["edges"])
    assert result["discoveries"] == []


def test_batched_pair_statistics_match_aligned_scalar_and_mask_missing_returns():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(100, 3))
    returns = {("p", str(index)): dict(enumerate(values[:, index])) for index in range(3)}
    result = causal_analysis._pair_statistics(returns, 0.98)
    a, b = values[:-1, 0], values[1:, 1]
    weights = 0.98 ** np.arange(len(a) - 1, -1, -1)
    effective = weights.sum() ** 2 / (weights @ weights)
    weights /= weights.sum()
    a, b = a - weights @ a, b - weights @ b
    correlation = (weights * a) @ b / np.sqrt((weights @ (a * a)) * (weights @ (b * b)))
    correction = sum(
        (1 - lag / len(a)) * (a[:-lag] @ a[lag:] / (a @ a)) * (b[:-lag] @ b[lag:] / (b @ b))
        for lag in range(1, 11)
    )
    effective /= max(1, 1 + 2 * correction)
    assert result["correlation"][0, 1] == pytest.approx(
        np.corrcoef(values[:-1, 0], values[1:, 1])[0, 1]
    )
    assert result["weighted_correlation"][0, 1] == pytest.approx(correlation)
    assert result["effective_samples"][0, 1] == pytest.approx(effective)
    for bucket in range(20, 45):
        del returns[("p", "1")][bucket]
    result = causal_analysis._pair_statistics(returns, 0.98)
    paired = [bucket for bucket in returns[("p", "1")] if bucket - 1 in returns[("p", "0")]]
    a = [returns[("p", "0")][bucket - 1] for bucket in paired]
    b = [returns[("p", "1")][bucket] for bucket in paired]
    assert result["samples"][0, 1] == len(paired)
    assert result["correlation"][0, 1] == pytest.approx(np.corrcoef(a, b)[0, 1])
    assert result["last_bucket"][0, 1] == max(paired)


def test_batched_constant_nonzero_returns_cannot_create_a_correlation():
    returns = {
        ("p", identity): dict.fromkeys(range(100), value)
        for identity, value in (("A", 0.1), ("B", 0.2))
    }
    result = causal_analysis._pair_statistics(returns, 0.95)
    assert not result["valid"][0, 1]


def test_prespecified_family_promotes_strong_well_sampled_lag_only_as_advisory():
    scout = CausalVenueScout(min_samples=5, forgetting=1)
    rng = np.random.default_rng(11)
    changes = rng.normal(0, 0.001, 140)
    prices = [100.0, 100.0]
    for stamp in range(130):
        for index, identity in enumerate(("A", "B")):
            prices[index] *= 1 + changes[stamp - index]
            scout.observe(trade(stamp, prices[index], identity), stamp)
    result = scout.summary(130)
    assert any(
        edge["source"][1] == "A" and edge["target"][1] == "B" for edge in result["discoveries"]
    )
    assert result["advisory_only"] is True
    assert all(edge["actionable"] is False for edge in result["discoveries"])


def test_state_three_retains_exact_input_times_and_loads_v2_as_unknown(tmp_path):
    scout = CausalVenueScout()
    scout.observe(trade(10.25), 10.25)
    path = tmp_path / "state.json"
    scout.save(path)
    envelope = json.loads(path.read_text())
    assert envelope["state"]["schema_version"] == 3
    restored = CausalVenueScout.load(path)
    assert restored.summary(10.5)["streams"][0]["input_age_seconds"] == 0.25
    state = envelope["state"]
    state["schema_version"] = 2
    del state["input_times"]
    envelope["sha256"] = hashlib.sha256(canonical_json(state).encode()).hexdigest()
    path.write_text(json.dumps(envelope))
    restored = CausalVenueScout.load(path)
    assert restored.summary(10.5)["streams"][0]["input_age_seconds"] is None
    restored.observe(trade(11), 11)
    assert restored.summary(11.5)["streams"][0]["input_age_seconds"] == 0.5
