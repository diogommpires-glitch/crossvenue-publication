"""Crash-boundary, backup provenance and replay-version regressions."""

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.market.events import MarketDataEvent, MarketEventKind
from src.runtime import recording
from src.runtime.application_service import PaperApplicationService
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.recording import SessionRecorder, iter_records
from src.runtime.replay import ReplayRunner, VirtualClock
from src.runtime.state_manager import StateManager
from src.utils.types import Side
from tests.test_command_replay_regressions import command

CONFIG = Path(__file__).parents[1] / "config/cross_asset_paper.example.json"
NOW = 1792497600.0


def service_for(tmp_path):
    config = load_cross_asset_config(CONFIG)
    clock = VirtualClock(NOW)
    manager = StateManager(str(tmp_path / "account.db"))
    engine = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    recorder = SessionRecorder(tmp_path / "recording", config=json.loads(CONFIG.read_text()))
    recorder.set_initial_state(engine._checkpoint_payload())
    service = PaperApplicationService(engine, manager, recorder=recorder, clock=clock)
    return config, clock, manager, engine, service


def test_execution_semantics_has_an_independent_version(tmp_path):
    assert getattr(recording, "EXECUTION_SEMANTICS_VERSION", None) == 2
    with SessionRecorder(tmp_path / "recording", config={}) as recorder:
        assert recorder.manifest["execution_semantics_version"] == 2


def test_writer_clock_binding_survives_ledger_clone(tmp_path):
    _config, _clock, _manager, engine, service = service_for(tmp_path)
    service._writer_time += 3
    assert engine.ledger.clock() == engine.clock() == NOW + 3
    assert engine.ledger.clone().clock() == NOW + 3


def test_writer_clock_protects_financial_order_at_mark_expiry_boundary(tmp_path):
    config = load_cross_asset_config(CONFIG)
    clock = VirtualClock(NOW)
    engine = CrossAssetPaperEngine(config, clock=clock)
    engine.ledger.record_fill(
        "initial", "CME:ESZ6", Side.BUY, 1, 6000, fee=0, fee_currency="USD", timestamp=NOW - 29.999
    )
    service = PaperApplicationService(engine, clock=clock)
    for kind, data in (
        (MarketEventKind.BOOK_SNAPSHOT, {"bids": ((199.99, 100),), "asks": ((200.01, 100),)}),
        (MarketEventKind.STATUS, {"status": "open"}),
        (MarketEventKind.TRADE, {"price": 200, "quantity": 10, "side": Side.BUY}),
    ):
        assert engine.on_market_event(
            MarketDataEvent("NASDAQ:AAPL", "fixture", kind, NOW, NOW, **data)
        )
    clock.advance_to(NOW + 0.01)
    outcome = service._execute_command(
        {
            "protocol_version": 1,
            "request_id": "boundary",
            "account_id": config.account_id,
            "expected_revision": 0,
            "command": "submit_paper_order",
            "payload": {
                "instrument_id": "NASDAQ:AAPL",
                "side": "BUY",
                "quantity": 1,
                "price": 199.99,
            },
        }
    )
    assert outcome["success"] and outcome["data"]["accepted"]


@pytest.mark.parametrize("failure", ["ack", "checkpoint"])
def test_committed_funding_keeps_exact_outcome_and_replays_after_late_failure(tmp_path, failure):
    config, clock, manager, engine, service = service_for(tmp_path)
    request = command(
        manager,
        config,
        "fund",
        "record_funding",
        {"funding_id": "funding", "currency": "USD", "amount": 100, "source": "fixture"},
    )
    with manager._get_connection() as conn:
        if failure == "ack":
            conn.execute("""CREATE TRIGGER fail_ack BEFORE UPDATE OF result_json ON paper_command_results
                BEGIN SELECT RAISE(ABORT, 'ack write failed'); END""")
        else:
            conn.execute("""CREATE TRIGGER fail_checkpoint BEFORE INSERT ON paper_account_journal
                WHEN NEW.event_type='account_checkpoint'
                BEGIN SELECT RAISE(ABORT, 'checkpoint write failed'); END""")
    outcome = service._execute_command(request)
    assert outcome["success"] and outcome["code"] == "ok" and outcome["data"] is True
    assert outcome["persistence_error"]["stage"] == failure
    assert engine.ledger.cash_by_currency["USD"] == 40100
    assert service._execute_command(request) == outcome
    conflicting = dict(request, payload=dict(request["payload"], amount=200))
    with pytest.raises(ValueError, match="reused"):
        service._execute_command(conflicting)
    recovered = StateManager(str(tmp_path / "account.db")).get_command_result(
        config.account_id, "fund"
    )
    assert recovered == outcome
    service.recorder.close(complete=True)
    commands = [
        row["payload"] for row in iter_records(tmp_path / "recording") if row["kind"] == "command"
    ]
    assert commands[0]["outcome"] == outcome and commands[1]["duplicate"]
    replay = ReplayRunner(config).run(tmp_path / "recording")
    assert replay["financial"]["cash"]["USD"] == 40100
    assert replay["command_outcomes"][:2] == [outcome, outcome]
    assert manager.get_account_revision(config.account_id) == outcome["revision"]
    assert clock.value == NOW


def test_partial_import_rejection_is_durable_even_when_ack_fails(tmp_path):
    config, clock, manager, engine, service = service_for(tmp_path)
    action = {
        "action_id": "partial",
        "action_type": "financing",
        "instrument_id": "OANDA:EUR_USD",
        "source": "fixture",
        "effective_ts": clock.value + 100,
        "payload": {"amount": -2, "currency": "USD"},
    }
    request = command(
        manager, config, "import", "import_account_actions", {"actions": [action, {}]}
    )
    with manager._get_connection() as conn:
        conn.execute("""CREATE TRIGGER fail_ack BEFORE UPDATE OF result_json ON paper_command_results
            BEGIN SELECT RAISE(ABORT, 'ack write failed'); END""")
    outcome = service._execute_command(request)
    assert not outcome["success"] and outcome["code"] == "rejected"
    assert [row["action_id"] for row in engine.get_account_actions()] == ["partial"]
    recovered = StateManager(str(tmp_path / "account.db")).get_command_result(
        config.account_id, "import"
    )
    assert recovered == outcome
    service.recorder.close(complete=True)
    instances = []

    def factory(configuration, *, clock, id_factory, **kwargs):
        instance = CrossAssetPaperEngine(
            configuration, clock=clock, id_factory=id_factory, **kwargs
        )
        instances.append(instance)
        return instance

    ReplayRunner(config, engine_factory=factory).run(tmp_path / "recording")
    assert [row["action_id"] for row in instances[-1].get_account_actions()] == ["partial"]


def test_post_dispatch_revision_read_cannot_erase_committed_funding(tmp_path, monkeypatch):
    config, _clock, manager, engine, service = service_for(tmp_path)
    request = command(
        manager,
        config,
        "revision-fault",
        "record_funding",
        {"funding_id": "funding", "currency": "USD", "amount": 100, "source": "fixture"},
    )
    original = manager.get_account_revision
    reads = []

    def fail_post_dispatch(account_id):
        reads.append(account_id)
        if len(reads) > 1:
            raise sqlite3.OperationalError("post-dispatch revision read failed")
        return original(account_id)

    monkeypatch.setattr(manager, "get_account_revision", fail_post_dispatch)
    outcome = service._execute_command(request)
    assert outcome["success"] and outcome["data"] is True
    assert outcome["revision"] == original(config.account_id)
    assert engine.ledger.cash_by_currency["USD"] == 40100
    service.recorder.close(complete=not service._incomplete)
    replay = ReplayRunner(config).run(tmp_path / "recording")
    assert replay["financial"]["cash"]["USD"] == 40100


def test_partial_rejection_replay_verifies_actual_ordered_committed_effects(tmp_path):
    config, clock, manager, _engine, service = service_for(tmp_path)
    action = {
        "action_id": "before-reject",
        "action_type": "financing",
        "instrument_id": "OANDA:EUR_USD",
        "source": "fixture",
        "effective_ts": clock.value + 100,
        "payload": {"amount": -2, "currency": "USD"},
    }
    outcome = service._execute_command(
        command(
            manager, config, "partial-stage", "import_account_actions", {"actions": [action, {}]}
        )
    )
    service.recorder.close(complete=True)

    class RejectBeforeEffect(CrossAssetPaperEngine):
        def schedule_account_action(self, payload):
            raise ValueError(outcome["message"])

    ReplayRunner(config).run(tmp_path / "recording")
    with pytest.raises(ValueError, match="committed effects"):
        ReplayRunner(config, engine_factory=RejectBeforeEffect).run(tmp_path / "recording")


def test_crash_after_partial_effect_does_not_invent_dispatch_success(tmp_path):
    config, clock, manager, engine, service = service_for(tmp_path)
    request = command(
        manager,
        config,
        "interrupted",
        "record_funding",
        {
            "funding_id": "interrupted-funding",
            "currency": "USD",
            "amount": 100,
            "source": "fixture",
        },
    )
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with manager.command_scope(config.account_id, request["request_id"], request_hash):
        engine.record_funding(**request["payload"])
    recovered = StateManager(str(tmp_path / "account.db")).get_command_result(
        config.account_id, request["request_id"]
    )
    assert recovered["code"] == "reconciliation_required" and not recovered["success"]
    assert recovered["effects_committed"]
    assert service._execute_command(request) == recovered
    assert engine.ledger.cash_by_currency["USD"] == 40100
    assert clock.value == NOW


def test_restarted_unknown_outcome_reconciliation_halt_replays_rejected_ack(tmp_path):
    config, clock, manager, engine, service = service_for(tmp_path)
    request = command(
        manager,
        config,
        "unknown",
        "record_funding",
        {"funding_id": "unknown-funding", "currency": "USD", "amount": 100, "source": "fixture"},
    )
    fingerprint = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with manager.command_scope(config.account_id, "unknown", fingerprint):
        engine.record_funding(**request["payload"])
    service.recorder.close(complete=False)

    store = StateManager(str(tmp_path / "account.db"))
    restored = CrossAssetPaperEngine(config, clock=clock, state_manager=store)
    directory = tmp_path / "restart-recording"
    recorder = SessionRecorder(directory, config=json.loads(CONFIG.read_text()))
    recorder.set_initial_state(restored._checkpoint_payload())
    restarted = PaperApplicationService(restored, store, clock=clock, recorder=recorder)
    assert restored.circuit_breaker_reason == "command_outcome_reconciliation_required"
    ack = command(store, config, "ack", "acknowledge_risk_circuit_breaker", {})
    rejected = restarted._execute_command(ack)
    assert not rejected["success"] and "outcome reconciliation" in rejected["message"]
    recorder.close(complete=True)
    instances = []

    def factory(configuration, *, clock, id_factory, **kwargs):
        instance = CrossAssetPaperEngine(
            configuration, clock=clock, id_factory=id_factory, **kwargs
        )
        instances.append(instance)
        return instance

    result = ReplayRunner(config, engine_factory=factory).run(directory)
    assert result["financial"]["cash"]["USD"] == 40100
    assert result["command_outcomes"] == [rejected]
    assert result["committed_effects"] == []
    assert instances[-1].circuit_breaker_active
    assert instances[-1].circuit_breaker_reason == "command_outcome_reconciliation_required"


def test_unrecoverable_outcome_storage_does_not_certify_recording(tmp_path):
    config, _clock, manager, engine, service = service_for(tmp_path)
    with manager._get_connection() as conn:
        conn.execute("""CREATE TRIGGER fail_outcome BEFORE INSERT ON paper_command_outcomes
            BEGIN SELECT RAISE(ABORT, 'outcome write failed'); END""")
        conn.execute("""CREATE TRIGGER fail_ack BEFORE UPDATE OF result_json ON paper_command_results
            BEGIN SELECT RAISE(ABORT, 'ack write failed'); END""")
    request = command(
        manager,
        config,
        "unacknowledged",
        "record_funding",
        {"funding_id": "funding", "currency": "USD", "amount": 100, "source": "fixture"},
    )
    outcome = service._execute_command(request)
    assert outcome["success"] and service._incomplete
    assert service._execute_command(request) == outcome
    assert engine.ledger.cash_by_currency["USD"] == 40100
    service.recorder.close(complete=not service._incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        list(iter_records(tmp_path / "recording"))


@pytest.mark.parametrize("fail_outcome_storage", [False, True])
def test_final_checkpoint_halt_survives_restart_and_unrepaired_acknowledgement(
    tmp_path, fail_outcome_storage
):
    config, clock, manager, _engine, service = service_for(tmp_path)
    with manager._get_connection() as conn:
        conn.execute("""CREATE TRIGGER fail_checkpoint BEFORE INSERT ON paper_account_journal
            WHEN NEW.event_type='account_checkpoint'
            BEGIN SELECT RAISE(ABORT, 'checkpoint write failed'); END""")
        if fail_outcome_storage:
            conn.execute("""CREATE TRIGGER fail_outcome BEFORE INSERT ON paper_command_outcomes
                BEGIN SELECT RAISE(ABORT, 'outcome write failed'); END""")
    request = command(
        manager,
        config,
        "halted-funding",
        "record_funding",
        {"funding_id": "funding", "currency": "USD", "amount": 100, "source": "fixture"},
    )
    assert service._execute_command(request)["success"]
    restored_store = StateManager(str(tmp_path / "account.db"))
    restored = CrossAssetPaperEngine(config, clock=clock, state_manager=restored_store)
    restarted = PaperApplicationService(restored, restored_store, clock=clock)
    assert restored.circuit_breaker_active and not restored.is_running
    assert restored.circuit_breaker_reason == "paper_history_persistence_failed"
    assert restored.ledger.cash_by_currency["USD"] == 40100
    with restored_store._get_connection() as conn:
        conn.execute("DROP TRIGGER fail_checkpoint")
    acknowledgement = command(restored_store, config, "ack", "acknowledge_risk_circuit_breaker", {})
    rejected = restarted._execute_command(acknowledgement)
    assert not rejected["success"] and "must be repaired" in rejected["message"]
    assert (
        restored_store.get_command_persistence_halt(config.account_id)
        == "paper_history_persistence_failed"
    )


def test_recorder_failure_preserves_committed_outcome_without_certifying_capture(
    tmp_path, monkeypatch
):
    config, _clock, manager, engine, service = service_for(tmp_path)

    def fail_record(*args, **kwargs):
        raise OSError("recording disk full")

    monkeypatch.setattr(service.recorder, "append", fail_record)
    request = command(
        manager,
        config,
        "capture-failure",
        "record_funding",
        {"funding_id": "funding", "currency": "USD", "amount": 100, "source": "fixture"},
    )
    outcome = service._execute_command(request)
    assert outcome["success"] and service._incomplete and service.recorder is None
    assert service._execute_command(request) == outcome
    assert engine.ledger.cash_by_currency["USD"] == 40100
    with pytest.raises(ValueError, match="incomplete"):
        list(iter_records(tmp_path / "recording"))


def test_daily_backup_repairs_unrelated_schema_and_separates_same_stems(tmp_path):
    one = StateManager(str(tmp_path / "one" / "same.db"))
    two = StateManager(str(tmp_path / "two" / "same.db"))
    backups = tmp_path / "backups"
    first = Path(one.run_daily_backup(backups, now=NOW))
    second = Path(two.run_daily_backup(backups, now=NOW))
    assert first != second
    first.unlink()
    with closing(sqlite3.connect(str(first))) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")
    assert Path(one.run_daily_backup(backups, now=NOW)) == first
    with closing(sqlite3.connect(str(first))) as conn:
        assert conn.execute("SELECT COUNT(*) FROM paper_schema_versions").fetchone()[0] >= 2
    second_bytes = second.read_bytes()
    first.write_bytes(second_bytes)
    one.run_daily_backup(backups, now=NOW)
    with closing(sqlite3.connect(str(first))) as conn, one._get_connection() as source:
        assert (
            conn.execute("SELECT value FROM paper_store_metadata WHERE key='source_id'").fetchone()[
                0
            ]
            == source.execute(
                "SELECT value FROM paper_store_metadata WHERE key='source_id'"
            ).fetchone()[0]
        )
    latest = Path(one.run_daily_backup(backups, now=NOW + 10 * 86400))
    assert latest.exists() and not first.exists()
    assert second.read_bytes() == second_bytes
    old = backups / first.name
    old.write_bytes(b"expired")
    one.run_daily_backup(backups, now=NOW + 10 * 86400)
    assert not old.exists()


def test_backup_copy_failure_preserves_previous_file(tmp_path, monkeypatch):
    manager = StateManager(str(tmp_path / "account.db"))
    destination = Path(manager.run_daily_backup(tmp_path / "backups", now=NOW))
    destination.write_bytes(b"damaged but retained until replacement succeeds")

    def fail_copy(path):
        Path(path).write_bytes(b"incomplete")
        raise OSError("copy interrupted")

    monkeypatch.setattr(manager, "backup_database", fail_copy)
    with pytest.raises(OSError, match="copy interrupted"):
        manager.run_daily_backup(tmp_path / "backups", now=NOW)
    assert destination.read_bytes() == b"damaged but retained until replacement succeeds"
    assert list(destination.parent.iterdir()) == [destination]


def test_daily_backup_replaces_hardlink_to_live_database_with_independent_copy(tmp_path):
    manager = StateManager(str(tmp_path / "account.db"))
    destination = Path(manager.run_daily_backup(tmp_path / "backups", now=NOW))
    destination.unlink()
    os.link(manager.db_path, destination)
    assert destination.samefile(manager.db_path)
    assert Path(manager.run_daily_backup(tmp_path / "backups", now=NOW)) == destination
    assert not destination.samefile(manager.db_path)
    with closing(sqlite3.connect(str(destination))) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_legacy_schema2_replay_requires_explicit_semantics_override(tmp_path):
    config = load_cross_asset_config(CONFIG)
    directory = tmp_path / "recording"
    with SessionRecorder(directory, config=json.loads(CONFIG.read_text())) as recorder:
        recorder.append("timer", {}, processing_ts=NOW)
    manifest = directory / "manifest.json"
    raw = json.loads(manifest.read_text())
    raw.pop("execution_semantics_version", None)
    manifest.write_text(json.dumps(raw))
    assert len(list(iter_records(directory))) == 1
    with pytest.raises(ValueError, match="execution semantics"):
        ReplayRunner(config).run(directory)
    result = ReplayRunner(config, research_assumption_override=True).run(directory)
    assert result["execution_semantics_version"] == 2
    assert result["recorded_execution_semantics_version"] == 1
    assert result["execution_semantics_overridden"] and result["assumptions_overridden"]
    assert result["replay_mode"] == "corrected_re_evaluation"


def test_legacy_initial_checkpoint_requires_explicit_corrected_re_evaluation(tmp_path):
    config = load_cross_asset_config(CONFIG)
    engine = CrossAssetPaperEngine(config, clock=VirtualClock(NOW))
    state = engine._checkpoint_payload()
    state["engine"]["execution_assumptions"]["_portfolio"].pop("execution_semantics_version")
    directory = tmp_path / "recording"
    with SessionRecorder(directory, config=json.loads(CONFIG.read_text())) as recorder:
        recorder.set_initial_state(state)
        recorder.append("timer", {}, processing_ts=NOW)
    with pytest.raises(ValueError, match="execution semantics|fill-model assumptions"):
        ReplayRunner(config).run(directory)
    corrected = ReplayRunner(config, research_assumption_override=True).run(directory)
    assert corrected["financial"]["cash"] == config.starting_balances
    assert corrected["execution_semantics_overridden"] and corrected["assumptions_overridden"]
    assert corrected["recorded_initial_execution_semantics_version"] == 1


def test_unhealthy_provider_repeats_preserve_research_throttle(tmp_path):
    _config, clock, manager, engine, _unused = service_for(tmp_path)

    class Research:
        calls = 0

        def summary(self, now):
            self.calls += 1
            return {"advisory_only": True, "processing_ts": now}

        def provider_status(self, *args):
            pass

    research = Research()
    service = PaperApplicationService(engine, manager, clock=clock, research=research)
    service.start()
    try:
        first = research.calls
        for _ in range(5):
            clock.advance_to(clock.value + 0.1)
            service.enqueue_provider_status(
                {"provider_id": "fixture", "generation": 1, "state": "retrying"}
            )
            service.flush()
        assert research.calls == first
    finally:
        service.stop()
