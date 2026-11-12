from copy import deepcopy
import sqlite3

import pytest

from src.runtime.state_manager import StateManager


def snapshot(cash=100, signature=None):
    return {"schema_version": 1, "ledger": {
        "schema_version": 3, "cash": {"USD": cash}, "positions": {},
        "instrument_specs": {}, "fill_signatures": signature or {},
        "cost_ids": {}, "cash_adjustment_ids": {}, "corporate_action_ids": {},
    }, "engine": {"trades": [], "config_epoch": 1}}


def test_recovery_uses_sequence_even_for_backdated_accounting(tmp_path):
    manager = StateManager(str(tmp_path / "paper.db"))
    manager.commit_cross_asset_event("a", "one", "cash", 100, {}, snapshot(100), 100)
    manager.commit_cross_asset_event("a", "two", "cash", 5, {}, snapshot(98), 101)
    recovered = StateManager(str(tmp_path / "paper.db")).load_cross_asset_snapshot("a")
    assert recovered["account_seq"] == 2
    assert recovered["snapshot"]["ledger"]["cash"]["USD"] == 98
    assert manager.get_cross_asset_events("a")[0]["event_id"] == "two"
    assert not manager.commit_cross_asset_event("a", "two", "cash", 5, {}, snapshot(96), 102)
    assert manager.load_cross_asset_snapshot("a")["snapshot"]["ledger"]["cash"]["USD"] == 98


def test_idempotent_retry_does_not_advance_revision(tmp_path):
    manager = StateManager(str(tmp_path / "paper.db"))
    assert manager.commit_cross_asset_event("a", "one", "cash", 1, {}, snapshot(), 1)
    assert not manager.commit_cross_asset_event("a", "one", "cash", 1, {}, snapshot(), 10)
    assert manager.get_account_revision("a") == 1
    with pytest.raises(ValueError, match="conflicts"):
        manager.commit_cross_asset_event("a", "one", "cash", 1, {"different": True}, snapshot(), 10)


def test_public_recovery_state_cannot_mutate_cached_account_state(tmp_path):
    manager = StateManager(str(tmp_path / "paper.db"))
    manager.commit_cross_asset_event("a", "one", "cash", 1, {}, snapshot(), 1)
    exposed = manager.load_cross_asset_snapshot("a")
    exposed["snapshot"]["ledger"]["cash"]["USD"] = 0
    assert manager.load_cross_asset_snapshot("a")["snapshot"]["ledger"]["cash"]["USD"] == 100


def test_signature_history_is_durable_but_checkpoint_is_compact(tmp_path):
    path = str(tmp_path / "paper.db")
    manager = StateManager(path)
    state = snapshot(signature={"old-fill": ["BTC", "buy", 1, 100, 0, "USD", 1]})
    manager.commit_cross_asset_event("a", "one", "fill", 1, {}, state, 1)
    with sqlite3.connect(path) as conn:
        stored = conn.execute("SELECT snapshot_json FROM paper_checkpoints").fetchone()[0]
    assert "old-fill" not in stored
    restored = StateManager(path)
    assert restored.get_cross_asset_signature("a", "fill_signatures", "old-fill")[0] == "BTC"
    assert restored.load_cross_asset_snapshot("a")["snapshot"]["ledger"]["fill_signatures"] == {}


def test_corrupt_latest_checkpoint_falls_back_and_replays_every_committed_event(tmp_path):
    path = str(tmp_path / "paper.db")
    manager = StateManager(path)
    for index in range(1, 5):
        manager.commit_cross_asset_event("a", str(index), "cash", index, {}, snapshot(100 - index), index * 31)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE paper_checkpoints SET snapshot_json='corrupt' WHERE account_seq=4")
    restored = StateManager(path).load_cross_asset_snapshot("a")
    assert restored["account_seq"] == 4
    assert restored["snapshot"]["ledger"]["cash"]["USD"] == 96


def test_corrupt_journal_fails_closed_instead_of_recovering_wrong_balances(tmp_path):
    path = str(tmp_path / "paper.db")
    manager = StateManager(path)
    manager.commit_cross_asset_event("a", "one", "cash", 1, {}, snapshot(), 1)
    manager.commit_cross_asset_event("a", "two", "cash", 2, {}, snapshot(98), 2)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE paper_account_journal SET delta_json='{}' WHERE account_seq=2")
    with pytest.raises(RuntimeError, match="checksum"):
        StateManager(path).load_cross_asset_snapshot("a")


def test_failure_rolls_back_journal_signatures_and_cached_state(tmp_path):
    path = str(tmp_path / "paper.db")
    manager = StateManager(path)
    manager.commit_cross_asset_event("a", "one", "cash", 1, {}, snapshot(), 1)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TRIGGER reject_paper BEFORE INSERT ON paper_account_journal BEGIN SELECT RAISE(ABORT,'simulated disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        manager.commit_cross_asset_event("a", "two", "cash", 2, {}, snapshot(0, {"new": [1]}), 2)
    assert manager.get_account_revision("a") == 1
    assert manager.get_cross_asset_signature("a", "fill_signatures", "new") is None
    assert manager.load_cross_asset_snapshot("a")["snapshot"]["ledger"]["cash"]["USD"] == 100


def test_lost_acknowledgment_recovers_committed_command_without_reexecuting(tmp_path):
    path = str(tmp_path / "paper.db")
    manager = StateManager(path)
    with manager.command_scope("a", "request-1", "hash-1"):
        manager.commit_cross_asset_event("a", "one", "cash", 1, {}, snapshot(97), 1)
    restarted = StateManager(path)
    result = restarted.get_command_result("a", "request-1")
    assert not result["ok"] and result["recovered"] and result["revision"] == 1
    assert result["code"] == "reconciliation_required" and result["effects_committed"]
    with pytest.raises(ValueError, match="different command"):
        with restarted.command_scope("a", "request-1", "hash-2"):
            pass


def test_interrupted_command_without_commit_is_not_silently_reexecuted(tmp_path):
    path = str(tmp_path / "paper.db")
    manager = StateManager(path)
    with manager.command_scope("a", "request-1", "hash-1"):
        pass
    result = StateManager(path).get_command_result("a", "request-1")
    assert not result["ok"]
    assert result["status"] == "interrupted_before_commit"


def test_account_writer_rejects_second_process_lock_and_releases(tmp_path):
    manager = StateManager(str(tmp_path / "paper.db"))
    with manager.acquire_account_writer("a"):
        with pytest.raises(RuntimeError, match="another process"):
            with manager.acquire_account_writer("a"):
                pass
    with manager.acquire_account_writer("a"):
        pass


def test_writer_connections_reuse_only_within_their_thread_and_close_at_lease_exit(tmp_path):
    import threading
    manager = StateManager(str(tmp_path / "paper.db"))
    connections = []
    with manager.acquire_account_writer("a"):
        with manager._get_connection() as main:
            connections.append(main)
        with manager._get_connection() as again:
            assert again is main
        def worker():
            with manager._get_connection() as own:
                connections.append(own)
                assert own is not main
                own.execute("SELECT 1")
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(connections) == 2
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_backup_restore_and_observation_ranges_preserve_account(tmp_path):
    manager = StateManager(str(tmp_path / "paper.db"))
    manager.commit_cross_asset_event("a", "one", "cash", 1, {}, snapshot(95), 1)
    for stamp in range(5):
        manager.record_account_observation("a", stamp, {"equity": 95 + stamp})
    assert [row["timestamp"] for row in manager.get_account_history("a", 1, 3)] == [1, 2, 3]
    backup = manager.backup_database(tmp_path / "backup.db")
    restored_path = manager.restore_database(backup, tmp_path / "restored.db")
    assert StateManager(restored_path).load_cross_asset_snapshot("a")["snapshot"]["ledger"]["cash"]["USD"] == 95
    with pytest.raises(ValueError):
        manager.restore_database(backup, restored_path)


def test_legacy_checkpoint_and_tables_survive_new_journal_migration(tmp_path):
    path = str(tmp_path / "paper.db")
    manager = StateManager(path)
    import json
    legacy = snapshot(123, {"legacy-fill": [1, 2, 3]})
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO cross_asset_snapshots(account_id,timestamp,snapshot_json) VALUES(?,?,?)",
                     ("a", 100, json.dumps(legacy)))
        conn.execute("INSERT INTO trades(timestamp,order_id,side,quantity,price,trader_id,is_aggressor) VALUES(1,'legacy','buy',1,100,'old',0)")
    assert manager.load_cross_asset_snapshot("a")["snapshot"]["ledger"]["cash"]["USD"] == 123
    manager.commit_cross_asset_event("a", "new", "cash", 2, {}, snapshot(120), 101)
    restored = StateManager(path)
    assert restored.load_cross_asset_snapshot("a")["snapshot"]["ledger"]["cash"]["USD"] == 120
    assert restored.get_cross_asset_signature("a", "fill_signatures", "legacy-fill") == (1, 2, 3)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
