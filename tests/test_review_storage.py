import json

import pytest

from src.runtime.state_manager import StateManager


def _snapshot(cash):
    return {"schema_version": 1, "ledger": {"cash": {"USD": cash}}, "engine": {}}


def _legacy_account(manager):
    with manager._get_connection() as conn:
        # Deliberately noncanonical JSON represents databases produced by
        # older code; retries compare semantic JSON rather than whitespace.
        conn.execute("INSERT INTO cross_asset_events VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)",
                     ("paper", "old-fill", "fill", 100, '{"amount": 5, "source": "old"}'))
        conn.execute("INSERT INTO cross_asset_snapshots(account_id,timestamp,snapshot_json) VALUES(?,?,?)",
                     ("paper", 100, json.dumps(_snapshot(105))))


def test_legacy_event_retry_cannot_apply_a_second_financial_transition(tmp_path):
    database = tmp_path / "legacy.db"
    manager = StateManager(str(database))
    _legacy_account(manager)
    assert manager.commit_cross_asset_event(
        "paper", "old-fill", "fill", 100, {"source": "old", "amount": 5},
        _snapshot(110), 200) is False
    assert manager.get_account_revision("paper") == 0
    recovered = StateManager(str(database)).load_cross_asset_snapshot("paper")
    assert recovered["snapshot"]["ledger"]["cash"]["USD"] == 105


@pytest.mark.parametrize("event_type,timestamp,payload", [
    ("cash", 100, {"amount": 5, "source": "old"}),
    ("fill", 101, {"amount": 5, "source": "old"}),
    ("fill", 100, {"amount": 6, "source": "old"}),
])
def test_conflicting_legacy_event_reuse_fails_without_advancing_account(
        tmp_path, event_type, timestamp, payload):
    manager = StateManager(str(tmp_path / "legacy.db"))
    _legacy_account(manager)
    with pytest.raises(ValueError, match="conflicts"):
        manager.commit_cross_asset_event(
            "paper", "old-fill", event_type, timestamp, payload, _snapshot(110), 200)
    assert manager.get_account_revision("paper") == 0
    assert manager.load_cross_asset_snapshot("paper")["snapshot"]["ledger"]["cash"]["USD"] == 105


def test_new_journal_history_keeps_older_events_visible_once_each(tmp_path):
    manager = StateManager(str(tmp_path / "legacy.db"))
    _legacy_account(manager)
    manager.append_cross_asset_event("paper", "old-backdated", "cash", 50, {"amount": 1})
    manager.commit_cross_asset_event("paper", "new-fill", "fill", 10, {"amount": 2}, _snapshot(107), 200)
    history = manager.get_cross_asset_events("paper")
    assert [item["event_id"] for item in history] == ["new-fill", "old-backdated", "old-fill"]
    assert history[0]["account_seq"] == 1
    assert history[1]["account_seq"] is None
    assert len({item["event_id"] for item in history}) == 3


def _policy_snapshot(cash=100, epoch=1, fee=0.001):
    state = _snapshot(cash)
    state["ledger"]["reporting_currency"] = "USD"
    state["ledger"]["instrument_specs"] = {"AAPL": {"contract_multiplier": 1}}
    state["engine"] = {"config_epoch": epoch, "execution_assumptions": {"AAPL": {"maker_fee": fee}}}
    return state


def test_configuration_epoch_records_policy_and_rejects_silent_overwrite(tmp_path):
    manager = StateManager(str(tmp_path / "epochs.db"))
    manager.commit_cross_asset_event("paper", "one", "cash", 1, {}, _policy_snapshot(), 1)
    with pytest.raises(ValueError, match="without advancing"):
        manager.commit_cross_asset_event("paper", "bad", "cash", 2, {}, _policy_snapshot(fee=0.002), 2)
    assert manager.get_account_revision("paper") == 1
    manager.commit_cross_asset_event("paper", "two", "cash", 2, {}, _policy_snapshot(epoch=2, fee=0.002), 2)
    with manager._get_connection() as conn:
        rows = conn.execute("SELECT epoch,config_json FROM paper_config_epochs ORDER BY epoch").fetchall()
    assert [row[0] for row in rows] == [1, 2]
    assert json.loads(rows[0][1])["execution_assumptions"]["AAPL"]["maker_fee"] == 0.001
    assert json.loads(rows[1][1])["execution_assumptions"]["AAPL"]["maker_fee"] == 0.002


def test_configuration_epoch_insert_rolls_back_with_failed_financial_commit(tmp_path):
    manager = StateManager(str(tmp_path / "epochs.db"))
    manager.commit_cross_asset_event("paper", "one", "cash", 1, {}, _policy_snapshot(), 1)
    with manager._get_connection() as conn:
        conn.execute("CREATE TRIGGER reject_next BEFORE INSERT ON paper_account_journal BEGIN SELECT RAISE(ABORT,'disk failure'); END")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        manager.commit_cross_asset_event("paper", "two", "cash", 2, {}, _policy_snapshot(epoch=2, fee=0.002), 2)
    with manager._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM paper_config_epochs").fetchone()[0] == 1
    assert manager.get_account_revision("paper") == 1


def test_event_tail_uses_committed_sequence_for_backdated_funding(tmp_path):
    manager = StateManager(str(tmp_path / "funding.db"))
    manager.commit_cross_asset_event("paper", "one", "fill", 100, {}, _snapshot(100), 100)
    manager.commit_cross_asset_event("paper", "two", "funding", 5, {"external_flow_reporting": 50}, _snapshot(150), 101)
    manager.commit_cross_asset_event("paper", "three", "fill", 102, {}, _snapshot(151), 102)
    tail = manager.get_account_events_after("paper", 1, event_type="funding")
    assert len(tail) == 1
    assert tail[0]["account_seq"] == 2
    assert tail[0]["effective_timestamp"] == 5
    assert tail[0]["processed_timestamp"] == 101
    assert tail[0]["payload"]["external_flow_reporting"] == 50


def test_corrupt_checkpoint_sequence_cannot_skip_a_committed_debit(tmp_path):
    database = tmp_path / "checksum.db"
    manager = StateManager(str(database))
    manager.commit_cross_asset_event("paper", "one", "cash", 1, {}, _snapshot(100), 1)
    manager.commit_cross_asset_event("paper", "two", "cash", 2, {}, _snapshot(98), 2)
    with manager._get_connection() as conn:
        conn.execute("UPDATE paper_checkpoints SET account_seq=2 WHERE account_seq=1")
    recovered = StateManager(str(database)).load_cross_asset_snapshot("paper")
    assert recovered["account_seq"] == 2
    assert recovered["snapshot"]["ledger"]["cash"]["USD"] == 98


@pytest.mark.parametrize("column,value", [("event_id", "corrupted-id"),
    ("config_epoch", 7), ("request_id", "wrong-request")])
def test_corrupt_journal_identity_fails_closed(tmp_path, column, value):
    database = tmp_path / "checksum.db"
    manager = StateManager(str(database))
    manager.commit_cross_asset_event("paper", "one", "cash", 1, {}, _snapshot(100), 1)
    manager.commit_cross_asset_event("paper", "two", "cash", 2, {}, _snapshot(98), 2)
    with manager._get_connection() as conn:
        conn.execute("UPDATE paper_account_journal SET " + column + "=? WHERE account_seq=2", (value,))
    with pytest.raises(RuntimeError, match="checksum"):
        StateManager(str(database)).load_cross_asset_snapshot("paper")


def test_migration_rejects_opening_changes_using_original_not_traded_cash(tmp_path):
    manager = StateManager(str(tmp_path / "migration.db"))
    state = _policy_snapshot(cash=80)
    state["ledger"].update(starting_equity=100, opening_balances={"USD": 100})
    manager.commit_cross_asset_event("paper", "one", "cash", 1, {}, state, 1)
    config = {"starting_capital": 100, "starting_balances": {"USD": 100},
              "reporting_currency": "USD", "instruments": [{"instrument_id": "AAPL"}]}
    assert manager.preview_account_migration("paper", config)["allowed"]
    changed = dict(config, starting_capital=120, starting_balances={"USD": 120})
    result = manager.preview_account_migration("paper", changed)
    assert not result["allowed"]
    assert any("funding" in error for error in result["errors"])
    changed_allocation = dict(config, starting_balances={"USD": 50, "USDT": 50})
    assert not manager.preview_account_migration("paper", changed_allocation)["allowed"]


def test_writer_scope_keeps_wal_open_without_leaking_a_database_handle(tmp_path):
    database = tmp_path / "writer.db"
    manager = StateManager(str(database))
    from pathlib import Path
    with manager.acquire_account_writer("paper"):
        manager.commit_cross_asset_event("paper", "one", "cash", 1, {}, _snapshot(100), 1)
        assert Path(str(database) + "-wal").is_file()
        manager.commit_cross_asset_event("paper", "two", "cash", 2, {}, _snapshot(98), 2)
        assert manager.get_account_revision("paper") == 2
    # Windows rejects this replacement if the keeper leaked beyond its scope.
    renamed = database.with_name("closed-writer.db")
    database.rename(renamed)
    assert renamed.is_file()
