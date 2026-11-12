import json
import sqlite3
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.runtime.application_service import PaperApplicationService
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.recording import SessionRecorder, load_manifest
from src.runtime.replay import ReplayRunner, VirtualClock, _InitialStateManager
from src.runtime.state_manager import StateManager

CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"


@pytest.mark.parametrize("amount,include_history", [(100, True), (101, True), (100, False)])
def test_restored_funding_idempotency_survives_recording(tmp_path, amount, include_history):
    config = load_cross_asset_config(CONFIG)
    clock = VirtualClock(1792497600)
    database = str(tmp_path / "account.db")
    manager = StateManager(database)
    engine = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    assert engine.record_funding("prior-deposit", "USD", 100, "fixture")
    manager = StateManager(database)
    engine = CrossAssetPaperEngine(config, clock=clock, state_manager=manager)
    assert not engine.ledger.export_state()["cash_adjustment_ids"]
    directory = tmp_path / "recording"
    recorder = SessionRecorder(directory, config=json.loads(CONFIG.read_text()))
    recorder.set_initial_state(
        engine._checkpoint_payload(),
        signature_history=(
            manager.iter_cross_asset_signatures(config.account_id) if include_history else None
        ),
    )
    service = PaperApplicationService(engine, manager, recorder=recorder, clock=clock)
    result = service._execute_command(
        {
            "protocol_version": 1,
            "account_id": config.account_id,
            "request_id": "new-request",
            "expected_revision": manager.get_account_revision(config.account_id),
            "command": "record_funding",
            "payload": {
                "funding_id": "prior-deposit",
                "currency": "USD",
                "amount": amount,
                "source": "fixture",
            },
        }
    )
    recorder.close()
    assert result["success"] is (amount == 100)
    expected_cash = engine.ledger.export_state()["cash"]
    assert ReplayRunner(config).run(directory)["financial"]["cash"] == expected_cash


@pytest.fixture
def signature_recording(tmp_path):
    manager = StateManager(str(tmp_path / "source.db"))
    signatures = {
        namespace: {"historical-id": [namespace, 1, "Bearer private-material"]}
        for namespace in manager.SIGNATURE_NAMES
    }
    manager.commit_cross_asset_event(
        "account", "event", "fixture", 1, {}, {"ledger": signatures, "engine": {}}
    )
    checkpoint = manager.load_cross_asset_snapshot("account")["snapshot"]
    directory = tmp_path / "recording"
    recorder = SessionRecorder(directory, config={})
    recorder.set_initial_state(
        checkpoint, signature_history=manager.iter_cross_asset_signatures("account")
    )
    recorder.close()
    return directory, checkpoint, signatures


def test_all_durable_namespaces_have_sanitized_read_only_recording_lookup(signature_recording):
    directory, checkpoint, signatures = signature_recording
    manifest = load_manifest(directory)
    assert manifest["schema_version"] == 2
    history = manifest["initial_signature_history"]
    assert history["records"] == len(signatures)
    assert all(not checkpoint["ledger"][name] for name in signatures)
    path = directory / history["file"]
    recorded = path.read_bytes()
    assert b"private-material" not in recorded
    manager = _InitialStateManager(checkpoint, 1, path)
    for namespace in signatures:
        assert manager.get_cross_asset_signature("account", namespace, "historical-id") == (
            namespace,
            1,
            "Bearer [redacted]",
        )
    assert manager.get_cross_asset_signature("account", "cost_ids", "missing-id") is None
    assert path.read_bytes() == recorded
    assert "source.db" not in (directory / "manifest.json").read_text()


@pytest.mark.parametrize("damage", ["missing", "bytes", "count", "schema", "path"])
def test_recording_signature_history_damage_fails_closed(signature_recording, damage):
    directory, _, _ = signature_recording
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    history = manifest["initial_signature_history"]
    path = directory / history["file"]
    if damage == "missing":
        path.unlink()
    elif damage == "bytes":
        with path.open("ab") as stream:
            stream.write(b"corruption")
    else:
        field, value = {
            "count": ("records", 99),
            "schema": ("schema_version", 2),
            "path": ("file", "../source.db"),
        }[damage]
        history[field] = value
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_manifest(directory)


def test_conflicting_signature_rows_cannot_certify_a_recording(tmp_path):
    directory = tmp_path / "recording"
    recorder = SessionRecorder(directory, config={})
    with pytest.raises(sqlite3.IntegrityError):
        recorder.set_initial_state(
            {}, signature_history=iter([("cost_ids", "same-id", [1]), ("cost_ids", "same-id", [2])])
        )
    recorder.close()
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(directory)
