import hashlib
import json
from pathlib import Path

from src.config.cross_asset_config import load_cross_asset_config
from src.runtime.account_admin import migrate, preview
from src.runtime.cross_asset_paper_engine import CrossAssetPaperEngine
from src.runtime.state_manager import StateManager


CONFIG = Path(__file__).parents[1] / "config" / "cross_asset_paper.example.json"


def test_preview_uses_a_copy_and_explicit_migration_preserves_account(tmp_path):
    database = tmp_path / "account.db"
    original = load_cross_asset_config(CONFIG)
    manager = StateManager(str(database))
    engine = CrossAssetPaperEngine(original, state_manager=manager)
    engine.stop_strategy()
    cash = engine.ledger.export_state()["cash"]
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["instruments"][0]["fees"]["maker_rate"] *= 2
    config_path = tmp_path / "changed.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert preview(database, config_path)["allowed"]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    result = migrate(database, config_path)
    assert result["applied"] and Path(result["backup"]).is_file()
    restored = StateManager(str(database)).load_cross_asset_snapshot(original.account_id)
    assert restored["snapshot"]["ledger"]["cash"] == cash
    assert restored["snapshot"]["engine"]["config_epoch"] >= 2


def test_daily_backup_does_not_prune_unrelated_or_migration_files(tmp_path):
    manager = StateManager(str(tmp_path / "paper.db"))
    directory = tmp_path / "backups"
    directory.mkdir()
    protected = directory / "migration-old.db"
    protected.write_bytes(b"preserve")
    first = Path(manager.run_daily_backup(directory, now=1700000000))
    latest = Path(manager.run_daily_backup(directory, now=1700000000 + 10 * 86400))
    assert latest.exists() and not first.exists()
    assert protected.read_bytes() == b"preserve"
