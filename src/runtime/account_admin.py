"""Offline paper-account administration. No provider is constructed here."""

import argparse
from contextlib import contextmanager, closing
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid

from ..config.cross_asset_config import load_cross_asset_config
from .account_store import AccountWriterLock
from .state_manager import StateManager


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError("unsupported configuration value")


def configuration_document(config):
    return json.loads(json.dumps(asdict(config), default=_json_default))


def _copy_read_only(source_path, target_path):
    source = Path(source_path).resolve()
    if not source.is_file():
        raise ValueError("paper database was not found")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as conn:
        with closing(sqlite3.connect(str(target_path))) as target:
            conn.backup(target)


@contextmanager
def inspection_copy(database):
    """Run migration/recovery inspection without changing the source database."""
    with tempfile.TemporaryDirectory(prefix="paper-account-preview-") as folder:
        copied = Path(folder) / "preview.db"
        _copy_read_only(database, copied)
        yield StateManager(str(copied))


def preview(database, config_path):
    config = load_cross_asset_config(config_path)
    if not Path(database).is_file():
        return {"allowed": True, "new_account": True, "errors": [],
                "account_id": config.account_id}
    with inspection_copy(database) as manager:
        result = manager.preview_account_migration(config.account_id, configuration_document(config))
    result["account_id"] = config.account_id
    return result


def migrate(database, config_path):
    """Explicit stopped-account migration with a pre-change consistent backup."""
    from .cross_asset_paper_engine import CrossAssetPaperEngine
    config = load_cross_asset_config(config_path)
    path = Path(database).resolve()
    if not path.is_file():
        raise ValueError("migration requires an existing paper database")
    with AccountWriterLock(path, config.account_id):
        result = preview(path, config_path)
        if not result["allowed"]:
            raise ValueError("migration rejected: " + "; ".join(result["errors"]))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = path.parent / "backups" / ("migration-" + stamp + "-" + uuid.uuid4().hex[:8] + ".db")
        backup.parent.mkdir(parents=True, exist_ok=True)
        _copy_read_only(path, backup)
        manager = StateManager(str(path))
        engine = CrossAssetPaperEngine(config, state_manager=manager, allow_config_migration=True)
        engine.stop_strategy()
        return dict(result, applied=True, backup=str(backup),
                    config_epoch=engine.config_epoch, revision=manager.get_account_revision(config.account_id))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline local paper-account administration")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "migrate"):
        item = sub.add_parser(name)
        item.add_argument("--database", required=True)
        item.add_argument("--config", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--database", required=True)
    backup.add_argument("--destination", required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--backup", required=True)
    restore.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preview":
            result = preview(args.database, args.config)
        elif args.command == "migrate":
            result = migrate(args.database, args.config)
        elif args.command == "backup":
            destination = Path(args.destination).resolve()
            if destination.exists():
                raise ValueError("backup destination must be a new file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_read_only(args.database, destination)
            result = {"backup": str(destination)}
        else:
            result = {"restored_database": StateManager.restore_database(args.backup, args.destination)}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({"ok": False, "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
