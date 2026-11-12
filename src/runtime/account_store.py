"""Sequence-based durable paper accounts; legacy SQLite tables are preserved.

Economic timestamps never decide recovery order. Journal state transitions and
command effects commit together, and compact checkpoints are only accelerators.
"""

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checkpoint_checksum(account_id, sequence, timestamp, encoded):
    return digest(canonical([account_id, int(sequence), float(timestamp), encoded]))


def journal_checksum(account_id, sequence, event_id, event_type, effective,
                     processed, payload_json, delta_json, epoch, request_id):
    return digest(canonical([account_id, int(sequence), event_id, event_type,
                             float(effective), float(processed), payload_json,
                             delta_json, int(epoch), request_id]))


def state_delta(before, after, path=()):
    changes = {"set": [], "delete": []}
    if before == after:
        return changes
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() - after.keys()):
            changes["delete"].append(list(path + (key,)))
        for key in sorted(after):
            if key not in before:
                changes["set"].append([list(path + (key,)), after[key]])
            else:
                child = state_delta(before[key], after[key], path + (key,))
                changes["set"].extend(child["set"])
                changes["delete"].extend(child["delete"])
    elif before != after:
        changes["set"].append([list(path), after])
    return changes


def apply_delta(state, delta):
    result = deepcopy(state)
    for path in delta["delete"]:
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        parent.pop(path[-1], None)
    for path, value in delta["set"]:
        if not path:
            result = deepcopy(value)
            continue
        parent = result
        for key in path[:-1]:
            parent = parent.setdefault(key, {})
        parent[path[-1]] = deepcopy(value)
    return result


class DurableAccountStore:
    """Mixin used by StateManager; all writes use its transaction context."""

    _get_connection: Callable[[], AbstractContextManager[sqlite3.Connection]]
    db_path: Path

    SIGNATURE_NAMES = ("fill_signatures", "cost_ids", "cash_adjustment_ids",
                       "corporate_action_ids")

    def _initialize_account_store(self):
        self._account_store_lock = threading.RLock()
        self._account_cache = {}
        self._signature_hot = {}
        self._fill_hot = {}
        self._command_local = threading.local()
        self._store_instance_id = uuid.uuid4().hex
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            for sql in (
                "CREATE TABLE IF NOT EXISTS paper_schema_versions(version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)",
                """CREATE TABLE IF NOT EXISTS paper_account_journal(
                    account_id TEXT NOT NULL, account_seq INTEGER NOT NULL,
                    event_id TEXT NOT NULL, event_type TEXT NOT NULL,
                    effective_ts REAL NOT NULL, processed_ts REAL NOT NULL,
                    payload_json TEXT NOT NULL, delta_json TEXT NOT NULL,
                    checksum TEXT NOT NULL, config_epoch INTEGER NOT NULL DEFAULT 1,
                    request_id TEXT, PRIMARY KEY(account_id,account_seq),
                    UNIQUE(account_id,event_id))""",
                """CREATE TABLE IF NOT EXISTS paper_checkpoints(
                    account_id TEXT NOT NULL, account_seq INTEGER NOT NULL,
                    timestamp REAL NOT NULL, snapshot_json TEXT NOT NULL,
                    checksum TEXT NOT NULL, PRIMARY KEY(account_id,account_seq))""",
                """CREATE TABLE IF NOT EXISTS paper_signatures(
                    account_id TEXT NOT NULL, namespace TEXT NOT NULL,
                    event_id TEXT NOT NULL, signature_json TEXT NOT NULL,
                    PRIMARY KEY(account_id,namespace,event_id))""",
                """CREATE TABLE IF NOT EXISTS paper_command_results(
                    account_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL, owner TEXT NOT NULL,
                    result_json TEXT, PRIMARY KEY(account_id,request_id))""",
                """CREATE TABLE IF NOT EXISTS paper_command_effects(
                    account_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    account_seq INTEGER NOT NULL,
                    PRIMARY KEY(account_id,request_id,account_seq))""",
                """CREATE TABLE IF NOT EXISTS paper_command_outcomes(
                    account_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL, result_json TEXT NOT NULL,
                    PRIMARY KEY(account_id,request_id))""",
                "CREATE TABLE IF NOT EXISTS paper_store_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
                """CREATE TABLE IF NOT EXISTS paper_observations(
                    account_id TEXT NOT NULL, timestamp REAL NOT NULL,
                    payload_json TEXT NOT NULL, PRIMARY KEY(account_id,timestamp))""",
                """CREATE TABLE IF NOT EXISTS paper_config_epochs(
                    account_id TEXT NOT NULL, epoch INTEGER NOT NULL,
                    effective_ts REAL NOT NULL, config_json TEXT NOT NULL,
                    checksum TEXT NOT NULL, reason TEXT NOT NULL,
                    PRIMARY KEY(account_id,epoch))""",
                """CREATE TABLE IF NOT EXISTS paper_fill_history(
                    account_id TEXT NOT NULL, fill_key TEXT NOT NULL,
                    timestamp REAL NOT NULL, payload_json TEXT NOT NULL,
                    PRIMARY KEY(account_id,fill_key))""",
                "CREATE INDEX IF NOT EXISTS idx_paper_journal_type_sequence ON paper_account_journal(account_id,event_type,account_seq)",
            ):
                conn.execute(sql)
            conn.execute("INSERT OR IGNORE INTO paper_schema_versions(version) VALUES(2)")
            conn.execute("INSERT OR IGNORE INTO paper_schema_versions(version) VALUES(3)")
            conn.execute("INSERT OR IGNORE INTO paper_schema_versions(version) VALUES(4)")
            conn.execute("INSERT OR IGNORE INTO paper_store_metadata VALUES('source_id',?)", (uuid.uuid4().hex,))

    @staticmethod
    def _validate_account(account_id):
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("account_id must be a non-empty string")

    def _compact(self, snapshot, conn, account_id):
        # Remove lifetime diagnostics before serializing account recovery state.
        shallow = dict(snapshot)
        ledger = dict(shallow.get("ledger", {}))
        engine = dict(shallow.get("engine", {}))
        shallow["ledger"], shallow["engine"] = ledger, engine
        for namespace in self.SIGNATURE_NAMES:
            signatures = ledger.pop(namespace, {})
            for key, signature in signatures.items():
                encoded = canonical(signature)
                hot_key = (account_id, namespace, key)
                if self._signature_hot.get(hot_key) == encoded:
                    continue
                old = conn.execute("SELECT signature_json FROM paper_signatures WHERE account_id=? AND namespace=? AND event_id=?",
                                   (account_id, namespace, key)).fetchone()
                if old is not None and old[0] != encoded:
                    raise ValueError("durable idempotency signature conflicts with prior event")
                conn.execute("INSERT OR IGNORE INTO paper_signatures VALUES(?,?,?,?)",
                             (account_id, namespace, key, encoded))
                self._signature_hot[hot_key] = encoded
                if len(self._signature_hot) > 16384:
                    self._signature_hot.pop(next(iter(self._signature_hot)))
        trades = engine.pop("trades", [])
        for trade in trades:
            encoded = canonical(trade)
            key = str(trade.get("fill_id") or digest(encoded))
            hot_key = (account_id, key)
            if self._fill_hot.get(hot_key) == encoded:
                continue
            conn.execute("INSERT OR IGNORE INTO paper_fill_history VALUES(?,?,?,?)",
                         (account_id, key, float(trade.get("timestamp", 0)), encoded))
            self._fill_hot[hot_key] = encoded
            if len(self._fill_hot) > 4096:
                self._fill_hot.pop(next(iter(self._fill_hot)))
        return json.loads(canonical(shallow))

    def _expand(self, state, conn, account_id):
        result = deepcopy(state)
        for name in self.SIGNATURE_NAMES:
            result.setdefault("ledger", {}).setdefault(name, {})
        rows = conn.execute("SELECT payload_json FROM paper_fill_history WHERE account_id=? ORDER BY timestamp DESC,rowid DESC LIMIT 2000",
                            (account_id,)).fetchall()
        result.setdefault("engine", {})["trades"] = [json.loads(row[0]) for row in reversed(rows)]
        return result

    def _recover(self, conn, account_id):
        head = conn.execute("SELECT COALESCE(MAX(account_seq),0) FROM paper_account_journal WHERE account_id=?",
                            (account_id,)).fetchone()[0]
        cached = self._account_cache.get(account_id)
        if cached is not None and cached[0] == head:
            # Internal recovery state is immutable by convention. Public
            # callers receive an independent copy through _expand().
            return cached[0], cached[1], cached[2]
        sequence, stamp, state = 0, 0.0, {}
        checkpoints = conn.execute("SELECT * FROM paper_checkpoints WHERE account_id=? ORDER BY account_seq DESC",
                                   (account_id,)).fetchall()
        recovered_checkpoint = False
        for checkpoint in checkpoints:
            if checkpoint_checksum(account_id, checkpoint["account_seq"],
                                   checkpoint["timestamp"], checkpoint["snapshot_json"]) != checkpoint["checksum"]:
                continue
            try:
                state = json.loads(checkpoint["snapshot_json"])
            except (ValueError, TypeError):
                continue
            sequence, stamp = checkpoint["account_seq"], checkpoint["timestamp"]
            recovered_checkpoint = True
            break
        if not recovered_checkpoint:
            legacy = conn.execute("SELECT timestamp,snapshot_json FROM cross_asset_snapshots WHERE account_id=?",
                                  (account_id,)).fetchone()
            if legacy is not None:
                stamp = float(legacy["timestamp"])
                state = self._compact(json.loads(legacy["snapshot_json"]), conn, account_id)
        rows = conn.execute("SELECT * FROM paper_account_journal WHERE account_id=? AND account_seq>? ORDER BY account_seq",
                            (account_id, sequence)).fetchall()
        for row in rows:
            if row["account_seq"] != sequence + 1:
                raise RuntimeError("paper account journal has a sequence gap")
            checksum = journal_checksum(account_id, row["account_seq"], row["event_id"],
                row["event_type"], row["effective_ts"], row["processed_ts"],
                row["payload_json"], row["delta_json"], row["config_epoch"], row["request_id"])
            if checksum != row["checksum"]:
                raise RuntimeError("paper account journal checksum failed")
            state = apply_delta(state, json.loads(row["delta_json"]))
            sequence, stamp = row["account_seq"], row["processed_ts"]
        self._account_cache[account_id] = (sequence, stamp, state)
        return sequence, stamp, state

    def _checkpoint(self, conn, account_id, sequence, timestamp, state):
        encoded = canonical(state)
        conn.execute("INSERT OR REPLACE INTO paper_checkpoints VALUES(?,?,?,?,?)",
                     (account_id, sequence, timestamp, encoded,
                      checkpoint_checksum(account_id, sequence, timestamp, encoded)))
        conn.execute("DELETE FROM paper_checkpoints WHERE account_id=? AND account_seq NOT IN (SELECT account_seq FROM paper_checkpoints WHERE account_id=? ORDER BY account_seq DESC LIMIT 3)",
                     (account_id, account_id))

    @staticmethod
    def _epoch_document(state):
        engine, ledger = state.get("engine", {}), state.get("ledger", {})
        return {"execution_assumptions": engine.get("execution_assumptions", {}),
                "instrument_specs": ledger.get("instrument_specs", {}),
                "reporting_currency": ledger.get("reporting_currency")}

    def _record_transition_epoch(self, conn, account_id, state, effective_ts, reason):
        """Bind each epoch to its configuration in the financial transaction."""
        epoch = int(state.get("engine", {}).get("config_epoch", 1))
        if epoch < 1:
            raise ValueError("configuration epoch must be positive")
        document = self._epoch_document(state)
        encoded = canonical(document)
        existing = conn.execute(
            "SELECT config_json,checksum FROM paper_config_epochs WHERE account_id=? AND epoch=?",
            (account_id, epoch)).fetchone()
        if existing is not None:
            if digest(existing[0]) != existing[1]:
                raise RuntimeError("configuration epoch checksum failed")
            if existing[0] == encoded:
                return
            old_document = json.loads(existing[0])
            # A minimal schema-bootstrap checkpoint may have no policy at all.
            # Real policy values are immutable within an already used epoch.
            if any(value not in (None, {}) for value in old_document.values()):
                raise ValueError("configuration changed without advancing its epoch")
            conn.execute(
                "UPDATE paper_config_epochs SET effective_ts=?,config_json=?,checksum=?,reason=? WHERE account_id=? AND epoch=?",
                (float(effective_ts), encoded, digest(encoded), "configuration_schema_bootstrap", account_id, epoch))
        else:
            conn.execute("INSERT INTO paper_config_epochs VALUES(?,?,?,?,?,?)",
                         (account_id, epoch, float(effective_ts), encoded, digest(encoded), reason))

    def commit_cross_asset_event(self, account_id, event_id, event_type, timestamp,
                                 payload, snapshot, snapshot_timestamp=None):
        self._validate_account(account_id)
        if not all(isinstance(value, str) and value.strip() for value in (event_id, event_type)):
            raise ValueError("event_id and event_type must be non-empty strings")
        effective = float(timestamp)
        processed = effective if snapshot_timestamp is None else float(snapshot_timestamp)
        if not math.isfinite(effective) or not math.isfinite(processed):
            raise ValueError("event timestamps must be finite")
        event_json = canonical(payload)
        with self._account_store_lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    old = conn.execute("SELECT event_type,effective_ts,payload_json FROM paper_account_journal WHERE account_id=? AND event_id=?",
                                       (account_id, event_id)).fetchone()
                    if old is not None:
                        if old[0] != event_type or old[1] != effective or old[2] != event_json:
                            raise ValueError("cross-asset event ID conflicts with previously stored data")
                        return False
                    # The pre-upgrade journal remains authoritative for its
                    # event IDs. Mirroring with INSERT OR IGNORE below cannot
                    # protect against applying an old financial event twice.
                    legacy = conn.execute(
                        "SELECT event_type,timestamp,payload_json FROM cross_asset_events WHERE account_id=? AND event_id=?",
                        (account_id, event_id)).fetchone()
                    if legacy is not None:
                        if (legacy[0] != event_type or legacy[1] != effective
                                or canonical(json.loads(legacy[2])) != event_json):
                            raise ValueError("cross-asset event ID conflicts with previously stored data")
                        return False
                    sequence, previous_stamp, previous = self._recover(conn, account_id)
                    candidate = self._compact(snapshot, conn, account_id)
                    if previous:
                        self._record_transition_epoch(conn, account_id, previous, previous_stamp,
                                                      "legacy_configuration_bootstrap")
                    self._record_transition_epoch(conn, account_id, candidate, processed,
                                                  "configuration_epoch_committed")
                    delta_json = canonical(state_delta(previous, candidate))
                    sequence += 1
                    scope = getattr(self._command_local, "scope", None)
                    request_id = scope[1] if scope and scope[0] == account_id else None
                    epoch = int(candidate.get("engine", {}).get("config_epoch", 1))
                    checksum = journal_checksum(account_id, sequence, event_id, event_type,
                        effective, processed, event_json, delta_json, epoch, request_id)
                    conn.execute("INSERT INTO paper_account_journal VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                 (account_id, sequence, event_id, event_type, effective, processed,
                                  event_json, delta_json, checksum, epoch, request_id))
                    conn.execute("INSERT OR IGNORE INTO cross_asset_events(account_id,event_id,event_type,timestamp,payload_json) VALUES(?,?,?,?,?)",
                                 (account_id, event_id, event_type, effective, event_json))
                    if request_id:
                        conn.execute("INSERT OR IGNORE INTO paper_command_effects VALUES(?,?,?)",
                                     (account_id, request_id, sequence))
                    last = conn.execute("SELECT account_seq,timestamp FROM paper_checkpoints WHERE account_id=? ORDER BY account_seq DESC LIMIT 1",
                                        (account_id,)).fetchone()
                    if last is None or sequence - last[0] >= 500 or processed - last[1] >= 30:
                        self._checkpoint(conn, account_id, sequence, processed, candidate)
                self._account_cache[account_id] = (sequence, processed, candidate)
                receipts = getattr(self._command_local, "effects", None)
                if request_id and receipts is not None:
                    receipts.append({"account_seq": sequence, "event_id": event_id,
                                     "event_type": event_type, "effective_timestamp": effective,
                                     "payload": deepcopy(payload)})
                return True
            except Exception:
                self._account_cache.pop(account_id, None)
                self._signature_hot.clear()
                self._fill_hot.clear()
                raise

    def save_cross_asset_snapshot(self, account_id, timestamp, snapshot):
        self._validate_account(account_id)
        # This is a control/mark transition too: a checkpoint alone would make
        # fallback to an earlier checkpoint unable to recover the newer state.
        event_id = "checkpoint:" + uuid.uuid4().hex
        self.commit_cross_asset_event(account_id, event_id, "account_checkpoint",
                                      timestamp, {}, snapshot, timestamp)
        with self._account_store_lock, self._get_connection() as conn:
            sequence, stamp, state = self._recover(conn, account_id)
            self._checkpoint(conn, account_id, sequence, stamp, state)

    def load_cross_asset_snapshot(self, account_id):
        self._validate_account(account_id)
        with self._account_store_lock, self._get_connection() as conn:
            sequence, stamp, state = self._recover(conn, account_id)
            if not state:
                return None
            return {"timestamp": float(stamp), "snapshot": self._expand(state, conn, account_id),
                    "account_seq": sequence}

    def get_account_revision(self, account_id):
        with self._get_connection() as conn:
            return int(conn.execute("SELECT COALESCE(MAX(account_seq),0) FROM paper_account_journal WHERE account_id=?",
                                    (account_id,)).fetchone()[0])

    def get_cross_asset_event(self, account_id, event_id):
        with self._get_connection() as conn:
            row = conn.execute("SELECT event_type,effective_ts,payload_json,account_seq FROM paper_account_journal WHERE account_id=? AND event_id=?",
                               (account_id, event_id)).fetchone()
            if row is not None:
                return {"event_id": event_id, "event_type": row[0], "timestamp": row[1],
                        "payload": json.loads(row[2]), "account_seq": row[3]}
            legacy = conn.execute("SELECT event_type,timestamp,payload_json FROM cross_asset_events WHERE account_id=? AND event_id=?",
                                  (account_id, event_id)).fetchone()
            return None if legacy is None else {"event_id": event_id, "event_type": legacy[0],
                "timestamp": legacy[1], "payload": json.loads(legacy[2])}

    def get_account_events_after(self, account_id, account_seq, event_type=None):
        """Read committed arrival order, including backdated funding events."""
        sequence = int(account_seq)
        if sequence < 0:
            raise ValueError("account sequence must be non-negative")
        conditions, arguments = ["account_id=?", "account_seq>?"], [account_id, sequence]
        if event_type is not None:
            conditions.append("event_type=?")
            arguments.append(event_type)
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT account_seq,event_id,event_type,effective_ts,processed_ts,payload_json "
                "FROM paper_account_journal WHERE " + " AND ".join(conditions) + " ORDER BY account_seq",
                arguments).fetchall()
            return [{"account_seq": row[0], "event_id": row[1], "event_type": row[2],
                     "effective_timestamp": float(row[3]), "timestamp": float(row[3]),
                     "processed_timestamp": float(row[4]), "payload": json.loads(row[5])} for row in rows]

    def get_cross_asset_signature(self, account_id, namespace, event_id):
        with self._get_connection() as conn:
            row = conn.execute("SELECT signature_json FROM paper_signatures WHERE account_id=? AND namespace=? AND event_id=?",
                               (account_id, namespace, event_id)).fetchone()
            return None if row is None else tuple(json.loads(row[0]))

    def iter_cross_asset_signatures(self, account_id: str) -> Iterator[tuple[str, str, list[Any]]]:
        """Stream recording provenance while the caller holds the account writer lease."""
        self._validate_account(account_id)
        with self._account_store_lock, self._get_connection() as conn:
            for row in conn.execute(
                    "SELECT namespace,event_id,signature_json FROM paper_signatures WHERE account_id=?",
                    (account_id,)):
                yield row[0], row[1], json.loads(row[2])

    @contextmanager
    def command_scope(self, account_id, request_id, request_hash):
        self._validate_account(account_id)
        if not request_id or not request_hash:
            raise ValueError("command request ID and fingerprint are required")
        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO paper_command_results VALUES(?,?,?,?,NULL)",
                         (account_id, request_id, request_hash, self._store_instance_id))
            row = conn.execute("SELECT request_hash FROM paper_command_results WHERE account_id=? AND request_id=?",
                               (account_id, request_id)).fetchone()
            if row[0] != request_hash:
                raise ValueError("request ID reused with a different command")
        previous = getattr(self._command_local, "scope", None)
        previous_effects = getattr(self._command_local, "effects", None)
        self._command_local.scope = (account_id, request_id, request_hash)
        self._command_local.effects = []
        try:
            yield
        finally:
            self._command_local.scope = previous
            self._command_local.effects = previous_effects

    def current_command_effects(self) -> list[dict[str, Any]] | None:
        """Return writer-local receipts populated only after each SQLite commit."""
        effects: list[dict[str, Any]] | None = getattr(self._command_local, "effects", None)
        return deepcopy(effects)

    def get_command_result(self, account_id, request_id):
        with self._get_connection() as conn:
            outcome = conn.execute(
                "SELECT result_json FROM paper_command_outcomes WHERE account_id=? AND request_id=?",
                (account_id, request_id)).fetchone()
            if outcome is not None:
                result = json.loads(outcome[0])
                intent = conn.execute("SELECT request_hash FROM paper_command_results WHERE account_id=? AND request_id=?",
                                      (account_id, request_id)).fetchone()
                if (intent is None or result.get("request_hash") != intent[0]
                        or result.get("account_id") != account_id or result.get("request_id") != request_id):
                    raise RuntimeError("durable command outcome does not match its intent")
                return result
            row = conn.execute("SELECT * FROM paper_command_results WHERE account_id=? AND request_id=?",
                               (account_id, request_id)).fetchone()
            if row is None:
                return None
            if row["result_json"] is not None:
                result = json.loads(row["result_json"])
                result.setdefault("request_hash", row["request_hash"])
                return result
            effect = conn.execute("SELECT MAX(account_seq) FROM paper_command_effects WHERE account_id=? AND request_id=?",
                                  (account_id, request_id)).fetchone()[0]
            if effect is not None:
                return {"ok": False, "success": False, "code": "reconciliation_required",
                        "protocol_version": 1, "status": "outcome_unknown_after_commit", "recovered": True,
                        "effects_committed": True,
                        "request_id": request_id, "account_id": account_id, "revision": effect,
                        "request_hash": row["request_hash"]}
            if row["owner"] != self._store_instance_id:
                return {"ok": False, "success": False, "code": "interrupted_before_commit",
                        "protocol_version": 1, "status": "interrupted_before_commit", "request_id": request_id,
                        "account_id": account_id, "request_hash": row["request_hash"],
                        "message": "The command was interrupted; review current state and submit a new request."}
            return None

    def get_command_effects(self, account_id: str, request_id: str) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            return [dict(row) for row in conn.execute(
                """SELECT j.account_seq,j.event_id,j.event_type FROM paper_command_effects e
                   JOIN paper_account_journal j ON j.account_id=e.account_id AND j.account_seq=e.account_seq
                   WHERE e.account_id=? AND e.request_id=? ORDER BY e.account_seq""", (account_id, request_id))]

    def get_command_persistence_halt(self, account_id: str) -> str | None:
        with self._get_connection() as conn:
            unresolved = conn.execute("""SELECT 1 FROM paper_command_effects e
                JOIN paper_command_results r ON r.account_id=e.account_id AND r.request_id=e.request_id
                LEFT JOIN paper_command_outcomes o ON o.account_id=e.account_id AND o.request_id=e.request_id
                WHERE e.account_id=? AND r.result_json IS NULL AND o.request_id IS NULL LIMIT 1""",
                (account_id,)).fetchone()
            if unresolved is not None:
                return "command_outcome_reconciliation_required"
            row = conn.execute("SELECT value FROM paper_store_metadata WHERE key=?",
                               ("persistence_halt:" + account_id,)).fetchone()
            if row is None:
                return None
            value = row[0]
            if not isinstance(value, str):
                raise ValueError("persisted command halt reason must be text")
            return value

    def save_command_outcome(self, account_id: str, request_id: str, request_hash: str,
                             result: dict[str, Any]) -> None:
        """Persist exact dispatch semantics independently of acknowledgement storage."""
        encoded = canonical(result)
        with self._get_connection() as conn:
            intent = conn.execute(
                "SELECT request_hash FROM paper_command_results WHERE account_id=? AND request_id=?",
                (account_id, request_id)).fetchone()
            if intent is None or intent[0] != request_hash or result.get("request_hash") != request_hash:
                raise ValueError("command outcome requires its matching durable intent")
            prior = conn.execute("SELECT result_json FROM paper_command_outcomes WHERE account_id=? AND request_id=?",
                                 (account_id, request_id)).fetchone()
            if prior is not None:
                before = json.loads(prior[0])
                after = dict(result)
                before_effects = before.pop("committed_effects", [])
                after_effects = after.pop("committed_effects", [])
                if after_effects[:len(before_effects)] != before_effects:
                    raise ValueError("command committed effects cannot change")
                for key in ("persistence_error", "persistence_halt", "revision", "revision_status"):
                    before.pop(key, None)
                    after.pop(key, None)
                if before != after:
                    raise ValueError("command outcome cannot change its dispatch decision")
            conn.execute("""INSERT INTO paper_command_outcomes VALUES(?,?,?,?)
                ON CONFLICT(account_id,request_id) DO UPDATE SET result_json=excluded.result_json""",
                (account_id, request_id, request_hash, encoded))
            self._save_command_halt(conn, account_id, result)

    @staticmethod
    def _save_command_halt(conn: sqlite3.Connection, account_id: str, result: dict[str, Any]) -> None:
        if result.get("persistence_halt"):
            conn.execute("INSERT OR REPLACE INTO paper_store_metadata VALUES(?,?)",
                         ("persistence_halt:" + account_id, result["persistence_halt"]))
        elif result.get("success") and result.get("action") in {
                "acknowledge_risk_circuit_breaker", "acknowledge_risk_halt"}:
            conn.execute("DELETE FROM paper_store_metadata WHERE key=?", ("persistence_halt:" + account_id,))

    def save_command_result(self, account_id, request_id, request_hash, result):
        encoded = canonical(result)
        with self._get_connection() as conn:
            row = conn.execute("SELECT request_hash FROM paper_command_results WHERE account_id=? AND request_id=?",
                               (account_id, request_id)).fetchone()
            if row is not None and row[0] != request_hash:
                raise ValueError("request ID reused with a different command")
            conn.execute("INSERT INTO paper_command_results VALUES(?,?,?,?,?) ON CONFLICT(account_id,request_id) DO UPDATE SET result_json=excluded.result_json",
                         (account_id, request_id, request_hash, self._store_instance_id, encoded))
            self._save_command_halt(conn, account_id, result)

    def record_account_observation(self, account_id, timestamp, payload):
        self._validate_account(account_id)
        stamp = float(timestamp)
        if not math.isfinite(stamp):
            raise ValueError("observation timestamp must be finite")
        with self._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO paper_observations VALUES(?,?,?)",
                         (account_id, stamp, canonical(payload)))

    def get_account_history(self, account_id, start=None, end=None, limit=1000):
        count = min(100000, max(0, int(limit)))
        conditions, arguments = ["account_id=?"], [account_id]
        if start is not None:
            conditions.append("timestamp>=?")
            arguments.append(float(start))
        if end is not None:
            conditions.append("timestamp<=?")
            arguments.append(float(end))
        arguments.append(count)
        with self._get_connection() as conn:
            rows = conn.execute("SELECT timestamp,payload_json FROM paper_observations WHERE " +
                                " AND ".join(conditions) + " ORDER BY timestamp DESC LIMIT ?", arguments).fetchall()
            return [dict(json.loads(row[1]), timestamp=float(row[0])) for row in reversed(rows)]

    def register_config_epoch(self, account_id, config_payload, effective_ts, reason="configuration_migration"):
        encoded = canonical(config_payload)
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            last = conn.execute("SELECT epoch,checksum FROM paper_config_epochs WHERE account_id=? ORDER BY epoch DESC LIMIT 1",
                                (account_id,)).fetchone()
            if last is not None and last[1] == digest(encoded):
                return int(last[0])
            epoch = 1 if last is None else int(last[0]) + 1
            conn.execute("INSERT INTO paper_config_epochs VALUES(?,?,?,?,?,?)",
                         (account_id, epoch, float(effective_ts), encoded, digest(encoded), reason))
            return epoch

    def preview_account_migration(self, account_id, new_config_dict):
        loaded = self.load_cross_asset_snapshot(account_id)
        if loaded is None:
            return {"allowed": True, "new_account": True, "added_instruments": [], "errors": []}
        saved = loaded["snapshot"]["ledger"]
        old_specs = saved.get("instrument_specs", {})
        raw = new_config_dict.get("instruments", [])
        items = raw.values() if isinstance(raw, dict) else raw
        proposed = {}
        for item in items:
            spec = item.get("instrument", item.get("spec", item))
            proposed[spec["instrument_id"]] = spec
        errors = []
        warnings = []
        initial_capital = saved.get("starting_equity")
        requested_capital = new_config_dict.get("starting_capital")
        if (initial_capital is not None and requested_capital is not None
                and not math.isclose(float(initial_capital), float(requested_capital), rel_tol=0, abs_tol=1e-8)):
            errors.append("opening capital cannot be changed by migration; post a sourced funding event instead")
        reporting = saved.get("reporting_currency")
        requested_reporting = new_config_dict.get("reporting_currency", reporting)
        if reporting is not None and reporting != requested_reporting:
            errors.append("reporting currency requires a separate reporting view or new account")
        opening = saved.get("opening_balances")
        if opening is not None and "starting_balances" in new_config_dict:
            requested = new_config_dict["starting_balances"] or {
                requested_reporting: requested_capital}
            requested = {str(currency).upper(): float(amount) for currency, amount in requested.items()
                         if float(amount) != 0}
            original = {str(currency).upper(): float(amount) for currency, amount in opening.items()
                        if float(amount) != 0}
            if requested != original:
                errors.append("opening currency balances cannot be changed by migration; use sourced funding and explicit currency conversion events")
        elif opening is None:
            warnings.append("original opening currency allocation was not recorded by the legacy version; configured opening balances are not applied to this existing account")
        removed = sorted(set(old_specs) - set(proposed))
        for instrument_id in removed:
            quantity = saved.get("positions", {}).get(instrument_id, {}).get("quantity", 0)
            errors.append("{} must remain configured{}; disable quoting to retain its history".format(
                instrument_id, " because it has exposure" if quantity else ""))
        immutable = ("asset_class", "venue", "base_currency", "quote_currency",
                     "settlement_currency", "contract_multiplier", "expiry")
        for instrument_id in set(old_specs) & set(proposed):
            for name in immutable:
                if name in proposed[instrument_id] and old_specs[instrument_id].get(name) != proposed[instrument_id][name]:
                    errors.append("{} changes {}; use a new instrument ID or explicit migration".format(instrument_id, name))
        return {"allowed": not errors, "new_account": False,
                "added_instruments": sorted(set(proposed) - set(old_specs)),
                "removed_instruments": removed, "errors": errors,
                "warnings": warnings,
                "opening_balance_verification": "recorded" if opening is not None else "unverified_legacy",
                "requires_backup": True, "requires_stopped_writer": True,
                "revision": loaded.get("account_seq", 0)}

    def backup_database(self, destination):
        import sqlite3
        target = Path(destination).resolve()
        if target == self.db_path.resolve() or target.exists():
            raise ValueError("backup destination must be a new file distinct from the live database")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as source:
            with closing(sqlite3.connect(str(target))) as backup:
                source.backup(backup)
                if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("backup integrity check failed")
        return str(target)

    def run_daily_backup(self, directory=None, now=None, retention_days=7):
        """Rotate only this function's explicitly named daily backup files."""
        import re
        from datetime import UTC, datetime
        stamp = time.time() if now is None else float(now)
        root = (Path(directory) if directory is not None else self.db_path.parent / "backups").resolve()
        root.mkdir(parents=True, exist_ok=True)
        day = datetime.fromtimestamp(stamp, UTC).strftime("%Y-%m-%d")
        with self._get_connection() as source:
            source_id = source.execute("SELECT value FROM paper_store_metadata WHERE key='source_id'").fetchone()[0]
            schema = source.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
            schema = [tuple(row) for row in schema]
        prefix = self.db_path.stem + "-" + source_id + "-paper-daily-"
        destination = root / (prefix + day + ".db")

        def valid_backup(path: Path) -> bool:
            if not path.is_file() or path.samefile(self.db_path) or path.stat().st_size == 0:
                return False
            try:
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as check:
                    identity = check.execute("SELECT value FROM paper_store_metadata WHERE key='source_id'").fetchone()
                    tables = check.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
                    return (identity == (source_id,) and tables == schema
                            and check.execute("PRAGMA quick_check").fetchall() == [("ok",)])
            except sqlite3.Error:
                return False

        if not valid_backup(destination):
            temporary = root / ("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                self.backup_database(temporary)
                if not valid_backup(temporary):
                    raise RuntimeError("daily backup source identity or schema verification failed")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        cutoff = stamp - max(1, int(retention_days)) * 86400
        for candidate in root.iterdir():
            if candidate.is_file() and re.fullmatch(re.escape(prefix) + r"\d{4}-\d{2}-\d{2}\.db", candidate.name):
                try:
                    date = datetime.strptime(candidate.stem[len(prefix):], "%Y-%m-%d").replace(tzinfo=UTC)
                except ValueError:
                    continue
                if date.timestamp() < cutoff and candidate.resolve().parent == root:
                    candidate.unlink()
        return str(destination)

    @staticmethod
    def restore_database(backup_path, destination):
        """Restore to a new database; never overwrite an account in place."""
        import sqlite3
        backup, target = Path(backup_path).resolve(), Path(destination).resolve()
        if not backup.is_file() or target.exists() or backup == target:
            raise ValueError("restore requires an existing backup and a new destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(backup.as_uri() + "?mode=ro", uri=True)) as source:
            if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("backup integrity check failed")
            with closing(sqlite3.connect(str(target))) as restored:
                source.backup(restored)
        return str(target)

    @contextmanager
    def acquire_account_writer(self, account_id):
        """Hold the OS writer lease and a nontransacting WAL connection.

        Closing SQLite's last connection checkpoints the WAL. Keeping one
        connection open only for the writer lifetime prevents that expensive
        work on every financial commit without weakening FULL durability.
        """
        with AccountWriterLock(self.db_path, account_id) as lease:
            with self._connection_registry_condition:
                if self._writer_connections is not None or self._writer_connections_closing:
                    raise RuntimeError("this StateManager already has an active paper writer scope")
                self._writer_connections = {}
            try:
                with self._get_connection() as keeper:
                    keeper.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
                    yield lease
            finally:
                with self._connection_registry_condition:
                    self._writer_connections_closing = True
                    while self._connection_active:
                        self._connection_registry_condition.wait()
                    connections = list(self._writer_connections.values())
                    self._writer_connections = None
                errors = []
                try:
                    for connection in connections:
                        try:
                            connection.close()
                        except Exception as exc:
                            errors.append(exc)
                finally:
                    with self._connection_registry_condition:
                        self._writer_connections_closing = False
                        self._connection_registry_condition.notify_all()
                if errors:
                    raise errors[0]


class AccountWriterLock:
    """OS-held lock is released after a crash; stale lock files are harmless."""

    def __init__(self, database, account_id):
        self.path = Path(str(Path(database).resolve()) + "." + digest(account_id)[:16] + ".writer.lock")
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+b")
        self.file.seek(0)
        if self.path.stat().st_size == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise RuntimeError("another process is already writing this paper account") from exc
        return self

    def __exit__(self, *_):
        if self.file is not None:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            self.file = None
