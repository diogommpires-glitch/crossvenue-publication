"""Versioned, checksummed recordings of the authoritative paper input stream.

Record after inputs enter the account writer. Arrival order is authoritative;
source timestamps must never be used to reorder delayed provider messages.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import closing
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from ..market.events import MarketDataEvent, MarketEventKind
from ..utils.types import Side

SCHEMA_VERSION = 2
EXECUTION_SEMANTICS_VERSION = 2
QUOTE_POLICY_VERSION = 2
KINDS = frozenset({"market", "command", "timer", "provider", "configuration"})
_SECRET_KEYS = frozenset({"api_key", "api_secret", "secret", "password", "access_token",
                          "refresh_token", "authorization", "session_token", "token"})


def sanitize(value: Any) -> Any:
    """Convert to JSON values and remove credentials before touching disk."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): ("[redacted]" if str(key).lower() in _SECRET_KEYS
                          or str(key).lower().endswith(("_password", "_secret", "_token"))
                          else sanitize(item)) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((sanitize(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, (datetime, Path)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, str):
        # Account-specific request URLs and bearer material are never captured.
        value = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [redacted]", value)
        value = re.sub(r"(?i)(https?://[^\s/]+/[^\s]*accounts/)[^/\s?]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)([?&](?:token|access_token|api_key|key)=)[^&\s]+", r"\1[redacted]", value)
        return value
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("recordings cannot contain non-finite numbers")
        return value
    raise TypeError("unsupported recording value: {}".format(type(value).__name__))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_checksum(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _signature_history_path(directory: str | Path, manifest: dict[str, Any]) -> Path | None:
    history = manifest.get("initial_signature_history")
    if history is None:
        return None
    if not isinstance(history, dict):
        raise ValueError("unsupported recording signature history metadata")
    if (history.get("schema_version") != 1 or history.get("file") != "initial-signatures.sqlite3"
            or type(history.get("records")) is not int or history["records"] < 0):
        raise ValueError("unsupported recording signature history metadata")
    root = Path(directory).resolve()
    path = (root / "initial-signatures.sqlite3").resolve()
    if path.parent != root:
        raise ValueError("recording signature history escaped its directory")
    if _file_checksum(path) != history.get("sha256"):
        raise ValueError("recording signature history checksum mismatch")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        if (conn.execute("PRAGMA user_version").fetchone()[0] != 1
                or conn.execute("SELECT COUNT(*) FROM signatures").fetchone()[0] != history["records"]):
            raise ValueError("recording signature history metadata mismatch")
    return path


def event_to_dict(event: MarketDataEvent) -> Dict[str, Any]:
    if not isinstance(event, MarketDataEvent):
        raise TypeError("event must be MarketDataEvent")
    result = asdict(event)
    result["kind"] = event.kind.value
    result["side"] = event.side.value if event.side is not None else None
    return sanitize(result)


def event_from_dict(payload: Dict[str, Any]) -> MarketDataEvent:
    values = dict(payload)
    values["kind"] = MarketEventKind(values["kind"])
    if values.get("side") is not None:
        raw = values["side"]
        text = str(raw).lower()
        values["side"] = Side.BUY if text == "buy" else Side.SELL if text == "sell" else Side(raw)
    values["bids"] = tuple(tuple(level) for level in values.get("bids", ()))
    values["asks"] = tuple(tuple(level) for level in values.get("asks", ()))
    return MarketDataEvent(**values)


class SessionRecorder:
    """Append input records to compressed hourly chunks and an atomic manifest.

    A newly created manifest is incomplete until ``close(complete=True)``.
    Interrupted or disk-full sessions can never silently certify completeness.
    """

    def __init__(self, directory: Union[str, Path], *, config: Any,
                 code_version: str = "unknown", dependency_hash: str = "unknown",
                 seed: int = 0, max_chunk_records: int = 100_000,
                 source_kind: str = "unspecified") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "manifest.json").exists():
            raise FileExistsError("recording already exists; use a new directory")
        if max_chunk_records <= 0:
            raise ValueError("max_chunk_records must be positive")
        self._lock = threading.RLock()
        self._stream = None
        self._raw = None
        self._hour = None
        self._chunk_count = 0
        self._chunk_first = 0
        self._chunk_file = None
        self._max_chunk_records = int(max_chunk_records)
        self._closed = False
        self._last_time = None
        safe_config = sanitize(config)
        self.manifest = {"schema_version": SCHEMA_VERSION, "format": "paper-input-jsonl-gzip",
                         "execution_semantics_version": EXECUTION_SEMANTICS_VERSION,
                         "quote_policy_version": QUOTE_POLICY_VERSION,
                         "config": safe_config,
                         "config_hash": hashlib.sha256(canonical_json(safe_config).encode()).hexdigest(),
                         "code_version": sanitize(code_version), "dependency_hash": dependency_hash,
                         "seed": int(seed), "complete": False, "records": 0, "chunks": [],
                         "source_kind": source_kind,
                         "time_unit": "epoch_seconds_float64", "ordering": "ingest_seq"}
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(canonical_json(self.manifest) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")

    def set_initial_state(self, checkpoint: dict[str, Any],
                          provider_bindings: dict[str, Any] | None = None,
                          signature_history: Iterable[tuple[str, str, list[Any] | tuple[Any, ...]]] | None = None) -> None:
        """Capture a restored account before the first authoritative input."""
        with self._lock:
            if self._closed or self.manifest["records"]:
                raise RuntimeError("initial account state must precede every input")
            state = sanitize(checkpoint)
            self.manifest["initial_state"] = state
            self.manifest["initial_state_hash"] = hashlib.sha256(canonical_json(state).encode()).hexdigest()
            if provider_bindings is not None:
                self.manifest["provider_bindings"] = sanitize(provider_bindings)
            if signature_history is not None:
                path = self.directory / "initial-signatures.sqlite3"
                if path.exists():
                    raise FileExistsError("recording signature history already exists")
                # A failed export cannot become a complete recording on close.
                self.manifest["initial_signature_history"] = {
                    "schema_version": 1, "file": path.name, "sha256": None, "records": 0}
                count = 0
                with closing(sqlite3.connect(str(path))) as conn, conn:
                    conn.execute("PRAGMA user_version=1")
                    conn.execute("CREATE TABLE signatures(namespace TEXT NOT NULL,event_id TEXT NOT NULL,signature_json TEXT NOT NULL,PRIMARY KEY(namespace,event_id))")
                    for namespace, event_id, signature in signature_history:
                        namespace, event_id, signature = sanitize([namespace, event_id, signature])
                        if not isinstance(namespace, str) or not isinstance(event_id, str) or not isinstance(signature, list):
                            raise ValueError("recording signatures require namespace, event ID and signature array")
                        conn.execute("INSERT INTO signatures VALUES(?,?,?)",
                                     (namespace, event_id, canonical_json(signature)))
                        count += 1
                self.manifest["initial_signature_history"].update(
                    sha256=_file_checksum(path), records=count)
            self._write_manifest()

    def _finish_chunk(self) -> None:
        if self._stream is None:
            return
        self._stream.close()
        self._raw.close()
        path = self.directory / self._chunk_file
        self.manifest["chunks"].append({"file": self._chunk_file,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": self._chunk_count, "first_seq": self._chunk_first,
            "last_seq": self.manifest["records"]})
        self._stream = self._raw = None
        self._write_manifest()

    def append(self, kind: str, payload: Any, processing_ts: float, *,
               provider_id: Optional[str] = None, generation: int = 0) -> int:
        if kind not in KINDS:
            raise ValueError("unsupported recording input kind")
        stamp = float(processing_ts)
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError("processing_ts must be finite and non-negative")
        if isinstance(generation, bool) or int(generation) != generation or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        safe_payload = sanitize(payload)
        with self._lock:
            if self._closed:
                raise RuntimeError("recording is closed")
            if self._last_time is not None and stamp < self._last_time:
                raise ValueError("writer processing time cannot move backwards")
            hour = datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y%m%dT%H")
            if self._stream is not None and (hour != self._hour or self._chunk_count >= self._max_chunk_records):
                self._finish_chunk()
            sequence = self.manifest["records"] + 1
            if self._stream is None:
                self._hour = hour
                self._chunk_file = "{}-{:06d}.jsonl.gz".format(hour, len(self.manifest["chunks"]))
                self._raw = (self.directory / self._chunk_file).open("xb")
                self._stream = gzip.GzipFile(filename="", fileobj=self._raw, mode="wb", mtime=0)
                self._chunk_count = 0
                self._chunk_first = sequence
            record = {"schema_version": SCHEMA_VERSION, "ingest_seq": sequence,
                      "kind": kind, "processing_ts": stamp, "payload": safe_payload,
                      "provider_id": provider_id, "generation": int(generation)}
            self._stream.write((canonical_json(record) + "\n").encode("utf-8"))
            self._chunk_count += 1
            self.manifest["records"] = sequence
            self._last_time = stamp
            return sequence

    def record_market(self, event: MarketDataEvent, *, provider_id: Optional[str] = None,
                      generation: int = 0, processing_ts: Optional[float] = None) -> int:
        return self.append("market", event_to_dict(event),
                           event.receive_ts if processing_ts is None else processing_ts,
                           provider_id=provider_id, generation=generation)

    def close(self, complete: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._finish_chunk()
                self.manifest["complete"] = bool(complete)
                self._write_manifest()
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(complete=exc_type is None)


def load_manifest(directory: Union[str, Path], *, allow_incomplete: bool = False) -> Dict[str, Any]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") not in (1, SCHEMA_VERSION) or manifest.get("ordering") != "ingest_seq":
        raise ValueError("unsupported recording schema or ordering")
    if not manifest.get("complete") and not allow_incomplete:
        raise ValueError("recording is incomplete and cannot certify reproducibility")
    semantics = manifest.get("execution_semantics_version", 1)
    if type(semantics) is not int or semantics < 1:
        raise ValueError("recording execution semantics version must be a positive integer")
    quote_policy = manifest.get("quote_policy_version", 1)
    if type(quote_policy) is not int or quote_policy < 1:
        raise ValueError("recording quote policy version must be a positive integer")
    expected = hashlib.sha256(canonical_json(manifest["config"]).encode()).hexdigest()
    if expected != manifest.get("config_hash"):
        raise ValueError("recording configuration checksum mismatch")
    if "initial_state" in manifest:
        state_hash = hashlib.sha256(canonical_json(manifest["initial_state"]).encode()).hexdigest()
        if state_hash != manifest.get("initial_state_hash"):
            raise ValueError("initial account state checksum mismatch")
    _signature_history_path(root, manifest)
    return manifest


def iter_records(directory: Union[str, Path], *, allow_incomplete: bool = False) -> Iterator[Dict[str, Any]]:
    root = Path(directory).resolve()
    manifest = load_manifest(root, allow_incomplete=allow_incomplete)
    next_sequence = 1
    last_time = None
    for chunk in manifest["chunks"]:
        path = (root / chunk["file"]).resolve()
        if path.parent != root or path.suffix != ".gz":
            raise ValueError("recording chunk escaped its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]:
            raise ValueError("recording chunk checksum mismatch")
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if (record.get("schema_version") != manifest["schema_version"]
                        or record.get("ingest_seq") != next_sequence or record.get("kind") not in KINDS):
                    raise ValueError("recording has a missing/duplicate sequence or invalid kind")
                stamp = float(record["processing_ts"])
                if not math.isfinite(stamp) or (last_time is not None and stamp < last_time):
                    raise ValueError("recording processing clock moved backwards")
                last_time = stamp
                next_sequence += 1
                count += 1
                yield record
        if count != chunk["records"]:
            raise ValueError("recording chunk count mismatch")
    if next_sequence - 1 != manifest["records"]:
        raise ValueError("recording total count mismatch")
