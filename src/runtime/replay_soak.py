"""Real wall-clock repeated replay qualification with durable progress evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import tracemalloc

from ..config.cross_asset_config import load_cross_asset_config, parse_cross_asset_config
from .recording import load_manifest
from .replay import ReplayRunner


def source_fingerprint():
    root = Path(__file__).parents[1]
    result = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        result.update(path.relative_to(root).as_posix().encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def _write(path, evidence):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_soak(directory, *, duration_seconds=86400, progress_path, config_path=None,
             pause_seconds=0.25, research_assumption_override=False):
    if (not math.isfinite(duration_seconds) or not math.isfinite(pause_seconds)
            or duration_seconds <= 0 or pause_seconds < 0):
        raise ValueError("duration must be positive and pause non-negative")
    manifest = load_manifest(directory)
    config = load_cross_asset_config(config_path) if config_path else parse_cross_asset_config(manifest["config"])
    source_hash = source_fingerprint()
    started = time.monotonic()
    evidence = {"schema_version": 1, "kind": "wall_clock_repeated_replay_soak",
        "status": "running", "process_id": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_wall_seconds": duration_seconds, "elapsed_wall_seconds": 0,
        "completed_replays": 0, "all_digests_equal": True, "source_hash": source_hash,
        "recording_config_hash": manifest["config_hash"],
        "input_kind": manifest.get("source_kind", "unspecified_recording"),
        "reference_machine": {"platform": platform.platform(), "python": platform.python_version()},
        "live_acceptance": "pending", "qualifies_24_hour_gate": False}
    _write(progress_path, evidence)
    owned_tracer = not tracemalloc.is_tracing()
    if owned_tracer:
        tracemalloc.start(1)
    last_write = started
    try:
        while time.monotonic() - started < duration_seconds:
            result = ReplayRunner(config, research_assumption_override=research_assumption_override).run(directory)
            if evidence["completed_replays"] == 0:
                evidence["reference_digest"] = result["digest"]
                evidence["fills_per_replay"] = len(result["fills"])
                evidence["execution_semantics_version"] = result["execution_semantics_version"]
                evidence["recorded_execution_semantics_version"] = result["recorded_execution_semantics_version"]
                evidence["assumptions_overridden"] = result["assumptions_overridden"]
                evidence["replay_mode"] = result["replay_mode"]
            elif result["digest"] != evidence["reference_digest"]:
                evidence["all_digests_equal"] = False
                raise RuntimeError("replay output changed under the same recording and configuration")
            evidence["completed_replays"] += 1
            now = time.monotonic()
            evidence["elapsed_wall_seconds"] = now - started
            if now - last_write >= 5:
                current, peak = tracemalloc.get_traced_memory()
                evidence.update(python_heap_bytes=current, python_heap_peak_bytes=peak)
                _write(progress_path, evidence)
                last_write = now
            time.sleep(min(pause_seconds, max(0.0, duration_seconds - (time.monotonic() - started))))
        if source_fingerprint() != source_hash:
            raise RuntimeError("source changed during soak; rerun against the release version")
        evidence.update(status="passed", elapsed_wall_seconds=time.monotonic() - started,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        evidence["qualifies_24_hour_gate"] = duration_seconds >= 86400 and evidence["elapsed_wall_seconds"] >= 86400
    except BaseException as exc:
        evidence.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=str(exc), elapsed_wall_seconds=time.monotonic() - started)
        _write(progress_path, evidence)
        raise
    finally:
        current, peak = tracemalloc.get_traced_memory()
        evidence.update(python_heap_bytes=current, python_heap_peak_bytes=peak)
        _write(progress_path, evidence)
        if owned_tracer:
            tracemalloc.stop()
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real wall-clock deterministic replay soak; no provider connections")
    parser.add_argument("recording")
    parser.add_argument("--duration", type=float, default=86400)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--config")
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument("--research-assumption-override", action="store_true",
                        help="Explicit corrected re-evaluation of historical execution semantics")
    args = parser.parse_args(argv)
    result = run_soak(args.recording, duration_seconds=args.duration, progress_path=args.progress,
                      config_path=args.config, pause_seconds=args.pause,
                      research_assumption_override=args.research_assumption_override)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
