import json
from pathlib import Path
from types import SimpleNamespace

from src.runtime import replay_soak
from src.runtime.replay import VirtualClock
from src.runtime.replay_soak import main, run_soak


def test_short_wall_clock_soak_never_claims_twenty_four_hour_gate(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "representative_recording"
    evidence = run_soak(fixture, duration_seconds=0.1, pause_seconds=0,
                        research_assumption_override=True,
                        progress_path=tmp_path / "progress.json")
    assert evidence["status"] == "passed"
    assert evidence["completed_replays"] >= 1
    assert evidence["elapsed_wall_seconds"] >= 0.1
    assert evidence["qualifies_24_hour_gate"] is False
    assert evidence["input_kind"] == "synthetic_fixture"
    assert json.loads((tmp_path / "progress.json").read_text())["all_digests_equal"]


def test_standalone_soak_cli_explicitly_labels_historical_re_evaluation(tmp_path, monkeypatch):
    clock = VirtualClock()
    # Advance after one real replay so setup cost cannot exhaust this CLI flag test.
    monkeypatch.setattr(
        replay_soak,
        "time",
        SimpleNamespace(monotonic=clock, sleep=lambda _: clock.advance_to(0.01)),
    )
    fixture = Path(__file__).parent / "fixtures" / "representative_recording"
    progress = tmp_path / "cli-progress.json"
    assert (
        main(
            [
                str(fixture),
                "--duration",
                "0.01",
                "--pause",
                "0",
                "--progress",
                str(progress),
                "--research-assumption-override",
            ]
        )
        == 0
    )
    evidence = json.loads(progress.read_text())
    assert evidence["completed_replays"] == 1
    assert evidence["status"] == "passed" and evidence["assumptions_overridden"]
    assert evidence["replay_mode"] == "corrected_re_evaluation"
