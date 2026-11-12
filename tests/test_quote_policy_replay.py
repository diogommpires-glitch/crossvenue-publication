"""Quote decisions have provenance separate from account financial semantics."""

import json
from pathlib import Path

import pytest

from src.runtime.recording import SessionRecorder, load_manifest
from src.runtime.replay import ReplayRunner

CONFIG = Path(__file__).parents[1] / "config/cross_asset_paper.example.json"


def capture(path):
    with SessionRecorder(path, config=json.loads(CONFIG.read_text())) as recorder:
        recorder.append("timer", {"timestamp": 1792497600.0}, 1792497600.0)


def test_new_recording_declares_quote_policy_and_replays_without_override(tmp_path):
    capture(tmp_path)
    manifest = load_manifest(tmp_path)
    assert manifest.get("quote_policy_version") == 2
    result = ReplayRunner(CONFIG).run(tmp_path)
    assert result["quote_policy_version"] == 2
    assert result["quote_policy_overridden"] is False
    assert result["execution_semantics_version"] == 2
    assert result["replay_mode"] == "current_semantics"


def test_legacy_v2_quote_policy_requires_explicit_corrected_re_evaluation(tmp_path):
    capture(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.pop("quote_policy_version", None)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quote policy"):
        ReplayRunner(CONFIG).run(tmp_path)
    corrected = ReplayRunner(CONFIG, research_assumption_override=True).run(tmp_path)
    assert corrected["recorded_quote_policy_version"] == 1
    assert corrected["quote_policy_overridden"] is True
    assert corrected["assumptions_overridden"] is True
    assert corrected["execution_semantics_overridden"] is False
    assert corrected["replay_mode"] == "corrected_re_evaluation"


@pytest.mark.parametrize("bad", [True, 0, -1, "2", None])
def test_malformed_quote_policy_is_not_treated_as_current(tmp_path, bad):
    capture(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["quote_policy_version"] = bad
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quote policy"):
        load_manifest(tmp_path)
