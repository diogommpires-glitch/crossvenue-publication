from dataclasses import replace
from pathlib import Path

import pytest

from src.config.cross_asset_config import load_cross_asset_config
from src.runtime.replay import ReplayRunner


ROOT = Path(__file__).parents[1]


def test_replay_rejects_unannounced_fee_changes_and_labels_explicit_research_assumptions():
    config = load_cross_asset_config(ROOT / "config" / "cross_asset_paper.example.json")
    instruments = dict(config.instruments)
    identity = "BINANCE:BTCUSDT"
    original = instruments[identity]
    instruments[identity] = replace(original, fees=replace(original.fees, maker_rate=0.001))
    changed = replace(config, instruments=instruments)
    recording = ROOT / "tests" / "fixtures" / "representative_recording"
    with pytest.raises(ValueError, match="differs from recording"):
        ReplayRunner(changed).run(recording)
    study = ReplayRunner(changed, research_assumption_override=True).run(recording)
    assert study["assumptions_overridden"] is True
    assert study["execution_config_hash"] != study["recorded_execution_config_hash"]
    assert study["config_hash"] == study["execution_config_hash"]
