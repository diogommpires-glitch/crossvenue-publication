"""The fee command is offline and uses no account configuration."""

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest


def test_fee_cli_reports_exact_break_even_and_native_fx_without_account(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.runtime.paper_cli",
            "fees",
            "--pricing",
            "fixed",
            "--quantity",
            "10",
            "--price",
            "100",
            "--stamp-rate",
            "0",
            "--finra-pass-through",
            "zero",
            "--prefund-usd",
            "10000",
            "--eur-per-usd",
            "0.9",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert float(report["flat_price_cost_usd"]) == pytest.approx(2.02066)
    assert float(report["exact_nonnegative_break_even_gain_usd"]) == pytest.approx(
        2.02070162645, abs=1e-10
    )
    assert Decimal(report["fx_prefunding"]["commission_eur"]) == Decimal("1.80")
    assert report["private_account_access"] is False
    assert report["provider_connections"] is False
    assert report["source_snapshot"] == "2026-11-11"
    assert list(tmp_path.iterdir()) == []


def test_fee_cli_does_not_assume_a_portuguese_tax_exemption(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.runtime.paper_cli",
            "fees",
            "--pricing",
            "fixed",
            "--quantity",
            "10",
            "--price",
            "100",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "stamp-rate" in result.stderr
