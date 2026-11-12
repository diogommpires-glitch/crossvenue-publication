"""Windows credential setup with synthetic secrets and a disposable child app."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows DPAPI credential helper")
ROOT = Path(__file__).parents[1]


def sandbox_project(tmp_path):
    root = tmp_path / "project"
    script = root / "scripts" / "windows" / "provider_access.ps1"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/windows/provider_access.ps1", script)
    runtime = root / ".venv-paper"
    (runtime / "Scripts").mkdir(parents=True)
    # Use an existing Windows venv redirector and its base-runtime metadata.
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if not cfg.is_file():
        pytest.skip("Windows test requires a dedicated virtual environment")
    shutil.copyfile(sys.executable, runtime / "Scripts/python.exe")
    shutil.copyfile(cfg, runtime / "pyvenv.cfg")
    (root / "config").mkdir()
    config = {"providers": [{"type": "oanda", "environment": "live"}],
              "starting_balances": {"USDT": 10000}, "risk": {"max_gross_exposure": 10000}}
    (root / "config/cross_asset_paper.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "launch.py").write_text(
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "expected={'DATABENTO_API_KEY':'fixture-databento-value',"
        "'OANDA_ACCOUNT_ID':'fixture-account-value','OANDA_ACCESS_TOKEN':'fixture-token-value'}\n"
        "result={'all_loaded':all(os.environ.get(k)==v for k,v in expected.items()),"
        "'no_secret_arguments':not any(v in ' '.join(sys.argv) for v in expected.values())}\n"
        "Path('child-result.json').write_text(json.dumps(result))\n",
        encoding="utf-8",
    )
    return root, script, tmp_path / "secrets/provider-access.clixml"


def powershell(tmp_path, script):
    harness = tmp_path / "harness.ps1"
    harness.write_text(script, encoding="utf-8")
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                          capture_output=True, text=True, timeout=30)


def test_encrypted_credentials_load_into_child_without_changing_account_funds(tmp_path):
    root, helper, store = sandbox_project(tmp_path)
    result = powershell(tmp_path, f"""
$ErrorActionPreference='Stop'
function Read-Host {{
    param([string]$Prompt, [switch]$AsSecureString)
    if (-not $AsSecureString) {{ return 'practice' }}
    $value = if ($Prompt -like 'Databento*') {{ 'fixture-databento-value' }}
             elseif ($Prompt -like '*account ID*') {{ 'fixture-account-value' }}
             else {{ 'fixture-token-value' }}
    return ConvertTo-SecureString $value -AsPlainText -Force
}}
& '{helper}' -Action Configure -CredentialsPath '{store}'
& '{helper}' -Action Check -CredentialsPath '{store}'
$env:DATABENTO_API_KEY=$null
$env:OANDA_ACCOUNT_ID=$null
$env:OANDA_ACCESS_TOKEN=$null
& '{helper}' -Action Run -CredentialsPath '{store}'
if ($LASTEXITCODE -ne 0) {{ exit $LASTEXITCODE }}
if ($env:DATABENTO_API_KEY -or $env:OANDA_ACCOUNT_ID -or $env:OANDA_ACCESS_TOKEN) {{ throw 'Child credentials leaked to parent environment' }}
""")
    assert result.returncode == 0, result.stdout + result.stderr
    stored_xml_text = "".join(ET.parse(store).getroot().itertext())
    for value in ("fixture-databento-value", "fixture-account-value", "fixture-token-value"):
        assert value not in stored_xml_text
        assert value not in result.stdout + result.stderr
    assert json.loads((root / "child-result.json").read_text()) == {
        "all_loaded": True, "no_secret_arguments": True}
    config = json.loads((root / "config/cross_asset_paper.json").read_text())
    assert config["providers"][0]["environment"] == "practice"
    assert config["starting_balances"] == {"USDT": 10000}
    assert config["risk"] == {"max_gross_exposure": 10000}
    assert list((root / "data/backups").glob("provider-config-*.json"))


def test_empty_setup_does_not_create_secrets_or_change_private_configuration(tmp_path):
    root, helper, store = sandbox_project(tmp_path)
    config = (root / "config/cross_asset_paper.json").read_bytes()
    result = powershell(tmp_path, f"""
function Read-Host {{ param([string]$Prompt,[switch]$AsSecureString)
    if($AsSecureString){{ return [Security.SecureString]::new() }}
    return ''
}}
& '{helper}' -Action Configure -CredentialsPath '{store}'
""")
    assert result.returncode == 0, result.stderr
    assert not store.exists()
    assert (root / "config/cross_asset_paper.json").read_bytes() == config


def test_corrupt_credential_store_fails_before_starting_child(tmp_path):
    root, helper, store = sandbox_project(tmp_path)
    store.parent.mkdir()
    store.write_text("invalid encrypted store", encoding="utf-8")
    result = powershell(tmp_path, f"& '{helper}' -Action Run -CredentialsPath '{store}'")
    assert result.returncode != 0
    assert not (root / "child-result.json").exists()
    assert "Cannot decrypt" in result.stderr


def test_default_credential_store_uses_crossvenue_without_changing_account_funds(tmp_path):
    root, helper, _ = sandbox_project(tmp_path)
    config_path = root / "config/cross_asset_paper.json"
    original_config = config_path.read_bytes()
    local_app_data = tmp_path / "local-app-data"
    store = local_app_data / "CrossVenue/provider-access.clixml"
    secret = "fixture-default-databento-value"
    result = powershell(
        tmp_path,
        f"""
$ErrorActionPreference='Stop'
$env:LOCALAPPDATA='{local_app_data}'
$env:DATABENTO_API_KEY=$null
function Read-Host {{
    param([string]$Prompt, [switch]$AsSecureString)
    return ConvertTo-SecureString '{secret}' -AsPlainText -Force
}}
& '{helper}' -Action Configure -Provider Databento
& '{helper}' -Action Check -Provider Databento
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert store.is_file()
    stored_xml_text = "".join(ET.parse(store).getroot().itertext())
    assert secret not in stored_xml_text
    assert secret not in result.stdout + result.stderr
    assert "DATABENTO_API_KEY: configured (value hidden)" in result.stdout
    assert config_path.read_bytes() == original_config
