import gzip
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("release_audit", Path(__file__).parents[1] / "tools" / "release_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_detects_token_without_disclosing_it(tmp_path):
    token = "gh" + "p_" + "A" * 36
    (tmp_path / "sample.py").write_text('value = "' + token + '"', encoding="utf-8")
    result = module.audit(tmp_path)
    assert result["status"] == "failed"
    assert result["findings"][0]["rule"] == "github_token"
    assert token not in str(result)


def test_audit_checks_decompressed_fixtures_and_private_state(tmp_path):
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    with gzip.open(fixtures / "input.jsonl.gz", "wt") as stream:
        stream.write("contact = 'private" + "@" + "personal.invalid-domain.com'\n")
    (tmp_path / ".env").write_text("", encoding="utf-8")
    rules = {item["rule"] for item in module.audit(tmp_path)["findings"]}
    assert {"email_address", "private_runtime_file"} <= rules


def test_public_environment_template_and_reserved_example_email_are_allowed(tmp_path):
    (tmp_path / ".env.example").write_text("DATABENTO_API_KEY=\nOANDA_ACCOUNT_ID=\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Contact example@example.com", encoding="utf-8")
    assert module.audit(tmp_path)["status"] == "passed"
