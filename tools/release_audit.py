"""Local source-export privacy checks. Findings never include matched values."""

import argparse
import gzip
import json
import math
from pathlib import Path
import re


SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
              ".venv-paper", "venv", ".python", ".tools", ".uv-cache", ".tmp", ".hallmark"}
PRIVATE_PARTS = {"data", "logs", "results", "recordings", "delivery_evidence", "context"}
PRIVATE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx", ".log"}
RULES = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "service_token": re.compile(r"\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    "personal_home_path": re.compile(r"(?i)(?:[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"']+|/" + "home" + r"/[^/\s]+|/" + "Users" + r"/[^/\s]+)"),
    "account_identifier": re.compile(r"\b\d{3}-\d{3}-\d{7}-\d{3}\b"),
}
EMAIL = re.compile(r"[A-Za-z0-9_.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
ASSIGNMENT = re.compile(r"(?i)\b(?:api[_-]?key|api[_-]?secret|access[_-]?token|password|authorization|session_token)\b[\"']?\s*[:=]\s*[\"']([^\"'\r\n]{20,})[\"']")
PLACEHOLDERS = ("example", "placeholder", "fixture", "dummy", "unused", "redacted",
                "never-write", "your-", "your_", "your ", "synthetic")


def entropy(value):
    return -sum((value.count(char) / len(value)) * math.log2(value.count(char) / len(value))
                for char in set(value))


def audit(root, personal_terms=()):
    root = Path(root).resolve()
    findings, count = [], 0
    personal = [re.compile(r"(?i)\b" + re.escape(term) + r"\b") for term in personal_terms if term]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            findings.append({"file": relative.as_posix(), "line": 0, "rule": "symlink_not_exportable"})
            continue
        if not path.is_file():
            continue
        count += 1
        name = path.name.lower()
        private = (any(part in PRIVATE_PARTS for part in relative.parts)
                   or path.suffix.lower() in PRIVATE_SUFFIXES
                   or (name.startswith(".env") and name != ".env.example")
                   or name in {"cross_asset_paper.json", "settings.json", "trading_config.json"}
                   or ".db-" in name)
        if private:
            findings.append({"file": relative.as_posix(), "line": 0, "rule": "private_runtime_file"})
            continue
        try:
            if path.suffix == ".gz":
                if "fixtures" not in relative.parts:
                    raise ValueError("only reviewed fixture archives belong in the export")
                with gzip.open(path, "rb") as stream:
                    content = stream.read(32 * 1024 * 1024 + 1)
                if len(content) > 32 * 1024 * 1024:
                    raise ValueError("fixture decompression limit exceeded")
            else:
                content = path.read_bytes()
            text = content.decode("utf-8-sig")
        except (UnicodeError, OSError, ValueError):
            findings.append({"file": relative.as_posix(), "line": 0, "rule": "unreviewed_binary_or_archive"})
            continue
        for number, line in enumerate(text.splitlines(), 1):
            matched = [rule for rule, pattern in RULES.items() if pattern.search(line)]
            if any(pattern.search(line) for pattern in personal):
                matched.append("personal_term")
            for match in EMAIL.finditer(line):
                domain = match.group(1).lower()
                if domain not in {"example.com", "example.org", "example.net"} and not domain.endswith((".invalid", ".test")):
                    matched.append("email_address")
            for match in ASSIGNMENT.finditer(line):
                value = match.group(1)
                if not any(word in value.lower() for word in PLACEHOLDERS) and entropy(value) >= 3.5:
                    matched.append("high_entropy_secret_assignment")
            findings.extend({"file": relative.as_posix(), "line": number, "rule": rule}
                            for rule in sorted(set(matched)))
    return {"status": "passed" if not findings else "failed", "files_scanned": count,
            "findings": findings, "matched_values_included": False,
            "scope": "current_source_tree_and_decompressed_synthetic_fixtures"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--personal-term", action="append", default=[])
    parser.add_argument("--report", help="Write the audit outside the public source tree")
    args = parser.parse_args(argv)
    result = audit(args.root, args.personal_term)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
