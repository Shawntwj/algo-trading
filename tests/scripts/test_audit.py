"""Smoke + JSON-validity tests for ``scripts/audit.py``.

The CLI is meant to be run in any environment (ClickHouse up or down) and
to never raise. These tests assert it returns 0 and produces non-empty
output in both human and JSON modes. JSON output must round-trip through
``json.loads``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT = ROOT / "scripts" / "audit.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_audit_runs_and_emits_human_output() -> None:
    res = _run()
    assert res.returncode == 0, f"stderr={res.stderr!r}"
    out = res.stdout
    assert out.strip(), "expected non-empty stdout"
    # Each top-level section header should appear in the human output.
    for section in (
        "Data (ClickHouse",
        "Strategies",
        "Tests",
        "API endpoints",
        "Frontend",
        "Reports",
        "IMPROVEMENTS.md",
    ):
        assert section in out, f"missing section {section!r} in audit output"


def test_audit_json_is_valid() -> None:
    res = _run("--json")
    assert res.returncode == 0, f"stderr={res.stderr!r}"
    payload = json.loads(res.stdout)  # raises on invalid JSON
    # Spot-check the required top-level keys.
    for key in (
        "generated_at",
        "data",
        "strategies",
        "tests",
        "api",
        "frontend",
        "reports",
        "improvements",
    ):
        assert key in payload, f"missing key {key!r}"
    # The strategies section is independent of any external service — it
    # must always populate.
    assert payload["strategies"]["count"] >= 1
    assert isinstance(payload["strategies"]["names"], list)
    # IMPROVEMENTS.md is a tracked file in this repo; should report >0 entries.
    assert payload["improvements"]["total_entries"] > 0


def test_audit_data_section_degrades_gracefully() -> None:
    """Whether ClickHouse is up or down, the data section must be present
    and shaped predictably (reachable bool + tickers list)."""
    res = _run("--json")
    payload = json.loads(res.stdout)
    data = payload["data"]
    assert "reachable" in data
    assert isinstance(data["reachable"], bool)
    # tickers is always a list (empty when unreachable).
    assert isinstance(data.get("tickers", []), list)
