"""CLI: print a short summary of the platform's current state.

Use this to see drift at a glance: per-ticker row counts in ClickHouse,
the number of strategies registered, test counts, API endpoints, frontend
components, recent reports, and the IMPROVEMENTS.md backlog size.

Usage:
    python scripts/audit.py
    python scripts/audit.py --json   # machine-readable

Designed to finish in under ~10s on a warm laptop (excluding ClickHouse
latency). Test runtime intentionally avoids a full ``pytest`` invocation;
see ``_collect_tests``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------- pretty printing (no colorama dep) ---------------------------------

def _h1(title: str) -> str:
    bar = "=" * max(8, len(title))
    return f"\n{title}\n{bar}"


def _fact(key: str, value: Any) -> str:
    return f"  {key:.<32} {value}"


# ---------- section collectors ------------------------------------------------

@dataclass
class AuditReport:
    generated_at: str
    data: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, Any] = field(default_factory=dict)
    tests: dict[str, Any] = field(default_factory=dict)
    api: dict[str, Any] = field(default_factory=dict)
    frontend: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    improvements: dict[str, Any] = field(default_factory=dict)


def _collect_data() -> dict[str, Any]:
    """Per-ticker rows + last-bar timestamp from ClickHouse.

    If ClickHouse is unreachable we degrade gracefully — the platform is still
    a useful research tool offline (price CSVs etc.).
    """
    out: dict[str, Any] = {"reachable": False, "tickers": [], "total_rows": 0}
    try:
        from data.clickhouse_client import get_client  # local import — heavy
    except Exception as exc:
        out["error"] = f"could not import clickhouse client: {exc}"
        return out

    try:
        client = get_client()
        rows = client.query(
            "SELECT ticker, count() AS n, max(timestamp) AS last_ts "
            "FROM bars GROUP BY ticker ORDER BY n DESC"
        ).result_rows
    except Exception as exc:
        out["error"] = (
            "ClickHouse not reachable — start with "
            "`docker compose up -d clickhouse`. "
            f"({type(exc).__name__}: {exc})"
        )
        return out

    out["reachable"] = True
    out["tickers"] = [
        {
            "ticker": r[0],
            "rows": int(r[1]),
            "last_bar": r[2].isoformat() if hasattr(r[2], "isoformat") else str(r[2]),
        }
        for r in rows
    ]
    out["total_rows"] = sum(t["rows"] for t in out["tickers"])
    return out


def _collect_strategies() -> dict[str, Any]:
    try:
        from strategies import REGISTRY  # noqa: PLC0415
    except Exception as exc:
        return {"count": 0, "names": [], "error": str(exc)}
    names = sorted(REGISTRY.keys())
    return {"count": len(names), "names": names}


def _collect_tests() -> dict[str, Any]:
    """Count test files + pytest-collected tests.

    We use option (b) from BRIEF: ``pytest --collect-only -q`` is fast (<5s
    on this repo) but a full ``pytest -q`` run is in the tens of seconds.
    The audit prints a pointer to run ``pytest -q`` separately for pass rate.
    """
    tests_dir = ROOT / "tests"
    files = sorted(p for p in tests_dir.rglob("test_*.py"))
    out: dict[str, Any] = {
        "test_files": len(files),
        "collected": None,
        "note": "Run `pytest -q` for pass rate; audit avoids the full run.",
    }
    try:
        # Use the venv's python if active, else current interp.
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Last non-empty line of pytest's terse output: "N tests collected".
        # Or with deselects: "N/M tests collected (X deselected)".
        m = re.search(r"(\d+)\s+tests?\s+collected", res.stdout)
        if m:
            out["collected"] = int(m.group(1))
    except Exception as exc:
        out["collect_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _collect_api() -> dict[str, Any]:
    try:
        from api.app import app  # noqa: PLC0415
    except Exception as exc:
        return {"routes": 0, "error": str(exc)}
    # Filter out the auto-mounted /openapi.json, /docs, /redoc, /docs/oauth2-redirect.
    builtins = {"/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"}
    user_routes = [
        {"path": r.path, "methods": sorted(getattr(r, "methods", []) or [])}
        for r in app.routes
        if getattr(r, "path", None) and r.path not in builtins
    ]
    return {"routes": len(user_routes), "paths": user_routes}


def _collect_frontend() -> dict[str, Any]:
    components_dir = ROOT / "frontend" / "src" / "components"
    if not components_dir.exists():
        return {"components": 0, "dist_present": False, "error": "components dir missing"}
    components = sorted(p.name for p in components_dir.glob("*.tsx"))
    dist = ROOT / "frontend" / "dist"
    dist_present = dist.exists() and any(dist.iterdir())
    dist_mtime = None
    if dist_present:
        # mtime of newest file under dist/.
        latest = max((p.stat().st_mtime for p in dist.rglob("*") if p.is_file()), default=None)
        if latest:
            dist_mtime = datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()
    return {
        "components": len(components),
        "component_files": components,
        "dist_present": bool(dist_present),
        "dist_built_at": dist_mtime,
    }


def _collect_reports() -> dict[str, Any]:
    out_dir = ROOT / "reports" / "output"
    if not out_dir.exists():
        return {"html_reports": 0, "latest": None}
    htmls = sorted(out_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = None
    if htmls:
        st = htmls[0].stat()
        latest = {
            "name": htmls[0].name,
            "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        }
    return {"html_reports": len(htmls), "latest": latest}


def _collect_improvements() -> dict[str, Any]:
    path = ROOT / "IMPROVEMENTS.md"
    if not path.exists():
        return {"total_entries": 0, "by_section": {}, "error": "IMPROVEMENTS.md missing"}
    text = path.read_text(encoding="utf-8")
    sections: dict[str, int] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, 0)
        elif current is not None and line.startswith("- **"):
            sections[current] = sections.get(current, 0) + 1
    return {"total_entries": sum(sections.values()), "by_section": sections}


# ---------- main --------------------------------------------------------------

def build_report() -> AuditReport:
    rpt = AuditReport(generated_at=datetime.now(timezone.utc).isoformat())
    rpt.data = _collect_data()
    rpt.strategies = _collect_strategies()
    rpt.tests = _collect_tests()
    rpt.api = _collect_api()
    rpt.frontend = _collect_frontend()
    rpt.reports = _collect_reports()
    rpt.improvements = _collect_improvements()
    return rpt


def render_human(rpt: AuditReport) -> str:
    lines: list[str] = []
    lines.append(f"algo-trading platform audit — {rpt.generated_at}")

    # Data
    lines.append(_h1("Data (ClickHouse `bars`)"))
    d = rpt.data
    if not d.get("reachable"):
        lines.append(f"  {d.get('error', 'unknown error')}")
    else:
        lines.append(_fact("tickers backfilled", len(d["tickers"])))
        lines.append(_fact("total rows", f"{d['total_rows']:,}"))
        top = d["tickers"][:20]
        for t in top:
            lines.append(_fact(t["ticker"], f"{t['rows']:>8,} rows, last={t['last_bar']}"))
        rest = len(d["tickers"]) - len(top)
        if rest > 0:
            lines.append(f"  ...and {rest} more")

    # Strategies
    lines.append(_h1("Strategies"))
    s = rpt.strategies
    lines.append(_fact("registered", s.get("count", 0)))
    for name in s.get("names", []):
        lines.append(f"  - {name}")

    # Tests
    lines.append(_h1("Tests"))
    t = rpt.tests
    lines.append(_fact("test files", t.get("test_files", 0)))
    if t.get("collected") is not None:
        lines.append(_fact("collected tests", t["collected"]))
    else:
        lines.append(_fact("collected tests", "n/a (collect failed)"))
    lines.append(f"  ({t.get('note', '')})")

    # API
    lines.append(_h1("API endpoints"))
    a = rpt.api
    lines.append(_fact("routes", a.get("routes", 0)))
    for r in a.get("paths", []):
        lines.append(f"  - {','.join(r['methods']):<8} {r['path']}")

    # Frontend
    lines.append(_h1("Frontend"))
    f = rpt.frontend
    lines.append(_fact("components (.tsx)", f.get("components", 0)))
    lines.append(_fact("dist/ present", "yes" if f.get("dist_present") else "no"))
    if f.get("dist_built_at"):
        lines.append(_fact("dist last built", f["dist_built_at"]))

    # Reports
    lines.append(_h1("Reports"))
    r = rpt.reports
    lines.append(_fact("HTML reports", r.get("html_reports", 0)))
    if r.get("latest"):
        lines.append(_fact("most recent", f"{r['latest']['name']} @ {r['latest']['modified']}"))

    # Improvements
    lines.append(_h1("IMPROVEMENTS.md backlog"))
    imp = rpt.improvements
    lines.append(_fact("total entries", imp.get("total_entries", 0)))
    for sec, n in imp.get("by_section", {}).items():
        lines.append(_fact(sec, n))

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "audit")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = p.parse_args(argv)

    rpt = build_report()
    if args.json:
        # asdict-style — dataclass + plain dicts, all JSON-safe.
        payload = {
            "generated_at": rpt.generated_at,
            "data": rpt.data,
            "strategies": rpt.strategies,
            "tests": rpt.tests,
            "api": rpt.api,
            "frontend": rpt.frontend,
            "reports": rpt.reports,
            "improvements": rpt.improvements,
        }
        sys.stdout.write(json.dumps(payload, indent=2, default=str))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_human(rpt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
