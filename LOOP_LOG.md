# Loop Log

One entry per task attempt. Append-only.

Format: `## YYYY-MM-DDTHH:MM:SSZ — Task <id> [<status>]`
- Agent ID:
- Summary (1 paragraph):
- Branch / commit:

---

## 2026-06-25T00:48:00Z — Bootstrap [info]
- Orchestrator initialized.
- Repo had no `.git`; ran `git init` locally. No remote configured (push step deferred — see IMPROVEMENTS).
- Plan: split Task 1 across 3 subagents (1a FastAPI backend, 1b Vite scaffold + data fetch, 1c charts + sweep UI). Then 7 → 2 → 3 → 4 → 5 → 6.

## 2026-06-25T09:00:00Z — Task 1a [success]
- Summary: Built `api/` (FastAPI + Pydantic v2) exposing `/health`, `/tickers`, `/strategies`, `/backtest`, `/sweep`. Reuses the existing `backtest` engine, `sweep` harness, and `strategies.REGISTRY` — no logic reimplemented. Services layer keeps HTTP concerns separate; numpy/NaN scrubbing lives in `_jsonable`. CORS opened to `:5173` and `:3000`. Tests under `tests/api/test_endpoints.py` use `TestClient`; data-backed cases run against the local ClickHouse (all 7 pass, 12 total in repo). Requirements appended with fastapi / uvicorn[standard] / pydantic>=2 / httpx.
- Commit: 30b3d9b
- Cut for scope: no tmp-schema fixture for ClickHouse tests (skip-when-unreachable instead); no LRU cache for repeated bar loads in sweep; no streaming/pagination on `/sweep` results; per-ticker metric slice falls back to `None` for vbt scalar returns. All logged in IMPROVEMENTS.md.

## 2026-06-25T09:30:00Z — Task 1b [success]
- Summary: Scaffolded `frontend/` with Vite 5 + React 18 + TypeScript (pinned create-vite@5 because Node 20.11 predates create-vite@9's engines floor). Added TanStack Query, axios, and Tailwind v3. `api/client.ts` wraps `/health`, `/tickers`, `/strategies`, `/backtest`, `/sweep` against `VITE_API_URL` (default `http://localhost:8000`); types in `api/types.ts` hand-mirror `api/schemas.py`. `Sidebar.tsx` fetches tickers + strategies, renders a checkbox multi-select, date pickers, strategy dropdown, an auto-generated params form prefilled from `default_params` (number inputs where the default is numeric), and a Single/Sweep radio (Sweep is a stub for 1c). The Run button POSTs `/backtest`, logs to console, and routes the response into `Main.tsx` which renders it inside a `<pre>`. `npm run build` succeeds (236 KB JS / 8 KB CSS, gzipped 77/2.3 KB); `npm run dev` boots on :5173. README documents quickstart.
- Commit: a38ab9e
- Cut for scope: no Vitest/Playwright smoke test (deferred to 1c); no shared TS schema codegen (hand-maintained types); no charts, metrics grid, or sweep results UI (1c); ticker selector is a flat checkbox list rather than a search combobox; param form has no validation against `param_grid`. All logged in IMPROVEMENTS.md.
