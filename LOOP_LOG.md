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

## 2026-06-25T01:05:27Z — Task 1c [success]
- Summary: Replaced the Task 1b `<pre>` dump with the real research UI. Added `MetricsTable` (TanStack Table v8 — per-ticker Sharpe/return/drawdown/win rate with sortable columns), `EquityChart` (Recharts `LineChart`, one line per ticker on a shared time X-axis), and `PriceChart` (per-ticker line + up/down triangle markers at entry/exit timestamps, with a ticker pill selector when >1 ticker). The Sidebar's Sweep mode now POSTs `/sweep` and renders a `SweepTable` (params + sharpe/return/dd/win/n_trades/exposure) plus `SharpeHeatmap` (HTML-table heatmap with HSL colour ramp) when exactly 2 params are swept — otherwise a "needs exactly 2" note. The sweep grid editor in the sidebar lets the user toggle individual values per `param_grid` key (default = all). Added `scripts/dev.sh` (bash, traps SIGINT, polls because macOS bash lacks `wait -n`) and a root `Makefile` (`make dev` / `make test` / `make build`). Deprecated `dashboard/app.py` with a top-of-file comment block and rewrote the README "Stack" + "Quickstart" + the old `## 4. Launch the dashboard` section. Vitest 2 + jsdom 25 + MSW 2 wired up (downgraded from Vitest 4 because Node 20.11.1 in this env lacks `node:util#styleText`); one smoke test at `frontend/src/__tests__/Sidebar.test.tsx` mounts Sidebar with mocked `/tickers` + `/strategies` and asserts the params form renders — passes. `make dev` was booted end-to-end in the sandbox: both servers come up cleanly (FastAPI on :8000, Vite on :5173) and shut down on SIGTERM. `pytest -q` still 12/12; `npm run build` succeeds (693 KB JS / 209 KB gzip — Recharts is the bulk).
- Commit: <pending>
- Cut for scope: PriceChart plots equity-curve rather than raw close price (no `/price` endpoint exists); heatmap is plain HTML, no interpolation; marker tooltip uses Recharts default (no entry-vs-exit label); no Playwright e2e; one smoke test only (no Run-button / sweep-flow coverage); bundle is one 692 KB chunk (no code-splitting); Vitest pinned to v2 because of Node 20.11 / rolldown engine gap. All logged in IMPROVEMENTS.md.
