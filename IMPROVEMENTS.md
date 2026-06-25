# Improvements

Living backlog. Each entry: **Title** — why-it-matters (1 line) — effort (S/M/L) — pointer(s).

Owned by Task 6 once that runs, but populated incrementally by every task.

## Data

## Strategies

## Backtest

## Live

## UX
- **`/backtest` and `/sweep` reload bars per call** — every request hits ClickHouse + `polars_to_wide`. For the same (tickers, range, interval) used during sweep tuning this is wasteful. Add a small LRU cache keyed on the load args, or an explicit `/load` endpoint returning a token reused across runs. — M — `api/services.py::load_wide`.
- **No pagination / size cap on `/sweep`** — large grids (e.g. RSI: 3×4×3 = 36 combos) return a single big JSON blob. Either stream NDJSON or cap result count and expose top-N. — S — `api/app.py::sweep_endpoint`.
- **Per-ticker metric slice is best-effort** — `api/services.py::_ticker_metrics` falls back to `None` when vectorbt returns a scalar instead of a per-ticker Series. Promote this to use `pf[ticker]` sub-portfolios so single-ticker metrics are always populated. — S — `api/services.py`.
- **Frontend TS types are hand-maintained** — `frontend/src/api/types.ts` mirrors `api/schemas.py` by hand. Drift is a real risk once the schemas grow. Add `datamodel-code-generator` (Pydantic → TS via OpenAPI) or `openapi-typescript` against `/openapi.json` and wire it into a `make types` target. — M — `frontend/src/api/types.ts`.
- **Sidebar param form is flat** — Task 1b renders one `<input>` per `default_params` key (number-typed when the default is numeric). No validation against `param_grid` ranges, no support for list/enum params, no help text. Promote to a schema-driven form once the strategy registry exposes richer param metadata. — S — `frontend/src/components/Sidebar.tsx`.
- **Ticker multi-select is a raw checkbox list** — fine for a handful of symbols, but will not scale past ~50. Swap for react-select / combobox with search once the universe grows. — S — `frontend/src/components/Sidebar.tsx`.
- **PriceChart plots equity, not raw price** — `/backtest` returns `equity_curve` per ticker but no underlying close-price series. Entry/exit triangles are placed on the equity line at the nearest equity timestamp. Add a `/price` endpoint (or include `close` in `TickerBacktest`) and switch the line to true close so the markers sit on actual price. — S — `api/services.py::serialize_backtest`, `frontend/src/components/PriceChart.tsx`.
- **Heatmap is plain HTML cells, no interpolation** — `SharpeHeatmap` uses an HSL hue ramp on raw values, no contour smoothing, no axes-aware spacing if grid values are non-uniform. Good enough for 3x3..5x5 grids; for larger grids consider a real Recharts ScatterChart with `Cell` colors and a proper colorbar. — S — `frontend/src/components/SharpeHeatmap.tsx`.
- **Marker tooltip is generic Recharts default** — entry/exit triangles share the line's tooltip; no per-marker label distinguishing entry from exit timestamp. Custom `<Tooltip content=…>` would fix this. — S — `frontend/src/components/PriceChart.tsx`.
- **No e2e coverage (Playwright)** — only one Vitest smoke test (`Sidebar.test.tsx`) covering data loading + params form render. No coverage for Run-button → mutation, table sorting, sweep flow, or chart rendering. Add MSW-backed component tests and a Playwright run-through against `make dev`. — M — `frontend/src/__tests__/`.
- **Dev launcher uses busy-wait polling** — macOS bash 3.2 lacks `wait -n`, so `scripts/dev.sh` polls with `kill -0` every second. Negligible cost, but if a contributor rewrites it in zsh / bash 5 it could go back to `wait -n`. — S — `scripts/dev.sh`.
- **Frontend dep tree pinned older than upstream latest** — Vitest pinned to ^2.1.9 (not v4) and jsdom to ^25 because Node 20.11.1 in this env lacks `node:util#styleText` (needed by rolldown / vitest 4) and ESM-only encoding shims used by jsdom 29. Bumping Node to ≥20.19 unlocks both. — S — `frontend/package.json`.
- **Bundle is one 692 KB chunk** — Vite warns at >500 KB. Recharts + d3 dominate. Code-split with `React.lazy` on the chart components, or `manualChunks` for `recharts` + `@tanstack/*`. — S — `frontend/vite.config.ts`.

## Tests
- **API tests skip when ClickHouse is unreachable** — `tests/api/test_endpoints.py` skips `/tickers`, `/backtest`, and `/sweep` tests if `list_tickers()` fails or returns empty, per the BRIEF's no-mock principle. A tmp-schema fixture (insert synthetic bars, run against an isolated DB, drop on teardown) would let these run in CI without a populated `algo.bars`. — M — `tests/api/test_endpoints.py`.
- **No frontend tests yet** — Task 1b deferred Vitest/Playwright. A smoke test covering "Sidebar fetches tickers + strategies, Run button posts to `/backtest`" with MSW mocks would catch regressions cheaply. Add in Task 1c alongside the chart work. — S — `frontend/`.

## Infra / Repo
- **No git remote configured** — repo was bootstrapped with a local `git init`; remote push is deferred. Add a remote (`git remote add origin …`) and switch the loop to push per task. — S — `LOOP_LOG.md` bootstrap entry.
