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

## Tests
- **API tests skip when ClickHouse is unreachable** — `tests/api/test_endpoints.py` skips `/tickers`, `/backtest`, and `/sweep` tests if `list_tickers()` fails or returns empty, per the BRIEF's no-mock principle. A tmp-schema fixture (insert synthetic bars, run against an isolated DB, drop on teardown) would let these run in CI without a populated `algo.bars`. — M — `tests/api/test_endpoints.py`.
- **No frontend tests yet** — Task 1b deferred Vitest/Playwright. A smoke test covering "Sidebar fetches tickers + strategies, Run button posts to `/backtest`" with MSW mocks would catch regressions cheaply. Add in Task 1c alongside the chart work. — S — `frontend/`.

## Infra / Repo
- **No git remote configured** — repo was bootstrapped with a local `git init`; remote push is deferred. Add a remote (`git remote add origin …`) and switch the loop to push per task. — S — `LOOP_LOG.md` bootstrap entry.
