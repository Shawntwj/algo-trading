# algo-trading — equities signal research platform

A runnable MVP for **discovering trading signals**, iterating on strategies, and
comparing them visually. Polars for data, ClickHouse for storage, Dagster for
orchestration, vectorbt for backtests, and a React + FastAPI research SPA.

## Stack

- **API**: FastAPI + Pydantic v2 (in `api/`) — wraps the vectorbt engine.
- **Frontend**: Vite + React 18 + TypeScript + Tailwind (in `frontend/`).
- **Data fetching / cache**: TanStack Query + axios.
- **Tables**: TanStack Table v8. **Charts**: Recharts.
- **Data**: Polars → ClickHouse (`ReplacingMergeTree`).
- **Orchestration**: Dagster (`workspace.yaml`).
- **Backtests**: vectorbt via `backtest/`.

## Quickstart

```bash
make dev   # boots FastAPI on :8000 and Vite on :5173 in parallel
```

Open <http://localhost:5173>. Ctrl-C tears both servers down.

`make test` runs `pytest` + the Vitest frontend smoke test.

> The legacy **Streamlit dashboard** in `dashboard/app.py` is **deprecated** —
> the React SPA above replaces it. The file is kept for reference until the
> live-runner task ships.

```
algo-trading/
├── config/           tickers, intervals, costs, ClickHouse conn
├── data/             DataSource ABC + YFinance impl, ClickHouse client, backfill, queries
├── orchestration/    Dagster assets/jobs/schedules — thin wrapper over data/
├── strategies/       Strategy ABC + ma_crossover + rsi_mean_reversion
├── backtest/         vectorbt engine, metrics, parameter sweeps
├── dashboard/        Streamlit UI
├── execution/        Paper/live STUB (interface only — not wired)
├── scripts/          CLI entry points
└── tests/            pytest
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Start ClickHouse

```bash
docker compose up -d
# wait for: docker compose ps  (clickhouse should be healthy)
```

ClickHouse HTTP is on `localhost:8123`. Schema (`bars` table, `algo` DB) is
created lazily on first write.

## 3. Backfill data

**Direct (no Dagster):**

```bash
python scripts/backfill.py                            # whole universe in tickers.yaml
python scripts/backfill.py --tickers AAPL,MSFT \
                          --start 2022-01-01 --end 2024-12-31
```

**Via Dagster:**

```bash
dagster dev -w workspace.yaml
# open http://localhost:3000
# Assets → bars → "Materialize all" (one partition per ticker)
```

The Dagster `bars` asset is **ticker-partitioned** — materializing a partition
just calls `data.backfill.backfill_ticker(...)`. Same code, two entry points.

A **daily schedule** (`daily_update_schedule`, 21:30 UTC, weekdays, disabled by
default) calls `data.update_latest(...)` to refresh recent bars. Enable it from
the Dagster UI when you're ready.

## 4. Launch the UI

```bash
make dev   # FastAPI :8000 + Vite :5173
```

Then open <http://localhost:5173>. In the sidebar:
- pick tickers, date range
- pick a strategy
- **Single** mode runs one config; **Sweep** mode runs the cartesian product
  of the values you check per param
- Sharpe heatmap appears when exactly 2 params are swept

The deprecated Streamlit dashboard (`streamlit run dashboard/app.py`) still
works but is no longer maintained.

## 5. Add a new strategy in one file

Drop a file in `strategies/`, e.g. `strategies/my_signal.py`:

```python
import pandas as pd
from .base import Signals, Strategy

class MySignal(Strategy):
    name = "my_signal"

    @classmethod
    def default_params(cls):
        return {"window": 10}

    @classmethod
    def param_grid(cls):
        return {"window": [5, 10, 20, 50]}

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close = data["close"]
        zscore = (close - close.rolling(self.params["window"]).mean()) \
                 / close.rolling(self.params["window"]).std()
        entries = (zscore < -2) & (zscore.shift(1) >= -2)
        exits   = (zscore >  0) & (zscore.shift(1) <=  0)
        return Signals(entries.fillna(False), exits.fillna(False))
```

Register it in `strategies/__init__.py`:

```python
from .my_signal import MySignal
REGISTRY["my_signal"] = MySignal
```

It now appears in the dashboard, in sweeps, and is backtestable from code.
No changes to the engine.

## 6. Tests

```bash
pytest -q
```

Covers the data-layer schema/sort invariants and the backtest metrics.

## Architecture notes

- **Polars in the data layer**; conversion to pandas happens only at the
  vectorbt boundary (`backtest.engine.polars_to_wide`).
- **ClickHouse** uses `ReplacingMergeTree(ingested_at)` keyed on
  `(ticker, interval, timestamp)`, partitioned by `(ticker, month)`. Re-running
  a backfill is safe — `OPTIMIZE TABLE bars FINAL DEDUPLICATE` collapses
  overlap, and queries use `FROM bars FINAL`.
- **DataSource** is an ABC. Add e.g. `PolygonSource` without touching anything
  downstream.
- **Dagster** wraps the plain backfill functions; the platform works without
  it. The schedule is just `update_latest(...)` on a cron.
- **Execution** is a *stub*. No live orders. The interface is there so you can
  plug in a paper broker later.
