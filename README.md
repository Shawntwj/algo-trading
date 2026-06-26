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

### Evaluation UI (Task 7d)

The Main panel is tabbed:

- **Backtest** — per-ticker metrics table with a "?" hover on each metric
  revealing its 95% confidence interval (`/stats`), equity curves, price +
  entry/exit markers.
- **Sweep** — sweep table + Sharpe heatmap when exactly two params are swept.
- **Regimes** — splits the latest single-backtest returns by trend / volatility
  / drawdown regimes via `/regimes/split` (requires `SPY` and `^VIX` to be
  backfilled).
- **Significance** — PSR + DSR (Deflated Sharpe) badge; green when DSR > 0.95.
  Trial count and σ(Sharpe) are derived from the prior sweep when present.
- **Walk-forward** — train/test inputs + `/walkforward` scatter of IS vs OOS
  fold Sharpes with a y=x reference line; warns if decay slope < 0.3.
- **Explain** — enabled only after a `combined_explainable` run (the sidebar
  auto-routes that strategy through `/backtest/explain`). Click any entry /
  exit triangle on the price chart to load that trade's per-child breakdown:
  the plain-English `summary` line as a headline, a bar chart of contributing
  child signals (sorted by `|weight × signal|`, coloured by sign), a
  key-value table of weights / signals / contributions, and a
  "Copy as Markdown" button that yields the same shape as
  `backtest/explainability.py::to_journal` for one trade.

## Quickstart

```bash
make dev   # boots FastAPI on :8000 and Vite on :5173 in parallel
```

Open <http://localhost:5173>. Ctrl-C tears both servers down.

`make test` runs `pytest` + the Vitest frontend smoke test.

> The legacy **Streamlit dashboard** in `dashboard/app.py` is **deprecated** —
> the React SPA above replaces it. Kept for reference; do not extend.

```
algo-trading/
├── config/           tickers, intervals, costs, ClickHouse conn
├── data/             DataSource ABC + YFinance impl, ClickHouse client, backfill, queries
├── orchestration/    Dagster assets/jobs/schedules — thin wrapper over data/
├── strategies/       Strategy ABC + ma_crossover + rsi_mean_reversion
├── backtest/         vectorbt engine, metrics, parameter sweeps
├── dashboard/        Streamlit UI
├── execution/        Broker ABC + IBKR + Alpaca adapters + live_runner + risk/kill
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

## 7. Evaluation gauntlet (`reports/`)

Single-shot Sharpe numbers lie. The evaluation gauntlet runs every strategy
through **headline metrics with bootstrap CIs**, **PSR / DSR significance**,
**per-regime Sharpe**, **walk-forward IS-vs-OOS decay**, and **CAPM attribution
vs SPY** — all rendered into one self-contained dark-theme HTML file you can
share without a network.

```bash
# fast: skip walk-forward, use ClickHouse-backed prices
python -m reports.evaluate ma_crossover \
    --tickers AAPL,MSFT --start 2022-01-01 --end 2023-01-01 \
    --no-walk-forward

# full gauntlet, including walk-forward
python -m reports.evaluate ma_crossover --walk-forward
```

Output lands in `reports/output/<strategy>_<timestamp>.html`. The HTML is
self-contained — every chart is an inline base64 PNG, no CDNs.

### Honest numbers (placeholder — `ma_crossover` baseline)

After Task 4 (combined-explainable strategy) ships, this section will carry
the honest combined-strategy numbers. For now the baseline is `ma_crossover`,
intended as the *placeholder template* — populate by running the gauntlet on
your machine and pasting the values back in.

| Metric                          | Value (95% CI)         | Notes                              |
|---------------------------------|------------------------|------------------------------------|
| Sharpe (annualised)             | `<x.xx>` `[lo, hi]`    | stationary block bootstrap         |
| Total return                    | `<+x.x%>` `[lo, hi]`   |                                    |
| Max drawdown                    | `<-x.x%>` `[lo, hi]`   |                                    |
| PSR (P[true Sharpe > 0])        | `<0.xx>`               | Bailey & López de Prado 2012       |
| DSR (deflated PSR)              | `<0.xx>`               | green when > 0.95                  |
| Per-regime Sharpe — bull / bear | `<x.xx>` / `<x.xx>`    | SPY > 200d SMA split               |
| Per-regime Sharpe — low / mid / high vol | `<x.xx>` / `<x.xx>` / `<x.xx>` | window-relative VIX terciles |
| OOS-fold Sharpe distribution    | `<mean>` `[ci]`        | walk-forward, one point per fold   |
| Decay slope (OOS ~ IS)          | `<x.xx>`               | < 0.3 → overfit warning            |

> **No more headline numbers without CIs.** If a Sharpe shows up in this
> repo without a confidence interval next to it, treat it as suspicious.

### Overfit demo

```bash
python -m reports.overfit_demo
```

Runs a deliberately wide sweep on the in-sample window, promotes the winner,
and re-runs the full gauntlet on the winner. If the strategy was genuinely
overfit to the IS noise, the DSR badge will land **amber or red** (not
green) and the script prints:

```
✅ overfit gauntlet works (DSR < 0.95 — the gate fires)
```

If the sweep happens to find a real signal (e.g. a long-window MA crossover
on a real bull market) the demo will warn instead:

```
⚠️  DSR > 0.95 — sweep wasn't overfit enough. Try a wider grid...
```

— this is *also* expected behaviour. The honest gauntlet doesn't manufacture
overfit signals; it surfaces them when they exist.

## 8. Live trading (paper or real money)

The live runner (`execution/live_runner.py`) is a long-running process that
polls ClickHouse for new bars, asks the configured strategy for a target
position per ticker, diffs against the broker's actual positions, runs every
order through a risk gate, and submits via the configured broker adapter
(IBKR via `ib_insync` or Alpaca via `alpaca-py`). Every decision and every
order is written to two ClickHouse audit tables (`decisions`, `orders`)
created by:

```bash
python -m execution.migrate
```

Config lives at `config/live.yaml` — broker, paper flag, tickers, rebalance
cadence, risk caps, broker-credential **env-var names** (never the secrets
themselves), and the kill-switch flag-file path.

### ⚠ **LIVE TRADING WARNING**

> **This software can place real orders against real money.** Read this
> section twice before flipping `paper: false`. Bugs, network glitches, bad
> bars, stale positions, and overfit strategies all cost real cash in live
> mode. **The author and Claude take zero responsibility for losses.** You
> are the broker of record; you wear every fill.
>
> **Defaults are paper.** The runner refuses to start in live mode unless
> you set `ALGO_LIVE_CONFIRMED=yes` in the environment — this is the BRIEF's
> "never run live orders without my explicit go-live confirmation" rule, enforced.
>
> **Before going live, you must:**
> 1. Run on paper for at least a week, watching the IBKR / Alpaca client
>    portal to confirm orders land as expected.
> 2. Tighten `risk.max_position_usd`, `risk.max_daily_loss_usd`,
>    `risk.max_gross_exposure_usd`, and `risk.max_order_notional_usd` in
>    `config/live.yaml` to amounts you are prepared to lose **today**.
> 3. Know how to flip the kill switch (below) without typing.
> 4. Confirm `ALGO_LIVE_CONFIRMED=yes` is only set in the live shell — never
>    export it from `.zshrc`.

### Getting IBKR paper credentials

1. Open a free **paper trading account** at
   <https://www.interactivebrokers.com/en/trading/free-demo.php>.
   Account approval is automatic for paper.
2. Download **Trader Workstation (TWS)** or **IB Gateway** from
   <https://www.interactivebrokers.com/en/trading/tws.php>. Install and log
   in with your paper credentials.
3. In TWS: `File → Global Configuration → API → Settings`:
   * **Enable ActiveX and Socket Clients** ✓
   * **Read-Only API** ✗ (uncheck — we need to place orders)
   * **Socket port**: `7497` for TWS paper (default; live=7496, Gateway
     paper=4002, Gateway live=4001).
   * **Trusted IP Addresses**: add `127.0.0.1`.
4. Leave TWS running. The live runner talks to it over the socket — no
   IBKR creds touch the runner's environment.

### Getting Alpaca paper credentials

1. Sign up at <https://app.alpaca.markets/signup> (free).
2. Switch to the **Paper** tab in the top-right dashboard.
3. **API Keys → Generate New Key** → copy the key id + secret.
4. Export them in the shell that will run the live runner:
   ```bash
   export ALPACA_API_KEY=PK...
   export ALPACA_SECRET_KEY=...
   ```

### Launching the runner

One-shot acceptance test (the BRIEF's smoke check — runs one iteration and
exits):

```bash
# Paper, IBKR (TWS must be running on 7497)
python -m execution.live_runner --strategy combined_explainable --broker ibkr --paper --once

# Paper, Alpaca (env vars set as above)
python -m execution.live_runner --strategy combined_explainable --broker alpaca --paper --once
```

Long-running poll loop (`rebalance_seconds` from `config/live.yaml`):

```bash
python -m execution.live_runner --strategy combined_explainable --broker alpaca --paper
# Ctrl-C / SIGTERM → graceful shutdown after the in-flight iteration.
```

After a run, inspect the audit tables:

```sql
SELECT decided_at, ticker, target_position, current_position, diff_qty,
       risk_blocked, risk_reason
FROM decisions
ORDER BY decided_at DESC
LIMIT 20;

SELECT submitted_at, ticker, side, qty, broker_order_id, status
FROM orders
ORDER BY submitted_at DESC
LIMIT 20;
```

### Flipping from paper to live

```bash
# 1. In config/live.yaml:
#       paper: false
#       broker: ibkr   # or alpaca
#       # tighten the risk caps!
# 2. (IBKR) point TWS at the live port (7496 / 4001) and log in with a LIVE login.
# 3. In the shell:
export ALGO_LIVE_CONFIRMED=yes
python -m execution.live_runner --strategy combined_explainable --broker ibkr --live
```

If `ALGO_LIVE_CONFIRMED` is missing, the runner prints a loud banner and
exits with code 2. There is no `--force` flag.

### Kill switch

Two sources, OR'd together — either halts the runner:

* **File flag**: presence of the file at `kill_switch_file` (default
  `config/.kill_switch`) halts new orders mid-loop. The runner finishes the
  current iteration (no orders submitted) and writes a sentinel `decisions`
  row per ticker so the halt is auditable.
  ```bash
  touch config/.kill_switch     # halts
  rm    config/.kill_switch     # resumes on the next poll
  ```
  `config/.kill_switch` is in `.gitignore` so it never ends up in a commit.

* **Env var**: `ALGO_KILL=1` (or `yes` / `true` / `on`) in the runner's
  environment. Useful when launching from systemd / launchd. Cleared by
  unsetting the var and restarting.

  ```bash
  # Override the flag-file path too if you like:
  export ALGO_KILL_FLAG_PATH=/var/run/algo-trading/kill
  ```

To stop the runner entirely (rather than just halting orders): SIGINT
(Ctrl-C) or SIGTERM. The loop finishes the current iteration then exits.

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
- **Execution** ships paper + live adapters for IBKR (via `ib_insync`) and
  Alpaca (via `alpaca-py`) behind a single `Broker` ABC, plus a long-running
  `execution/live_runner.py` with risk caps (`RiskGate`), a kill switch
  (env var **and** file flag), and ClickHouse audit tables (`orders`,
  `decisions`). See **§8 Live trading** for the warning, credentials howto,
  and paper-to-live flip.
