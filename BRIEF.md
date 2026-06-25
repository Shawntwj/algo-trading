# Extend the existing algo-trading platform

You are extending an existing project at `/Users/shawnteo/Documents/GitHub/algo-trading`.
**Do not rebuild it.** Read the README and code first to understand what's there.

What exists today:
- ClickHouse `bars` table + YFinance backfill + Polars data layer
- `Strategy` ABC + 2 example strategies (MA crossover, RSI mean reversion)
- vectorbt backtest engine with parameter sweeps
- Streamlit dashboard at `dashboard/app.py`
- Dagster orchestration for ticker-partitioned backfills
- Stub `execution/broker.py` (interface only, no live wiring)

You will extend it with 7 tasks. **Task 1 must land fast. Tasks 2–7 take time — that's
expected.** Commit and push after each task so I can review incrementally. Don't bundle.

---

## Task 1 — Replace Streamlit with a React frontend (fast)

Goal: same UX (pick tickers/strategy/params → run → compare), but as a React SPA backed
by a FastAPI service. Streamlit was for the MVP; React is for the product.

Build:
- `api/` directory: FastAPI app exposing:
  - `GET /tickers` — list tickers in ClickHouse
  - `GET /strategies` — list registered strategies + their `default_params` and `param_grid`
  - `POST /backtest` — body: `{tickers, start, end, interval, strategy, params, commission, slippage}` → returns metrics + equity curve + entry/exit timestamps as JSON
  - `POST /sweep` — body: `{...strategy, grid}` → returns ranked results
  - CORS open to localhost dev
- `frontend/` directory: Vite + React + TypeScript + TanStack Query + Recharts (or Plotly.js)
  - Sidebar: ticker multi-select, date range, strategy dropdown, params (auto-rendered from `/strategies`), Single/Sweep toggle, Run button
  - Main: metrics table, overlaid equity-curve chart, price chart with entry/exit markers, Sharpe heatmap when 2 params are swept
  - Use TanStack Table for the metrics grid
  - Tailwind CSS or shadcn/ui — pick one and be consistent
- Add `frontend/README.md` with dev commands (`npm install`, `npm run dev`)
- Add a `Makefile` or a single `scripts/dev.sh` that starts FastAPI + Vite together
- Update root `README.md` to reflect the new stack
- Keep `dashboard/app.py` (Streamlit) for now but mark it deprecated in a comment

Acceptance: `make dev` (or equivalent) brings up both services, and I can run a backtest
from the React UI that hits the FastAPI backend and renders results.

---

## Task 2 — Survey state-of-the-art systematic strategies from arxiv (long)

Goal: a literature review that drives implementation, not just a doc.

Steps:
1. Search arxiv (q-fin.TR, q-fin.PM, q-fin.ST categories) for papers from the last
   ~3 years on equity trading strategies that report out-of-sample Sharpe.
2. Filter to strategies that are (a) implementable from public price/volume data,
   (b) report a clear, reproducible signal definition, (c) include realistic transaction
   costs in their results.
3. Produce `research/arxiv_survey.md` with: paper title, authors, year, link, signal
   description in plain English, reported Sharpe, and an "implementability" score 1–5.
4. Pick the top 3 by implementability × claimed Sharpe.
5. Implement each as a new file under `strategies/`, following the existing
   `Strategy` contract. Include the citation in a one-line module docstring.
6. Run them through the sweep harness on the existing universe (2018→today) and add
   their results to a new `research/arxiv_results.md` table.
7. Be honest in the writeup about which ones underperformed vs. the paper's claim
   and *why* you think that is (different universe, look-ahead bias in the paper,
   regime change, etc.).

Use WebFetch/WebSearch. Cite every paper. **Do not invent results** — if you can't
reproduce a paper, say so.

---

## Task 3 — Reverse-engineer top public stock pickers (long)

Goal: turn the publicly visible picks of well-known investors into quantifiable signals
that this platform can backtest.

Sources to use (all public, no paywall scraping):
- **SEC EDGAR 13F filings** for institutional managers (Buffett/Berkshire, Burry/Scion,
  Ackman/Pershing Square, Tepper/Appaloosa, etc.). Use the EDGAR JSON endpoints.
- Publicly logged picks where structured data exists (e.g. WhaleWisdom-style summaries
  via free APIs if available, or direct 13F parsing).
- Skip: anything that requires scraping paid newsletters, Discord servers, or X/Twitter
  content gated behind login. Flag what you skipped and why.

For each picker chosen (pick 3–5):
1. Pull their position history.
2. Reverse-engineer the *characteristics* of their picks: market cap bucket, sector,
   value vs growth (P/E, P/B), momentum (12-1), quality (ROE, debt/equity), volatility
   regime, etc. Use yfinance fundamentals or a free alternative.
3. Express each picker as a **factor profile** — a vector showing which characteristics
   their picks over-weight relative to S&P 500.
4. Write `research/pickers.md`: one section per picker with the profile, sample picks,
   notes on style.
5. Implement a `PickerCloneStrategy(picker_name)` under `strategies/` that ranks the
   investable universe by similarity to the picker's factor profile each rebalance
   period and goes long the top-N matches.
6. Backtest it against actually following the 13F (with the realistic ~45-day filing
   delay) and report the gap — that gap is the value of the factor-based signal vs.
   pure copying.

---

## Task 4 — Combine into one explainable strategy (long)

Goal: an ensemble that blends the arxiv signals (Task 2) and the picker-factor signals
(Task 3) into a single tradeable strategy, and — critically — can explain *why* any
given trade fired.

Build:
- `strategies/combined_explainable.py` — an ensemble strategy that:
  - Computes each child signal independently
  - Combines them (start with: weighted sum of normalized signal strengths, with weights
    learned by walk-forward Sharpe maximization on a holdout — no leakage)
  - Emits, alongside each entry/exit, a JSON "explanation" object listing which child
    signals contributed, their weights, and their values at the bar
- `backtest/explainability.py` — utility that turns the explanation stream into a
  human-readable trade journal
- Extend the React UI: when the user clicks a trade marker on the price chart, a panel
  shows the explanation for that trade ("Long because: momentum-12m percentile 92,
  picker-factor similarity 0.81, Bollinger reversion z-score −2.3")
- Walk-forward harness in `backtest/walkforward.py` with rolling train/test windows so
  the reported Sharpe is out-of-sample, not in-sample fitted

Acceptance: I can point at any trade in the UI and read a one-paragraph English
explanation of why the strategy took it.

---

## Task 5 — Live (or paper) trading via a real broker API (long)

Goal: I can flip a switch and the combined strategy from Task 4 starts placing orders
in a paper account, with the option to graduate to live later.

Brokers — important reality check:
- **IBKR** (Interactive Brokers): official API via TWS/IB Gateway, mature, supports
  paper. Recommended as primary. Use `ib_insync`.
- **Alpaca**: simplest official REST API, free paper account, US equities only.
  Recommended as the second adapter (fastest to wire up).
- **Webull**: NO official public API. Community wrappers (`webull-python`) exist but
  are reverse-engineered, fragile, and TOS-risky. **Skip Webull unless I explicitly
  ask for it.** Flag this in the README.

Build:
- Flesh out `execution/broker.py`:
  - Concrete `IBKRBroker` (via `ib_insync`) with `paper=True/False`
  - Concrete `AlpacaBroker`
  - Both implement the existing `Broker` ABC; submit, positions, cash, plus cancel
    and order-status query
- `execution/live_runner.py`: a long-running process that:
  - On each new bar (poll or websocket), runs the combined strategy
  - Diffs target positions vs current, generates orders, submits via the configured broker
  - Logs every decision and order to ClickHouse (new `orders` and `decisions` tables)
  - Honors a configurable kill switch (env var or file flag) that halts new orders
- Risk controls: max position size, max daily loss, max gross exposure — refuse to
  submit orders that would violate them
- `config/live.yaml` for broker creds (from env vars, never committed), risk limits,
  rebalance frequency
- README section: how to get IBKR paper creds, how to launch the live runner, how to
  flip from paper to live, and a **bold warning** about going live

Acceptance: with IBKR paper creds set, `python -m execution.live_runner` connects,
runs the strategy on one bar, places at least one paper order, and I can see it in
the IBKR client portal.

---

## Task 6 — Continuous improvement tracking

Goal: a living backlog of what could be better, surfaced in the repo so I (or future
loops) can pick up where you left off.

Build:
- `IMPROVEMENTS.md` at repo root, organized into sections:
  - **Data**: missing data sources, intraday coverage, corporate actions handling, etc.
  - **Strategies**: ideas you didn't have time to try, papers you skipped and why
  - **Backtest**: things vectorbt can't do well, walk-forward refinements, regime
    detection
  - **Live**: missing risk controls, slippage modeling vs. observed fills, broker gaps
  - **UX**: known frontend gaps
  - **Tests**: what's untested
- Each entry has: title, why-it-matters (1 line), effort estimate (S/M/L), and a
  pointer to the relevant file(s)
- Add a `scripts/audit.py` that prints a short summary of the platform's current
  state (rows per ticker, last backfill date, # strategies registered, # tests,
  test pass rate) so I can run it any time to see drift
- At the end of every other task above, append entries to `IMPROVEMENTS.md` for
  anything you cut for scope

---

## Task 7 — Evaluation infrastructure: are results actually real? (long)

Goal: every Sharpe number this platform reports should come with an honest answer to
"is this skill or luck?" Today the platform reports point estimates. After this task it
reports point estimates **with confidence intervals, regime breakdowns, and
multiple-testing corrections.**

Build:

1. **Benchmarks** — `backtest/benchmarks.py`
   - Buy-and-hold the same universe (equal-weight and cap-weight variants)
   - Buy-and-hold SPY
   - Random-entry strategy with matched exposure (Monte Carlo, ≥500 paths) — gives a
     null distribution to compare against
   - All metrics tables in the UI must show the strategy's numbers *alongside* these
     benchmarks, not in isolation

2. **Regime tagging** — `backtest/regimes.py`
   - Tag every bar with a regime label using public, deterministic rules:
     - **Trend regime**: SPY above/below its 200-day SMA → bull/bear
     - **Volatility regime**: VIX level (low/mid/high terciles) — backfill VIX as
       `^VIX` via the existing data layer
     - **Drawdown regime**: SPY drawdown depth from rolling 1y high (calm / mild /
       severe)
   - Split each backtest's stats by regime. Report per-regime Sharpe, return, max DD,
     and exposure. Strategies that "work" only in one regime should be obvious from
     the table.

3. **Statistical significance** — `backtest/stats.py`
   - **Probabilistic Sharpe Ratio (PSR)**: probability that the true Sharpe is above
     a threshold (default 0). Implement per Bailey & López de Prado (2012).
   - **Deflated Sharpe Ratio (DSR)**: PSR adjusted for the number of trials in a
     sweep. This is the only honest number to report after a parameter sweep —
     in-sample sweep Sharpes are nearly always inflated.
   - **Bootstrap confidence intervals** on Sharpe, total return, and max drawdown
     (block bootstrap, ≥1000 resamples, block length ~ √N).
   - **Reality check / SPA test** (Hansen 2005 or White 2000) for sweeps with many
     configurations — gates whether the best configuration is statistically
     distinguishable from the null benchmark.

4. **Walk-forward, properly** — extend `backtest/walkforward.py` from Task 4
   - Expanding window AND rolling window variants
   - For sweeps: per-fold winner selection, then evaluate the *selected* config on
     the next fold's test data. Report the distribution of test-fold Sharpes — that's
     the honest performance estimate.
   - Output an "OOS decay" chart: in-sample Sharpe vs out-of-sample Sharpe, one point
     per fold. The diagonal is the goal; everything below it is overfitting.

5. **Attribution** — `backtest/attribution.py`
   - Decompose total return into: market beta × SPY return + alpha + residual
   - For the combined-explainable strategy (Task 4), decompose alpha further by child
     signal — which signal actually paid?

6. **UI** — extend the React frontend
   - Every metric in the table grows a small "?" hover showing its 95% CI
   - New "Regime breakdown" tab: per-regime metrics table + a bar chart of Sharpe by
     regime
   - New "Significance" panel: PSR, DSR, p-value vs random-entry null. Color the
     result green only if DSR > 0.95.
   - New "OOS decay" chart on any walk-forward result

7. **Reports** — `reports/` directory
   - `python -m reports.evaluate <strategy>` produces a self-contained HTML report
     (use the same dark-theme styling as RUNBOOK.html) with every chart, table, and
     significance test for that strategy. One file, no JS framework needed, easy to
     share.

Acceptance criteria:
- The combined-explainable strategy from Task 4 is re-evaluated under this
  infrastructure and the README is updated with the honest numbers (PSR, DSR, per-regime
  breakdown, OOS-fold Sharpe distribution). No more headline numbers without CIs.
- A deliberately overfit strategy (e.g. the in-sample sweep winner) is run through the
  same gauntlet and **flagged as not significant** by DSR < 0.95. This proves the
  infrastructure actually catches overfitting rather than rubber-stamping it.
- IMPROVEMENTS.md gets entries for: factor-model attribution (Fama-French), transaction
  cost model from real fills (after Task 5), and any significance tests you skipped.

Citations to use in code docstrings (don't reinvent these — read them):
- Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012) — PSR/DSR
- Hansen, "A Test for Superior Predictive Ability" (2005) — SPA
- Politis & Romano, "The Stationary Bootstrap" (1994) — block bootstrap
- López de Prado, "Advances in Financial Machine Learning" (2018), chapters 11–14

---

## Execution model — one fresh context per task

**Critical:** do not do these tasks yourself in the main conversation. Each task gets
its own fresh agent context to keep the working memory small and focused.

Pattern for every task:

1. The orchestrator (you in the main loop) reads only the task header + acceptance
   criteria for the next task. Don't load all 7 tasks into your context.
2. Spawn a subagent with `Agent` tool, `subagent_type: "general-purpose"` (or
   `"claude"`), and a **self-contained prompt** that includes:
   - The full task section verbatim from `BRIEF.md`
   - "What exists today" summary at the top of `BRIEF.md`
   - The "Working principles" block
   - An explicit instruction: "Commit and push when done. Report back: branch name,
     commit SHA, what you cut for scope, and what to put in IMPROVEMENTS.md."
3. Wait for the subagent to finish. Read **only its summary**, not its full transcript.
4. If it succeeded: append its IMPROVEMENTS.md notes, commit them, schedule the next
   wakeup, and move on.
5. If it failed or got blocked: log the failure to `LOOP_LOG.md` with the agent's
   summary, then decide — retry with a tighter scope, skip and move on, or stop the
   loop and surface the blocker to the user.
6. **Never let one task's context leak into another.** Each task is a clean slate.
   This is the whole point of the per-task context window.

Maintain `LOOP_LOG.md` at the repo root: one entry per task attempt with timestamp,
agent ID, outcome, and the one-paragraph summary. This is your only persistent memory
between iterations — keep it concise.

Long-running tasks (2, 3, 4, 7) may need to be **split across multiple subagent runs**.
That's fine — split them yourself into sub-deliverables, each one its own agent.
Example for Task 7: one agent for benchmarks + regimes, one for significance stats,
one for walk-forward + attribution, one for UI integration, one for the HTML report.
Smaller agents finish faster and don't blow their own context.

If a single task subagent fails twice in a row, stop attempting it and write a
blocker entry in `LOOP_LOG.md`. Move to the next task. Don't burn the loop on one
problem.

## Usage-aware handover (don't get cut off mid-task)

Claude Code runs against a usage cap (rolling ~5-hour window on Max/Pro). A
long-running loop will hit this cap. When it does mid-task, work is lost and the
loop's context dies. Avoid that by **checkpointing and handing off via cron before
the cap hits.**

Rules for the orchestrator at the start of each iteration:

1. **Check usage** before spawning a subagent. In order of preference:
   - Run `ccusage blocks --json 2>/dev/null` and parse `tokensPercent` /
     `costPercent` of the active 5-hour block (community tool — install with
     `npm i -g ccusage` if missing, otherwise fall back).
   - Fallback: track wall-clock time since the *first* loop iteration in this
     window. Anthropic's window is ~5 hours. Assume cap proximity at ~4h 30m
     elapsed.
   - Fallback²: read `~/.claude/projects/.../usage.jsonl` if present and compute
     a rough utilization from token counts.
2. **If estimated usage ≥ 90%**, do NOT start the next task. Instead:
   - Write a `HANDOVER` entry to `LOOP_LOG.md` with: timestamp, last completed
     task ID, next task ID + sub-deliverable, branch name, current commit SHA,
     anything in-flight, and a one-paragraph "where to resume" note for the next
     agent.
   - Use `CronCreate` from the `schedule` skill to fire `<<autonomous-loop>>`
     (the CronCreate sentinel) at a time **after the current 5h window resets**
     — typically 6 hours from now to leave headroom, or read the actual reset
     time from `ccusage` if available and add 15 minutes of slack.
   - Schedule it as a **one-shot cron** (single fire), not recurring, with a
     short description like `algo-trading loop resume after cap reset`.
   - End the loop turn. Do NOT call ScheduleWakeup — that would just hit the
     same cap.
3. **On the next iteration** (whether triggered by cron or by you continuing):
   - First action: read `LOOP_LOG.md` from the bottom. If the last entry is a
     `HANDOVER`, resume from `next task ID + sub-deliverable` and clear the
     handover marker.
   - Re-verify ClickHouse is up before doing anything else
     (`docker compose ps clickhouse`). If not, bring it up.
   - Then proceed with the normal "spawn fresh subagent for the next task" flow.

Rules for the spawned subagents:

- Subagents do not handle handover themselves. If a subagent is doing serious
  work and hits its own usage limits, that's the orchestrator's problem to size
  correctly — split big tasks into smaller subagent runs (Task 7 already calls
  this out).
- Each subagent prompt must include: "If you finish a sub-deliverable and feel
  more work would not fit in remaining capacity, **stop and report**. Do not
  start a second sub-deliverable in the same context."

Heuristic for sizing:

- One subagent run targets ≤ ~20% of a usage window. That gives the orchestrator
  ~4 task slots per window with headroom for handover.
- Task 1 (React frontend) is the only task that may exceed this in a single
  agent — split into (a) FastAPI backend, (b) Vite scaffold + ticker/strategy
  fetch, (c) charts + sweep UI.

Why CronCreate and not ScheduleWakeup: ScheduleWakeup runs in this same Claude
session and shares its cap. CronCreate (autonomous-loop sentinel) spawns a fresh
session at the scheduled time — that's a new usage budget.


## Working principles

- **Read before you write.** Inspect the existing code first. Match its style.
- **Commit per task**, with a clear message. Don't bundle tasks 2–7 into one PR.
- **Tests for everything new.** Every new strategy gets a test that runs it on a
  small synthetic frame and asserts the signal contract holds.
- **No mocking the database** in tests that touch ClickHouse — use a tmp schema.
- **Be honest about what you couldn't do.** If arxiv claims don't reproduce, say so.
  If a broker integration is half-finished, say so. Add it to `IMPROVEMENTS.md`.
- **Walk-forward, not in-sample.** After Task 7 ships, no Sharpe number gets reported
  without a confidence interval and a regime breakdown — including in the README.
- **Never run live orders without my explicit "go live" confirmation.** Paper only
  unless I say otherwise, in writing, in chat.

Start by reading the existing repo, then post a short plan for Task 1 and begin. After
Task 1 ships, propose the order and granularity for Tasks 2–7 before diving in.

A reasonable default ordering for 2–7 (you can revise after Task 1):
**7 → 2 → 3 → 4 → 5 → 6 last**

Reasoning: build the evaluation gauntlet (Task 7) *before* you implement new strategies
so that every new signal from Tasks 2–4 gets evaluated honestly from day one, instead
of being added with inflated in-sample numbers that have to be redone later.
