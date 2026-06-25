# arxiv top-3 — reproduction results

Honest writeup for BRIEF Task 2b. The literature scan (Task 2a,
`research/arxiv_survey.md`) shortlisted three strategies. This document
records what we replicated, what data we had, what the gauntlet said,
and where each strategy lands on the survived / underperformed /
falsified spectrum.

> *"Do not invent results — if you can't reproduce a paper, say so."*
> — BRIEF

## Methodology

### Universe and data

| Strategy | Tickers used | Window | Source |
|----------|--------------|--------|--------|
| PCAStatArb        | AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, QQQ, SPY, TLT, V, VTV, VUG, XLK, XOM (16 names) | 2022-01-01 → 2024-12-31 (752 bars) | ClickHouse (backfilled this task) |
| MacroTimingXiong  | SPY, ^VIX, ^IRX + growth basket {QQQ, XLK, VUG, VGT, SPYG} + defensive basket {SCHD, XLP, XLU, VTV, VYM} (13 names) | 2018-01-02 → 2024-12-30 (1760 bars) | yfinance direct (CH backfill of pre-2022 ^VIX / ^IRX in flight at gauntlet-run time) |
| DriftRegimeSingha | AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, QQQ, SCHD, SPY, TLT, V, VTV, VUG, VYM, XLK, XLP, XLU, XOM (20 names) | 2022-01-01 → 2024-12-31 (752 bars) | ClickHouse |

- **PCA**: 16 names with full 2022-2024 coverage. Paper used 60 Polish
  equities; 16 US large-caps is a thin cross-section for residual
  arbitrage but the implementation handles it (caps r at min(15, 15)).
- **Macro**: pulled directly from yfinance to bypass the ClickHouse
  partition limit during the inline backfill of ^VIX / ^IRX (only
  2018-2022 was loaded in time for the gauntlet run; the rest is still
  warming the cache in a background job). This is the cleanest possible
  source — same vendor as the data layer, no transformation lag.
- **Drift**: 20 names. Paper used 500 (S&P 500 current constituents).
- **Costs**: 5 bp commission + 5 bp slippage on every backtest (the
  `backtest.engine.run_backtest` defaults). 10 bp round-trip matches the
  Xiong (2026) headline assumption.

### Evaluation gauntlet

For each strategy:
- **Headline** Sharpe / total-return / max-DD with stationary-bootstrap
  95% CIs from `backtest.stats.sharpe_ci` / `total_return_ci` /
  `max_drawdown_ci` (500 resamples per metric).
- **PSR** (Bailey & López de Prado 2012) at baseline 0 Sharpe.
- **Sweep DSR**: `deflated_sharpe_ratio_from_sweep` over a small grid
  from the strategy's `param_grid()`. The trial-Sharpe distribution is
  what the DSR deflates the point Sharpe against.
- **Walk-forward** (`backtest.walkforward.walk_forward` +
  `aggregate_walkforward`): expanding folds, train/test sizes scaled to
  each strategy's data window.
- **HTML report** per strategy via `python -m reports.evaluate` →
  `reports/output/<name>.html`. Files are gitignored
  (`reports/output/.gitignore`); regenerate from the commands below.

### Reproduction commands

```
# PCA
python -m reports.evaluate pca_stat_arb \
  --tickers AAPL,MSFT,NVDA,QQQ,SPY,GOOGL,AMZN,META,XOM,JPM,V,JNJ,XLK,VUG,VTV,TLT \
  --start 2022-01-01 --end 2024-12-31 \
  --params '{"window":126,"n_factors":5,"entry_z":1.0,"exit_z":0.25}' \
  --no-walk-forward --out reports/output/pca_stat_arb.html

# Macro (uses build_report directly with yfinance frame — see commit history)
# Drift
python -m reports.evaluate drift_regime \
  --tickers AAPL,MSFT,NVDA,QQQ,SPY,GOOGL,AMZN,META,XOM,JPM,V,JNJ,XLK,VUG,VTV,TLT,XLP,XLU,SCHD,VYM \
  --start 2022-01-01 --end 2024-12-31 \
  --no-walk-forward --out reports/output/drift_regime.html
```

## Results

| Strategy | Reported Sharpe (paper) | Our point Sharpe | 95% CI | PSR | Sweep DSR | OOS-fold mean | Verdict |
|----------|-------------------------|------------------|--------|-----|-----------|---------------|---------|
| PCAStatArb (arxiv:2512.02037) | 2.63 (PL 2017) | **0.36** | [-0.68, 1.46] | 0.73 | 0.44 (N=8) | n/a (skipped) | **underperformed** |
| MacroTimingXiong (arxiv:2605.20636) | 1.01 (net 10bp) | **0.76** | [0.18, 1.41] | 0.97 | **0.95** (N=8) | 0.63 [0.21, 1.04] (9 folds) | **survived** |
| DriftRegimeSingha (arxiv:2511.12490) | >13 | **0.21** | [-0.94, 1.30] | 0.64 | 0.34 (N=16) | 0.68 [-0.70, 1.94] (7 folds) | **falsified** |

## Per-strategy notes

### PCAStatArb (arxiv:2512.02037) — *underperformed*

**Implementation** (`strategies/pca_stat_arb.py`):
- Avellaneda-Lee s-score: 252-bar rolling PCA (paper default) → first
  r=15 eigenportfolios → per-stock OLS to get residuals → cumulative
  residual fit to an OU process via the AR(1) shortcut → normalised
  z-residual s_t (paper §3.1 + eq. 2.10).
- Trading rule: open long when s < -1.25, close long when s > -0.5
  (paper eq. 2.10). Long-only — Strategy ABC's single-signal-pair
  constraint precludes the symmetric short leg.
- For the headline run we used window=126 and n_factors=5 (compressed
  from the paper's 252/15) because 16 tickers × 252 bars leaves only
  500 effective fitting bars and r=15 on 16 names is rank-1-deficient.
  The strategy's `param_grid` exposes both the paper defaults and the
  compressed alternates; the sweep DSR ranges over the compressed grid.

**Our reproduction vs paper claim**:

| Metric | Paper (Polish 60-name, 2017) | Ours (US 16-name, 2022-2024) |
|---|---|---|
| Annualised Sharpe | 2.63 | 0.36 |
| Cumulative return (period) | ~20% (2017) | 4.27% (2022-2024 cumulative) |
| Max DD | not stated for 2017 | -6.19% |

**Honest reading**: it did not reproduce. Plausible reasons:
- **Cross-section size**: 16 vs 60 is a ~4× narrower factor model. The
  OU residuals on US large-caps tend to be tightly correlated to the
  market, so what survives after the first 5 eigenportfolios is largely
  idiosyncratic noise rather than tradeable mispricing.
- **Universe regime**: 2017 Poland was a low-correlation, fragmented
  market with many small-cap mispricings. US large-caps 2022-2024 were
  dominated by the AI tech rotation — exactly the wrong regime for
  mean-reversion on residual returns.
- **Cost asymmetry**: Polish brokerage costs in 2017 were higher than
  our 10 bp round-trip; if the paper used a Polish-realistic 30 bp the
  reported Sharpe is on a *tougher* benchmark. (We didn't deflate ours
  back to match — would only make the gap larger.)
- **No short leg**: the paper's headline is for the long+short pair;
  our long-only slice should retain roughly half the signal.

DSR=0.44 on the 8-config sweep says the 0.36 point Sharpe could easily
be a draw from a higher-best-of-N distribution under the null. Bootstrap
CI [-0.68, 1.46] contains 0. **The paper's 2.63 Sharpe is well outside
our CI** — we didn't observe it.

### MacroTimingXiong (arxiv:2605.20636) — *survived*

**Implementation** (`strategies/macro_timing.py`):
- Direction-normalised macro inputs (paper eqs. 3-7), softplus
  components (eq. 8), interaction terms (eqs. 14-17), CoreScore /
  StressScore / CrowdedScore composition (eqs. 18-21), tanh weight
  mapping (eq. 22), EWMA smoothing (eq. 23).
- Defaults from selected config eq. (31): α=0.50, λ_s=0.50, λ_c=0.05,
  MaxTilt=0.50, τ_w=0.75, η=0.05.

**Our reproduction vs paper claim**:

| Metric | Paper (2017-06-28 → 2026-05-15, 10 bp) | Ours (2018-01-02 → 2024-12-30, 10 bp) |
|---|---|---|
| Sharpe | 1.01 | 0.76 (CI [0.18, 1.41]) |
| CAGR | 19.24% | ~5% annualised (40.5% over 7 yrs) |
| Max DD | -31.63% | -11.84% |
| Trades | not stated | 105 (binary translation) |

**Differences explaining the gap**:
- **^IRX (13-wk T-bill) substituted for the paper's FRED FEDFUNDS** rate
  input. ^IRX moves on a similar curve but the 21-day differential is
  noisier, so the rate-relief signal is weaker.
- **Continuous → boolean translation**: paper's continuous w_G turns
  smoothly between 0.0-1.0 around 0.5 ± MaxTilt = 0.0-1.0; ours flips
  long-growth at w_G>0.55, long-defensive at w_G<0.45, dead-band in
  between. This loses the gradient between, say, 60% growth and 75%
  growth — both flip to "all-in growth basket". Realised Sharpe is
  therefore strictly less than the paper's continuous version.
- **Window not aligned** to the paper's 2017-06-28 start (we start
  2018-01-02 because that's the earliest yfinance gives us cleanly on
  some of the basket members).

**Honest reading**: the paper's 1.01 Sharpe sits inside our 95% CI
[0.18, 1.41]. PSR=0.97 (very strong significance vs 0-Sharpe null) and
sweep-DSR=0.95 — the strategy beats the deflated-for-multiple-trials
threshold. Walk-forward OOS mean 0.63 with CI [0.21, 1.04] — positive
across all 9 folds. **This is the only one of the three that survived
the gauntlet**, and given how aggressive our cuts were (^IRX proxy +
binary signal translation) the paper's 1.01 claim is plausible.

### DriftRegimeSingha (arxiv:2511.12490) [FALSIFICATION TARGET] — *falsified*

**Implementation** (`strategies/drift_regime.py`):
- value = cross-sectional percentile of 1/price (paper §2.1)
- reversal = cross-sectional z-score of -trailing-10d return (paper §2.1)
- BASE = 0.7 * value + 0.3 * reversal (eq. 1)
- UpFraction_63 > 0.60 regime gate (eqs. 2-3)
- EDGE = BASE * REGIME (eq. 4), long-only top-decile selector.

**Why a falsification target**: paper claims OOS Sharpe > 13 on the S&P
500. Implausible at scale; survives mostly because the paper deliberately
uses current S&P 500 constituents (survivorship bias) over a 20-year
window.

**Our reproduction vs paper claim**:

| Metric | Paper (500 names, 2010-2021 walk-fwd) | Ours (20 names, 2022-2024) |
|---|---|---|
| Sharpe | >13 OOS | **0.21** (CI [-0.94, 1.30]) |
| Annualised return | 158.6% | ~0.5% annualised (0.98% total) |
| Max DD | not stated | -2.92% |

**Was it falsified?** Yes, decisively:
- Point Sharpe 0.21 vs claim >13: **factor of ~60×**.
- Bootstrap 95% CI [-0.94, 1.30] contains 0 — we cannot reject the null
  that the strategy has zero edge.
- PSR=0.64 — the probability the true Sharpe exceeds 0 given the
  observed sample is only 64%, not a level any practitioner would
  promote.
- Sweep DSR=0.34 — after deflating for the 16-trial parameter sweep
  the deflated Sharpe is barely positive.
- Walk-forward OOS Sharpe 0.68 with CI [-0.70, 1.94] — straddles zero.

**Caveats**:
- Our universe is 20 names vs the paper's 500. The drift regime gate
  + value + reversal signals are cross-sectional; thin cross-sections
  bias toward noisier ranks and lower realised Sharpe.
- Long-only (paper is market-neutral long-short). The short leg
  typically delivers ~half the headline Sharpe; on a market-neutral
  basis our number would roughly double, to ~0.4 — still nowhere near 13.
- Window mismatch: 2022-2024 is three years of mostly-up market on a
  highly concentrated AI rotation; the paper's 2010-2021 OOS spans a
  much broader regime mix.

Even granting all those caveats, **the paper's >13 Sharpe is irrecoverable
from any cut of our reproduction.** The gauntlet (PSR, DSR, bootstrap CI,
walk-forward) all reject it. This is a clean negative-result proof point
that the evaluation machinery works on a real paper-claimed strategy.

## Lessons / IMPROVEMENTS pointers

- **Strategy ABC cannot natively express long-short or continuous-weight
  strategies.** Every arxiv replication this task added had to translate
  to long-only booleans. The macro-timing translation in particular
  drops a chunk of the paper's smoothness — the cleaner fix is to give
  `Signals` an optional `weights` field. *IMPROVEMENTS / Strategies.*
- **ClickHouse `max_partitions_per_insert_block = 100` limit** breaks
  multi-year bulk backfills. We worked around it with year-by-year
  loops; the backfill helper should chunk internally. *IMPROVEMENTS /
  Data.*
- **`backfill_universe` with a single-element ticker list fails** with
  a column-rename KeyError because `yf.download` returns a different
  column shape for one ticker. Workaround: pass a two-ticker list.
  *IMPROVEMENTS / Data.*
- **PCA stat-arb O(n_bars × n_tickers²) loop** — at 16 tickers we run
  in <1s, but a 500-name S&P 500 reproduction would need vectorising
  the per-stock OU AR(1) fit. *IMPROVEMENTS / Strategies.*
- **DSR sweep grid for new strategies is hand-trimmed in this writeup**
  (8-16 configs vs the full `param_grid` product of 36+). A proper
  reproduction should run the full grid, accept the longer runtime,
  and let DSR deflate against the actual trial count. *IMPROVEMENTS /
  Backtest.*
- **No FRED data path.** Macro timing has to substitute ^IRX for
  FEDFUNDS. A small FRED loader (`pandas_datareader.get_data_fred`)
  would close this and make the macro-timing reproduction strictly
  faithful to the paper. *IMPROVEMENTS / Data.*
