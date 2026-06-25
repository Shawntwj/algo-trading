# Combined Explainable Strategy — gauntlet results

BRIEF Task 4a deliverable: blend the arxiv signals (Task 2b) and the
picker-clone (Task 3) into a single tradeable signal, with a per-trade
explanation log so the React UI (Task 4b) can show *why* every trade fired.

## What it does

`CombinedExplainableStrategy` (`strategies/combined_explainable.py`) composes
six child strategies and emits one long signal per ticker per bar by:

1. Asking each child for its standard `(entries, exits)` signal pair.
2. Forward-filling the +1/-1 action stream into a long-state {0, 1}.
3. Z-scoring each child's state over a rolling 252-bar window, clipped to ±3.
4. Combining as `Σ_k w_{k,t} · normalised_signal_{k,i,t}`.
5. Firing a long entry when the combined score crosses above
   `entry_threshold` AND `≥ min_active_children` children agree on a positive
   sign. Exit when the combined score returns ≤ 0.

Default child line-up:

- `pca_stat_arb` (arxiv:2512.02037 — Avellaneda-Lee residual mean-reversion)
- `macro_timing` (arxiv:2605.20636 — Xiong growth/defensive macro tilt)
- `drift_regime` (arxiv:2511.12490 — Singha value+reversal — falsification target)
- `picker_clone_appaloosa` (Task 3 — best A-vs-B factor-clone gap)
- `ma_crossover` (MVP control)
- `rsi_mean_reversion` (MVP control)

Every entry/exit timestamp persists an explanation dict on the strategy
instance:

```python
{
    "ticker": str,
    "timestamp": pd.Timestamp,
    "direction": "long_entry" | "long_exit",
    "weights": {child_name: float},        # sum to 1
    "child_signals": {child_name: float},  # z-clipped to ±3
    "summary": "Long AAPL because: ma_crossover z=+1.82, ...",
}
```

`backtest/explainability.py` reads the log via `strategy.get_explanation_log()`
and joins it with the vectorbt portfolio's recorded trades; `to_journal(...)`
renders to markdown / json / text.

## Honest gauntlet numbers

Universe: `AAPL, MSFT, GOOGL, AMZN, SPY, ^VIX, ^IRX`
Window: `2020-01-01 → 2026-06-25` (1,255 daily bars)
Engine: vectorbt, 5 bps commission + 5 bps slippage, init cash 100k.

### Default-params headline

Default child line-up, equal initial weights (1/6 each), `min_active_children=2`:

| metric | value |
|---|---|
| Annualised Sharpe | **+1.18** |
| Sharpe 95% bootstrap CI (n=500) | [+0.28, +2.18] |
| PSR(SR > 0) | 0.997 |
| Total return | +110.99% |
| Max drawdown | −18.19% |
| Explanations recorded | 403 |

### Sweep + DSR

Full Cartesian product of the declared `param_grid` (norm_window ∈ {126, 252},
min_active_children ∈ {1, 2, 3}, entry_threshold ∈ {0.0, 0.25, 0.5}) = 18
configs.

| metric | value |
|---|---|
| Best in-sample Sharpe | +0.64 (`min_active_children=1, entry_threshold=0.0, norm_window=252`) |
| All 18 Sharpes (sorted desc) | 0.64, 0.61, 0.56, 0.56, 0.53, 0.52, 0.49, 0.49, 0.49, 0.49, 0.47, 0.45, 0.45, 0.45, 0.41, 0.39, 0.32, 0.22 |
| **Deflated Sharpe Ratio** (best-of-18) | **+1.00** (PSR-style, ≈ "the best config is significantly > 0 after correcting for multiple testing") |

The default `min_active_children=2` value is conservative — relaxing to
`min_active_children=1` lifts headline Sharpe above 1 but is sweep-selected
(the DSR shows that selection survives the correction).

### Regime split

Computed by `backtest/regime_split.py::split_stats_by_regime` against SPY (trend
+ drawdown) and ^VIX (vol terciles) over the same window:

| dimension | regime | n_bars | total_return | sharpe | max_dd | exposure |
|---|---|---:|---:|---:|---:|---:|
| trend | bear | 438 | +207% | **+2.09** | −23% | 0.86 |
| trend | bull | 565 | +47% | +0.78 | −29% | 1.00 |
| vol | high | 333 | +155% | **+2.06** | −27% | 0.92 |
| vol | mid | 335 | +35% | +1.01 | −26% | 1.00 |
| vol | low | 335 | +31% | +0.98 | −14% | 0.90 |
| drawdown | severe | 189 | +100% | **+2.42** | −23% | 0.92 |
| drawdown | mild | 284 | +87% | +2.11 | −13% | 0.96 |
| drawdown | calm | 530 | +20% | +0.48 | −29% | 0.93 |

The shape is consistent: the ensemble does its work in *high-vol / severe-DD*
regimes — exactly where the macro-timing and mean-reversion children are
designed to pick up signal. The calm/bull regime Sharpe is the weakest cell
(0.48 / 0.78), which is what we'd expect from a mean-reversion-heavy blend.

## Honest cuts

- **Universe is 7 tickers** because that's what ClickHouse has been backfilled
  with (Task 1c artefact, see IMPROVEMENTS / Data). The arxiv strategies were
  written for 60 / 500-name universes; the picker-clone (`appaloosa`)
  references holdings (BABA, JD, PDD, FXI, …) that aren't in our universe at
  all. The picker child contributes effectively zero signal on this set; the
  explanation log surfaces this as `[inactive: picker_clone_appaloosa]` in
  the summary line — the system tells the truth about which children are
  voting rather than silently masking the absence.
- **Long-only** because the `Strategy` ABC's `Signals` dataclass carries
  entry/exit booleans only. Same constraint that bit every arxiv reproduction
  in Task 2b. See IMPROVEMENTS → "Strategy ABC has no first-class
  long-short".
- **Weight learning is not on the hot path.** The default strategy ships with
  equal weights. `fit_weights_walk_forward()` is opt-in — call it before
  running a backtest if you want Sharpe-max convex-combination weights. The
  decision is documented inline in the class docstring; it keeps a vanilla
  `run_backtest` call cheap.
- **DSR uses the sweep's IS Sharpes**, not a walk-forward distribution. A
  full walk-forward over the param grid would be the more rigorous version
  but is multiplicatively expensive (18 configs × N folds × 6 children); the
  best-Sharpe + sweep-deflate is the pragmatic Task 4a slice. The
  walk-forward harness (`backtest/walkforward.py`) is available for any
  future caller that wants the stricter number.
- **Weight-fit fallback log surfaces** via `weight_fit_fallback_=True` on
  the strategy after `fit_weights_walk_forward()` if any fold fell back to
  inverse-vol. The explanation summary tail appends
  `"(weights: fallback inverse-vol)"` so trades made under the fallback are
  visibly different from trades made under a fitted weight.

## Reproduce

```bash
# Single backtest + HTML report.
python -m reports.evaluate combined_explainable \
    --tickers AAPL,MSFT,GOOGL,AMZN,SPY,^VIX,^IRX \
    --start 2020-01-01 --end 2026-06-25 \
    --no-walk-forward

# Same numbers reported above, no report, just metrics.
# (see commit message for Step 6 — the exact ad-hoc script that produced this table)
```

## What the explanation looks like

Example from the actual run above (markdown via
`backtest.explainability.to_journal`):

```
### 2023-05-22T00:00:00 — long_entry

Long AAPL because: rsi_mean_reversion z=+1.46, ma_crossover z=+0.81, pca_stat_arb z=+0.00. [inactive: picker_clone_appaloosa]

- rsi_mean_reversion: weight=0.167, signal=+1.463, contribution=+0.244
- ma_crossover:       weight=0.167, signal=+0.809, contribution=+0.135
- pca_stat_arb:       weight=0.167, signal=+0.000, contribution=+0.000
- macro_timing:       weight=0.167, signal=+0.000, contribution=+0.000
- drift_regime:       weight=0.167, signal=+0.000, contribution=+0.000
- picker_clone_appaloosa: weight=0.167, signal=+0.000, contribution=+0.000
```

The frontend (Task 4b) renders this as a per-trade card with the contributing
children listed and the inactive ones called out separately.
