"""Walk-forward analysis harness (BRIEF Task 7c).

This is the out-of-sample (OOS) leg of the evaluation gauntlet. A vectorbt
sweep over the full window will always find *some* parameter set with a great
in-sample (IS) Sharpe; walk-forward is how we tell whether that Sharpe survives
when we promote the chosen params to data they never saw.

Conventions:
  * `prices_wide` is the same wide pandas DataFrame consumed by
    `backtest.engine.run_backtest` — `MultiIndex(field, ticker)` columns, with
    at least a `close` field. Row index is timestamps, monotonic increasing.
  * Bar counts are **integer offsets into the row axis** of `prices_wide`. We
    deliberately do NOT mix calendar deltas and integer steps; if a caller
    wants "1y train / 3mo test" they convert to bars themselves (252 / 63).
  * Each fold reuses `backtest.sweep.sweep` on the train slice to pick a
    winner, then runs the chosen config (one combo) on the test slice through
    `run_backtest`. No signal logic is re-implemented here.
  * `FoldResult.test_returns` is the **portfolio-level per-bar return** of the
    chosen config on the test window. For multi-ticker portfolios we average
    across tickers (matching `backtest.metrics.summarize` which takes the
    cross-ticker mean for ratios), which is the same series you'd pass to
    `stats.annualised_sharpe` for an OOS Sharpe.

Cuts (see IMPROVEMENTS.md, Backtest):
  * No purged k-fold (López de Prado AFML §7.4) — overlap between train/test
    windows is handled only by non-overlapping `step = test_size`. For
    overlapping `step < test_size` callers we do NOT purge bars that leak
    information across the split.
  * No combinatorial purged CV — single chronological split per fold.
  * `decay_slope` uses plain OLS, no robust regression — one outlier fold can
    swing it. The point estimate is meant as a diagnostic, not a test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from strategies.base import Strategy

from .engine import BacktestResult, run_backtest
from .stats import annualised_sharpe, bootstrap_ci
from .sweep import sweep


# ─── Config / Result dataclasses ───────────────────────────────────────────
@dataclass
class WalkForwardConfig:
    """Knobs for `walk_forward`.

    Attributes
    ----------
    train_size : int
        Bars per train window (the IS slice the sweep optimises over).
    test_size : int
        Bars per test window (the OOS slice the winning params are evaluated on).
    step : int | None
        Bars to advance between fold starts. Defaults to `test_size` —
        non-overlapping test windows, the standard walk-forward convention.
    mode : Literal["expanding", "rolling"]
        - "expanding": train window grows fold-by-fold; first fold uses
          `train_size` bars, subsequent folds use everything up to the next
          test window's start.
        - "rolling": train window is always exactly `train_size` bars (the
          oldest bars roll off as we step forward).
    min_train : int | None
        Only consulted in "expanding" mode. If set, the first fold's train
        window is clipped to at most `min_train` bars (taken from the tail of
        the available history). Useful when the strategy needs a minimum
        lookback but the caller still wants more train data later folds.
    """

    train_size: int
    test_size: int
    step: int | None = None
    mode: Literal["expanding", "rolling"] = "expanding"
    min_train: int | None = None

    def __post_init__(self) -> None:
        if self.train_size < 2:
            raise ValueError("train_size must be >= 2")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.step is not None and self.step < 1:
            raise ValueError("step must be >= 1")
        if self.mode not in {"expanding", "rolling"}:
            raise ValueError(f"mode must be 'expanding' or 'rolling', got {self.mode!r}")
        if self.min_train is not None and self.min_train < 2:
            raise ValueError("min_train must be >= 2")


@dataclass
class FoldResult:
    """One walk-forward fold.

    `selected_params` is the highest-Sharpe combo on the *train* sweep; the
    same dict was used to instantiate the strategy that produced
    `test_returns` and `out_of_sample_sharpe`. We carry both Sharpes so the
    aggregate decay chart can plot one point per fold.
    """

    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    selected_params: dict[str, Any] = field(default_factory=dict)
    test_returns: np.ndarray = field(default_factory=lambda: np.empty(0))


# ─── helpers ───────────────────────────────────────────────────────────────
def _slice_wide(prices_wide: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Integer-positional slice of a `(field, ticker)`-columned wide frame.

    `end` is exclusive. The MultiIndex column structure is preserved so the
    slice remains a drop-in input to `run_backtest`."""
    return prices_wide.iloc[start:end]


def _portfolio_returns(result: BacktestResult) -> np.ndarray:
    """Per-bar portfolio return as a 1-D numpy array.

    For multi-ticker portfolios we average across columns (equal-weight) so the
    result matches `summarize` semantics and feeds straight into the Sharpe
    helpers in `stats.py`."""
    eq = result.portfolio.value()
    if isinstance(eq, pd.DataFrame):
        # Equal-weight composite equity: mean of per-ticker equity curves.
        eq_series = eq.mean(axis=1)
    else:
        eq_series = eq
    rets = eq_series.pct_change().fillna(0.0).to_numpy(dtype=float)
    return rets


def _best_combo(results: list[BacktestResult]) -> tuple[BacktestResult, float]:
    """Return `(best_result, best_sharpe)` from a sweep, breaking NaNs.

    Falls back to the first result if all Sharpes are NaN — better to evaluate
    *something* OOS than to silently drop the fold."""
    best_idx = 0
    best_sharpe = -np.inf
    found_finite = False
    for i, r in enumerate(results):
        s = r.portfolio.sharpe_ratio()
        if isinstance(s, pd.Series):
            s = float(s.mean())
        else:
            s = float(s)
        if not np.isfinite(s):
            continue
        found_finite = True
        if s > best_sharpe:
            best_sharpe = s
            best_idx = i
    if not found_finite:
        best_sharpe = float("nan")
    return results[best_idx], best_sharpe


# ─── main entry point ──────────────────────────────────────────────────────
def walk_forward(
    strategy_cls: type[Strategy],
    prices_wide: pd.DataFrame,
    param_grid: dict[str, list],
    config: WalkForwardConfig,
    periods_per_year: int = 252,
    backtest_kwargs: dict | None = None,
) -> list[FoldResult]:
    """Run a walk-forward over `prices_wide`.

    For each fold:
      1. Slice the train window (bars `[train_start, train_end)`).
      2. Run `sweep(strategy_cls, train_slice, param_grid)` → list of results.
      3. Pick the highest-Sharpe combo; record its IS Sharpe and params.
      4. Slice the test window (bars `[test_start, test_end)`) and run a
         single backtest with the chosen params.
      5. Record the OOS Sharpe and the OOS per-bar returns.

    Returns one `FoldResult` per fold, in chronological order.
    """
    if not isinstance(prices_wide, pd.DataFrame):
        raise TypeError("prices_wide must be a pandas DataFrame")
    n_bars = len(prices_wide)
    step = config.step if config.step is not None else config.test_size
    backtest_kwargs = dict(backtest_kwargs or {})

    folds: list[FoldResult] = []
    fold_idx = 0
    # First test window starts immediately after the first train window.
    test_start = config.train_size

    while test_start + config.test_size <= n_bars:
        test_end = test_start + config.test_size
        if config.mode == "rolling":
            train_start = test_start - config.train_size
            train_end = test_start
        else:  # expanding
            train_end = test_start
            if config.min_train is not None and fold_idx == 0:
                # Clip the very first fold's train window to `min_train` bars
                # taken from the tail of the available history. Later folds
                # are free to use all available history.
                train_start = max(0, train_end - config.min_train)
            else:
                train_start = 0

        train_slice = _slice_wide(prices_wide, train_start, train_end)
        test_slice = _slice_wide(prices_wide, test_start, test_end)

        # Run the sweep on the train fold. `sweep` filters invalid combos
        # (e.g. ma_crossover fast >= slow) for us.
        sweep_results = sweep(train_slice, strategy_cls, grid=param_grid, **backtest_kwargs)
        if not sweep_results:
            # Skip degenerate folds where every combo was rejected by sweep().
            test_start += step
            fold_idx += 1
            continue

        best, is_sharpe = _best_combo(sweep_results)
        chosen_params = dict(best.params)

        # Evaluate the chosen combo on the OOS slice.
        oos_strat = strategy_cls(**chosen_params)
        oos_result = run_backtest(test_slice, oos_strat, **backtest_kwargs)
        oos_returns = _portfolio_returns(oos_result)
        oos_sharpe = annualised_sharpe(oos_returns, periods_per_year=periods_per_year)

        folds.append(
            FoldResult(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                in_sample_sharpe=float(is_sharpe),
                out_of_sample_sharpe=float(oos_sharpe),
                selected_params=chosen_params,
                test_returns=oos_returns,
            )
        )

        test_start += step
        fold_idx += 1

    return folds


# ─── Aggregation ───────────────────────────────────────────────────────────
def _ols_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Plain OLS slope of `ys ~ a + b * xs`. NaN when undefined."""
    if xs.size < 2:
        return float("nan")
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = float(((xs - x_mean) ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float(((xs - x_mean) * (ys - y_mean)).sum() / denom)


def aggregate_walkforward(
    folds: list[FoldResult],
    periods_per_year: int = 252,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate per-fold results for the OOS decay chart and headline stats.

    Returns
    -------
    dict with keys:
      * `oos_sharpe_distribution`: np.ndarray, one OOS Sharpe per fold
      * `oos_sharpe_mean`: float
      * `oos_sharpe_ci`: (low, high) — stationary-bootstrap CI of the mean OOS
        Sharpe (block_length = sqrt(n_folds), the same rule-of-thumb used
        throughout `stats.py`)
      * `is_vs_oos`: list of (IS, OOS) tuples — raw data for the decay chart
      * `decay_slope`: OLS slope of OOS ~ IS. Closer to 1 → IS Sharpe predicts
        OOS well (healthy); ≪ 1 → the sweep is overfitting to train noise.
    """
    if not folds:
        return {
            "oos_sharpe_distribution": np.empty(0),
            "oos_sharpe_mean": float("nan"),
            "oos_sharpe_ci": (float("nan"), float("nan")),
            "is_vs_oos": [],
            "decay_slope": float("nan"),
            "n_folds": 0,
        }

    oos = np.array([f.out_of_sample_sharpe for f in folds], dtype=float)
    is_ = np.array([f.in_sample_sharpe for f in folds], dtype=float)

    # CI of the mean OOS Sharpe across folds. Stationary bootstrap is overkill
    # for an i.i.d.-ish per-fold series but keeps the library consistent.
    mean_oos = float(np.nanmean(oos))
    finite = oos[np.isfinite(oos)]
    if finite.size >= 2:
        _, lo, hi = bootstrap_ci(
            finite,
            metric_fn=lambda x: float(np.mean(x)),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        )
    else:
        lo, hi = float("nan"), float("nan")

    # Decay slope on the subset where both IS and OOS are finite.
    mask = np.isfinite(is_) & np.isfinite(oos)
    slope = _ols_slope(is_[mask], oos[mask]) if mask.any() else float("nan")

    return {
        "oos_sharpe_distribution": oos,
        "oos_sharpe_mean": mean_oos,
        "oos_sharpe_ci": (lo, hi),
        "is_vs_oos": list(zip(is_.tolist(), oos.tolist())),
        "decay_slope": slope,
        "n_folds": int(len(folds)),
    }
