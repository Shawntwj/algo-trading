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


# ─── OOS weight learner (Task 4a) ──────────────────────────────────────────
def _inverse_vol_weights(child_returns: dict[str, np.ndarray]) -> dict[str, float]:
    """Inverse-volatility weights, normalised to sum to 1.

    Used as the fallback when ``scipy.optimize.minimize`` fails to converge
    on the Sharpe-max convex problem (per BRIEF Task 4a spec). A child whose
    return series is all-zero (e.g. the strategy never traded on that fold)
    is given zero weight.
    """
    names = list(child_returns.keys())
    vols = np.array([float(np.std(child_returns[n], ddof=1)) for n in names])
    # Zero-vol children get zero weight; non-zero share the rest by 1/σ.
    inv = np.where(vols > 0, 1.0 / np.maximum(vols, 1e-12), 0.0)
    s = float(inv.sum())
    if s <= 0:
        # All children are dead; degenerate uniform fallback.
        return {n: 1.0 / len(names) for n in names}
    return {n: float(v / s) for n, v in zip(names, inv)}


def fit_sharpe_max_weights(
    child_returns: dict[str, np.ndarray],
    periods_per_year: int = 252,
) -> tuple[dict[str, float], bool]:
    """Fit convex-combination Sharpe-maximising weights.

    Solves: maximise Sharpe(Σ w_i * r_i)  s.t.  Σ w_i = 1, w_i ≥ 0.

    Returns
    -------
    (weights, used_fallback) : the per-child weight dict (sums to 1) and a
        flag set to True when scipy.optimize failed to converge and we fell
        back to inverse-vol weights. The caller (strategy) reads the flag
        and surfaces it in the explanation summary.
    """
    from scipy.optimize import minimize  # local import (heavy, only on fit)

    names = list(child_returns.keys())
    n = len(names)
    if n == 0:
        raise ValueError("child_returns must contain at least one child")
    # Stack into (T, n) matrix; reject if any series has different length.
    lengths = {n_: child_returns[n_].size for n_ in names}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"child_returns series have mismatched lengths: {lengths}"
        )
    R = np.column_stack([child_returns[n_] for n_ in names]).astype(float)

    # Reject pathological all-zero / all-NaN frames immediately → fallback.
    if not np.isfinite(R).all() or float(np.std(R)) <= 0:
        return _inverse_vol_weights(child_returns), True

    def neg_sharpe(w: np.ndarray) -> float:
        port = R @ w
        mu = port.mean()
        sd = port.std(ddof=1)
        if not np.isfinite(sd) or sd <= 1e-12:
            return 1e6  # huge penalty — scipy treats this as a bad point
        sr = (mu / sd) * np.sqrt(periods_per_year)
        return -float(sr)

    # x0 = equal weights; bounds [0, 1]; equality constraint sum(w) = 1.
    x0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    try:
        result = minimize(
            neg_sharpe,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-9},
        )
    except Exception:
        return _inverse_vol_weights(child_returns), True
    if not result.success or not np.all(np.isfinite(result.x)):
        return _inverse_vol_weights(child_returns), True
    w = np.clip(result.x, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return _inverse_vol_weights(child_returns), True
    w = w / s
    return ({n_: float(w[i]) for i, n_ in enumerate(names)}, False)


def fit_combined_weights_walk_forward(
    strategy,
    prices_wide: pd.DataFrame,
    config: WalkForwardConfig,
    periods_per_year: int = 252,
    backtest_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Walk-forward weight learner for a CombinedExplainableStrategy.

    For each fold:
      1. Slice train and test windows per ``config`` (reusing the same
         expanding/rolling/min_train semantics as ``walk_forward``).
      2. For each child strategy, run a single backtest on the **train**
         slice with the strategy's own default params and collect the per-
         bar portfolio returns. (One backtest per child, not a sweep — the
         child's parameters are frozen; we're only fitting the *weights*.)
      3. Fit Sharpe-max convex-combination weights on the train returns
         (with inverse-vol fallback).
      4. Hold those weights and evaluate the combined strategy on the test
         slice. Record per-fold OOS Sharpe + the chosen weights.

    Writes the final fold's weights back onto ``strategy.weights`` so the
    caller can use the strategy directly after the call, and returns a
    summary dict mirroring ``aggregate_walkforward`` for the OOS Sharpe.
    """
    if not hasattr(strategy, "children") or not hasattr(strategy, "weights"):
        raise TypeError(
            "fit_combined_weights_walk_forward needs a CombinedExplainableStrategy-shaped instance"
        )

    n_bars = len(prices_wide)
    step = config.step if config.step is not None else config.test_size
    backtest_kwargs = dict(backtest_kwargs or {})

    folds_out: list[dict[str, Any]] = []
    fold_idx = 0
    test_start = config.train_size
    any_fallback = False
    last_weights: dict[str, float] | None = None

    while test_start + config.test_size <= n_bars:
        test_end = test_start + config.test_size
        if config.mode == "rolling":
            train_start = test_start - config.train_size
            train_end = test_start
        else:
            train_end = test_start
            if config.min_train is not None and fold_idx == 0:
                train_start = max(0, train_end - config.min_train)
            else:
                train_start = 0

        train_slice = _slice_wide(prices_wide, train_start, train_end)
        test_slice = _slice_wide(prices_wide, test_start, test_end)

        # Per-child train-window return series. We instantiate each child
        # via the same path the strategy uses so picker / macro init args
        # carry through.
        children = strategy._instantiate_children()  # noqa: SLF001 — internal helper, documented
        child_returns: dict[str, np.ndarray] = {}
        for name, child in children.items():
            try:
                res = run_backtest(train_slice, child, **backtest_kwargs)
                child_returns[name] = _portfolio_returns(res)
            except Exception:
                # Child blew up on this fold (e.g. macro_timing missing
                # SPY in a slice that doesn't carry it) → zero series so
                # the fitter naturally assigns it ~0 weight.
                child_returns[name] = np.zeros(len(train_slice), dtype=float)

        weights, fallback = fit_sharpe_max_weights(
            child_returns, periods_per_year=periods_per_year
        )
        any_fallback = any_fallback or fallback
        last_weights = weights

        # OOS evaluation with the chosen weights.
        from strategies.combined_explainable import CombinedExplainableStrategy

        oos_strat = CombinedExplainableStrategy(
            children=strategy.children,
            child_params=strategy.child_params,
            weights=weights,
            **{k: v for k, v in strategy.params.items()},
        )
        oos_strat.weight_fit_fallback_ = fallback
        oos_result = run_backtest(test_slice, oos_strat, **backtest_kwargs)
        oos_returns = _portfolio_returns(oos_result)
        oos_sharpe = annualised_sharpe(oos_returns, periods_per_year=periods_per_year)

        folds_out.append(
            {
                "fold_idx": fold_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "weights": weights,
                "fallback": fallback,
                "out_of_sample_sharpe": float(oos_sharpe),
            }
        )

        test_start += step
        fold_idx += 1

    # Promote the last fold's weights onto the strategy so the caller can
    # use it immediately (most-recent-fold convention).
    if last_weights is not None:
        strategy.weights = last_weights
        strategy.weight_fit_fallback_ = any_fallback

    oos = np.array([f["out_of_sample_sharpe"] for f in folds_out], dtype=float)
    return {
        "folds": folds_out,
        "oos_sharpe_mean": float(np.nanmean(oos)) if oos.size else float("nan"),
        "final_weights": last_weights or dict(strategy.weights),
        "any_fallback": any_fallback,
        "n_folds": len(folds_out),
    }
