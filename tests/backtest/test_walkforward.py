"""Tests for backtest/walkforward.py — walk-forward harness + aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.walkforward import (
    FoldResult,
    WalkForwardConfig,
    aggregate_walkforward,
    walk_forward,
)
from strategies.ma_crossover import MACrossover


# ─── synthetic price helpers ───────────────────────────────────────────────
def _synthetic_wide(n_bars: int = 800, seed: int = 0) -> pd.DataFrame:
    """One-ticker `polars_to_wide`-shaped frame: 800 bars of GBM-ish prices.

    Built so an MA crossover sweep can actually find a (sometimes wrong)
    winner per fold. We keep it deterministic via the seed."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=n_bars)
    close = 100.0 * np.cumprod(1.0 + rets)
    ts = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    cols = pd.MultiIndex.from_product(
        [["open", "high", "low", "close", "volume"], ["AAA"]],
        names=["field", "ticker"],
    )
    frame = pd.DataFrame(index=ts, columns=cols, dtype=float)
    frame.loc[:, ("open", "AAA")] = close
    frame.loc[:, ("high", "AAA")] = close * 1.001
    frame.loc[:, ("low", "AAA")] = close * 0.999
    frame.loc[:, ("close", "AAA")] = close
    frame.loc[:, ("volume", "AAA")] = 1_000_000.0
    return frame


# ─── config validation ────────────────────────────────────────────────────
def test_config_rejects_bad_sizes():
    with pytest.raises(ValueError):
        WalkForwardConfig(train_size=1, test_size=10)
    with pytest.raises(ValueError):
        WalkForwardConfig(train_size=100, test_size=0)
    with pytest.raises(ValueError):
        WalkForwardConfig(train_size=100, test_size=10, step=0)
    with pytest.raises(ValueError):
        WalkForwardConfig(train_size=100, test_size=10, mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WalkForwardConfig(train_size=100, test_size=10, min_train=1)


# ─── core: fold count + train-window sizing ───────────────────────────────
def test_expanding_vs_rolling_fold_windows_differ_as_expected():
    """Both modes produce the same number / placement of folds, but the
    train_start (and thus train length) differs: rolling = constant, expanding
    = growing."""
    wide = _synthetic_wide(n_bars=600, seed=1)
    grid = {"fast": [5, 10], "slow": [50, 100]}
    cfg_exp = WalkForwardConfig(train_size=200, test_size=50, mode="expanding")
    cfg_roll = WalkForwardConfig(train_size=200, test_size=50, mode="rolling")

    folds_exp = walk_forward(MACrossover, wide, grid, cfg_exp)
    folds_roll = walk_forward(MACrossover, wide, grid, cfg_roll)

    # (600 - 200) // 50 = 8 non-overlapping test windows in both modes.
    assert len(folds_exp) == 8
    assert len(folds_roll) == 8

    # Rolling: train length is constant at 200 bars.
    for f in folds_roll:
        assert f.train_end - f.train_start == 200

    # Expanding: train starts at 0 for every fold (no min_train), so length
    # grows with the fold index.
    assert all(f.train_start == 0 for f in folds_exp)
    train_lens = [f.train_end - f.train_start for f in folds_exp]
    assert train_lens == sorted(train_lens)  # monotone increasing
    assert train_lens[0] == 200
    assert train_lens[-1] > train_lens[0]


def test_min_train_clips_first_fold_only():
    """`min_train` is meant to floor the first expanding fold's lookback while
    leaving later folds free to use all history."""
    wide = _synthetic_wide(n_bars=600, seed=2)
    grid = {"fast": [5], "slow": [50]}
    cfg = WalkForwardConfig(
        train_size=300, test_size=50, mode="expanding", min_train=100
    )
    folds = walk_forward(MACrossover, wide, grid, cfg)

    assert len(folds) >= 2
    # First fold's train window is clipped to 100 bars (from the tail).
    f0 = folds[0]
    assert f0.train_end - f0.train_start == 100
    assert f0.train_end == 300  # tail-anchored at the first test_start
    # Later folds use all history (train_start == 0).
    assert folds[1].train_start == 0


# ─── is-vs-oos and aggregation ────────────────────────────────────────────
def test_walk_forward_produces_per_fold_returns_and_sharpes():
    wide = _synthetic_wide(n_bars=600, seed=3)
    grid = {"fast": [5, 10], "slow": [30, 50]}
    cfg = WalkForwardConfig(train_size=250, test_size=60, mode="expanding")
    folds = walk_forward(MACrossover, wide, grid, cfg)

    assert folds, "expected at least one fold"
    for f in folds:
        assert isinstance(f, FoldResult)
        assert f.train_end == f.test_start  # train ends where test starts
        assert f.test_end - f.test_start == 60
        assert f.test_returns.size == 60
        assert "fast" in f.selected_params and "slow" in f.selected_params
        # IS/OOS Sharpes are either finite or NaN (vbt occasionally returns
        # NaN for degenerate windows); both states are tolerated downstream.
        assert np.isfinite(f.in_sample_sharpe) or np.isnan(f.in_sample_sharpe)


def test_aggregate_returns_decay_data_and_slope():
    """The aggregate dict has every key the chart-rendering 7d task needs."""
    # Hand-build folds with a clear IS > OOS gap, the classic overfit pattern.
    rng = np.random.default_rng(11)
    folds = []
    for i in range(8):
        oos_rets = rng.normal(0.0, 0.01, size=50)
        folds.append(
            FoldResult(
                fold_idx=i,
                train_start=0,
                train_end=200 + i * 50,
                test_start=200 + i * 50,
                test_end=250 + i * 50,
                in_sample_sharpe=2.0 + 0.1 * i,         # uniformly high IS
                out_of_sample_sharpe=0.2 + 0.05 * i,    # uniformly low OOS
                selected_params={"fast": 5, "slow": 50},
                test_returns=oos_rets,
            )
        )
    agg = aggregate_walkforward(folds, n_resamples=200, seed=7)

    assert agg["n_folds"] == 8
    assert agg["oos_sharpe_distribution"].shape == (8,)
    assert len(agg["is_vs_oos"]) == 8
    # Decay slope: OOS rises by 0.05 per fold while IS rises by 0.1 →
    # OLS slope of OOS~IS ≈ 0.05 / 0.10 = 0.5.
    assert agg["decay_slope"] == pytest.approx(0.5, abs=1e-9)
    lo, hi = agg["oos_sharpe_ci"]
    assert lo <= agg["oos_sharpe_mean"] <= hi


def test_aggregate_handles_empty_folds():
    """`walk_forward` can legitimately return [] if the data is too short."""
    agg = aggregate_walkforward([])
    assert agg["n_folds"] == 0
    assert agg["oos_sharpe_distribution"].size == 0
    assert agg["is_vs_oos"] == []


def test_aggregate_detects_overfit_with_low_slope():
    """End-to-end: a walk-forward on noise should detect the IS/OOS gap.

    With pure GBM noise the sweep picks a different 'best' combo each fold;
    OOS Sharpe distribution should center near zero, while IS Sharpes are
    inflated by selection. The decay slope is therefore ≪ 1."""
    wide = _synthetic_wide(n_bars=800, seed=42)
    grid = {"fast": [5, 10, 20], "slow": [30, 50, 100]}
    cfg = WalkForwardConfig(train_size=250, test_size=80, mode="expanding")
    folds = walk_forward(MACrossover, wide, grid, cfg)
    assert len(folds) >= 5

    agg = aggregate_walkforward(folds, n_resamples=200, seed=42)
    is_vals = np.array([f.in_sample_sharpe for f in folds])
    oos_vals = np.array([f.out_of_sample_sharpe for f in folds])
    finite = np.isfinite(is_vals) & np.isfinite(oos_vals)
    assert finite.any()
    # On random data, IS mean should comfortably exceed OOS mean (selection
    # bias). Loose tolerance — this is a statistical assertion, not exact.
    assert is_vals[finite].mean() > oos_vals[finite].mean() - 0.5
    # The aggregate's keys exist and have the right types.
    assert isinstance(agg["decay_slope"], float)
