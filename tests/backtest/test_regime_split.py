from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backtest.regime_split import split_stats_by_regime


def test_split_stats_matches_hand_computed_sharpe():
    # 10 bars: first 5 labelled "bull" with returns [+1%]*5, next 5 "bear" with [-1%]*5.
    rets = pl.Series(name="ret", values=[0.01] * 5 + [-0.01] * 5)
    regimes = pl.DataFrame({"trend": ["bull"] * 5 + ["bear"] * 5})

    out = split_stats_by_regime(rets, regimes).sort("regime")
    assert set(out.columns) == {
        "dimension",
        "regime",
        "n_bars",
        "total_return",
        "sharpe",
        "max_drawdown",
        "exposure",
    }

    bull = out.filter(pl.col("regime") == "bull").row(0, named=True)
    bear = out.filter(pl.col("regime") == "bear").row(0, named=True)

    assert bull["n_bars"] == 5
    assert bear["n_bars"] == 5
    # Constant +1% / bar → std=0 → sharpe is NaN by convention.
    assert np.isnan(bull["sharpe"])
    assert np.isnan(bear["sharpe"])

    # Total return: (1.01)^5 - 1 ≈ 0.0510100501
    assert bull["total_return"] == pytest.approx(1.01**5 - 1, rel=1e-9)
    # Bear: (0.99)^5 - 1 ≈ -0.049010
    assert bear["total_return"] == pytest.approx(0.99**5 - 1, rel=1e-9)


def test_split_stats_sharpe_finite_when_variance_present():
    # Construct returns so we know the per-regime mean and stdev.
    rng = np.random.default_rng(0)
    bull_rets = rng.normal(0.001, 0.01, size=100)
    bear_rets = rng.normal(-0.001, 0.01, size=100)
    rets_arr = np.concatenate([bull_rets, bear_rets])
    rets = pl.Series(name="ret", values=rets_arr)
    regimes = pl.DataFrame({"trend": ["bull"] * 100 + ["bear"] * 100})

    out = split_stats_by_regime(rets, regimes)
    bull = out.filter(pl.col("regime") == "bull").row(0, named=True)
    bear = out.filter(pl.col("regime") == "bear").row(0, named=True)

    # Hand compute: Sharpe = mean/std * sqrt(252) on each slice.
    expected_bull = bull_rets.mean() / bull_rets.std(ddof=1) * np.sqrt(252.0)
    expected_bear = bear_rets.mean() / bear_rets.std(ddof=1) * np.sqrt(252.0)
    assert bull["sharpe"] == pytest.approx(expected_bull, rel=1e-9)
    assert bear["sharpe"] == pytest.approx(expected_bear, rel=1e-9)


def test_split_stats_supports_dataframe_input():
    rets = pl.DataFrame({"ret": [0.01, -0.01, 0.02, -0.02]})
    regimes = pl.DataFrame({"vol": ["low", "low", "high", "high"]})
    out = split_stats_by_regime(rets, regimes)
    assert set(out["dimension"].to_list()) == {"vol"}
    assert set(out["regime"].to_list()) == {"low", "high"}


def test_split_stats_rejects_length_mismatch():
    rets = pl.Series(name="ret", values=[0.01, 0.02])
    regimes = pl.DataFrame({"trend": ["bull"] * 5})
    with pytest.raises(ValueError):
        split_stats_by_regime(rets, regimes)


def test_split_stats_rejects_unknown_dim_set():
    rets = pl.Series(name="ret", values=[0.01, 0.02])
    regimes = pl.DataFrame({"unused": ["a", "b"]})
    with pytest.raises(ValueError):
        split_stats_by_regime(rets, regimes)
