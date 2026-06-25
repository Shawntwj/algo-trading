from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backtest.benchmarks import (
    buy_and_hold,
    random_entry_monte_carlo,
)


def _synthetic_wide(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two-ticker random walk in the MultiIndex `(field, ticker)` shape."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.01, size=(n, 2))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    tickers = ["AAA", "BBB"]
    frames = {}
    for field in ["open", "high", "low", "close"]:
        frames[field] = pd.DataFrame(prices, index=idx, columns=tickers)
    frames["volume"] = pd.DataFrame(1_000_000.0, index=idx, columns=tickers)
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


def test_buy_and_hold_equal_weight_matches_average_price_ratio():
    wide = _synthetic_wide()
    close = wide["close"]
    init = 100_000.0

    eq_df, ret_df = buy_and_hold(wide, weights="equal", init_cash=init)

    assert set(eq_df.columns) == {"timestamp", "equity"}
    assert set(ret_df.columns) == {"timestamp", "ret"}
    assert eq_df.height == close.shape[0]

    # Expected: equal-weighted shares * final price, where shares = (init/N)/p0.
    p0 = close.iloc[0].to_numpy()
    pN = close.iloc[-1].to_numpy()
    shares = (init / 2) / p0
    expected_final = float((shares * pN).sum())

    final = float(eq_df["equity"].to_numpy()[-1])
    assert final == pytest.approx(expected_final, rel=1e-9)


def test_buy_and_hold_cap_weight_requires_caps():
    wide = _synthetic_wide()
    with pytest.raises(ValueError, match="caps"):
        buy_and_hold(wide, weights="cap")


def test_buy_and_hold_cap_weight_honours_supplied_caps():
    wide = _synthetic_wide()
    init = 100_000.0
    caps = {"AAA": 9.0, "BBB": 1.0}  # 90/10 split
    eq_df, _ = buy_and_hold(wide, weights="cap", caps=caps, init_cash=init)

    close = wide["close"]
    p0 = close.iloc[0].to_numpy()
    pN = close.iloc[-1].to_numpy()
    w = np.array([0.9, 0.1])
    shares = init * w / p0
    expected_final = float((shares * pN).sum())
    assert float(eq_df["equity"].to_numpy()[-1]) == pytest.approx(expected_final, rel=1e-9)


def test_random_entry_path_count_and_exposure_match_target():
    wide = _synthetic_wide(n=300, seed=7)
    n_paths = 500
    target = 0.5
    eq, summary = random_entry_monte_carlo(
        wide, n_paths=n_paths, exposure_target=target, seed=7
    )

    assert eq.shape == (n_paths, 300)
    assert summary.height == n_paths
    assert set(summary.columns) == {
        "path",
        "total_return",
        "sharpe",
        "max_drawdown",
        "exposure",
    }

    # Realised mean exposure should track the target tightly with 500 paths.
    realised = float(summary["exposure"].mean())
    assert abs(realised - target) <= 0.05, f"realised={realised}, target={target}"


def test_random_entry_is_deterministic_under_seed():
    wide = _synthetic_wide(n=100)
    eq1, s1 = random_entry_monte_carlo(wide, n_paths=20, exposure_target=0.4, seed=123)
    eq2, s2 = random_entry_monte_carlo(wide, n_paths=20, exposure_target=0.4, seed=123)
    np.testing.assert_array_equal(eq1, eq2)
    assert s1.equals(s2)


def test_random_entry_rejects_bad_exposure():
    wide = _synthetic_wide(n=20)
    with pytest.raises(ValueError):
        random_entry_monte_carlo(wide, n_paths=5, exposure_target=0.0)
    with pytest.raises(ValueError):
        random_entry_monte_carlo(wide, n_paths=5, exposure_target=1.0)
