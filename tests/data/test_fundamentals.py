"""Offline tests for the fundamentals fetcher. The yfinance side is injected
via the ``info=`` / ``close=`` kwargs so no network is required."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data import fundamentals as fnd


def _synthetic_close(n: int = 300, drift: float = 0.0005, vol: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, size=n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.Series(prices, index=idx)


def test_momentum_12_1_positive_for_uptrend():
    s = _synthetic_close(n=300, drift=0.002, seed=1)
    m = fnd.momentum_12_1(s)
    assert math.isfinite(m)
    assert m > 0.0  # strong drift over a year


def test_momentum_12_1_nan_when_short_series():
    s = _synthetic_close(n=100)
    assert math.isnan(fnd.momentum_12_1(s))


def test_realised_vol_60d_close_to_expected():
    s = _synthetic_close(n=300, vol=0.01, seed=2)
    rv = fnd.realised_vol_60d(s)
    # annualised σ ≈ 0.01 * sqrt(252) ≈ 0.159
    assert 0.10 < rv < 0.22


def test_realised_vol_nan_when_too_short():
    s = _synthetic_close(n=30)
    assert math.isnan(fnd.realised_vol_60d(s))


def test_fundamentals_for_uses_injected_info():
    info = {
        "marketCap": 3.0e12,
        "forwardPE": 28.5,
        "trailingPE": 30.0,
        "priceToBook": 45.2,
        "returnOnEquity": 1.5,   # 150%
        "debtToEquity": 150.0,   # scaled — should normalise to 1.5
    }
    close = _synthetic_close(n=400, drift=0.0008, vol=0.012, seed=3)
    fv = fnd.fundamentals_for("AAPL", end="2024-12-31", info=info, close=close)
    assert fv.ticker == "AAPL"
    assert math.isclose(fv.log_market_cap, math.log(3.0e12))
    assert fv.forward_pe == 28.5
    assert fv.pb_ratio == 45.2
    assert math.isclose(fv.debt_to_equity, 1.5, rel_tol=1e-6)
    assert math.isfinite(fv.momentum_12_1)
    assert math.isfinite(fv.realised_vol_60d)


def test_fundamentals_for_forward_pe_falls_back_to_trailing():
    info = {"marketCap": 1e10, "trailingPE": 17.0}
    fv = fnd.fundamentals_for("XYZ", end="2024-12-31", info=info, close=_synthetic_close())
    assert fv.forward_pe == 17.0


def test_fundamentals_for_empty_info_yields_nans():
    fv = fnd.fundamentals_for("ZZZ", end="2024-12-31", info={}, close=_synthetic_close())
    assert math.isnan(fv.log_market_cap)
    assert math.isnan(fv.forward_pe)
    assert math.isnan(fv.pb_ratio)
    assert math.isnan(fv.roe)
    assert math.isnan(fv.debt_to_equity)


def test_zscore_frame_preserves_shape_and_drops_constant_columns():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 5.0, 5.0, 5.0],  # constant -> NaN sd
            "c": [10.0, 20.0, 30.0, np.nan],
        }
    )
    z = fnd.zscore_frame(df, fill=0.0)
    assert z.shape == df.shape
    assert math.isclose(z["a"].mean(), 0.0, abs_tol=1e-9)
    # constant column collapses to 0 after fill
    assert (z["b"] == 0.0).all()
    # NaN survives the source NaN ⇒ filled to 0.
    assert z["c"].iloc[-1] == 0.0


def test_vectors_to_frame_shapes():
    vectors = {
        "AAPL": fnd.fundamentals_for(
            "AAPL", end="2024-12-31",
            info={"marketCap": 1e12, "forwardPE": 25.0, "priceToBook": 40.0,
                  "returnOnEquity": 1.5, "debtToEquity": 200.0},
            close=_synthetic_close(seed=10),
        ),
        "MSFT": fnd.fundamentals_for(
            "MSFT", end="2024-12-31",
            info={"marketCap": 2e12, "forwardPE": 30.0, "priceToBook": 12.0,
                  "returnOnEquity": 0.45, "debtToEquity": 80.0},
            close=_synthetic_close(seed=11),
        ),
    }
    frame = fnd.vectors_to_frame(vectors)
    assert list(frame.columns) == list(fnd.FACTOR_FIELDS)
    assert frame.index.tolist() == ["AAPL", "MSFT"]
