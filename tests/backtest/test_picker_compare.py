"""Tests for backtest.picker_compare. Holdings history is hand-built so
no EDGAR / yfinance round-trips are needed."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.picker_compare import Literal13FFollow, picker_compare
from data.fundamentals import FACTOR_FIELDS
from research.picker_profiles import PickerProfile
from strategies import Signals
from strategies.base import Strategy


def _wide_frame(
    n_bars: int = 400, n_tickers: int = 6, seed: int = 0,
    drift_per_ticker: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    if drift_per_ticker is None:
        drift_per_ticker = rng.uniform(-0.0003, 0.001, size=n_tickers)
    vol = 0.012
    factor = rng.normal(0.0, vol * 0.7, size=n_bars)
    idio = rng.normal(0.0, vol * 0.5, size=(n_bars, n_tickers))
    betas = rng.uniform(0.6, 1.4, size=n_tickers)
    rets = drift_per_ticker[None, :] + factor[:, None] * betas[None, :] + idio
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2021-01-04", periods=n_bars, freq="B")
    frames = {}
    for field in ("open", "high", "low", "close"):
        frames[field] = pd.DataFrame(prices, index=idx, columns=tickers)
    vol_scales = rng.uniform(1e6, 1e8, size=n_tickers)
    frames["volume"] = pd.DataFrame(
        np.broadcast_to(vol_scales[None, :], (n_bars, n_tickers)),
        index=idx, columns=tickers,
    )
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


def _trivial_profile(name: str) -> PickerProfile:
    return PickerProfile(
        name=name, holdings=("X", "Y"),
        profile={f: 0.0 for f in FACTOR_FIELDS},
        picker_means={f: 0.0 for f in FACTOR_FIELDS},
        benchmark_means={f: 0.0 for f in FACTOR_FIELDS},
        benchmark_stds={f: 1.0 for f in FACTOR_FIELDS},
        as_of="2024-12-31", n_holdings_with_data=2,
    )


def test_literal_13f_follow_holds_basket_until_next_event():
    data = _wide_frame(n_bars=120, n_tickers=4, seed=0)
    # Two rebalance events on actual bar dates.
    bar_dates = [d.date().isoformat() for d in data["close"].index]
    holdings_history = {
        bar_dates[10]: ["T00", "T01"],
        bar_dates[60]: ["T02", "T03"],
    }
    strat = Literal13FFollow(holdings_history=holdings_history)
    sig = strat.generate_signals(data)
    assert isinstance(sig, Signals)
    # Before the first event nobody is in basket → no entries pre-bar 10.
    pre_10 = sig.entries.iloc[:10].values.sum()
    assert pre_10 == 0
    # First event: T00 + T01 fire.
    assert sig.entries.iloc[10]["T00"]
    assert sig.entries.iloc[10]["T01"]
    # Second event: T00 + T01 exit; T02 + T03 enter.
    assert sig.exits.iloc[60]["T00"]
    assert sig.exits.iloc[60]["T01"]
    assert sig.entries.iloc[60]["T02"]
    assert sig.entries.iloc[60]["T03"]


def test_literal_13f_follow_drops_unknown_tickers():
    data = _wide_frame(n_bars=80, n_tickers=3, seed=1)
    bar_dates = [d.date().isoformat() for d in data["close"].index]
    holdings_history = {
        bar_dates[5]: ["T00", "TIBM_NOT_IN_UNIVERSE"],
    }
    strat = Literal13FFollow(holdings_history=holdings_history)
    sig = strat.generate_signals(data)
    # T00 enters; the unknown ticker is silently dropped.
    assert sig.entries.iloc[5]["T00"]
    assert int(sig.entries.values.sum()) == 1


def test_picker_compare_returns_gap():
    profile = _trivial_profile("comp_test")
    data = _wide_frame(n_bars=400, n_tickers=6, seed=2)
    bar_dates = [d.date().isoformat() for d in data["close"].index]
    holdings_history = {
        bar_dates[260]: ["T00", "T01", "T02"],
        bar_dates[320]: ["T03", "T04", "T05"],
    }
    out = picker_compare(
        "comp_test", data,
        holdings_history=holdings_history, profile=profile,
        top_n=3, rebalance_freq=21, lookback_for_factors=252,
    )
    assert out.picker_name == "comp_test"
    assert isinstance(out.sharpe_a, float)
    assert isinstance(out.sharpe_b, float)
    assert math.isclose(out.sharpe_gap, out.sharpe_a - out.sharpe_b, abs_tol=1e-9)
    # Both backtests produced trades.
    assert out.variant_a.portfolio.value().shape[0] == data["close"].shape[0]
    assert out.variant_b.portfolio.value().shape[0] == data["close"].shape[0]
