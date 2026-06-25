"""Smoke + contract tests for PickerCloneStrategy.

Profiles are constructed inline so tests don't depend on the committed JSONs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.fundamentals import FACTOR_FIELDS
from research.picker_profiles import PickerProfile
from strategies import REGISTRY, PICKER_CLONE_REGISTRY, Signals
from strategies.picker_clone import PickerCloneStrategy


def _wide_frame(
    n_bars: int = 400, n_tickers: int = 8, seed: int = 0,
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
    # Volume scaled so each ticker has a distinct "size" proxy.
    volume_scales = rng.uniform(1e6, 1e8, size=n_tickers)
    frames["volume"] = pd.DataFrame(
        np.broadcast_to(volume_scales[None, :], (n_bars, n_tickers)),
        index=idx,
        columns=tickers,
    )
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


def _profile(name: str, momentum: float, vol: float, size: float) -> PickerProfile:
    """Build a tiny PickerProfile (only the three runtime fields matter)."""
    base = {f: 0.0 for f in FACTOR_FIELDS}
    base["momentum_12_1"] = momentum
    base["realised_vol_60d"] = vol
    base["log_market_cap"] = size
    return PickerProfile(
        name=name,
        holdings=("X", "Y"),
        profile=base,
        picker_means={f: 0.0 for f in FACTOR_FIELDS},
        benchmark_means={f: 0.0 for f in FACTOR_FIELDS},
        benchmark_stds={f: 1.0 for f in FACTOR_FIELDS},
        as_of="2024-12-31",
        n_holdings_with_data=2,
    )


def test_registry_entries_present():
    for key in PICKER_CLONE_REGISTRY:
        assert key in REGISTRY
        assert REGISTRY[key] is PICKER_CLONE_REGISTRY[key]


def test_picker_clone_contract_no_nans():
    profile = _profile("test_profile", momentum=2.0, vol=-1.0, size=1.0)
    data = _wide_frame(n_bars=400, n_tickers=8, seed=42)
    strat = PickerCloneStrategy(
        picker_name="test_profile",
        profile=profile,
        top_n=3,
        rebalance_freq=21,
        lookback_for_factors=252,
    )
    sig = strat.generate_signals(data)
    assert isinstance(sig, Signals)
    close = data["close"]
    assert sig.entries.shape == close.shape
    assert sig.exits.shape == close.shape
    assert sig.entries.index.equals(close.index)
    assert list(sig.entries.columns) == list(close.columns)
    assert not sig.entries.isna().any().any()
    assert not sig.exits.isna().any().any()
    assert sig.entries.dtypes.eq(bool).all()


def test_picker_clone_fires_at_least_once():
    profile = _profile("test_profile", momentum=2.0, vol=-1.0, size=1.0)
    data = _wide_frame(n_bars=500, n_tickers=8, seed=7)
    strat = PickerCloneStrategy(
        picker_name="test_profile", profile=profile,
        top_n=3, rebalance_freq=21, lookback_for_factors=252,
    )
    sig = strat.generate_signals(data)
    assert int(sig.entries.values.sum()) > 0
    assert int(sig.exits.values.sum()) > 0


def test_picker_clone_picks_high_momentum_when_profile_demands():
    # Construct a universe where exactly 3 tickers are high-momentum drift
    # winners. A profile with momentum=+5 (strong momentum demand) should
    # prefer those 3 over the rest.
    drifts = np.array([0.0030, 0.0028, 0.0025] + [-0.0005] * 5)
    data = _wide_frame(n_bars=500, n_tickers=8, seed=11, drift_per_ticker=drifts)
    profile = _profile("momentum_chaser", momentum=5.0, vol=0.0, size=0.0)
    strat = PickerCloneStrategy(
        picker_name="momentum_chaser", profile=profile,
        top_n=3, rebalance_freq=63, lookback_for_factors=252,
    )
    sig = strat.generate_signals(data)
    # Sum of entries per ticker — the top-3 momentum names should accumulate
    # more entries than the bottom-5.
    entries_per_ticker = sig.entries.sum(axis=0)
    winners = entries_per_ticker.iloc[:3].sum()
    losers = entries_per_ticker.iloc[3:].sum()
    assert winners > losers


def test_picker_clone_short_data_returns_empty():
    profile = _profile("test", momentum=0.0, vol=0.0, size=0.0)
    data = _wide_frame(n_bars=100, n_tickers=4, seed=1)  # < lookback
    strat = PickerCloneStrategy(
        picker_name="test", profile=profile,
        top_n=2, rebalance_freq=21, lookback_for_factors=252,
    )
    sig = strat.generate_signals(data)
    assert int(sig.entries.values.sum()) == 0
    assert int(sig.exits.values.sum()) == 0


def test_picker_clone_missing_committed_profile_raises():
    with pytest.raises(ValueError):
        PickerCloneStrategy(picker_name="totally_not_a_committed_picker_name")


def test_picker_clone_loads_committed_berkshire_profile():
    """Smoke: the committed JSON loads and the strategy instantiates."""
    strat = PickerCloneStrategy(picker_name="berkshire")
    assert strat.profile.name == "berkshire"
    assert set(strat.profile.profile.keys()) >= {"momentum_12_1", "log_market_cap"}
