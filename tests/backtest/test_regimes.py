from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backtest.regimes import tag_all, tag_drawdown, tag_trend, tag_volatility


def test_tag_trend_bull_above_sma_bear_below():
    # 250 bars rising linearly then 50 bars dropping below the 200 SMA.
    n_rise = 250
    n_fall = 60
    rising = np.linspace(100.0, 200.0, n_rise)
    falling = np.linspace(200.0, 50.0, n_fall)  # crashes well below SMA
    series = pl.Series(np.concatenate([rising, falling]))

    tags = tag_trend(series, window=200).to_list()

    # The rising phase (after warmup) must be predominantly bull.
    rising_post_warmup = tags[200:n_rise]
    assert all(t == "bull" for t in rising_post_warmup), rising_post_warmup[:5]

    # The crash must end in bear.
    assert tags[-1] == "bear"


def test_tag_trend_warmup_is_labelled_bear_by_convention():
    series = pl.Series(np.linspace(100.0, 200.0, 100))
    tags = tag_trend(series, window=200).to_list()
    # No 200-bar SMA yet → no trend evidence → bear.
    assert set(tags) == {"bear"}


def test_tag_volatility_terciles_split_evenly():
    # 300 bars uniformly distributed over [10, 40] — terciles should land at
    # ~20 and ~30, giving ~100 bars per bucket.
    arr = np.linspace(10.0, 40.0, 300)
    tags = tag_volatility(pl.Series(arr)).to_list()
    counts = {label: tags.count(label) for label in ("low", "mid", "high")}
    assert counts["low"] >= 90 and counts["low"] <= 110
    assert counts["mid"] >= 90 and counts["mid"] <= 110
    assert counts["high"] >= 90 and counts["high"] <= 110

    # The lowest value must be 'low' and the highest 'high'.
    assert tags[0] == "low"
    assert tags[-1] == "high"


def test_tag_drawdown_thresholds_fire():
    # Hand-crafted: climb to 200, drop by stages.
    # Bar 0: 100; bars 1..49: 100->200; bar 50: 195 (~-2.5% dd → calm)
    # bar 51: 180 (~-10% dd → mild); bar 52: 160 (~-20% dd → severe).
    base = np.linspace(100.0, 200.0, 50)
    series = pl.Series(np.concatenate([base, [195.0, 180.0, 160.0]]))
    tags = tag_drawdown(series).to_list()

    # Early bars while climbing should be calm (we're always near the high).
    assert tags[10] == "calm"
    # The three crash bars hit each bucket in order.
    assert tags[-3] == "calm"  # -2.5%
    assert tags[-2] == "mild"  # -10%
    assert tags[-1] == "severe"  # -20%


def test_tag_all_returns_aligned_dataframe():
    n = 300
    spy = pl.Series(np.linspace(100.0, 200.0, n))
    vix = pl.Series(np.linspace(10.0, 40.0, n))
    ts = pl.Series([f"2022-01-{i:02d}" for i in range(1, 31)] * 10)  # length 300

    df = tag_all(spy, vix, timestamps=ts)
    assert df.height == n
    assert set(df.columns) >= {"trend", "vol", "drawdown", "timestamp"}


def test_tag_all_rejects_length_mismatch():
    spy = pl.Series(np.linspace(100.0, 200.0, 100))
    vix = pl.Series(np.linspace(10.0, 40.0, 50))
    with pytest.raises(ValueError):
        tag_all(spy, vix)
