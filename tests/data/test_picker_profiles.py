"""Tests for the picker factor-profile builder. Uses a stubbed fundamentals
function so no network is required."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from data.fundamentals import FACTOR_FIELDS, FactorVector
from research import picker_profiles as pp


def _stub_fundamentals(profile_map: dict[str, dict[str, float]]):
    """Return a fundamentals_for stand-in that yields fixed factor values."""

    def _fn(ticker: str, end: str) -> FactorVector:
        row = profile_map.get(ticker, {})
        return FactorVector(
            ticker=ticker.upper(),
            log_market_cap=row.get("log_market_cap", float("nan")),
            forward_pe=row.get("forward_pe", float("nan")),
            pb_ratio=row.get("pb_ratio", float("nan")),
            momentum_12_1=row.get("momentum_12_1", float("nan")),
            roe=row.get("roe", float("nan")),
            debt_to_equity=row.get("debt_to_equity", float("nan")),
            realised_vol_60d=row.get("realised_vol_60d", float("nan")),
        )

    return _fn


def test_build_profile_from_frame_overweight_megacap():
    # Picker holds two megacaps; benchmark is small + mega mixed.
    picker = pd.DataFrame(
        {f: [np.nan] * 2 for f in FACTOR_FIELDS},
        index=["AAPL", "MSFT"],
    )
    picker["log_market_cap"] = [30.0, 29.5]   # log(3T), log(~2T)
    picker["pb_ratio"] = [40.0, 12.0]
    picker["momentum_12_1"] = [0.20, 0.15]

    bench = pd.DataFrame(
        {f: [np.nan] * 4 for f in FACTOR_FIELDS},
        index=["A", "B", "C", "D"],
    )
    bench["log_market_cap"] = [25.0, 26.0, 27.0, 28.0]
    bench["pb_ratio"] = [3.0, 4.0, 5.0, 6.0]
    bench["momentum_12_1"] = [0.05, 0.10, 0.0, -0.05]

    prof = pp.build_profile_from_frame(
        "test", picker, bench, holdings=["AAPL", "MSFT"], as_of="2024-12-31",
    )
    # Megacap diff should be sharply positive (picker mean 29.75 vs bench 26.5).
    assert prof.profile["log_market_cap"] > 1.5
    assert prof.profile["pb_ratio"] > 1.5
    # Realised vol wasn't provided ⇒ benchmark std NaN ⇒ profile 0.
    assert prof.profile["realised_vol_60d"] == 0.0


def test_build_profile_from_holdings_with_stub():
    fundamentals_map = {
        # Picker holdings — high market cap, low P/B (Berkshire-ish profile).
        "X": {"log_market_cap": 30.0, "pb_ratio": 3.0, "momentum_12_1": 0.0,
              "roe": 0.5, "debt_to_equity": 0.5, "realised_vol_60d": 0.18, "forward_pe": 15.0},
        "Y": {"log_market_cap": 29.5, "pb_ratio": 2.5, "momentum_12_1": 0.05,
              "roe": 0.3, "debt_to_equity": 0.4, "realised_vol_60d": 0.16, "forward_pe": 18.0},
        # Benchmark — smaller, higher growth.
        "B1": {"log_market_cap": 26.0, "pb_ratio": 8.0, "momentum_12_1": 0.20,
               "roe": 0.2, "debt_to_equity": 1.5, "realised_vol_60d": 0.28, "forward_pe": 35.0},
        "B2": {"log_market_cap": 25.5, "pb_ratio": 7.0, "momentum_12_1": 0.15,
               "roe": 0.25, "debt_to_equity": 1.7, "realised_vol_60d": 0.30, "forward_pe": 40.0},
        "B3": {"log_market_cap": 27.0, "pb_ratio": 9.0, "momentum_12_1": 0.25,
               "roe": 0.18, "debt_to_equity": 1.6, "realised_vol_60d": 0.32, "forward_pe": 45.0},
    }
    stub = _stub_fundamentals(fundamentals_map)
    prof = pp.build_profile_from_holdings(
        "stub_value",
        holdings=["X", "Y"],
        benchmark=["B1", "B2", "B3"],
        as_of="2024-12-31",
        fundamentals_fn=stub,
    )
    assert prof.name == "stub_value"
    assert prof.n_holdings_with_data == 2
    # Value tilt: lower forward_pe than benchmark ⇒ negative diff.
    assert prof.profile["forward_pe"] < 0
    # Quality / size tilt: higher market cap than benchmark ⇒ positive.
    assert prof.profile["log_market_cap"] > 0


def test_profile_roundtrip_json(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "PROFILES_DIR", tmp_path)
    profile = pp.PickerProfile(
        name="rt",
        holdings=("AAPL", "MSFT"),
        profile={f: 0.5 for f in FACTOR_FIELDS},
        picker_means={f: 1.0 for f in FACTOR_FIELDS},
        benchmark_means={f: 0.0 for f in FACTOR_FIELDS},
        benchmark_stds={f: 1.0 for f in FACTOR_FIELDS},
        as_of="2024-12-31",
        n_holdings_with_data=2,
    )
    path = pp.save_profile(profile)
    assert path.exists()
    back = pp.load_profile("rt")
    assert back.holdings == ("AAPL", "MSFT")
    assert back.profile == profile.profile
    assert "rt" in pp.list_profiles()


def test_cosine_similarity_basic():
    v = np.array([1.0, 0.0, 0.0])
    assert math.isclose(pp.cosine_similarity(v, v), 1.0)
    assert math.isclose(pp.cosine_similarity(v, np.array([0.0, 1.0, 0.0])), 0.0, abs_tol=1e-9)
    assert math.isclose(pp.cosine_similarity(v, -v), -1.0)
    # Zero vector guard.
    assert pp.cosine_similarity(v, np.zeros(3)) == 0.0


def test_load_profile_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "PROFILES_DIR", tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError):
        pp.load_profile("definitely_not_a_picker")
