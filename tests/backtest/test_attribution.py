"""Tests for backtest/attribution.py — CAPM + child-signal attribution."""
from __future__ import annotations

import numpy as np
import pytest

from backtest.attribution import child_signal_attribution, market_attribution


# ─── market_attribution ────────────────────────────────────────────────────
def test_market_attribution_recovers_known_beta_and_alpha():
    """Construct strat = 0.7 * market + alpha + noise; check OLS recovers
    beta ≈ 0.7 and the alpha t-stat is positive (true alpha is positive)."""
    rng = np.random.default_rng(0)
    n = 1000
    market = rng.normal(0.0, 0.01, size=n)
    true_beta = 0.7
    true_alpha = 0.0005
    noise = rng.normal(0.0, 0.005, size=n)
    strat = true_beta * market + true_alpha + noise

    out = market_attribution(strat, market, periods_per_year=252)

    assert out["beta"] == pytest.approx(true_beta, abs=0.05)
    assert out["alpha"] == pytest.approx(true_alpha, abs=2e-4)
    assert out["alpha_t_stat"] > 0
    assert out["alpha_annualised"] == pytest.approx(out["alpha"] * 252)
    assert 0.0 <= out["r_squared"] <= 1.0
    assert out["n_obs"] == n
    # Decomposition arrays have the right length.
    assert out["residual_returns"].shape == (n,)
    assert out["systematic_returns"].shape == (n,)


def test_market_attribution_perfect_fit_r_squared_one():
    """When strat == market (alpha=0, beta=1), R² is ~1.0 exactly."""
    rng = np.random.default_rng(1)
    market = rng.normal(0.0, 0.01, size=500)
    out = market_attribution(market, market)
    assert out["beta"] == pytest.approx(1.0, abs=1e-9)
    assert out["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_market_attribution_rejects_length_mismatch():
    with pytest.raises(ValueError):
        market_attribution(np.zeros(100), np.zeros(99))


def test_market_attribution_accepts_array_risk_free():
    rng = np.random.default_rng(2)
    n = 200
    market = rng.normal(0.0, 0.01, size=n)
    rf = np.full(n, 0.0001)  # constant 1bp / day
    strat = market + 0.0005 + rng.normal(0.0, 0.002, size=n)
    out = market_attribution(strat, market, risk_free=rf)
    # Beta should still be ~1 because rf cancels in both excess series.
    assert out["beta"] == pytest.approx(1.0, abs=0.1)


def test_market_attribution_rejects_bad_risk_free_shape():
    with pytest.raises(ValueError):
        market_attribution(np.zeros(50), np.zeros(50), risk_free=np.zeros(40))


# ─── child_signal_attribution ──────────────────────────────────────────────
def test_child_attribution_sums_to_combined_for_consistent_inputs():
    """Build a 2-child case where the combined is exactly `w1*r1 + w2*r2`.
    The residual should be float-precision zero."""
    rng = np.random.default_rng(0)
    n = 200
    r1 = rng.normal(0.0, 0.01, size=n)
    r2 = rng.normal(0.0, 0.02, size=n)
    w1 = np.full(n, 0.6)
    w2 = np.full(n, 0.4)
    combined = w1 * r1 + w2 * r2

    out = child_signal_attribution(
        combined,
        child_returns={"a": r1, "b": r2},
        weights={"a": w1, "b": w2},
    )

    assert pytest.approx(out["a"]) == float((w1 * r1).sum())
    assert pytest.approx(out["b"]) == float((w2 * r2).sum())
    # Residual is essentially zero.
    assert abs(out["residual"]) < 1e-9
    # Sum of contributions equals combined total.
    explained = out["a"] + out["b"]
    assert pytest.approx(float(combined.sum())) == explained


def test_child_attribution_accepts_scalar_weights():
    """A scalar weight is broadcast to the full series."""
    rng = np.random.default_rng(1)
    n = 100
    r1 = rng.normal(0.0, 0.01, size=n)
    r2 = rng.normal(0.0, 0.01, size=n)
    combined = 0.5 * r1 + 0.5 * r2
    out = child_signal_attribution(
        combined,
        child_returns={"a": r1, "b": r2},
        weights={"a": 0.5, "b": 0.5},
    )
    assert abs(out["residual"]) < 1e-9


def test_child_attribution_exposes_gap_when_inputs_inconsistent():
    """Residual is the unexplained chunk when the caller hasn't given us a
    decomposition that exactly reproduces `combined`."""
    rng = np.random.default_rng(2)
    n = 50
    r1 = rng.normal(0.0, 0.01, size=n)
    r2 = rng.normal(0.0, 0.01, size=n)
    # Combined isn't equal to 0.5*r1 + 0.5*r2 — it has an extra constant.
    combined = 0.5 * r1 + 0.5 * r2 + 0.001
    out = child_signal_attribution(
        combined,
        child_returns={"a": r1, "b": r2},
        weights={"a": 0.5, "b": 0.5},
    )
    # The residual captures the constant-offset gap (n * 0.001).
    assert out["residual"] == pytest.approx(0.001 * n, abs=1e-9)


def test_child_attribution_rejects_key_mismatch():
    with pytest.raises(ValueError):
        child_signal_attribution(
            np.zeros(10),
            child_returns={"a": np.zeros(10)},
            weights={"b": np.zeros(10)},
        )


def test_child_attribution_rejects_length_mismatch():
    with pytest.raises(ValueError):
        child_signal_attribution(
            np.zeros(10),
            child_returns={"a": np.zeros(11)},
            weights={"a": np.zeros(10)},
        )


def test_child_attribution_rejects_empty_children():
    with pytest.raises(ValueError):
        child_signal_attribution(
            np.zeros(10),
            child_returns={},
            weights={},
        )
