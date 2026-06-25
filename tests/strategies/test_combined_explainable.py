"""Tests for CombinedExplainableStrategy (BRIEF Task 4a)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.walkforward import _inverse_vol_weights, fit_sharpe_max_weights
from strategies import REGISTRY, CombinedExplainableStrategy, Signals


# ─── synthetic frame ───────────────────────────────────────────────────────
def _wide(n_bars: int = 500, n_tickers: int = 3, seed: int = 0) -> pd.DataFrame:
    """Minimal multi-ticker GBM frame for the contract / weight-fit tests."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=(n_bars, n_tickers))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2022-01-03", periods=n_bars, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    frames = {f: pd.DataFrame(prices, index=idx, columns=tickers) for f in ("open", "high", "low", "close")}
    frames["volume"] = pd.DataFrame(1_000_000, index=idx, columns=tickers)
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


# ─── registration ──────────────────────────────────────────────────────────
def test_combined_registered():
    assert "combined_explainable" in REGISTRY
    assert REGISTRY["combined_explainable"] is CombinedExplainableStrategy


# ─── signal contract ───────────────────────────────────────────────────────
def test_signal_contract_no_nan_bool_aligned():
    data = _wide(n_bars=500, n_tickers=3, seed=11)
    strat = CombinedExplainableStrategy(
        children=("ma_crossover", "rsi_mean_reversion"),
    )
    sig = strat.generate_signals(data)
    close = data["close"]

    assert isinstance(sig, Signals)
    assert sig.entries.shape == close.shape
    assert sig.exits.shape == close.shape
    assert sig.entries.index.equals(close.index)
    assert list(sig.entries.columns) == list(close.columns)
    assert not sig.entries.isna().any().any()
    assert not sig.exits.isna().any().any()
    assert sig.entries.dtypes.eq(bool).all()
    assert sig.exits.dtypes.eq(bool).all()


def test_at_least_one_entry_has_complete_explanation():
    """The first entry produced must carry a fully-populated explanation
    with weights summing to 1 (±1e-6)."""
    data = _wide(n_bars=500, n_tickers=3, seed=23)
    children = ("ma_crossover", "rsi_mean_reversion")
    strat = CombinedExplainableStrategy(children=children, min_active_children=1)
    sig = strat.generate_signals(data)
    assert int(sig.entries.values.sum()) > 0

    # Pull one entry's explanation and assert the structure.
    entry_log = [
        v for v in strat.explanation_log.values() if v["direction"] == "long_entry"
    ]
    assert entry_log, "no long_entry explanations recorded"
    sample = entry_log[0]
    assert set(sample.keys()) == {
        "ticker",
        "timestamp",
        "direction",
        "weights",
        "child_signals",
        "summary",
    }
    assert set(sample["weights"].keys()) == set(children)
    assert set(sample["child_signals"].keys()) == set(children)
    assert abs(sum(sample["weights"].values()) - 1.0) < 1e-6
    assert isinstance(sample["summary"], str) and len(sample["summary"]) > 0


def test_unknown_child_raises():
    """Validation: bogus child names should fail loudly on instantiation /
    first call rather than silently dropping the child."""
    strat = CombinedExplainableStrategy(children=("does_not_exist",))
    with pytest.raises(KeyError, match="does_not_exist"):
        strat.generate_signals(_wide(n_bars=300))


def test_weights_must_be_normalised():
    with pytest.raises(ValueError, match="sum to 1"):
        CombinedExplainableStrategy(
            children=("ma_crossover", "rsi_mean_reversion"),
            weights={"ma_crossover": 0.7, "rsi_mean_reversion": 0.7},
        )
    with pytest.raises(ValueError, match="non-negative"):
        CombinedExplainableStrategy(
            children=("ma_crossover", "rsi_mean_reversion"),
            weights={"ma_crossover": 1.2, "rsi_mean_reversion": -0.2},
        )


# ─── weight learning ───────────────────────────────────────────────────────
def test_sharpe_max_weights_picks_best_child():
    """A child with a clearly higher Sharpe should attract more weight than
    a noise child."""
    rng = np.random.default_rng(42)
    good = rng.normal(0.002, 0.005, size=400)   # SR ≈ 0.4 per period × √252 ≈ 6+
    noise = rng.normal(0.0, 0.01, size=400)
    weights, fallback = fit_sharpe_max_weights({"good": good, "noise": noise})
    assert fallback is False
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["good"] > weights["noise"]


def test_sharpe_max_weights_falls_back_on_pathological():
    """All-zero / non-finite frames cannot be optimised → fallback to inverse-vol."""
    pathological = {"A": np.zeros(200), "B": np.zeros(200)}
    w, fb = fit_sharpe_max_weights(pathological)
    assert fb is True
    # Degenerate uniform from the inverse-vol fallback when every σ is 0.
    assert abs(sum(w.values()) - 1.0) < 1e-9

    # Contrived NaN / inf frame.
    nan_frame = {"A": np.full(100, np.nan), "B": np.ones(100)}
    w2, fb2 = fit_sharpe_max_weights(nan_frame)
    assert fb2 is True


def test_inverse_vol_assigns_more_to_quiet_series():
    """Inverse-vol means the lower-σ series should get the larger share."""
    rng = np.random.default_rng(7)
    quiet = rng.normal(0.0, 0.003, size=300)
    loud = rng.normal(0.0, 0.03, size=300)
    w = _inverse_vol_weights({"quiet": quiet, "loud": loud})
    assert w["quiet"] > w["loud"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_fit_weights_walk_forward_promotes_last_fold_weights():
    """Calling fit_weights_walk_forward should overwrite self.weights."""
    data = _wide(n_bars=700, n_tickers=2, seed=3)
    strat = CombinedExplainableStrategy(children=("ma_crossover", "rsi_mean_reversion"))
    original = dict(strat.weights)
    summary = strat.fit_weights_walk_forward(data, train_size=300, test_size=80)
    assert summary["n_folds"] >= 1
    assert set(strat.weights.keys()) == set(original.keys())
    assert abs(sum(strat.weights.values()) - 1.0) < 1e-6
