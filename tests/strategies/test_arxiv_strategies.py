"""Smoke + contract tests for the arxiv-replication strategies (Task 2b).

Each test asserts the existing ``Strategy`` ABC contract:
  * ``generate_signals(data)`` returns a ``Signals`` dataclass.
  * ``entries`` / ``exits`` are wide bool DataFrames aligned to ``data['close']``.
  * No NaNs in the signal frames.
  * Non-trivial activity on a realistic synthetic universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies import REGISTRY, Signals
from strategies.drift_regime import DriftRegimeSingha
from strategies.macro_timing import MacroTimingXiong
from strategies.pca_stat_arb import PCAStatArb


# ─── synthetic data helpers ───────────────────────────────────────────────
def _wide_frame(
    n_bars: int = 260,
    n_tickers: int = 8,
    seed: int = 0,
    drift: float = 0.0005,
    vol: float = 0.012,
) -> pd.DataFrame:
    """Geometric-Brownian-motion price grid with shared factor noise so the
    PCA fit isn't pure noise (residuals would all collapse to zero)."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, vol * 0.8, size=n_bars)
    idio = rng.normal(0.0, vol * 0.5, size=(n_bars, n_tickers))
    betas = rng.uniform(0.6, 1.4, size=n_tickers)
    rets = drift + factor[:, None] * betas[None, :] + idio
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))

    idx = pd.date_range("2022-01-03", periods=n_bars, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    frames = {}
    for field in ("open", "high", "low", "close"):
        frames[field] = pd.DataFrame(prices, index=idx, columns=tickers)
    frames["volume"] = pd.DataFrame(1_000_000, index=idx, columns=tickers)
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


def _macro_frame(n_bars: int = 320, seed: int = 1) -> pd.DataFrame:
    """Wide frame carrying SPY / VIX / IRX / GROW / DEF close series — enough
    columns for MacroTimingXiong to compute every signal."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    spy_rets = rng.normal(0.0004, 0.011, size=n_bars)
    spy_rets[100:140] -= 0.005
    spy = 350.0 * np.exp(np.cumsum(spy_rets))

    vix = np.empty(n_bars)
    vix[0] = 18.0
    for t in range(1, n_bars):
        vix[t] = max(9.0, vix[t - 1] + 0.1 * (18.0 - vix[t - 1]) + rng.normal(0.0, 1.5))
    vix[110:130] += 12.0

    irx = 1.0 + 0.005 * np.arange(n_bars) + rng.normal(0.0, 0.02, size=n_bars)
    irx[200:] += np.linspace(0.0, -1.0, n_bars - 200)

    growth = 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.014, size=n_bars)))
    defensive = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=n_bars)))

    tickers = ["SPY", "^VIX", "^IRX", "GROW", "DEF"]
    closes = pd.DataFrame(
        {"SPY": spy, "^VIX": vix, "^IRX": irx, "GROW": growth, "DEF": defensive},
        index=idx,
    )
    frames = {f: closes.copy() for f in ("open", "high", "low", "close")}
    frames["volume"] = pd.DataFrame(1_000_000, index=idx, columns=tickers)
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


# ─── PCAStatArb ───────────────────────────────────────────────────────────
def test_pca_stat_arb_registered():
    assert "pca_stat_arb" in REGISTRY
    assert REGISTRY["pca_stat_arb"] is PCAStatArb


def test_pca_stat_arb_contract_no_nans():
    data = _wide_frame(n_bars=320, n_tickers=8, seed=42)
    strat = PCAStatArb(window=126, n_factors=3, entry_z=1.0, exit_z=0.25)
    sig = strat.generate_signals(data)

    assert isinstance(sig, Signals)
    close = data["close"]
    assert sig.entries.index.equals(close.index)
    assert list(sig.entries.columns) == list(close.columns)
    assert sig.entries.shape == close.shape
    assert sig.exits.shape == close.shape

    assert not sig.entries.isna().any().any()
    assert not sig.exits.isna().any().any()
    assert sig.entries.dtypes.eq(bool).all()
    assert sig.exits.dtypes.eq(bool).all()


def test_pca_stat_arb_fires_at_least_once():
    data = _wide_frame(n_bars=400, n_tickers=8, seed=7)
    strat = PCAStatArb(window=126, n_factors=3, entry_z=0.5, exit_z=0.1)
    sig = strat.generate_signals(data)
    total_entries = int(sig.entries.values.sum())
    assert total_entries > 0, "PCA stat-arb produced zero entries on synthetic data"


def test_pca_stat_arb_too_few_tickers_returns_empty_signal():
    data = _wide_frame(n_bars=200, n_tickers=2, seed=3)
    strat = PCAStatArb()
    sig = strat.generate_signals(data)
    assert sig.entries.shape == data["close"].shape
    assert sig.entries.values.sum() == 0
    assert sig.exits.values.sum() == 0


# ─── MacroTimingXiong ─────────────────────────────────────────────────────
def test_macro_timing_registered():
    assert "macro_timing" in REGISTRY
    assert REGISTRY["macro_timing"] is MacroTimingXiong


def test_macro_timing_contract_no_nans():
    data = _macro_frame()
    strat = MacroTimingXiong(
        growth_tickers=("GROW",), defensive_tickers=("DEF",)
    )
    sig = strat.generate_signals(data)

    close = data["close"]
    assert sig.entries.shape == close.shape
    assert sig.exits.shape == close.shape
    assert sig.entries.index.equals(close.index)
    assert list(sig.entries.columns) == list(close.columns)
    assert not sig.entries.isna().any().any()
    assert not sig.exits.isna().any().any()


def test_macro_timing_targets_growth_basket():
    data = _macro_frame()
    strat = MacroTimingXiong(
        growth_tickers=("GROW",), defensive_tickers=("DEF",),
    )
    sig = strat.generate_signals(data)
    # At least one entry on either basket and one exit somewhere.
    growth_entries = int(sig.entries["GROW"].sum())
    defensive_entries = int(sig.entries["DEF"].sum())
    assert growth_entries + defensive_entries > 0
    assert int(sig.exits.values.sum()) > 0
    # Should not trade SPY / VIX / IRX (those are inputs only).
    for t in ("SPY", "^VIX", "^IRX"):
        assert int(sig.entries[t].sum()) == 0
        assert int(sig.exits[t].sum()) == 0


def test_macro_timing_target_weight_in_paper_range():
    data = _macro_frame()
    strat = MacroTimingXiong(
        growth_tickers=("GROW",), defensive_tickers=("DEF",)
    )
    _ = strat.generate_signals(data)
    # w_target = 0.5 + MaxTilt * tanh(...) ⇒ ∈ [0.5 - MaxTilt, 0.5 + MaxTilt].
    wG = strat.last_target_weight_
    lo, hi = 0.5 - 0.5, 0.5 + 0.5  # MaxTilt default = 0.5
    assert wG.min() >= lo - 1e-9
    assert wG.max() <= hi + 1e-9


def test_macro_timing_missing_inputs_returns_empty_signal():
    # No SPY / VIX / IRX columns → strategy can't compute the macro signals
    # and must degrade gracefully to zero-signal rather than crash.
    data = _wide_frame(n_bars=260, n_tickers=4, seed=5)
    strat = MacroTimingXiong()
    sig = strat.generate_signals(data)
    assert sig.entries.shape == data["close"].shape
    assert sig.entries.values.sum() == 0


# ─── DriftRegimeSingha ────────────────────────────────────────────────────
def test_drift_regime_registered():
    assert "drift_regime" in REGISTRY
    assert REGISTRY["drift_regime"] is DriftRegimeSingha


def test_drift_regime_contract_no_nans():
    data = _wide_frame(n_bars=260, n_tickers=6, seed=11)
    strat = DriftRegimeSingha()
    sig = strat.generate_signals(data)
    close = data["close"]
    assert sig.entries.shape == close.shape
    assert sig.exits.shape == close.shape
    assert sig.entries.index.equals(close.index)
    assert not sig.entries.isna().any().any()
    assert not sig.exits.isna().any().any()


def test_drift_regime_emits_signals_in_uptrend():
    # An uptrend frame should push UpFraction above 0.6 for several names →
    # the regime gate opens and entries appear.
    data = _wide_frame(
        n_bars=260, n_tickers=6, seed=13, drift=0.0015, vol=0.008
    )
    strat = DriftRegimeSingha(up_fraction_threshold=0.55, top_decile=0.4)
    sig = strat.generate_signals(data)
    assert int(sig.entries.values.sum()) > 0


def test_drift_regime_edge_score_is_finite():
    # EDGE = BASE * REGIME — should never carry NaN/Inf in the inspection
    # cache after our explicit NaN→0 substitution.
    data = _wide_frame(n_bars=260, n_tickers=6, seed=17)
    strat = DriftRegimeSingha()
    _ = strat.generate_signals(data)
    edge = strat.last_edge_
    assert edge is not None
    finite = np.isfinite(edge.to_numpy())
    assert finite.all()


def test_drift_regime_low_fraction_universe_no_entries():
    # Heavy downtrend → UpFraction stays well below 0.60 → regime gate
    # never opens → zero entries (proves the regime gate actually gates).
    data = _wide_frame(
        n_bars=260, n_tickers=6, seed=23, drift=-0.003, vol=0.008
    )
    strat = DriftRegimeSingha()
    sig = strat.generate_signals(data)
    assert int(sig.entries.values.sum()) == 0
