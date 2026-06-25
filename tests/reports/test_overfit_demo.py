"""Tests for ``reports.overfit_demo``.

The point of the demo is that a wide IS sweep + a single-ticker noise process
should produce a DSR < 0.95. We use a controlled, deterministic noise series
so the assertion is reliable in CI — if synthetic data ever fails to overfit
reliably, log the decision in IMPROVEMENTS and gate the assertion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reports.overfit_demo import run_overfit_demo


def _noise_wide(n_bars: int = 500, n_tickers: int = 2, seed: int = 7) -> pd.DataFrame:
    """A pure-noise process — drift zero, modest vol. A wide MA sweep over this
    will almost certainly find a spurious winner whose true Sharpe is ~0."""
    rng = np.random.default_rng(seed)
    tickers = [f"N{i}" for i in range(n_tickers)]
    ts = pd.date_range("2022-01-03", periods=n_bars, freq="B")
    fields = ["open", "high", "low", "close", "volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["field", "ticker"])
    frame = pd.DataFrame(index=ts, columns=cols, dtype=float)
    for t in tickers:
        rets = rng.normal(0.0, 0.012, size=n_bars)
        close = 100.0 * np.cumprod(1.0 + rets)
        frame.loc[:, ("open", t)] = close
        frame.loc[:, ("high", t)] = close * 1.001
        frame.loc[:, ("low", t)] = close * 0.999
        frame.loc[:, ("close", t)] = close
        frame.loc[:, ("volume", t)] = 1_000_000.0
    return frame


def test_overfit_demo_dsr_below_threshold(tmp_path: Path) -> None:
    """Wide sweep on pure noise → DSR should land below 0.95."""
    res = run_overfit_demo(
        strategy_name="ma_crossover",
        prices_wide=_noise_wide(n_bars=500),
        out_path=tmp_path / "overfit.html",
        n_resamples=80,
        n_random_paths=40,
    )
    assert res.out_path.exists()
    assert res.n_trials >= 5
    # DSR is a probability — must always be in [0, 1].
    assert 0.0 <= res.dsr <= 1.0
    # The overfit gate is DSR < 0.95. On noise this should hold.
    assert res.dsr < 0.95, (
        f"expected DSR < 0.95 (overfit signal), got {res.dsr:.3f}. "
        "If this fires reliably, widen the grid in `_wide_sweep_grid`."
    )
    # The patched HTML must mention the deflated number.
    html = res.out_path.read_text(encoding="utf-8")
    assert "Deflated Sharpe Ratio" in html
    assert f"{res.dsr:.3f}" in html
