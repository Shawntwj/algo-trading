"""Smoke tests for `reports.evaluate` (BRIEF Task 7e).

Runs against synthetic prices so we don't need ClickHouse populated. The
walk-forward leg is gated on a separate (slower) test to keep the headline
smoke fast — the fast test sets ``walk_forward=False`` and only verifies the
report renders end-to-end with all other sections present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reports.evaluate import ReportConfig, build_report
from backtest.walkforward import WalkForwardConfig


def _synthetic_wide(n_bars: int = 260, n_tickers: int = 2, seed: int = 0) -> pd.DataFrame:
    """Two-ticker, ~one-year frame in the same shape as `polars_to_wide`."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    ts = pd.date_range("2022-01-03", periods=n_bars, freq="B")
    fields = ["open", "high", "low", "close", "volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["field", "ticker"])
    frame = pd.DataFrame(index=ts, columns=cols, dtype=float)
    for t in tickers:
        rets = rng.normal(0.0005, 0.012, size=n_bars)
        close = 100.0 * np.cumprod(1.0 + rets)
        frame.loc[:, ("open", t)] = close
        frame.loc[:, ("high", t)] = close * 1.001
        frame.loc[:, ("low", t)] = close * 0.999
        frame.loc[:, ("close", t)] = close
        frame.loc[:, ("volume", t)] = 1_000_000.0
    return frame


def test_build_report_smoke(tmp_path: Path) -> None:
    """End-to-end: build the report from a synthetic frame and verify the
    output file exists, is non-empty, contains the expected section headers,
    and stays under 5 MB."""
    out = tmp_path / "smoke.html"
    cfg = ReportConfig(
        strategy="ma_crossover",
        tickers=["T0", "T1"],
        start="2022-01-03",
        end="2023-01-03",
        params={"fast": 10, "slow": 30},
        walk_forward=False,
        n_resamples=80,           # keep the bootstrap cheap
        n_random_paths=40,        # ditto for the random-entry null
        prices_wide=_synthetic_wide(),
        out_path=out,
    )
    path = build_report(cfg)
    assert path == out
    assert path.exists()
    size = path.stat().st_size
    assert size > 1_000  # non-empty
    assert size < 5_000_000  # under 5 MB

    html = path.read_text(encoding="utf-8")
    for marker in (
        "<h2>Headline metrics</h2>",
        "<h2>Equity curve vs benchmarks</h2>",
        "<h2>Price with entry / exit markers</h2>",
        "<h2>Significance</h2>",
        "<h2>Regime breakdown</h2>",
        "<h2>Market attribution (CAPM vs SPY)</h2>",
        "<h2>Reproducibility</h2>",
        "ma_crossover",
    ):
        assert marker in html, f"missing marker: {marker!r}"
    # Walk-forward section is skipped when --no-walk-forward.
    assert "<h2>Walk-forward (IS vs OOS)</h2>" not in html


def test_build_report_with_walkforward(tmp_path: Path) -> None:
    """A separate, smaller-grid test that asserts the walk-forward section
    renders when enabled. Two folds × tiny grid keep wall-clock under a few
    seconds."""
    out = tmp_path / "wf.html"
    cfg = ReportConfig(
        strategy="ma_crossover",
        tickers=["T0", "T1"],
        start="2022-01-03",
        end="2023-01-03",
        params={"fast": 10, "slow": 30},
        walk_forward=True,
        walk_forward_config=WalkForwardConfig(
            train_size=120, test_size=60, mode="expanding"
        ),
        walk_forward_grid={"fast": [10, 20], "slow": [30, 50]},
        n_resamples=40,
        n_random_paths=20,
        prices_wide=_synthetic_wide(n_bars=260),
        out_path=out,
    )
    path = build_report(cfg)
    html = path.read_text(encoding="utf-8")
    assert "<h2>Walk-forward (IS vs OOS)</h2>" in html
    assert "Decay slope" in html


def test_unknown_strategy_raises(tmp_path: Path) -> None:
    cfg = ReportConfig(
        strategy="not_a_strategy",
        tickers=["T0"],
        start="2022-01-03",
        end="2023-01-03",
        prices_wide=_synthetic_wide(),
        out_path=tmp_path / "x.html",
    )
    with pytest.raises(KeyError, match="not_a_strategy"):
        build_report(cfg)
