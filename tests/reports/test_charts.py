"""Unit tests for ``reports/charts.py``.

We don't try to decode the PNG byte-for-byte — that would over-couple to
matplotlib's renderer. Instead we assert the magic header and a non-trivial
size to make sure the chart actually produced an image.
"""
from __future__ import annotations

import base64

import pandas as pd

from reports import charts


def test_equity_curve_chart_returns_base64_png() -> None:
    ts = pd.date_range("2022-01-01", periods=50, freq="B")
    series = {
        "strategy": [100 + i * 0.5 for i in range(50)],
        "buy-and-hold": [100 + i * 0.3 for i in range(50)],
    }
    payload = charts.equity_curve_chart(ts, series, title="test")
    assert isinstance(payload, str)
    raw = base64.b64decode(payload)
    # PNG magic bytes — verifies we got an actual image, not an HTML error blob.
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(raw) > 1_000  # non-trivial image


def test_regime_sharpe_chart_handles_nan() -> None:
    payload = charts.regime_sharpe_chart(
        dimension="trend",
        regimes=["bull", "bear"],
        sharpes=[0.8, float("nan")],
    )
    raw = base64.b64decode(payload)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_walkforward_decay_chart_handles_empty() -> None:
    payload = charts.walkforward_decay_chart([], [], slope=float("nan"))
    raw = base64.b64decode(payload)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
