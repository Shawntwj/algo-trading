from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from data.yfinance_source import YFinanceSource


def _fake_index(n: int = 3) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range("2024-01-02", periods=n, freq="D"), name="Date")


def _flat_frame() -> pd.DataFrame:
    idx = _fake_index()
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Adj Close": [100.5, 101.5, 102.5],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=idx,
    )


def _multiindex_frame(ticker: str) -> pd.DataFrame:
    df = _flat_frame()
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def test_single_ticker_flat_columns_does_not_crash():
    src = YFinanceSource()
    with patch("data.yfinance_source.yf.download", return_value=_flat_frame()):
        out = src.fetch(["SPY"], "2024-01-01", "2024-01-05")
    assert out.height == 3
    assert set(out.columns) >= {"ticker", "timestamp", "open", "high", "low", "close", "volume"}
    assert out["ticker"].unique().to_list() == ["SPY"]


def test_single_ticker_multiindex_columns_does_not_crash():
    # Regression: yfinance 0.2+ sometimes returns MultiIndex even for n=1.
    src = YFinanceSource()
    with patch("data.yfinance_source.yf.download", return_value=_multiindex_frame("SPY")):
        out = src.fetch(["SPY"], "2024-01-01", "2024-01-05")
    assert out.height == 3
    assert out["ticker"].unique().to_list() == ["SPY"]
    assert out["close"].to_list() == [100.5, 101.5, 102.5]


def test_empty_response_returns_empty_frame_with_schema():
    src = YFinanceSource()
    with patch("data.yfinance_source.yf.download", return_value=pd.DataFrame()):
        out = src.fetch(["SPY"], "2024-01-01", "2024-01-05")
    assert out.height == 0
    assert "ticker" in out.columns
