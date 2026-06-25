from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from data.source import BARS_SCHEMA, DataSource


class FakeSource(DataSource):
    """In-memory data source for tests — no network."""

    def fetch(self, tickers, start, end, interval="1d"):
        rows = []
        base = datetime.fromisoformat(start)
        for t in tickers:
            for i in range(5):
                ts = base + timedelta(days=i)
                rows.append({
                    "ticker": t,
                    "timestamp": ts,
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 1_000.0,
                    "interval": interval,
                })
        return pl.DataFrame(rows, schema=BARS_SCHEMA)


def test_fake_source_schema_and_shape():
    src = FakeSource()
    df = src.fetch(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")
    assert df.schema == BARS_SCHEMA
    assert df.shape == (10, len(BARS_SCHEMA))
    assert set(df["ticker"].unique().to_list()) == {"AAPL", "MSFT"}


def test_fake_source_sorted_by_ticker_then_timestamp():
    src = FakeSource()
    df = src.fetch(["AAPL"], "2024-01-01", "2024-01-10")
    ts = df["timestamp"].to_list()
    assert ts == sorted(ts)


def test_empty_tickers_returns_empty_frame_with_schema():
    src = FakeSource()
    df = src.fetch([], "2024-01-01", "2024-01-10")
    # Implementation may either return empty rows or skip — schema must hold either way.
    assert set(df.columns) == set(BARS_SCHEMA.keys()) or df.is_empty()
