from __future__ import annotations

import polars as pl

from .clickhouse_client import get_client


def load_bars(
    tickers: list[str] | str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pl.DataFrame:
    """Return OHLCV bars as a Polars DataFrame sorted by (ticker, timestamp)."""
    if isinstance(tickers, str):
        tickers = [tickers]
    if not tickers:
        return pl.DataFrame()

    client = get_client()
    placeholders = ",".join(f"'{t}'" for t in tickers)
    sql = f"""
        SELECT ticker, timestamp, open, high, low, close, volume, interval
        FROM bars FINAL
        WHERE ticker IN ({placeholders})
          AND interval = %(interval)s
          AND timestamp BETWEEN %(start)s AND %(end)s
        ORDER BY ticker, timestamp
    """
    result = client.query(
        sql,
        parameters={"interval": interval, "start": start, "end": end},
    )
    if not result.result_rows:
        return pl.DataFrame()
    return pl.DataFrame(
        {col: [row[i] for row in result.result_rows] for i, col in enumerate(result.column_names)}
    )


def list_tickers() -> list[str]:
    client = get_client()
    rows = client.query("SELECT DISTINCT ticker FROM bars ORDER BY ticker").result_rows
    return [r[0] for r in rows]
