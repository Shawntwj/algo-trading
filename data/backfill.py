from __future__ import annotations

import logging
from datetime import date

import polars as pl

from .clickhouse_client import ensure_schema, get_client
from .source import BARS_SCHEMA, DataSource
from .yfinance_source import YFinanceSource

log = logging.getLogger(__name__)


def _default_source() -> DataSource:
    return YFinanceSource()


def _insert(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    client = get_client()
    cols = list(BARS_SCHEMA.keys())
    rows = df.select(cols).rows()
    client.insert("bars", rows, column_names=cols)
    # ReplacingMergeTree deduplicates lazily; force a merge on the touched partitions
    # so subsequent queries see the deduped state.
    client.command("OPTIMIZE TABLE bars FINAL DEDUPLICATE")
    return len(rows)


def backfill_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    source: DataSource | None = None,
) -> int:
    """Fetch a single ticker's history and upsert into ClickHouse. Idempotent
    via ReplacingMergeTree on (ticker, interval, timestamp)."""
    ensure_schema()
    src = source or _default_source()
    df = src.fetch([ticker], start=start, end=end, interval=interval)
    n = _insert(df)
    log.info("backfilled %s [%s..%s %s] -> %d rows", ticker, start, end, interval, n)
    return n


def backfill_universe(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    source: DataSource | None = None,
) -> dict[str, int]:
    """Bulk backfill. Returns rows-written per ticker."""
    ensure_schema()
    src = source or _default_source()
    df = src.fetch(tickers, start=start, end=end, interval=interval)
    counts: dict[str, int] = {}
    if df.is_empty():
        return {t: 0 for t in tickers}
    for ticker, part in df.partition_by("ticker", as_dict=True).items():
        # partition_by keys are tuples on newer Polars
        key = ticker[0] if isinstance(ticker, tuple) else ticker
        counts[key] = _insert(part)
    log.info("backfilled %d tickers", len(counts))
    return counts


def update_latest(
    tickers: list[str],
    interval: str = "1d",
    source: DataSource | None = None,
    lookback_days: int = 7,
) -> dict[str, int]:
    """Fetch the last `lookback_days` of bars and upsert. Safe to run on a schedule —
    ReplacingMergeTree collapses overlap."""
    from datetime import timedelta

    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    return backfill_universe(tickers, start=start, end=end, interval=interval, source=source)
