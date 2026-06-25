from __future__ import annotations

from dagster import (
    AssetExecutionContext,
    MetadataValue,
    StaticPartitionsDefinition,
    asset,
)

from config import load_settings
from data import backfill_ticker

_settings = load_settings()
ticker_partitions = StaticPartitionsDefinition(_settings.universe)


@asset(
    partitions_def=ticker_partitions,
    group_name="market_data",
    description="OHLCV bars in ClickHouse, partitioned by ticker. Materializing a "
                "partition runs backfill_ticker for that ticker over the configured range.",
)
def bars(context: AssetExecutionContext) -> None:
    ticker = context.partition_key
    settings = load_settings()
    rows = backfill_ticker(
        ticker=ticker,
        start=settings.backfill_start,
        end=settings.end_date,
        interval=settings.intervals[0],
    )
    context.add_output_metadata(
        {
            "ticker": ticker,
            "rows_written": MetadataValue.int(rows),
            "interval": settings.intervals[0],
            "start": settings.backfill_start,
            "end": settings.end_date,
        }
    )
