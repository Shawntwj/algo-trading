from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import polars as pl


@dataclass(frozen=True)
class Bar:
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str


BARS_SCHEMA = {
    "ticker": pl.Utf8,
    "timestamp": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "interval": pl.Utf8,
}


class DataSource(ABC):
    """Provider-agnostic OHLCV source. Implementations should return a Polars
    DataFrame matching BARS_SCHEMA, sorted by (ticker, timestamp)."""

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pl.DataFrame:
        ...
