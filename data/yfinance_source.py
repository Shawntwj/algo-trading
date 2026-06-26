from __future__ import annotations

import logging

import polars as pl
import yfinance as yf

from .source import BARS_SCHEMA, DataSource

log = logging.getLogger(__name__)


class YFinanceSource(DataSource):
    """yfinance-backed OHLCV source. Wraps yf.download and normalizes output
    into the canonical Polars schema."""

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pl.DataFrame:
        if not tickers:
            return pl.DataFrame(schema=BARS_SCHEMA)

        # group_by='ticker' gives a consistent MultiIndex regardless of len(tickers).
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if raw is None or raw.empty:
            log.warning("yfinance returned no rows for %s", tickers)
            return pl.DataFrame(schema=BARS_SCHEMA)

        frames: list[pl.DataFrame] = []

        if len(tickers) == 1:
            ticker = tickers[0]
            df = raw.copy()
            # yfinance 0.2+ returns MultiIndex columns even for n=1 in some
            # versions; flatten by dropping whichever level holds the ticker.
            if df.columns.nlevels > 1:
                level_0 = df.columns.get_level_values(0)
                drop = 0 if ticker in set(level_0) else -1
                df.columns = df.columns.droplevel(drop)
            df = df.reset_index()
            df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
            frames.append(_to_polars(df, ticker, interval))
        else:
            for ticker in tickers:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                sub = raw[ticker].dropna(how="all").reset_index()
                if sub.empty:
                    continue
                sub.columns = [c.lower() if isinstance(c, str) else c for c in sub.columns]
                frames.append(_to_polars(sub, ticker, interval))

        if not frames:
            return pl.DataFrame(schema=BARS_SCHEMA)

        return pl.concat(frames).sort(["ticker", "timestamp"])


def _to_polars(df, ticker: str, interval: str) -> pl.DataFrame:
    # yfinance uses "Date" for daily, "Datetime" for intraday.
    ts_col = "date" if "date" in df.columns else "datetime"
    df = df.rename(columns={ts_col: "timestamp"})
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[keep].copy()
    df["ticker"] = ticker
    df["interval"] = interval
    pdf = pl.from_pandas(df)
    pdf = pdf.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us")),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )
    return pdf.select(list(BARS_SCHEMA.keys()))
