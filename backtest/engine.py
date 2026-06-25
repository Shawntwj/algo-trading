from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl
import vectorbt as vbt

from strategies.base import Strategy


@dataclass
class BacktestResult:
    portfolio: vbt.Portfolio
    strategy_name: str
    params: dict[str, Any]
    tickers: list[str]

    def equity_curve(self) -> pd.DataFrame:
        return self.portfolio.value()

    def trades(self) -> pd.DataFrame:
        return self.portfolio.trades.records_readable

    def label(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.strategy_name}({params})"


def polars_to_wide(df: pl.DataFrame) -> pd.DataFrame:
    """Convert long-format ClickHouse output to a wide pandas frame with a
    MultiIndex on columns: (field, ticker). vectorbt-ready."""
    pdf = df.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    wide = pdf.pivot(index="timestamp", columns="ticker", values=["open", "high", "low", "close", "volume"])
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide.sort_index()


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    commission: float = 0.0005,
    slippage: float = 0.0005,
    init_cash: float = 100_000.0,
    freq: str = "1D",
) -> BacktestResult:
    """Run a single strategy/parameter combo across all tickers in `data`."""
    signals = strategy.generate_signals(data)
    close = data["close"]

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=signals.entries,
        exits=signals.exits,
        init_cash=init_cash,
        fees=commission,
        slippage=slippage,
        freq=freq,
    )

    return BacktestResult(
        portfolio=portfolio,
        strategy_name=strategy.name,
        params=dict(strategy.params),
        tickers=list(close.columns),
    )
