from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─── Health ─────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(description="Overall API status")
    clickhouse: str = Field(description="'ok' if ClickHouse responded, else 'down'")


# ─── Strategies ─────────────────────────────────────────────────────────────
class StrategyInfo(BaseModel):
    name: str
    default_params: dict[str, Any]
    param_grid: dict[str, list[Any]]


# ─── Backtest ───────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    start: str
    end: str
    interval: str = "1d"
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    commission: float = 0.0005
    slippage: float = 0.0005


class EquityPoint(BaseModel):
    timestamp: str
    value: float


class TickerBacktest(BaseModel):
    ticker: str
    metrics: dict[str, Any]
    equity_curve: list[EquityPoint]
    entries: list[str]
    exits: list[str]


class BacktestResponse(BaseModel):
    strategy: str
    params: dict[str, Any]
    label: str
    portfolio_metrics: dict[str, Any]
    results: list[TickerBacktest]


# ─── Sweep ──────────────────────────────────────────────────────────────────
class SweepRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    start: str
    end: str
    interval: str = "1d"
    strategy: str
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    commission: float = 0.0005
    slippage: float = 0.0005


class SweepEntry(BaseModel):
    params: dict[str, Any]
    label: str
    metrics: dict[str, Any]


class SweepResponse(BaseModel):
    strategy: str
    results: list[SweepEntry]


# ─── Benchmarks ─────────────────────────────────────────────────────────────
class BenchmarkRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    start: str
    end: str
    interval: str = "1d"
    weights: str = Field(
        default="equal",
        description="'equal' or 'cap'. 'cap' requires a non-empty `caps` mapping.",
    )
    caps: dict[str, float] | None = Field(
        default=None,
        description="Optional ticker→market-cap mapping; required when weights='cap'.",
    )
    init_cash: float = 100_000.0
    include_spy: bool = Field(
        default=False,
        description="If true, also return the SPY buy-and-hold curve for the same window.",
    )


class BenchmarkCurve(BaseModel):
    name: str
    equity_curve: list[EquityPoint]


class BenchmarkResponse(BaseModel):
    weights: str
    tickers: list[str]
    curves: list[BenchmarkCurve]
