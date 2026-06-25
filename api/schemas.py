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
