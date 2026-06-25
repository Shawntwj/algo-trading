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


# ─── Stats ──────────────────────────────────────────────────────────────────
class StatsRequest(BaseModel):
    returns: list[float] = Field(
        min_length=2,
        description="Simple periodic returns (NOT equity, NOT log returns).",
    )
    sr_benchmark: float = Field(
        default=0.0,
        description="Annualised Sharpe threshold for PSR; defaults to 0.",
    )
    periods_per_year: int = Field(
        default=252,
        description="Annualisation factor. 252 for daily; pass 252*78 for 5-min RTH.",
    )
    n_resamples: int = Field(
        default=1000,
        ge=100,
        description="Number of bootstrap resamples for the CIs.",
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="(1 - alpha) confidence level for bootstrap CIs.",
    )
    seed: int = Field(default=42, description="Bootstrap RNG seed (reproducibility).")


class CIBlock(BaseModel):
    point: float
    low: float
    high: float


class StatsResponse(BaseModel):
    sharpe: float
    sharpe_ci: CIBlock
    psr: float
    max_dd: float
    max_dd_ci: CIBlock
    total_return: float
    total_return_ci: CIBlock


# ─── Walk-forward (Task 7c) ─────────────────────────────────────────────────
class WalkForwardRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    start: str
    end: str
    interval: str = "1d"
    strategy: str
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    train_size: int = Field(gt=1, description="Bars per train window.")
    test_size: int = Field(gt=0, description="Bars per test window.")
    step: int | None = Field(
        default=None, description="Bars between fold starts. Defaults to test_size."
    )
    mode: str = Field(
        default="expanding",
        description="'expanding' (train grows) or 'rolling' (fixed-length train).",
    )
    min_train: int | None = Field(
        default=None,
        description="Floor for the first expanding-fold train window (in bars).",
    )
    periods_per_year: int = 252
    commission: float = 0.0005
    slippage: float = 0.0005
    n_resamples: int = Field(default=500, ge=100)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    seed: int = 42


class FoldEntry(BaseModel):
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    in_sample_sharpe: float | None
    out_of_sample_sharpe: float | None
    selected_params: dict[str, Any]


class WalkForwardResponse(BaseModel):
    n_folds: int
    oos_sharpe_mean: float | None
    oos_sharpe_ci: CIBlock
    decay_slope: float | None
    is_vs_oos: list[list[float | None]]
    folds: list[FoldEntry]


# ─── Attribution (Task 7c) ──────────────────────────────────────────────────
class AttributionRequest(BaseModel):
    strategy_returns: list[float] = Field(min_length=2)
    market_returns: list[float] = Field(min_length=2)
    risk_free: float = 0.0
    periods_per_year: int = 252


class AttributionResponse(BaseModel):
    alpha: float
    alpha_annualised: float
    beta: float
    alpha_t_stat: float | None
    r_squared: float | None
    n_obs: int


# ─── Regime split (Task 7d) ─────────────────────────────────────────────────
class RegimeSplitRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    start: str
    end: str
    interval: str = "1d"
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    commission: float = 0.0005
    slippage: float = 0.0005
    spy_ticker: str = Field(
        default="SPY",
        description="Symbol used as the SPY proxy for trend/drawdown regimes.",
    )
    vix_ticker: str = Field(
        default="^VIX",
        description="Symbol used as the VIX proxy for the volatility regime.",
    )


class RegimeStat(BaseModel):
    dimension: str
    regime: str
    n_bars: int
    total_return: float | None
    sharpe: float | None
    max_drawdown: float | None
    exposure: float | None


class RegimeSplitResponse(BaseModel):
    strategy: str
    regimes: list[RegimeStat]


# ─── Combined explainable (Task 4a) ────────────────────────────────────────
class TradeExplanationModel(BaseModel):
    """JSON-safe shape of a backtest.explainability.TradeExplanation."""

    ticker: str
    timestamp: str
    direction: str = Field(
        description="long_entry | long_exit | short_entry | short_exit"
    )
    weights: dict[str, float]
    child_signals: dict[str, float]
    summary: str


class BacktestExplainResponse(BacktestResponse):
    """BacktestResponse + per-trade explanation list (Task 4a only)."""

    explanations: list[TradeExplanationModel]


class ExplanationSchemaResponse(BaseModel):
    """JSON Schema (Draft 2020-12) describing the explanation object.

    Frontend (Task 4b) uses this to render fields generically. Only emitted
    for strategies that implement the explanation contract — currently just
    ``combined_explainable``.
    """

    strategy: str
    schema_: dict[str, Any] = Field(
        alias="schema",
        description="JSON Schema for one TradeExplanation entry.",
    )
    children: list[str] = Field(
        description="Names of the child strategies the explanation will reference."
    )

    model_config = {"populate_by_name": True}
