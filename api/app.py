from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import services
from .schemas import (
    BacktestRequest,
    BacktestResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    HealthResponse,
    StatsRequest,
    StatsResponse,
    StrategyInfo,
    SweepRequest,
    SweepResponse,
)

log = logging.getLogger(__name__)

app = FastAPI(
    title="Algo Trading API",
    version="0.1.0",
    description="Backend for the research SPA. Wraps the existing vectorbt engine.",
)

# Open CORS to the two common local frontend dev ports (Vite + CRA/Next).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", clickhouse=services.clickhouse_health())


@app.get("/tickers", response_model=list[str])
def tickers() -> list[str]:
    try:
        return services.get_tickers()
    except Exception as exc:
        log.exception("list_tickers failed")
        raise HTTPException(status_code=503, detail=f"ClickHouse unavailable: {exc}")


@app.get("/strategies", response_model=list[StrategyInfo])
def strategies() -> list[StrategyInfo]:
    return [StrategyInfo(**s) for s in services.get_strategies()]


@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    try:
        payload = services.run_single_backtest(
            tickers=req.tickers,
            start=req.start,
            end=req.end,
            interval=req.interval,
            strategy=req.strategy,
            params=req.params,
            commission=req.commission,
            slippage=req.slippage,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {req.strategy}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("backtest failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return BacktestResponse(**payload)


@app.post("/benchmarks", response_model=BenchmarkResponse)
def benchmarks(req: BenchmarkRequest) -> BenchmarkResponse:
    """Buy-and-hold benchmark equity curves for the requested universe.

    Used by Task 7d to show strategy returns alongside an equal/cap-weight
    baseline (and optionally SPY) — never in isolation.
    """
    try:
        payload = services.run_benchmarks(
            tickers=req.tickers,
            start=req.start,
            end=req.end,
            interval=req.interval,
            weights=req.weights,
            caps=req.caps,
            init_cash=req.init_cash,
            include_spy=req.include_spy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("benchmarks failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return BenchmarkResponse(**payload)


@app.post("/stats", response_model=StatsResponse)
def stats_endpoint(req: StatsRequest) -> StatsResponse:
    """Significance statistics (BRIEF Task 7b) for a single returns series.

    Pure compute — no DB hit. Caller supplies a vector of periodic returns and
    receives Sharpe / PSR / max-DD / total return with stationary-bootstrap CIs.
    """
    try:
        payload = services.run_stats(
            returns=req.returns,
            sr_benchmark=req.sr_benchmark,
            periods_per_year=req.periods_per_year,
            n_resamples=req.n_resamples,
            alpha=req.alpha,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("stats failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return StatsResponse(**payload)


@app.post("/sweep", response_model=SweepResponse)
def sweep_endpoint(req: SweepRequest) -> SweepResponse:
    try:
        payload = services.run_sweep(
            tickers=req.tickers,
            start=req.start,
            end=req.end,
            interval=req.interval,
            strategy=req.strategy,
            grid=req.grid,
            commission=req.commission,
            slippage=req.slippage,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {req.strategy}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("sweep failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return SweepResponse(**payload)
