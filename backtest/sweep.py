from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from strategies.base import Strategy
from .engine import BacktestResult, run_backtest


def _grid(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not params:
        return [{}]
    keys = list(params.keys())
    return [dict(zip(keys, combo)) for combo in product(*[params[k] for k in keys])]


def sweep(
    data: pd.DataFrame,
    strategy_cls: type[Strategy],
    grid: dict[str, list[Any]] | None = None,
    commission: float = 0.0005,
    slippage: float = 0.0005,
    init_cash: float = 100_000.0,
    freq: str = "1D",
) -> list[BacktestResult]:
    """Run a strategy across the cartesian product of its parameter grid.

    Pass `grid=None` to use the strategy's declared `param_grid()`."""
    grid = grid if grid is not None else strategy_cls.param_grid()
    combos = _grid(grid)

    results: list[BacktestResult] = []
    for params in combos:
        # Skip invalid combos (e.g. fast >= slow in MA crossover).
        if "fast" in params and "slow" in params and params["fast"] >= params["slow"]:
            continue
        strat = strategy_cls(**params)
        results.append(
            run_backtest(
                data,
                strat,
                commission=commission,
                slippage=slippage,
                init_cash=init_cash,
                freq=freq,
            )
        )
    return results
