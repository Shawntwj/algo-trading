from .engine import run_backtest, BacktestResult, polars_to_wide
from .metrics import summarize, compare
from .sweep import sweep
from .benchmarks import buy_and_hold, buy_and_hold_spy, random_entry_monte_carlo
from .regimes import tag_trend, tag_volatility, tag_drawdown, tag_all
from .regime_split import split_stats_by_regime

__all__ = [
    "run_backtest",
    "BacktestResult",
    "polars_to_wide",
    "summarize",
    "compare",
    "sweep",
    "buy_and_hold",
    "buy_and_hold_spy",
    "random_entry_monte_carlo",
    "tag_trend",
    "tag_volatility",
    "tag_drawdown",
    "tag_all",
    "split_stats_by_regime",
]
