from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def _scalar(val) -> float:
    if isinstance(val, (pd.Series, pd.DataFrame)):
        try:
            return float(val.mean())
        except Exception:
            return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def summarize(result: BacktestResult) -> dict[str, float]:
    """Aggregate per-ticker portfolio stats into a single row of metrics.

    Returns the cross-ticker mean for ratios (Sharpe, win rate, exposure) and the
    cross-ticker mean for return / drawdown. Trade count is summed."""
    pf = result.portfolio
    trades = pf.trades

    total_return = _scalar(pf.total_return())
    sharpe = _scalar(pf.sharpe_ratio())
    max_dd = _scalar(pf.max_drawdown())
    win_rate = _scalar(trades.win_rate())

    pos = pf.asset_value()
    if isinstance(pos, pd.DataFrame):
        exposure = float((pos != 0).mean().mean())
    else:
        exposure = float((pos != 0).mean())

    n_trades = int(np.nan_to_num(_scalar(trades.count()), nan=0))

    return {
        "strategy": result.strategy_name,
        "label": result.label(),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "exposure": exposure,
        **{f"param_{k}": v for k, v in result.params.items()},
    }


def compare(results: list[BacktestResult]) -> pd.DataFrame:
    rows = [summarize(r) for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("sharpe", ascending=False).reset_index(drop=True)
