"""Thin glue between FastAPI handlers and the existing strategy/backtest code.

Keeps HTTP-layer concerns (serialization, error mapping) out of the engine.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from backtest import compare, polars_to_wide, run_backtest, sweep
from backtest.benchmarks import buy_and_hold, buy_and_hold_spy
from backtest.engine import BacktestResult
from backtest.metrics import summarize
from data import list_tickers, load_bars
from strategies import REGISTRY


# ─── Lookups ────────────────────────────────────────────────────────────────
def get_tickers() -> list[str]:
    return list_tickers()


def get_strategies() -> list[dict[str, Any]]:
    """Introspect the strategy REGISTRY — mirrors dashboard/app.py usage."""
    out: list[dict[str, Any]] = []
    for name, cls in REGISTRY.items():
        out.append(
            {
                "name": name,
                "default_params": cls.default_params(),
                "param_grid": cls.param_grid(),
            }
        )
    return out


def get_strategy_class(name: str):
    if name not in REGISTRY:
        raise KeyError(name)
    return REGISTRY[name]


# ─── JSON-safe coercion ─────────────────────────────────────────────────────
def _jsonable(value: Any) -> Any:
    """Coerce numpy / pandas scalars to JSON-safe Python primitives.

    NaN / inf become None so the response stays valid JSON."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _interval_to_freq(interval: str) -> str:
    return "1D" if interval == "1d" else interval


# ─── Data loading ───────────────────────────────────────────────────────────
def load_wide(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    df = load_bars(tickers, start=start, end=end, interval=interval)
    if df.is_empty():
        raise ValueError("No bars in ClickHouse for that selection.")
    return polars_to_wide(df)


# ─── Single backtest serialization ──────────────────────────────────────────
def _ticker_metrics(pf, ticker: str) -> dict[str, Any]:
    """Per-ticker metric slice. Mirrors what summarize() does, but for one column."""

    def _scalar(val):
        if isinstance(val, (pd.Series, pd.DataFrame)):
            try:
                if ticker in val.index:
                    return _jsonable(val.loc[ticker])
                # Single-ticker portfolios may collapse to a scalar Series
                if len(val) == 1:
                    return _jsonable(val.iloc[0])
            except Exception:
                pass
            return None
        return _jsonable(val)

    trades = pf.trades
    return {
        "total_return": _scalar(pf.total_return()),
        "sharpe": _scalar(pf.sharpe_ratio()),
        "max_drawdown": _scalar(pf.max_drawdown()),
        "win_rate": _scalar(trades.win_rate()),
    }


def serialize_backtest(result: BacktestResult, wide: pd.DataFrame) -> dict[str, Any]:
    pf = result.portfolio
    eq = pf.value()
    close = wide["close"]

    # Per-strategy entry/exit signal masks (recompute via the strategy class —
    # cheaper than digging through vbt internals).
    strat_cls = REGISTRY[result.strategy_name]
    strat = strat_cls(**result.params)
    signals = strat.generate_signals(wide)

    ticker_blocks: list[dict[str, Any]] = []
    for ticker in result.tickers:
        if isinstance(eq, pd.DataFrame):
            eq_series = eq[ticker]
        else:
            eq_series = eq

        ent_mask = signals.entries[ticker].astype(bool)
        exit_mask = signals.exits[ticker].astype(bool)

        ticker_blocks.append(
            {
                "ticker": ticker,
                "metrics": _ticker_metrics(pf, ticker),
                "equity_curve": [
                    {"timestamp": ts.isoformat(), "value": _jsonable(v)}
                    for ts, v in eq_series.items()
                    if _jsonable(v) is not None
                ],
                "entries": [ts.isoformat() for ts in close.index[ent_mask.values]],
                "exits": [ts.isoformat() for ts in close.index[exit_mask.values]],
            }
        )

    portfolio_metrics = {k: _jsonable(v) for k, v in summarize(result).items()}

    return {
        "strategy": result.strategy_name,
        "params": result.params,
        "label": result.label(),
        "portfolio_metrics": portfolio_metrics,
        "results": ticker_blocks,
    }


# ─── Sweep ──────────────────────────────────────────────────────────────────
def serialize_sweep(results: list[BacktestResult]) -> dict[str, Any]:
    if not results:
        return {"strategy": "", "results": []}

    metrics_df = compare(results)
    # compare() sorts by Sharpe desc. Map back to the matching result by label.
    by_label = {r.label(): r for r in results}

    out: list[dict[str, Any]] = []
    for _, row in metrics_df.iterrows():
        label = row["label"]
        r = by_label[label]
        out.append(
            {
                "params": r.params,
                "label": label,
                "metrics": {k: _jsonable(v) for k, v in row.to_dict().items()},
            }
        )
    return {"strategy": results[0].strategy_name, "results": out}


# ─── Runners ────────────────────────────────────────────────────────────────
def run_single_backtest(
    *,
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    strategy: str,
    params: dict[str, Any],
    commission: float,
    slippage: float,
) -> dict[str, Any]:
    strat_cls = get_strategy_class(strategy)
    wide = load_wide(tickers, start, end, interval)
    strat = strat_cls(**params)
    result = run_backtest(
        wide,
        strat,
        commission=commission,
        slippage=slippage,
        freq=_interval_to_freq(interval),
    )
    return serialize_backtest(result, wide)


def run_sweep(
    *,
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    strategy: str,
    grid: dict[str, list[Any]],
    commission: float,
    slippage: float,
) -> dict[str, Any]:
    strat_cls = get_strategy_class(strategy)
    wide = load_wide(tickers, start, end, interval)
    results = sweep(
        wide,
        strat_cls,
        grid=grid or None,
        commission=commission,
        slippage=slippage,
        freq=_interval_to_freq(interval),
    )
    return serialize_sweep(results)


# ─── Benchmarks ─────────────────────────────────────────────────────────────
def _curve_to_json(eq_df) -> list[dict[str, Any]]:
    """polars equity-curve DataFrame -> [{timestamp, value}, ...]."""
    out: list[dict[str, Any]] = []
    ts = eq_df["timestamp"].to_list()
    eq = eq_df["equity"].to_list()
    for t, v in zip(ts, eq):
        val = _jsonable(v)
        if val is None:
            continue
        out.append({"timestamp": t.isoformat() if hasattr(t, "isoformat") else str(t), "value": val})
    return out


def run_benchmarks(
    *,
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    weights: str,
    caps: dict[str, float] | None = None,
    init_cash: float = 100_000.0,
    include_spy: bool = False,
) -> dict[str, Any]:
    if weights not in {"equal", "cap"}:
        raise ValueError(f"weights must be 'equal' or 'cap', got {weights!r}")
    wide = load_wide(tickers, start, end, interval)
    eq_df, _ = buy_and_hold(wide, weights=weights, caps=caps, init_cash=init_cash)

    curves = [
        {
            "name": f"buy_and_hold_{weights}",
            "equity_curve": _curve_to_json(eq_df),
        }
    ]
    if include_spy:
        try:
            spy_eq, _ = buy_and_hold_spy(start=start, end=end, interval=interval, init_cash=init_cash)
            curves.append({"name": "buy_and_hold_spy", "equity_curve": _curve_to_json(spy_eq)})
        except Exception as exc:
            logging.getLogger(__name__).warning("SPY benchmark unavailable: %s", exc)

    return {"weights": weights, "tickers": tickers, "curves": curves}


# ─── Health ─────────────────────────────────────────────────────────────────
def clickhouse_health() -> str:
    """Return 'ok' / 'down' — never raises."""
    try:
        from data.clickhouse_client import get_client

        get_client().command("SELECT 1")
        return "ok"
    except Exception:
        return "down"
