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
from data import list_tickers, load_bars  # noqa: F401  (load_bars used elsewhere too)
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


# ─── Stats ──────────────────────────────────────────────────────────────────
def run_stats(
    *,
    returns: list[float],
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute Sharpe / PSR / max-DD / total return with bootstrap CIs."""
    from backtest.stats import (  # local import — keeps cold-start lean
        annualised_sharpe,
        max_drawdown,
        max_drawdown_ci,
        probabilistic_sharpe_ratio,
        sharpe_ci,
        total_return_ci,
    )

    arr = np.asarray(returns, dtype=float)
    sharpe = annualised_sharpe(arr, periods_per_year=periods_per_year)
    psr = probabilistic_sharpe_ratio(
        arr, sr_benchmark=sr_benchmark, periods_per_year=periods_per_year
    )

    s_pt, s_lo, s_hi = sharpe_ci(
        arr,
        periods_per_year=periods_per_year,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )
    dd_pt, dd_lo, dd_hi = max_drawdown_ci(
        arr, n_resamples=n_resamples, alpha=alpha, seed=seed
    )
    tr_pt, tr_lo, tr_hi = total_return_ci(
        arr, n_resamples=n_resamples, alpha=alpha, seed=seed
    )

    return {
        "sharpe": _jsonable(sharpe),
        "sharpe_ci": {"point": _jsonable(s_pt), "low": _jsonable(s_lo), "high": _jsonable(s_hi)},
        "psr": _jsonable(psr),
        "max_dd": _jsonable(max_drawdown(np.cumprod(1.0 + arr))),
        "max_dd_ci": {"point": _jsonable(dd_pt), "low": _jsonable(dd_lo), "high": _jsonable(dd_hi)},
        "total_return": _jsonable(float(np.prod(1.0 + arr) - 1.0)),
        "total_return_ci": {"point": _jsonable(tr_pt), "low": _jsonable(tr_lo), "high": _jsonable(tr_hi)},
    }


# ─── Walk-forward (Task 7c) ─────────────────────────────────────────────────
def run_walkforward(
    *,
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    strategy: str,
    grid: dict[str, list[Any]],
    train_size: int,
    test_size: int,
    step: int | None,
    mode: str,
    min_train: int | None,
    periods_per_year: int,
    commission: float,
    slippage: float,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Wire the walk-forward harness to the API surface.

    Hits ClickHouse for bars, runs `walk_forward` + `aggregate_walkforward`,
    then JSON-scrubs the aggregate (NaN → None) so the response is valid JSON.
    """
    from backtest.walkforward import (  # local import — keeps cold-start lean
        WalkForwardConfig,
        aggregate_walkforward,
        walk_forward,
    )

    strat_cls = get_strategy_class(strategy)
    wide = load_wide(tickers, start, end, interval)
    cfg = WalkForwardConfig(
        train_size=train_size,
        test_size=test_size,
        step=step,
        mode=mode,  # type: ignore[arg-type]
        min_train=min_train,
    )
    folds = walk_forward(
        strat_cls,
        wide,
        param_grid=grid or strat_cls.param_grid(),
        config=cfg,
        periods_per_year=periods_per_year,
        backtest_kwargs={
            "commission": commission,
            "slippage": slippage,
            "freq": _interval_to_freq(interval),
        },
    )
    agg = aggregate_walkforward(
        folds,
        periods_per_year=periods_per_year,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )
    lo, hi = agg["oos_sharpe_ci"]
    mean = agg["oos_sharpe_mean"]
    return {
        "n_folds": agg["n_folds"],
        "oos_sharpe_mean": _jsonable(mean),
        "oos_sharpe_ci": {
            "point": _jsonable(mean),
            "low": _jsonable(lo),
            "high": _jsonable(hi),
        },
        "decay_slope": _jsonable(agg["decay_slope"]),
        # is_vs_oos: list of [IS, OOS] pairs; NaN → None.
        "is_vs_oos": [[_jsonable(a), _jsonable(b)] for a, b in agg["is_vs_oos"]],
        "folds": [
            {
                "fold_idx": f.fold_idx,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "in_sample_sharpe": _jsonable(f.in_sample_sharpe),
                "out_of_sample_sharpe": _jsonable(f.out_of_sample_sharpe),
                # Strategy params are already JSON-friendly (int/float/str).
                "selected_params": dict(f.selected_params),
            }
            for f in folds
        ],
    }


# ─── Attribution (Task 7c) ──────────────────────────────────────────────────
def run_attribution(
    *,
    strategy_returns: list[float],
    market_returns: list[float],
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Pure-compute CAPM regression — no DB hit. Synthetic-array friendly."""
    from backtest.attribution import market_attribution  # local import

    out = market_attribution(
        np.asarray(strategy_returns, dtype=float),
        np.asarray(market_returns, dtype=float),
        risk_free=risk_free,
        periods_per_year=periods_per_year,
    )
    # `residual_returns` / `systematic_returns` are large arrays — the brief
    # only asks for the scalar block, so we drop them from the API surface.
    return {
        "alpha": _jsonable(out["alpha"]),
        "alpha_annualised": _jsonable(out["alpha_annualised"]),
        "beta": _jsonable(out["beta"]),
        "alpha_t_stat": _jsonable(out["alpha_t_stat"]),
        "r_squared": _jsonable(out["r_squared"]),
        "n_obs": int(out["n_obs"]),
    }


# ─── Regime split (Task 7d) ─────────────────────────────────────────────────
def run_regime_split(
    *,
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    strategy: str,
    params: dict[str, Any],
    commission: float,
    slippage: float,
    spy_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
) -> dict[str, Any]:
    """Backtest the strategy, tag regimes from SPY/VIX, then split stats by regime.

    Strategy returns are the bar-over-bar percent change of the portfolio
    equity curve, aligned to the SPY/VIX index via inner-join on timestamp.
    """
    import polars as pl

    from backtest.regimes import tag_all
    from backtest.regime_split import split_stats_by_regime

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

    # Strategy returns: aggregate portfolio equity → pct-change.
    eq = result.portfolio.value()
    if isinstance(eq, pd.DataFrame):
        eq_series = eq.sum(axis=1)
    else:
        eq_series = eq
    rets = eq_series.pct_change().dropna()

    # SPY + VIX bars (independent of tickers list) for the same window.
    spy_df = load_bars([spy_ticker], start=start, end=end, interval=interval)
    vix_df = load_bars([vix_ticker], start=start, end=end, interval=interval)
    if spy_df.is_empty():
        raise ValueError(f"No SPY bars (`{spy_ticker}`) in the requested range.")
    if vix_df.is_empty():
        raise ValueError(f"No VIX bars (`{vix_ticker}`) in the requested range.")

    spy_pd = spy_df.to_pandas().assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
    vix_pd = vix_df.to_pandas().assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
    spy_pd = spy_pd.set_index("timestamp")[["close"]].rename(columns={"close": "spy"})
    vix_pd = vix_pd.set_index("timestamp")[["close"]].rename(columns={"close": "vix"})

    # Align everything on common timestamps (inner join).
    rets.index = pd.to_datetime(rets.index)
    aligned = rets.to_frame("ret").join(spy_pd, how="inner").join(vix_pd, how="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlap between strategy returns and SPY/VIX bars.")

    regimes_df = tag_all(
        spy_close=pl.Series(values=aligned["spy"].to_numpy()),
        vix_close=pl.Series(values=aligned["vix"].to_numpy()),
        timestamps=pl.Series(values=aligned.index.astype("int64").to_numpy()),
    )
    stats = split_stats_by_regime(
        returns=pl.Series(values=aligned["ret"].to_numpy()),
        regimes_df=regimes_df,
    )

    rows: list[dict[str, Any]] = []
    for r in stats.iter_rows(named=True):
        rows.append(
            {
                "dimension": r["dimension"],
                "regime": r["regime"],
                "n_bars": int(r["n_bars"]),
                "total_return": _jsonable(r["total_return"]),
                "sharpe": _jsonable(r["sharpe"]),
                "max_drawdown": _jsonable(r["max_drawdown"]),
                "exposure": _jsonable(r["exposure"]),
            }
        )
    return {"strategy": strategy, "regimes": rows}


# ─── Combined explainable (Task 4a) ─────────────────────────────────────────
def run_backtest_explain(
    *,
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    params: dict[str, Any],
    commission: float,
    slippage: float,
) -> dict[str, Any]:
    """Run combined_explainable end-to-end and return the standard payload
    plus the per-trade explanation stream.

    The endpoint is hard-coded to combined_explainable because the
    explanation contract only exists on that strategy. Other strategies
    do not persist a per-bar log.
    """
    from backtest.explainability import explain_trades

    strat_cls = REGISTRY["combined_explainable"]
    wide = load_wide(tickers, start, end, interval)
    strat = strat_cls(**params)
    result = run_backtest(
        wide,
        strat,
        commission=commission,
        slippage=slippage,
        freq=_interval_to_freq(interval),
    )
    payload = serialize_backtest(result, wide)
    # serialize_backtest re-instantiates the strategy internally; the
    # explanation log we want lives on the *first* strat that we ran
    # through run_backtest. Re-run a quick signal pass on the original
    # strat (already done above), and pull its log.
    explanations = explain_trades(result, strat)
    payload["explanations"] = [e.as_dict() for e in explanations]
    return payload


def get_explanation_schema(strategy: str) -> dict[str, Any]:
    """Return the JSON schema of a TradeExplanation for `strategy`.

    Only implemented for combined_explainable. Other strategy names raise
    KeyError so the FastAPI handler can map to 404 with a clear message.
    """
    if strategy != "combined_explainable":
        raise KeyError(strategy)

    strat_cls = REGISTRY["combined_explainable"]
    default_children = list(strat_cls().children)
    # Draft 2020-12-flavoured JSON Schema. Hand-rolled so the frontend
    # doesn't need pydantic-on-the-wire.
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TradeExplanation",
        "type": "object",
        "required": [
            "ticker",
            "timestamp",
            "direction",
            "weights",
            "child_signals",
            "summary",
        ],
        "properties": {
            "ticker": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "direction": {
                "type": "string",
                "enum": ["long_entry", "long_exit", "short_entry", "short_exit"],
            },
            "weights": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": "Weight per child strategy at this bar; sums to 1.",
            },
            "child_signals": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": "Normalised signal value per child at this bar.",
            },
            "summary": {
                "type": "string",
                "description": "Plain-English one-line summary.",
            },
        },
    }
    return {
        "strategy": strategy,
        "schema": schema,
        "children": default_children,
    }


# ─── Health ─────────────────────────────────────────────────────────────────
def clickhouse_health() -> str:
    """Return 'ok' / 'down' — never raises."""
    try:
        from data.clickhouse_client import get_client

        get_client().command("SELECT 1")
        return "ok"
    except Exception:
        return "down"
