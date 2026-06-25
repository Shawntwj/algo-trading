"""Self-contained HTML evaluation report (BRIEF Task 7e).

Run via:

    python -m reports.evaluate ma_crossover \
        --tickers AAPL,MSFT --start 2022-01-01 --end 2023-01-01 \
        --no-walk-forward

The output is a single ``.html`` file with every chart embedded as a base64
PNG — no external CDNs, no JS framework. Matches the RUNBOOK.html dark theme.

The module also exposes ``build_report(...)`` so the overfit demo and tests can
drive it without the argparse layer.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd
import polars as pl

from backtest.attribution import market_attribution
from backtest.benchmarks import buy_and_hold, buy_and_hold_spy, random_entry_monte_carlo
from backtest.engine import BacktestResult, polars_to_wide, run_backtest
from backtest.regime_split import split_stats_by_regime
from backtest.regimes import tag_all
from backtest.stats import (
    annualised_sharpe,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_sweep,
    max_drawdown_ci,
    probabilistic_sharpe_ratio,
    sharpe_ci,
    total_return_ci,
)
from backtest.sweep import sweep
from backtest.walkforward import WalkForwardConfig, aggregate_walkforward, walk_forward
from strategies import REGISTRY
from strategies.base import Strategy

from . import charts as _charts

_LOGGER = logging.getLogger(__name__)


# ─── data classes ──────────────────────────────────────────────────────────
@dataclass
class ReportConfig:
    """Inputs to ``build_report``.

    ``prices_wide`` is optional — when omitted the runner loads from
    ClickHouse using ``tickers / start / end / interval``. Tests pass a
    pre-built synthetic frame so they don't need a populated DB.
    """

    strategy: str
    tickers: list[str]
    start: str
    end: str
    interval: str = "1d"
    params: dict[str, Any] = field(default_factory=dict)
    commission: float = 0.0005
    slippage: float = 0.0005
    walk_forward: bool = False
    n_resamples: int = 1000
    n_random_paths: int = 200
    walk_forward_config: WalkForwardConfig | None = None
    walk_forward_grid: dict[str, list] | None = None
    out_path: Path | None = None
    prices_wide: pd.DataFrame | None = None
    spy_close: pd.Series | None = None
    vix_close: pd.Series | None = None


# ─── helpers ───────────────────────────────────────────────────────────────
def _resolve_strategy(name: str) -> type[Strategy]:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown strategy {name!r}. Known: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]


def _load_prices_wide(
    tickers: list[str], start: str, end: str, interval: str
) -> pd.DataFrame:
    """ClickHouse → wide pandas frame. Lazy import so synthetic-data tests
    don't need a running DB."""
    from data import load_bars  # local import: same pattern used in benchmarks.py

    df = load_bars(tickers, start=start, end=end, interval=interval)
    if df.is_empty():
        raise RuntimeError(
            f"No bars in ClickHouse for {tickers} between {start} and {end}. "
            "Backfill via `python scripts/backfill.py --tickers ...`."
        )
    return polars_to_wide(df)


def _composite_returns(result: BacktestResult) -> np.ndarray:
    """Equal-weight cross-ticker mean of per-bar portfolio returns.

    Mirrors ``walkforward._portfolio_returns`` so every Sharpe in the report is
    consistent. The first bar's NaN is forced to 0.
    """
    eq = result.portfolio.value()
    if isinstance(eq, pd.DataFrame):
        eq_series = eq.mean(axis=1)
    else:
        eq_series = eq
    return eq_series.pct_change().fillna(0.0).to_numpy(dtype=float)


def _composite_equity(result: BacktestResult) -> pd.Series:
    eq = result.portfolio.value()
    if isinstance(eq, pd.DataFrame):
        return eq.mean(axis=1)
    return eq


def _safe_float(x: Any) -> float:
    try:
        f = float(x)
    except Exception:
        return float("nan")
    return f


# ─── significance ──────────────────────────────────────────────────────────
def _significance_block(
    strat_returns: np.ndarray,
    prices_wide: pd.DataFrame,
    n_resamples: int,
    n_random_paths: int,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """PSR / DSR / random-entry p-value for the single best-config run.

    Since this is a *single* config (not a sweep), DSR collapses to PSR — we
    surface both numbers so the badge stays honest. The overfit demo plumbs in
    a real sweep via ``deflated_sharpe_ratio_from_sweep`` separately.
    """
    psr = probabilistic_sharpe_ratio(strat_returns, sr_benchmark=0.0,
                                     periods_per_year=periods_per_year)
    dsr = deflated_sharpe_ratio(
        strat_returns, n_trials=1, sr_trials_std=0.0,
        periods_per_year=periods_per_year,
    )

    # Random-entry null. Match the strategy's realised exposure (fraction of
    # non-zero-return bars) as a coarse proxy, clipped to (0.05, 0.95) to keep
    # the Bernoulli well-defined.
    strat_sharpe = annualised_sharpe(strat_returns, periods_per_year=periods_per_year)
    realised_exposure = float((strat_returns != 0).mean())
    realised_exposure = max(0.05, min(0.95, realised_exposure))
    random_p = None
    try:
        _, summary = random_entry_monte_carlo(
            prices_wide,
            n_paths=n_random_paths,
            exposure_target=realised_exposure,
            seed=42,
        )
        mc_sharpes = summary["sharpe"].to_numpy()
        # p-value = P(MC Sharpe >= strategy Sharpe under the random-entry null).
        random_p = float((mc_sharpes >= strat_sharpe).mean())
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("random-entry null skipped: %s", exc)

    return {
        "psr": float(psr),
        "dsr": float(dsr),
        "n_trials": 1,
        "sr_trials_std": 0.0,
        "random_p_value": random_p,
        "n_random_paths": n_random_paths,
    }


# ─── regime / attribution helpers ──────────────────────────────────────────
def _try_load_spy_vix(start: str, end: str, interval: str) -> tuple[pd.Series | None, pd.Series | None]:
    """Best-effort SPY+VIX close fetch. Returns (None, None) on any failure."""
    try:
        from data import load_bars  # noqa: PLC0415
    except Exception:
        return None, None
    try:
        spy = load_bars(["SPY"], start=start, end=end, interval=interval)
    except Exception:
        spy = pl.DataFrame()
    try:
        vix = load_bars(["^VIX"], start=start, end=end, interval=interval)
    except Exception:
        vix = pl.DataFrame()

    def _to_close(df: pl.DataFrame) -> pd.Series | None:
        if df.is_empty():
            return None
        p = df.to_pandas()
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        p = p.set_index("timestamp").sort_index()
        return p["close"]

    return _to_close(spy), _to_close(vix)


def _regimes_block(
    strat_returns: np.ndarray,
    timestamps: pd.Index,
    spy_close: pd.Series | None,
    vix_close: pd.Series | None,
) -> dict[str, Any] | None:
    """Per-regime stats + bar charts. Returns ``None`` if SPY/VIX unavailable
    or the alignment yields fewer than a handful of bars."""
    if spy_close is None or vix_close is None:
        return None
    aligned = (
        pd.DataFrame({"ret": strat_returns}, index=timestamps)
        .join(spy_close.rename("spy"), how="inner")
        .join(vix_close.rename("vix"), how="inner")
        .dropna()
    )
    if len(aligned) < 30:
        return None

    regime_df = tag_all(
        spy_close=pl.Series(aligned["spy"].to_numpy()),
        vix_close=pl.Series(aligned["vix"].to_numpy()),
        timestamps=pl.Series(aligned.index.astype("int64").to_numpy()),
    )
    stats = split_stats_by_regime(
        returns=pl.Series(aligned["ret"].to_numpy()),
        regimes_df=regime_df,
    )
    rows = stats.to_dicts()

    # One bar chart per dimension.
    chart_pngs: dict[str, str] = {}
    for dim in ("trend", "vol", "drawdown"):
        dim_rows = [r for r in rows if r["dimension"] == dim]
        if not dim_rows:
            continue
        chart_pngs[dim] = _charts.regime_sharpe_chart(
            dimension=dim,
            regimes=[r["regime"] for r in dim_rows],
            sharpes=[r["sharpe"] for r in dim_rows],
        )
    return {"rows": rows, "charts": chart_pngs}


def _attribution_block(
    strat_returns: np.ndarray,
    timestamps: pd.Index,
    spy_close: pd.Series | None,
    periods_per_year: int = 252,
) -> dict[str, Any] | None:
    """Plain-OLS CAPM attribution vs SPY. None when SPY unavailable."""
    if spy_close is None:
        return None
    spy_ret = spy_close.pct_change().fillna(0.0)
    aligned = (
        pd.DataFrame({"strat": strat_returns}, index=timestamps)
        .join(spy_ret.rename("spy"), how="inner")
        .dropna()
    )
    if len(aligned) < 30:
        return None
    out = market_attribution(
        strategy_returns=aligned["strat"].to_numpy(),
        market_returns=aligned["spy"].to_numpy(),
        periods_per_year=periods_per_year,
    )
    out = {k: v for k, v in out.items() if k not in {"residual_returns", "systematic_returns"}}
    out["periods_per_year"] = periods_per_year
    return out


# ─── walk-forward block ────────────────────────────────────────────────────
def _walkforward_block(
    strategy_cls: type[Strategy],
    prices_wide: pd.DataFrame,
    config: WalkForwardConfig,
    grid: dict[str, list] | None,
    n_resamples: int,
) -> dict[str, Any]:
    grid = grid if grid is not None else strategy_cls.param_grid()
    folds = walk_forward(strategy_cls, prices_wide, grid, config)
    agg = aggregate_walkforward(folds, n_resamples=n_resamples)
    is_xs = [pair[0] for pair in agg["is_vs_oos"]]
    oos_ys = [pair[1] for pair in agg["is_vs_oos"]]
    chart = (
        _charts.walkforward_decay_chart(is_xs, oos_ys, agg["decay_slope"])
        if folds
        else None
    )
    return {
        "n_folds": agg["n_folds"],
        "oos_sharpe_mean": agg["oos_sharpe_mean"],
        "oos_sharpe_ci_lo": agg["oos_sharpe_ci"][0],
        "oos_sharpe_ci_hi": agg["oos_sharpe_ci"][1],
        "decay_slope": agg["decay_slope"],
        "chart": chart,
    }


# ─── headline / equity / price ─────────────────────────────────────────────
def _headline_metrics(
    result: BacktestResult,
    strat_returns: np.ndarray,
    n_resamples: int,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Sharpe / total return / max DD with stationary-bootstrap CIs, plus
    win rate / n_trades / exposure (point estimates only)."""
    sharpe_pt, sharpe_lo, sharpe_hi = sharpe_ci(
        strat_returns, periods_per_year=periods_per_year, n_resamples=n_resamples
    )
    tr_pt, tr_lo, tr_hi = total_return_ci(strat_returns, n_resamples=n_resamples)
    dd_pt, dd_lo, dd_hi = max_drawdown_ci(strat_returns, n_resamples=n_resamples)

    pf = result.portfolio
    try:
        win_rate = _safe_float(pf.trades.win_rate())
    except Exception:
        win_rate = float("nan")
    try:
        n_trades_raw = pf.trades.count()
        if hasattr(n_trades_raw, "sum"):
            n_trades = int(n_trades_raw.sum())
        else:
            n_trades = int(n_trades_raw)
    except Exception:
        n_trades = 0
    try:
        pos = pf.asset_value()
        if isinstance(pos, pd.DataFrame):
            exposure = float((pos != 0).mean().mean())
        else:
            exposure = float((pos != 0).mean())
    except Exception:
        exposure = float("nan")

    def _safe_pct(x: float) -> float:
        return 0.0 if not math.isfinite(x) else x

    return {
        "sharpe": {"point": sharpe_pt, "lo": sharpe_lo, "hi": sharpe_hi},
        "total_return": {"point": tr_pt, "lo": tr_lo, "hi": tr_hi},
        "max_drawdown": {"point": dd_pt, "lo": dd_lo, "hi": dd_hi},
        "win_rate": _safe_pct(win_rate),
        "n_trades": n_trades,
        "exposure": _safe_pct(exposure),
    }


def _equity_chart(
    result: BacktestResult,
    prices_wide: pd.DataFrame,
    start: str,
    end: str,
    interval: str,
) -> tuple[str | None, bool]:
    """Strategy equity vs equal-weight buy-and-hold (+ SPY if available)."""
    strat_eq = _composite_equity(result)
    if strat_eq.empty:
        return None, False
    series: dict[str, list[float]] = {"strategy": strat_eq.tolist()}
    timestamps = list(strat_eq.index)

    # Equal-weight universe buy-and-hold.
    try:
        bh_eq, _ = buy_and_hold(prices_wide, weights="equal", init_cash=float(strat_eq.iloc[0]))
        bh_pdf = bh_eq.to_pandas().set_index("timestamp")["equity"]
        bh_pdf.index = pd.to_datetime(bh_pdf.index)
        bh_pdf = bh_pdf.reindex(strat_eq.index, method="nearest")
        series["buy-and-hold (equal)"] = bh_pdf.tolist()
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("buy-and-hold benchmark skipped: %s", exc)

    # SPY buy-and-hold (optional — needs the SPY backfill).
    have_spy = False
    try:
        spy_eq, _ = buy_and_hold_spy(start=start, end=end, interval=interval,
                                     init_cash=float(strat_eq.iloc[0]))
        spy_pdf = spy_eq.to_pandas().set_index("timestamp")["equity"]
        spy_pdf.index = pd.to_datetime(spy_pdf.index)
        spy_pdf = spy_pdf.reindex(strat_eq.index, method="nearest")
        series["SPY"] = spy_pdf.tolist()
        have_spy = True
    except Exception as exc:  # noqa: BLE001
        _LOGGER.info("SPY benchmark unavailable: %s", exc)

    return _charts.equity_curve_chart(timestamps, series), have_spy


def _price_chart(
    result: BacktestResult,
    prices_wide: pd.DataFrame,
    first_ticker: str,
) -> str | None:
    if first_ticker not in prices_wide["close"].columns:
        return None
    close = prices_wide["close"][first_ticker].dropna()
    if close.empty:
        return None

    try:
        trades = result.portfolio.trades.records_readable
    except Exception:
        trades = pd.DataFrame()

    entry_ts: list = []
    exit_ts: list = []
    if not trades.empty:
        col_set = set(trades.columns)
        # vectorbt's `records_readable` uses "Column" for ticker; filter when present.
        if "Column" in col_set:
            tr = trades[trades["Column"] == first_ticker]
        else:
            tr = trades
        for col_name, bucket in [
            ("Entry Timestamp", entry_ts),
            ("Exit Timestamp", exit_ts),
        ]:
            if col_name in tr.columns:
                bucket.extend(pd.to_datetime(tr[col_name]).tolist())
    return _charts.price_with_markers_chart(
        timestamps=close.index.tolist(),
        close=close.tolist(),
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        ticker=first_ticker,
    )


# ─── orchestration ─────────────────────────────────────────────────────────
def _render_template(context: dict[str, Any]) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=str(Path(__file__).parent / "templates")),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("report.html.j2")
    return tmpl.render(**context)


def build_report(cfg: ReportConfig) -> Path:
    """End-to-end: run the backtest, compute stats, render HTML, write file."""
    strategy_cls = _resolve_strategy(cfg.strategy)

    # 1. Resolve prices.
    prices_wide = cfg.prices_wide
    if prices_wide is None:
        prices_wide = _load_prices_wide(
            cfg.tickers, cfg.start, cfg.end, cfg.interval
        )
    if prices_wide.empty:
        raise RuntimeError("prices_wide is empty — nothing to backtest")

    # 2. Run the backtest.
    params = cfg.params or strategy_cls.default_params()
    strategy = strategy_cls(**params)
    result = run_backtest(
        prices_wide, strategy,
        commission=cfg.commission, slippage=cfg.slippage,
    )
    strat_returns = _composite_returns(result)
    timestamps = prices_wide.index

    # 3. Sections.
    headline = _headline_metrics(result, strat_returns, n_resamples=cfg.n_resamples)
    equity_png, have_spy_bench = _equity_chart(
        result, prices_wide, cfg.start, cfg.end, cfg.interval
    )
    tickers_in_data = list(prices_wide["close"].columns)
    first_ticker = tickers_in_data[0] if tickers_in_data else cfg.tickers[0]
    price_png = _price_chart(result, prices_wide, first_ticker)
    significance = _significance_block(
        strat_returns, prices_wide,
        n_resamples=cfg.n_resamples, n_random_paths=cfg.n_random_paths,
    )

    spy_close = cfg.spy_close
    vix_close = cfg.vix_close
    if spy_close is None or vix_close is None:
        loaded_spy, loaded_vix = _try_load_spy_vix(cfg.start, cfg.end, cfg.interval)
        spy_close = spy_close if spy_close is not None else loaded_spy
        vix_close = vix_close if vix_close is not None else loaded_vix

    regimes = _regimes_block(strat_returns, timestamps, spy_close, vix_close)
    attribution = _attribution_block(strat_returns, timestamps, spy_close)

    walkforward = None
    if cfg.walk_forward:
        wf_cfg = cfg.walk_forward_config or WalkForwardConfig(
            train_size=max(60, len(prices_wide) // 3),
            test_size=max(20, len(prices_wide) // 6),
            mode="expanding",
        )
        try:
            walkforward = _walkforward_block(
                strategy_cls, prices_wide, wf_cfg, cfg.walk_forward_grid,
                n_resamples=cfg.n_resamples,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("walk-forward skipped: %s", exc)

    # 4. Render.
    generated_at = _dt.datetime.now().isoformat(timespec="seconds")
    context = {
        "strategy_name": strategy_cls.name,
        "tickers": tickers_in_data,
        "start": cfg.start,
        "end": cfg.end,
        "interval": cfg.interval,
        "generated_at": generated_at,
        "params_json": json.dumps(params, sort_keys=True),
        "commission": cfg.commission,
        "slippage": cfg.slippage,
        "n_resamples": cfg.n_resamples,
        "headline": headline,
        "charts": {"equity": equity_png, "price": price_png},
        "benchmarks_have_spy": have_spy_bench,
        "first_ticker": first_ticker,
        "significance": significance,
        "regimes": regimes,
        "attribution": attribution,
        "walkforward": walkforward,
    }
    html = _render_template(context)

    out_path = cfg.out_path
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{strategy_cls.name}_{ts}.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ─── CLI ───────────────────────────────────────────────────────────────────
def _parse_params(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    return json.loads(s)


def _introspect_default_tickers(limit: int = 5) -> list[str]:
    """First ``limit`` tickers in ClickHouse, or a sensible fallback."""
    try:
        from data import list_tickers  # noqa: PLC0415

        ts = list_tickers()
        return ts[:limit] if ts else ["AAPL"]
    except Exception:  # noqa: BLE001
        return ["AAPL"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reports.evaluate",
        description="Generate a self-contained HTML evaluation report.",
    )
    parser.add_argument("strategy", help=f"Strategy registry key (one of {sorted(REGISTRY.keys())})")
    parser.add_argument("--tickers", default=None,
                        help="Comma-separated. Default: first 5 from ClickHouse.")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--params", default=None,
                        help="JSON string overriding the strategy's default_params.")
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--out", default=None, help="Output HTML path.")
    parser.add_argument("--n-resamples", type=int, default=1000,
                        help="Bootstrap resamples for CIs.")
    parser.add_argument("--n-random-paths", type=int, default=200,
                        help="Monte Carlo paths for the random-entry null.")
    wf = parser.add_mutually_exclusive_group()
    wf.add_argument("--walk-forward", dest="walk_forward", action="store_true")
    wf.add_argument("--no-walk-forward", dest="walk_forward", action="store_false")
    parser.set_defaults(walk_forward=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else _introspect_default_tickers()
    )
    cfg = ReportConfig(
        strategy=args.strategy,
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        params=_parse_params(args.params),
        commission=args.commission,
        slippage=args.slippage,
        walk_forward=args.walk_forward,
        n_resamples=args.n_resamples,
        n_random_paths=args.n_random_paths,
        out_path=Path(args.out) if args.out else None,
    )
    out = build_report(cfg)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
