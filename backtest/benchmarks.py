"""Benchmark equity curves and null distributions.

Every Sharpe number the platform reports should be compared against a benchmark
(see BRIEF Task 7.1). This module produces:

  * buy-and-hold the same universe (equal- or cap-weighted)
  * buy-and-hold SPY (single instrument convenience wrapper around the data layer)
  * random-entry Monte Carlo with matched average exposure — the null distribution
    for "is the strategy better than coin-flipping in the same market?"

Conventions:
  * `prices_wide` is the same wide pandas DataFrame consumed by `backtest.engine`:
    MultiIndex columns `(field, ticker)` with at least a `close` field.
  * Equity curves are returned as polars DataFrames with a `timestamp` column and
    one `equity` (or per-path) column. Per-bar returns are returned alongside as a
    second polars DataFrame so downstream stats code can avoid re-differencing.
  * Monte Carlo bulk output stays in numpy (n_paths × n_bars matrix); converting
    half a million floats to polars adds no value for the stats consumer.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl


# ─── helpers ────────────────────────────────────────────────────────────────
def _extract_close(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """Return the close-price slice as a tickers-wide pandas frame.

    Accepts the MultiIndex `(field, ticker)` shape produced by `polars_to_wide`,
    or a plain `(timestamps × tickers)` close frame for callers that already
    pre-sliced."""
    if isinstance(prices_wide.columns, pd.MultiIndex):
        if "close" not in prices_wide.columns.get_level_values(0):
            raise ValueError("prices_wide is MultiIndex but has no `close` field")
        return prices_wide["close"]
    return prices_wide


def _equity_to_polars(equity: pd.Series | pd.DataFrame, name: str = "equity") -> pl.DataFrame:
    if isinstance(equity, pd.Series):
        return pl.DataFrame(
            {"timestamp": list(equity.index), name: equity.to_numpy().astype(float)}
        )
    out = {"timestamp": list(equity.index)}
    for col in equity.columns:
        out[str(col)] = equity[col].to_numpy().astype(float)
    return pl.DataFrame(out)


# ─── 1. Buy-and-hold the universe ───────────────────────────────────────────
def buy_and_hold(
    prices_wide: pd.DataFrame,
    weights: Literal["equal", "cap"] = "equal",
    caps: dict[str, float] | pl.Series | None = None,
    init_cash: float = 100_000.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Static buy-and-hold portfolio across the input universe.

    weights="equal" assigns 1/N to each ticker; weights="cap" requires the caller
    to pass `caps` (a dict ticker→cap or a polars Series indexed by ticker). We
    deliberately do NOT fetch live market caps here — that's a caller concern.

    Returns
    -------
    equity_df : polars.DataFrame
        columns = ["timestamp", "equity"]
    returns_df : polars.DataFrame
        columns = ["timestamp", "ret"] — simple per-bar return of the portfolio.
    """
    close = _extract_close(prices_wide).sort_index()
    if close.empty:
        raise ValueError("prices_wide is empty")

    # Forward-fill so a stale ticker doesn't drag the whole curve to NaN, then
    # drop leading rows where no ticker has yet started trading.
    close = close.ffill().dropna(how="all")
    tickers = list(close.columns)

    if weights == "equal":
        w = np.full(len(tickers), 1.0 / len(tickers))
    elif weights == "cap":
        if caps is None:
            raise ValueError("weights='cap' requires a `caps` mapping")
        if isinstance(caps, pl.Series):
            cap_map = dict(zip(caps.to_list(), caps.to_list()))  # pragma: no cover
            raise ValueError(
                "caps as pl.Series must have ticker→value semantics; pass a dict instead"
            )
        try:
            cap_vec = np.array([float(caps[t]) for t in tickers], dtype=float)
        except KeyError as exc:
            raise ValueError(f"caps missing ticker {exc!s}") from exc
        if (cap_vec <= 0).any():
            raise ValueError("caps must be strictly positive")
        w = cap_vec / cap_vec.sum()
    else:
        raise ValueError(f"unknown weights={weights!r} (use 'equal' or 'cap')")

    # Shares purchased at the first valid price per ticker. A ticker that has no
    # data at t0 enters at its first non-NaN bar.
    first_price = close.bfill().iloc[0].to_numpy(dtype=float)
    if (first_price <= 0).any():
        raise ValueError("non-positive first price encountered")
    dollar_alloc = init_cash * w
    shares = dollar_alloc / first_price

    # Daily portfolio value = Σ shares_i × price_i(t)
    values = close.to_numpy(dtype=float) @ shares
    equity = pd.Series(values, index=close.index, name="equity")
    rets = equity.pct_change().fillna(0.0).rename("ret")

    return _equity_to_polars(equity, "equity"), _equity_to_polars(rets, "ret")


# ─── 2. SPY buy-and-hold (data-layer wrapper) ───────────────────────────────
def buy_and_hold_spy(
    start: str,
    end: str,
    interval: str = "1d",
    init_cash: float = 100_000.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Buy-and-hold SPY for the requested window.

    Loads from ClickHouse first; on a miss (no rows / table empty), falls back
    to a one-shot YFinance backfill via `data.backfill.backfill_ticker` so the
    next call is cached. Never re-implements the fetch path.
    """
    # Local imports keep the module importable in environments without a
    # configured ClickHouse / yfinance install (e.g. unit-test of `buy_and_hold`).
    from data import load_bars  # noqa: PLC0415

    df = load_bars(["SPY"], start=start, end=end, interval=interval)
    if df.is_empty():
        try:
            from data import backfill_ticker  # noqa: PLC0415

            backfill_ticker("SPY", start=start, end=end, interval=interval)
            df = load_bars(["SPY"], start=start, end=end, interval=interval)
        except Exception as exc:  # pragma: no cover - network / CH-down path
            raise RuntimeError(f"SPY unavailable from data layer: {exc}") from exc
    if df.is_empty():
        raise RuntimeError("SPY backfill returned no rows")

    pdf = df.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.set_index("timestamp").sort_index()
    close = pdf[["close"]].rename(columns={"close": "SPY"})
    close.columns = pd.MultiIndex.from_product([["close"], close.columns], names=["field", "ticker"])
    return buy_and_hold(close, weights="equal", init_cash=init_cash)


# ─── 3. Random-entry Monte Carlo with matched exposure ──────────────────────
def random_entry_monte_carlo(
    prices_wide: pd.DataFrame,
    n_paths: int = 500,
    exposure_target: float = 0.5,
    seed: int = 42,
    init_cash: float = 100_000.0,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Null distribution: random in/out flips with average exposure = target.

    For each path we draw a Bernoulli(p=exposure_target) mask per bar per ticker
    (equal-weight across the universe at any held bar). The portfolio's per-bar
    return is `exposure * underlying_return` where `underlying_return` is the
    equal-weight mean return of the universe.

    Returns
    -------
    equity_matrix : np.ndarray, shape (n_paths, n_bars)
        Each row is one MC path's equity curve in dollars.
    summary : polars.DataFrame
        Per-path summary with columns ["path", "total_return", "sharpe",
        "max_drawdown", "exposure"]. Exposure is the realised mean — should be
        within ±5% of `exposure_target` for n_paths in the hundreds.
    """
    if not (0.0 < exposure_target < 1.0):
        raise ValueError("exposure_target must be in (0, 1)")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    close = _extract_close(prices_wide).ffill().dropna(how="all")
    if close.empty:
        raise ValueError("prices_wide is empty")
    rets = close.pct_change().fillna(0.0).to_numpy(dtype=float)  # (n_bars, n_tickers)
    n_bars, n_tickers = rets.shape
    if n_bars < 2:
        raise ValueError("need at least 2 bars to compute returns")

    rng = np.random.default_rng(seed)
    # Per-path, per-bar, per-ticker hold mask. For wide universes / long
    # windows this scales as n_paths × n_bars × n_tickers — fine for the
    # default 500 × ~1000 × 10 = 5M booleans (~5 MB).
    holds = rng.random((n_paths, n_bars, n_tickers)) < exposure_target

    # Equal-weight on held tickers each bar (avoid div0 with where-clause).
    held_count = holds.sum(axis=2)  # (n_paths, n_bars)
    weights = np.where(
        held_count[..., None] > 0,
        holds / np.maximum(held_count, 1)[..., None],
        0.0,
    )
    # Per-path per-bar portfolio return.
    path_rets = (weights * rets[None, :, :]).sum(axis=2)  # (n_paths, n_bars)

    equity_matrix = init_cash * np.cumprod(1.0 + path_rets, axis=1)

    # Per-path stats. Sharpe is annualised assuming daily bars (252) — good
    # enough for the null benchmark; the rigorous Sharpe machinery is Task 7b.
    mu = path_rets.mean(axis=1)
    sigma = path_rets.std(axis=1, ddof=1)
    sharpe = np.where(sigma > 0, mu / sigma * np.sqrt(252.0), 0.0)
    total_return = equity_matrix[:, -1] / init_cash - 1.0
    running_max = np.maximum.accumulate(equity_matrix, axis=1)
    drawdown = equity_matrix / running_max - 1.0
    max_dd = drawdown.min(axis=1)
    # Exposure := fraction of the universe held, averaged across bars per path.
    # With i.i.d. Bernoulli(p) holds, E[exposure] = p = exposure_target.
    realised_exposure = (held_count / n_tickers).mean(axis=1)

    summary = pl.DataFrame(
        {
            "path": np.arange(n_paths, dtype=np.int64),
            "total_return": total_return.astype(float),
            "sharpe": sharpe.astype(float),
            "max_drawdown": max_dd.astype(float),
            "exposure": realised_exposure.astype(float),
        }
    )

    return equity_matrix, summary
