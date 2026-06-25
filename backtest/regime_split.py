"""Split a strategy's return stream by regime tag (BRIEF Task 7.2 continued).

Pairs with `backtest.regimes.tag_all`. Given a per-bar strategy returns series
and a regime DataFrame produced by `tag_all`, computes per-regime Sharpe, total
return, max drawdown, and exposure.

Kept as its own file rather than extending `engine.py` because `engine.py` is
the vectorbt boundary — adding pure-Polars analytics there would muddy that
contract. `metrics.py` is the existing place for cross-result aggregation; this
file is the per-strategy regime-conditional analogue and lives alongside it.
"""
from __future__ import annotations

import numpy as np
import polars as pl


_DIM_COLS = ("trend", "vol", "drawdown")
_ANNUALISATION = 252.0  # daily bars; coarse but matches the rest of the harness


def _sharpe(rets: np.ndarray) -> float:
    if rets.size < 2:
        return float("nan")
    sigma = float(rets.std(ddof=1))
    if sigma == 0.0:
        return float("nan")
    return float(rets.mean() / sigma * np.sqrt(_ANNUALISATION))


def _max_drawdown(rets: np.ndarray) -> float:
    if rets.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + rets)
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return float(dd.min())


def _exposure(rets: np.ndarray) -> float:
    if rets.size == 0:
        return float("nan")
    # Approximation: fraction of bars with a non-zero return. Good enough as a
    # regime-relative "how active was the strategy here" signal. A true exposure
    # (held positions / time) lives in the engine — call sites that need it can
    # pass that pre-computed.
    return float((rets != 0).mean())


def _total_return(rets: np.ndarray) -> float:
    if rets.size == 0:
        return float("nan")
    return float(np.prod(1.0 + rets) - 1.0)


def split_stats_by_regime(
    returns: pl.Series | pl.DataFrame,
    regimes_df: pl.DataFrame,
) -> pl.DataFrame:
    """Per-regime stats for every regime dimension present in `regimes_df`.

    Parameters
    ----------
    returns : pl.Series or pl.DataFrame
        Per-bar returns. If a DataFrame, must contain a `ret` column (and may
        optionally contain `timestamp` for safety alignment, but length-equality
        with `regimes_df` is the binding contract).
    regimes_df : pl.DataFrame
        Output of `regimes.tag_all`. Any of {"trend", "vol", "drawdown"} columns
        present will be split on.

    Returns
    -------
    pl.DataFrame with columns
        ["dimension", "regime", "n_bars", "total_return", "sharpe",
         "max_drawdown", "exposure"]
        sorted by (dimension, regime).
    """
    if isinstance(returns, pl.DataFrame):
        if "ret" not in returns.columns:
            raise ValueError("returns DataFrame must contain a 'ret' column")
        rets = returns["ret"].to_numpy().astype(float)
    else:
        rets = returns.to_numpy().astype(float)

    if rets.size != regimes_df.height:
        raise ValueError(
            f"returns length ({rets.size}) != regimes_df height ({regimes_df.height})"
        )

    dims = [d for d in _DIM_COLS if d in regimes_df.columns]
    if not dims:
        raise ValueError(f"regimes_df has none of {_DIM_COLS}")

    rows: list[dict] = []
    for dim in dims:
        labels = regimes_df[dim].to_numpy()
        # Preserve original label ordering so downstream tables are stable.
        seen: list[str] = []
        for lbl in labels:
            if lbl not in seen:
                seen.append(lbl)
        for label in seen:
            mask = labels == label
            sub = rets[mask]
            rows.append(
                {
                    "dimension": dim,
                    "regime": str(label),
                    "n_bars": int(mask.sum()),
                    "total_return": _total_return(sub),
                    "sharpe": _sharpe(sub),
                    "max_drawdown": _max_drawdown(sub),
                    "exposure": _exposure(sub),
                }
            )

    return pl.DataFrame(rows).sort(["dimension", "regime"])
