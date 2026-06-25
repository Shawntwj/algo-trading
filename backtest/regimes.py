"""Deterministic per-bar regime tags (BRIEF Task 7.2).

All rules are public, fully specified, and reproducible from SPY + VIX closes:

  * trend     — bull if SPY > 200-day SMA, else bear
  * volatility — terciles of VIX *over the input window* (not a fixed lookback);
                 documented as low / mid / high. Window-relative terciles keep
                 the labels meaningful across very different historical periods.
  * drawdown  — depth below the rolling 1-year (252-bar) high of SPY:
                  > -5%               -> "calm"
                  -5%  >= dd > -15%   -> "mild"
                  <= -15%             -> "severe"
                Thresholds are intentionally conservative; tune in a later task
                once we calibrate against realised regime breaks (see IMPROVEMENTS).

`tag_all` returns a single timestamp-aligned polars DataFrame so the regime-split
helper can join on `timestamp` without re-aligning indexes.

VIX note: `^VIX` is a yfinance ticker like any other; the existing data layer
handles it without modification (see `data/yfinance_source.py`). The BRIEF asks
for it to be backfilled via the existing pipeline — that's exactly
`backfill_ticker("^VIX", ...)` or `python scripts/backfill.py --tickers ^VIX`.
No special-casing is required because yfinance accepts the caret-prefixed
index ticker directly and `data/yfinance_source.py` is provider-agnostic about
the symbol it receives.
"""
from __future__ import annotations

import numpy as np
import polars as pl


# ─── helpers ────────────────────────────────────────────────────────────────
def _to_numpy(series: pl.Series) -> np.ndarray:
    return series.to_numpy().astype(float)


def _ensure_series(x) -> pl.Series:
    if isinstance(x, pl.Series):
        return x
    return pl.Series(values=x)


# ─── trend ──────────────────────────────────────────────────────────────────
def tag_trend(spy_close: pl.Series, window: int = 200) -> pl.Series:
    """Bull if SPY > rolling 200-day SMA, else bear.

    The first `window-1` bars have an undefined SMA — we label those "bear"
    conservatively (no trend evidence yet). Returns a polars Utf8 series the
    same length as the input.
    """
    s = _ensure_series(spy_close)
    arr = _to_numpy(s)
    if arr.size == 0:
        return pl.Series(name="trend", values=[], dtype=pl.Utf8)

    sma = (
        pl.Series(arr)
        .rolling_mean(window_size=window, min_samples=window)
        .to_numpy()
    )
    out = np.where(arr > sma, "bull", "bear")
    # Pre-warmup bars: sma is NaN → comparison is False → labelled bear above.
    return pl.Series(name="trend", values=out, dtype=pl.Utf8)


# ─── volatility ─────────────────────────────────────────────────────────────
def tag_volatility(vix_close: pl.Series) -> pl.Series:
    """Tercile-split VIX over the supplied window into low / mid / high.

    Window-relative on purpose: VIX 18 was "high" in 2017 and "low" in 2020.
    """
    s = _ensure_series(vix_close)
    arr = _to_numpy(s)
    if arr.size == 0:
        return pl.Series(name="vol", values=[], dtype=pl.Utf8)

    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        # Not enough points to split into terciles — single-bucket fallback.
        return pl.Series(name="vol", values=["mid"] * arr.size, dtype=pl.Utf8)

    q33, q66 = np.quantile(finite, [1 / 3, 2 / 3])
    out = np.where(arr <= q33, "low", np.where(arr <= q66, "mid", "high"))
    # NaNs land in "high" via the chained where above; explicitly relabel.
    out = np.where(np.isfinite(arr), out, "mid")
    return pl.Series(name="vol", values=out, dtype=pl.Utf8)


# ─── drawdown ───────────────────────────────────────────────────────────────
# Documented thresholds (see module docstring): tune in a later task.
_DD_CALM = -0.05
_DD_MILD = -0.15


def tag_drawdown(spy_close: pl.Series, window: int = 252) -> pl.Series:
    """Drawdown depth from the rolling `window`-bar high of SPY.

    dd >  -5%               -> "calm"
    -5% >= dd >  -15%       -> "mild"
    dd <= -15%              -> "severe"
    """
    s = _ensure_series(spy_close)
    arr = _to_numpy(s)
    if arr.size == 0:
        return pl.Series(name="drawdown", values=[], dtype=pl.Utf8)

    # Rolling max with an *expanding* fallback so the first bars get a sane
    # baseline instead of NaN (otherwise every early bar would be labelled
    # "calm" by default which is fine, but expanding is more honest).
    rolling_max = (
        pl.Series(arr)
        .rolling_max(window_size=window, min_samples=1)
        .to_numpy()
    )
    dd = arr / rolling_max - 1.0
    out = np.where(dd > _DD_CALM, "calm", np.where(dd > _DD_MILD, "mild", "severe"))
    return pl.Series(name="drawdown", values=out, dtype=pl.Utf8)


# ─── combined ───────────────────────────────────────────────────────────────
def tag_all(
    spy_close: pl.Series,
    vix_close: pl.Series,
    timestamps: pl.Series | None = None,
) -> pl.DataFrame:
    """Combine all three taggers into one DataFrame.

    `spy_close` and `vix_close` must already be timestamp-aligned (same length,
    same bar grid). Pass `timestamps` if you want a `timestamp` column in the
    output — required for joining with strategy returns in `regime_split`.
    """
    spy = _ensure_series(spy_close)
    vix = _ensure_series(vix_close)
    if len(spy) != len(vix):
        raise ValueError(
            f"spy_close and vix_close must be the same length (got {len(spy)} vs {len(vix)})"
        )

    cols: dict[str, pl.Series] = {
        "trend": tag_trend(spy),
        "vol": tag_volatility(vix),
        "drawdown": tag_drawdown(spy),
    }
    if timestamps is not None:
        if len(timestamps) != len(spy):
            raise ValueError("timestamps length must match spy_close length")
        cols = {"timestamp": _ensure_series(timestamps), **cols}
    return pl.DataFrame(cols)
