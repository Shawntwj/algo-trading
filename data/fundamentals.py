"""yfinance-backed fundamentals fetcher for factor profiles (Task 3).

Pulls the seven fields the picker factor profile needs (BRIEF spec):

    log_market_cap, forward_pe (or trailing fallback), pb_ratio,
    momentum_12_1, roe, debt_to_equity, realised_vol_60d

The price-derived fields (momentum, realised vol) come from ClickHouse if the
ticker is backfilled, with a yfinance fallback when ClickHouse is empty or
the symbol is unknown. The balance-sheet / valuation fields come straight
from ``yfinance.Ticker(symbol).info`` — that endpoint can be flaky, so the
fetcher swallows individual failures and returns ``NaN`` for the missing
field rather than crashing the whole profile.

Output schema: ``Mapping[ticker, FactorVector]`` where ``FactorVector`` is a
``dict[str, float]`` with the seven keys above. NaNs are preserved so the
profile builder can decide whether to drop or impute.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# The seven factor fields. Order matters for any vector / cosine-similarity
# downstream so we centralise the canonical list here.
FACTOR_FIELDS: tuple[str, ...] = (
    "log_market_cap",
    "forward_pe",
    "pb_ratio",
    "momentum_12_1",
    "roe",
    "debt_to_equity",
    "realised_vol_60d",
)


@dataclass(frozen=True)
class FactorVector:
    """One ticker's factor profile."""

    ticker: str
    log_market_cap: float
    forward_pe: float
    pb_ratio: float
    momentum_12_1: float
    roe: float
    debt_to_equity: float
    realised_vol_60d: float

    def as_dict(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in FACTOR_FIELDS}

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in FACTOR_FIELDS], dtype=float)


# ─── price-derived fields ──────────────────────────────────────────────────
def momentum_12_1(close: pd.Series) -> float:
    """12-1 momentum: return from t-252 to t-21 days (BRIEF spec). Returns NaN
    when the series isn't long enough."""
    s = close.dropna()
    if len(s) < 252 + 1:
        return float("nan")
    p_now = s.iloc[-21]
    p_then = s.iloc[-252]
    if p_then <= 0 or not math.isfinite(p_then) or not math.isfinite(p_now):
        return float("nan")
    return float(p_now / p_then - 1.0)


def realised_vol_60d(close: pd.Series, annualise: bool = True) -> float:
    """Annualised 60-day realised vol of log returns. NaN if <60 obs."""
    s = close.dropna()
    if len(s) < 61:
        return float("nan")
    rets = np.log(s).diff().dropna().tail(60)
    if rets.empty:
        return float("nan")
    sd = float(rets.std(ddof=1))
    if not math.isfinite(sd):
        return float("nan")
    return sd * math.sqrt(252.0) if annualise else sd


def _load_close_from_clickhouse(
    ticker: str, end: str, lookback_days: int = 400
) -> pd.Series | None:
    """Best-effort ClickHouse fetch — returns None on any failure so the
    caller can fall back to yfinance."""
    try:
        from data.queries import load_bars  # noqa: PLC0415
    except Exception:
        return None
    try:
        end_ts = pd.Timestamp(end)
        start_ts = end_ts - pd.Timedelta(days=lookback_days * 2)  # weekends pad
        df = load_bars([ticker], start=start_ts.date().isoformat(),
                       end=end_ts.date().isoformat())
        if df.is_empty():
            return None
        pdf = df.to_pandas().sort_values("timestamp")
        return pd.Series(pdf["close"].to_numpy(), index=pd.to_datetime(pdf["timestamp"]))
    except Exception as exc:  # noqa: BLE001
        log.debug("ClickHouse close fetch failed for %s: %s", ticker, exc)
        return None


def _load_close_from_yfinance(
    ticker: str, end: str, lookback_days: int = 400
) -> pd.Series | None:
    try:
        import yfinance as yf  # noqa: PLC0415
    except Exception:
        return None
    try:
        end_ts = pd.Timestamp(end)
        start_ts = end_ts - pd.Timedelta(days=lookback_days * 2)
        df = yf.download(
            ticker,
            start=start_ts.date().isoformat(),
            end=end_ts.date().isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"][ticker] if ticker in df["Close"].columns else df["Close"].iloc[:, 0]
        else:
            close = df["Close"]
        return close.dropna()
    except Exception as exc:  # noqa: BLE001
        log.debug("yfinance close fetch failed for %s: %s", ticker, exc)
        return None


def get_close_series(
    ticker: str, end: str, lookback_days: int = 400
) -> pd.Series | None:
    """ClickHouse → yfinance lookup chain. None when both miss."""
    s = _load_close_from_clickhouse(ticker, end, lookback_days)
    if s is None or s.empty:
        s = _load_close_from_yfinance(ticker, end, lookback_days)
    return s


# ─── fundamentals via yfinance.Ticker.info ─────────────────────────────────
def _safe_float(x) -> float:
    try:
        f = float(x)
    except Exception:
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _info_block(ticker: str) -> dict:
    """Wraps ``yfinance.Ticker(t).info`` with full-exception swallow. Returns
    an empty dict on failure so downstream code uniformly sees NaNs."""
    try:
        import yfinance as yf  # noqa: PLC0415
    except Exception:
        return {}
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as exc:  # noqa: BLE001
        log.debug("yfinance.info failed for %s: %s", ticker, exc)
        return {}


def fundamentals_for(
    ticker: str,
    end: str,
    *,
    info: dict | None = None,
    close: pd.Series | None = None,
) -> FactorVector:
    """Build one ticker's :class:`FactorVector`. Both ``info`` and ``close``
    are injectable to keep the function fully testable offline."""
    info = info if info is not None else _info_block(ticker)
    close = close if close is not None else get_close_series(ticker, end)

    market_cap = _safe_float(info.get("marketCap"))
    log_mc = math.log(market_cap) if market_cap > 0 else float("nan")

    fwd_pe = _safe_float(info.get("forwardPE"))
    if not math.isfinite(fwd_pe):
        fwd_pe = _safe_float(info.get("trailingPE"))

    pb = _safe_float(info.get("priceToBook"))

    roe = _safe_float(info.get("returnOnEquity"))
    debt_eq = _safe_float(info.get("debtToEquity"))
    # yfinance reports debt/equity as a percent (e.g. 150 ≈ 1.5x). Normalise
    # to a ratio if it looks scaled.
    if math.isfinite(debt_eq) and abs(debt_eq) > 10.0:
        debt_eq = debt_eq / 100.0

    mom = momentum_12_1(close) if close is not None else float("nan")
    rv = realised_vol_60d(close) if close is not None else float("nan")

    return FactorVector(
        ticker=ticker.upper(),
        log_market_cap=log_mc,
        forward_pe=fwd_pe,
        pb_ratio=pb,
        momentum_12_1=mom,
        roe=roe,
        debt_to_equity=debt_eq,
        realised_vol_60d=rv,
    )


def fundamentals_batch(
    tickers: Iterable[str], end: str
) -> dict[str, FactorVector]:
    """Build factor vectors for ``tickers``. Failures yield NaN-filled rows."""
    out: dict[str, FactorVector] = {}
    for t in tickers:
        try:
            out[t.upper()] = fundamentals_for(t, end)
        except Exception as exc:  # noqa: BLE001
            log.warning("fundamentals failed for %s: %s", t, exc)
            out[t.upper()] = FactorVector(
                ticker=t.upper(),
                log_market_cap=float("nan"),
                forward_pe=float("nan"),
                pb_ratio=float("nan"),
                momentum_12_1=float("nan"),
                roe=float("nan"),
                debt_to_equity=float("nan"),
                realised_vol_60d=float("nan"),
            )
    return out


# ─── z-scoring helper ──────────────────────────────────────────────────────
def zscore_frame(df: pd.DataFrame, *, fill: float = 0.0) -> pd.DataFrame:
    """Column-wise z-score; NaN-tolerant (each column uses its non-NaN mean
    and std). ``fill`` replaces any remaining NaN after scaling (default 0,
    i.e. "neutral against the universe mean")."""
    mu = df.mean(axis=0, skipna=True)
    sd = df.std(axis=0, ddof=1, skipna=True).replace(0, np.nan)
    z = (df - mu) / sd
    if fill is not None:
        z = z.fillna(fill)
    return z


def vectors_to_frame(vectors: dict[str, FactorVector]) -> pd.DataFrame:
    """Stack a dict of FactorVectors into a (ticker × factor) DataFrame."""
    rows = {t: v.as_dict() for t, v in vectors.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(FACTOR_FIELDS))
