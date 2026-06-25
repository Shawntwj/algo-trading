"""Variant-A (factor clone) vs Variant-B (literal 13F follow) comparison.

Drives both halves of BRIEF Task 3 step 6 and returns the *gap* — the
spread between A's and B's risk-adjusted returns. A positive gap means
the factor-based signal beat naive copying; negative means literal
copying did better.

Variant B implementation:
  * At each rebalance bar, pick the most recent picker 13F filing whose
    ``filing_date`` is on/before ``bar - filing_delay_days`` (BRIEF default
    45d). That filing's top-``top_n`` positions (by reported USD value)
    become the basket — equal-weight on the intersection with our investable
    universe (tickers without ClickHouse coverage are dropped silently).
  * Held until the next rebalance.

The variant B side uses a tiny ``_Literal13FFollow`` Strategy subclass
constructed inline (it isn't worth a registry entry — the only sensible
way to instantiate it is with a holdings_history closure that depends on
the comparison's `start` / `end` window).
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, run_backtest
from backtest.stats import annualised_sharpe
from data import edgar
from data.cusip_to_ticker import resolve as resolve_cusips
from research.picker_profiles import load_profile
from strategies.base import Signals, Strategy
from strategies.picker_clone import PickerCloneStrategy

log = logging.getLogger(__name__)


@dataclass
class CompareResult:
    """Holds both sides + the derived gap."""

    picker_name: str
    variant_a: BacktestResult        # factor clone
    variant_b: BacktestResult        # literal 13F follow
    sharpe_a: float
    sharpe_b: float
    sharpe_gap: float                # A - B (positive = factor signal wins)
    holdings_history: dict[str, list[str]]  # rebalance_date_iso -> tickers held


# ─── Variant B: literal 13F follower ───────────────────────────────────
class Literal13FFollow(Strategy):
    """Boolean entries/exits driven by a date→tickers map.

    The strategy doesn't compute anything itself; it consumes the holdings
    map the caller built (typically via :func:`build_holdings_history_live`
    or a hand-crafted snapshot for tests).
    """

    name = "literal_13f_follow"

    def __init__(self, holdings_history: Mapping[str, Sequence[str]], **params):
        super().__init__(**params)
        # ISO-date string → set of tickers
        self.holdings_history = {k: tuple(v) for k, v in holdings_history.items()}

    @classmethod
    def default_params(cls) -> dict:
        return {}

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close: pd.DataFrame = data["close"]
        in_basket = pd.DataFrame(False, index=close.index, columns=close.columns)

        # Walk the index; at each bar look up the latest holdings_history
        # entry with date <= bar.
        sorted_dates = sorted(self.holdings_history.keys())
        if not sorted_dates:
            return Signals(
                entries=in_basket.copy(),
                exits=in_basket.copy(),
            )

        idx_pos = 0
        current_tickers: tuple[str, ...] = ()
        col_index = in_basket.columns.tolist()
        # Pre-flatten holdings to column-index lookups for speed.
        bar_dates = [d.date().isoformat() for d in close.index]
        for i, bar_iso in enumerate(bar_dates):
            # Advance idx_pos while next event is on/before this bar.
            while idx_pos < len(sorted_dates) and sorted_dates[idx_pos] <= bar_iso:
                current_tickers = self.holdings_history[sorted_dates[idx_pos]]
                idx_pos += 1
            if not current_tickers:
                continue
            keep = [t for t in current_tickers if t in col_index]
            if not keep:
                continue
            in_basket.iloc[i, in_basket.columns.get_indexer(keep)] = True

        in_basket_prev = in_basket.shift(1, fill_value=False)
        entries = in_basket & ~in_basket_prev
        exits = ~in_basket & in_basket_prev
        return Signals(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
        )


# ─── helpers ────────────────────────────────────────────────────────────
def build_holdings_history_live(
    picker_name: str,
    start: str,
    end: str,
    *,
    top_n: int = 15,
    filing_delay_days: int = 45,
) -> dict[str, list[str]]:
    """For each picker filing within [start, end], emit the holdings list
    available ``filing_delay_days`` after the file date.

    Returned dict is keyed by the *effective date* (filing_date + delay,
    rounded forward to the calendar day). Slow — pulls EDGAR live.
    """
    cik = edgar.picker_cik(picker_name)
    filings = edgar.list_13f_filings(cik)
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)

    out: dict[str, list[str]] = {}
    for filing in filings:
        filed = dt.date.fromisoformat(filing.filing_date)
        effective = filed + dt.timedelta(days=filing_delay_days)
        if effective < start_d or effective > end_d:
            continue
        try:
            holdings = edgar.fetch_13f_holdings(cik, filing)
        except Exception as exc:  # noqa: BLE001
            log.warning("13F fetch failed for %s: %s", filing.accession, exc)
            continue
        # Equity only; aggregate by CUSIP; take top-N by value.
        equity = [h for h in holdings
                  if (h.put_call is None or h.put_call.lower() not in ("put", "call"))]
        by_cusip: dict[str, float] = {}
        for h in equity:
            by_cusip[h.cusip] = by_cusip.get(h.cusip, 0.0) + h.value_usd
        top_cusips = [c for c, _ in sorted(by_cusip.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
        resolved = resolve_cusips(top_cusips)
        tickers = [t for t in (resolved.get(c) for c in top_cusips) if t]
        out[effective.isoformat()] = tickers
    return out


def _composite_returns(result: BacktestResult) -> np.ndarray:
    eq = result.portfolio.value()
    if isinstance(eq, pd.DataFrame):
        eq = eq.mean(axis=1)
    return eq.pct_change().fillna(0.0).to_numpy(dtype=float)


# ─── orchestrator ───────────────────────────────────────────────────────
def picker_compare(
    picker_name: str,
    prices_wide: pd.DataFrame,
    *,
    holdings_history: Mapping[str, Sequence[str]] | None = None,
    profile=None,
    top_n: int = 5,
    rebalance_freq: int = 21,
    lookback_for_factors: int = 252,
    filing_delay_days: int = 45,
    commission: float = 0.0005,
    slippage: float = 0.0005,
) -> CompareResult:
    """Run Variant A + Variant B against the same ``prices_wide`` window.

    ``prices_wide`` must already be a wide pandas frame with MultiIndex
    columns (field, ticker) — i.e. the standard ``Strategy`` ABC input.

    ``holdings_history`` lets the caller pass a pre-built map (e.g.
    snapshot fixtures in tests). When omitted the function pulls it live
    from EDGAR using the price index's date range.
    """
    if profile is None:
        profile = load_profile(picker_name)

    if holdings_history is None:
        start = prices_wide.index.min().date().isoformat()
        end = prices_wide.index.max().date().isoformat()
        holdings_history = build_holdings_history_live(
            picker_name, start, end,
            top_n=top_n, filing_delay_days=filing_delay_days,
        )

    # Variant A: factor clone.
    clone = PickerCloneStrategy(
        picker_name=picker_name, profile=profile,
        top_n=top_n, rebalance_freq=rebalance_freq,
        lookback_for_factors=lookback_for_factors,
    )
    result_a = run_backtest(prices_wide, clone, commission=commission, slippage=slippage)

    # Variant B: literal follow.
    literal = Literal13FFollow(holdings_history=holdings_history)
    result_b = run_backtest(prices_wide, literal, commission=commission, slippage=slippage)

    ra = _composite_returns(result_a)
    rb = _composite_returns(result_b)
    sharpe_a = float(annualised_sharpe(ra))
    sharpe_b = float(annualised_sharpe(rb))

    return CompareResult(
        picker_name=picker_name,
        variant_a=result_a,
        variant_b=result_b,
        sharpe_a=sharpe_a,
        sharpe_b=sharpe_b,
        sharpe_gap=sharpe_a - sharpe_b,
        holdings_history={k: list(v) for k, v in holdings_history.items()},
    )
