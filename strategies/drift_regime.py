"""arxiv:2511.12490 — Singha (2025) drift-regime factor (FALSIFICATION TARGET).

Picked from the arxiv survey precisely *because* the paper's headline
out-of-sample Sharpe of >13 over 20 years on the S&P 500 is implausible.
The strategy itself is simple enough to implement exactly as the paper
describes (§2.1, equations 1-4); we then let the existing evaluation
gauntlet (PSR / DSR / bootstrap CI / walk-forward) examine the claim.

If the gauntlet rejects it, that's a clean negative result — proof the
PSR/DSR machinery does its job on a real paper-claimed strategy, not just
synthetic overfit demos. If by some chance it survives, we report that
honestly too.

Paper signal definition (§2.1):

  value_{i,t}    = cross-sectional percentile of (1 / price_{i,t})
                   ∈ [0, 1]                                  (paper §2.1)

  reversal_{i,t} = cross-sectional z-score of  -ret_{i, t-10..t}
                                                          (paper §2.1)

  BASE_{i,t}     = 0.7 * value + 0.3 * reversal               (eq. 1)

  UpFraction_{i,t} = (1/63) Σ_{k=1..63} 1[ret_{i,t-k} > 0]    (eq. 2)

  REGIME_{i,t}   = 1[UpFraction_{i,t} > 0.60]                 (eq. 3)

  EDGE_{i,t}     = BASE_{i,t} * REGIME_{i,t}                  (eq. 4)

Portfolio: market-neutral long-short, normalised to 50% long + 50% short
gross exposure. For the boolean Strategy ABC contract we keep just the
long leg — long_only entries on stocks whose EDGE z-score lands in the
top decile (paper builds market-neutral; this is the conservative long-
only slice). The short leg is logged as a cut in the docstring + tests.

CUTS / honest notes:
  * Long-only translation (engine constraint — `Signals` is a single
    boolean entry/exit pair). The paper's headline Sharpe is for the
    market-neutral long-short version. Our reproduction therefore tests
    a strictly weaker version of the strategy; the gauntlet finding is
    valid evidence about the long-leg edge specifically.
  * Singha (2025) uses current S&P 500 constituents over 2004–2024 (~500
    names, deliberately survivorship-biased to "ensure liquid universe").
    Our ClickHouse coverage is ~5 names — wildly undersampled. The
    DSR/PSR conclusion will reflect that, not the paper's claim.
  * No kill-switch / vol-and-DD scaling (paper eq. 5) — those are
    risk-management overlays, not signal definitions, and the gauntlet
    measures raw signal quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signals, Strategy


class DriftRegimeSingha(Strategy):
    """Drift-regime + value + reversal factor (Singha 2025) — long-only slice."""

    name = "drift_regime"
    last_edge_: pd.DataFrame | None = None  # set by generate_signals for inspection

    @classmethod
    def default_params(cls) -> dict:
        # Paper §2.1 defaults verbatim.
        return {
            "value_weight": 0.7,
            "reversal_weight": 0.3,
            "reversal_window": 10,
            "regime_window": 63,
            "up_fraction_threshold": 0.60,
            "top_decile": 0.10,  # long-only slice: top 10% of EDGE z-score
        }

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        # The paper froze the parameters; we expose a small grid so the
        # walk-forward harness has something to chew on without straying
        # too far from the paper's claim.
        return {
            "reversal_window": [5, 10, 21],
            "regime_window": [42, 63, 126],
            "up_fraction_threshold": [0.55, 0.60, 0.65],
            "top_decile": [0.10, 0.20],
        }

    # ─── core signal -------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close: pd.DataFrame = data["close"]
        rets = close.pct_change(fill_method=None)

        # value: cross-sectional percentile of inverse price (paper §2.1).
        inv_price = 1.0 / close.replace(0, np.nan)
        value = inv_price.rank(axis=1, pct=True)

        # reversal: cross-sectional z-score of -trailing-10d return.
        rev_window = int(self.params["reversal_window"])
        trailing = close.pct_change(rev_window, fill_method=None)
        contrarian = -trailing
        # cross-sectional z-score per timestamp.
        row_mu = contrarian.mean(axis=1)
        row_sd = contrarian.std(axis=1).replace(0, np.nan)
        reversal = contrarian.sub(row_mu, axis=0).div(row_sd, axis=0)

        # BASE (eq. 1).
        vw = float(self.params["value_weight"])
        rw = float(self.params["reversal_weight"])
        base = vw * value + rw * reversal

        # Drift regime gate (eqs. 2-3).
        regime_window = int(self.params["regime_window"])
        positive = (rets > 0).astype(float)
        up_fraction = positive.rolling(regime_window, min_periods=regime_window).mean()
        regime = (up_fraction > float(self.params["up_fraction_threshold"])).astype(float)

        # EDGE (eq. 4).
        edge = base * regime
        # Replace BASE NaNs that came from the leading windows with 0 so we
        # don't accidentally include a single non-NaN survivor in the top
        # decile. We keep NaN handling explicit and centralised here.
        edge = edge.where(~edge.isna(), 0.0)
        self.last_edge_ = edge

        # Long-only top-decile selector. We re-rank EDGE cross-sectionally
        # each bar (rank=1 means highest), and treat the top `top_decile`
        # fraction as the long basket. Position-open = top-decile transition
        # (entered the basket); position-close = left the basket.
        decile = float(self.params["top_decile"])
        # rank in [0, 1]; top-decile means rank ≥ 1 - decile.
        ranks = edge.rank(axis=1, pct=True)
        in_basket = ranks >= (1.0 - decile)

        # Mask out timestamps where no stock has a valid EDGE (no regime
        # active anywhere) — those rows would otherwise mark every stock as
        # "tied for top decile" because their EDGE is all-0.
        regime_active_any = (regime.sum(axis=1) > 0)
        in_basket = in_basket.where(regime_active_any, other=False)

        in_basket_prev = in_basket.shift(1, fill_value=False)
        entries = in_basket & ~in_basket_prev
        exits = ~in_basket & in_basket_prev

        return Signals(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
        )
