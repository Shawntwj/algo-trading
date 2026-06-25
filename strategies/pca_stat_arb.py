"""arxiv:2512.02037 — Adamczyk & Dąbrowski (2025) PCA residual mean-reversion.

Classical Avellaneda & Lee (2010) statistical-arbitrage, as re-described in
section 3.1 (PCA setup) and section 2.3 / eq. (2.10) (OU + trading rules) of
Adamczyk & Dąbrowski (2025). The paper sticks to Avellaneda-Lee's empirically
optimal cut-offs for ETF residuals (eq. 2.10):

    g_open_long  = -1.25   open long when normalised OU residual < -1.25
    g_open_short = +1.25   open short when normalised OU residual > +1.25
    g_close_long = -0.5    close long when residual > -0.5
    g_close_short= +0.75   close short when residual <  +0.75

The PCA setup (paper §3.1):
  * rolling window of W = 252 trading days of returns
  * fit PCA on cross-sectional return matrix, take the first r = 15 components
    (paper sticks with Avellaneda-Lee's r=15 — explains ~50% of total variance
    on a 60-name Polish basket; the same fraction holds on US large-caps).
  * each stock is regressed on the r eigen-portfolios to produce residual
    returns ε_t. The cumulative residual X_t = Σ ε_s is fit to an OU process,
    which yields a per-stock equilibrium mean m_i and stddev σ_eq_i.
  * the trade signal is the normalised residual s_i,t = (X_t - m_i) / σ_eq_i
    (the "g" in the paper's eq. 2.10).

Per the Strategy ABC contract this module emits **per-stock long/short entry
and exit booleans** the engine can run through vectorbt. The harness's
``run_backtest`` does not natively support shorts (it consumes the signal as a
single (entries, exits) boolean pair) — we therefore emit long-only entries
from the long-side of the rule (residual < -entry_z) and exits from the
long-close rule, which is the conservative half of the strategy. Short
half is documented in IMPROVEMENTS as a follow-up.

CUTS / honest notes:
  * Universe in the paper is 60 Polish equities (WIG20+mWIG40). Our
    ClickHouse coverage is ~5 US large-caps. With <15 names the PCA fit is
    rank-deficient and we cap r at min(n_factors, n_tickers - 1). When
    n_tickers < 3 the strategy emits empty signals and lets the gauntlet
    show the cut.
  * Long-only execution per the above engine constraint.
  * OU fit uses the AR(1) regression shortcut (paper eq. between §2.3),
    consistent with Avellaneda-Lee's discrete approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signals, Strategy


class PCAStatArb(Strategy):
    """PCA residual stat-arb (Avellaneda-Lee 2010 / Adamczyk-Dąbrowski 2025)."""

    name = "pca_stat_arb"

    @classmethod
    def default_params(cls) -> dict:
        # Defaults straight from the paper (eq. 2.10 / §3.1).
        return {
            "window": 252,        # PCA + OU lookback in bars (paper: 252)
            "n_factors": 15,      # eigenportfolios kept (paper: r=15)
            "entry_z": 1.25,      # |s| > entry_z opens a position (eq. 2.10)
            "exit_z": 0.5,        # |s| < exit_z closes a long (eq. 2.10 g_cl)
            "min_kappa": 0.0,     # discard fits with κ ≤ this (mean-reversion speed)
        }

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        # Tunable knobs. Avellaneda-Lee's defaults sit inside the grid so the
        # walk-forward picker can recover them.
        return {
            "window": [126, 252],
            "n_factors": [3, 5, 15],
            "entry_z": [1.0, 1.25, 1.5],
            "exit_z": [0.25, 0.5, 0.75],
        }

    # ─── core signal -------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close: pd.DataFrame = data["close"].copy()
        rets = close.pct_change(fill_method=None).fillna(0.0)
        n_bars, n_tickers = rets.shape

        # Per paper §3.1 we need at least n_factors + a few names to keep
        # the PCA fit non-degenerate; otherwise emit zero-signal so the
        # gauntlet (and the test) can see the cut.
        if n_tickers < 3:
            empty = pd.DataFrame(False, index=close.index, columns=close.columns)
            return Signals(entries=empty, exits=empty.copy())

        window = int(self.params["window"])
        # Cap r at min(requested, n_tickers - 1) — paper's r=15 needs ≥16 names.
        r = max(1, min(int(self.params["n_factors"]), n_tickers - 1))
        entry_z = float(self.params["entry_z"])
        exit_z = float(self.params["exit_z"])
        min_kappa = float(self.params["min_kappa"])

        # Pre-allocate the s-score frame, NaN until the first window completes.
        s = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        rets_arr = rets.to_numpy(dtype=float)
        for t in range(window, n_bars):
            R = rets_arr[t - window:t]  # shape (window, n_tickers)
            R_mean = R.mean(axis=0, keepdims=True)
            R_std = R.std(axis=0, ddof=1, keepdims=True)
            R_std = np.where(R_std == 0, 1e-12, R_std)
            Y = (R - R_mean) / R_std  # standardised returns

            # PCA via SVD on standardised returns — eigen-portfolios are rows
            # of Vt scaled by 1/σ_i (Avellaneda-Lee 2010 §3, eq. 13).
            try:
                _, _, Vt = np.linalg.svd(Y, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            r_eff = min(r, Vt.shape[0])
            # Eigen-portfolio loadings (weights on each stock):
            # Q_k,i = V_k,i / σ_i  (so portfolio return = Σ_i Q_k,i * R_i)
            Q = Vt[:r_eff] / R_std  # shape (r_eff, n_tickers)

            # Factor returns over the window: F_t,k = R_t · Q_k.
            F = R @ Q.T  # shape (window, r_eff)

            # Per-stock OLS regression: r_i = α_i + Σ_k β_i,k F_k + ε_i.
            # Build the design once; fit per stock for residuals.
            X = np.column_stack([np.ones(window), F])  # (window, 1 + r_eff)
            try:
                # Solve all stocks at once: β has shape (1 + r_eff, n_tickers).
                betas, *_ = np.linalg.lstsq(X, R, rcond=None)
            except np.linalg.LinAlgError:
                continue
            residuals = R - X @ betas  # (window, n_tickers)

            # Cumulative residual process X_t per stock (paper §2.3).
            cum = residuals.cumsum(axis=0)  # (window, n_tickers)

            # OU via AR(1) on cum: cum_t = a + b * cum_{t-1} + ζ.
            # κ = -ln(b) * 252; m = a / (1 - b); σ_eq = sqrt(Var(ζ) / (1-b²)).
            X_prev = cum[:-1]
            X_curr = cum[1:]
            ones = np.ones_like(X_prev[:, :1])
            for i in range(n_tickers):
                x0 = X_prev[:, i]
                x1 = X_curr[:, i]
                # 2x2 OLS in closed form (cheap, avoids per-stock lstsq).
                A = np.column_stack([ones[:, 0], x0])
                try:
                    coef, *_ = np.linalg.lstsq(A, x1, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                a, b = coef
                if not (0.0 < b < 1.0):
                    continue  # not mean-reverting on this window
                kappa = -np.log(b) * 252.0
                if kappa <= min_kappa:
                    continue
                m = a / (1.0 - b)
                resid = x1 - (a + b * x0)
                var_zeta = float(resid.var(ddof=1))
                denom = 1.0 - b * b
                if denom <= 0 or var_zeta <= 0:
                    continue
                sigma_eq = np.sqrt(var_zeta / denom)
                if sigma_eq <= 0:
                    continue
                s.iat[t, i] = (cum[-1, i] - m) / sigma_eq

        # Long-only side (per docstring): enter when s crosses below -entry_z,
        # exit when s crosses above -exit_z. Match the engine's signal shape.
        s_prev = s.shift(1)
        entries = (s < -entry_z) & (s_prev >= -entry_z)
        exits = (s > -exit_z) & (s_prev <= -exit_z)
        return Signals(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
        )
