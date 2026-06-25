"""Return attribution (BRIEF Task 7c).

Two decompositions, both deliberately simple:

1. **Market attribution** — CAPM regression of strategy excess returns on
   market excess returns. Produces `alpha`, `beta`, `alpha_t_stat`, `r_squared`
   plus the residual (alpha + noise) and systematic (beta * market) series so
   the UI can stack them.

2. **Child-signal attribution** — given per-bar returns of each child signal
   and the weight each signal received at each bar, decompose the combined
   return as `Σ_i w_i,t * r_i,t`. The "combined" strategy (Task 4 — not yet
   built) will plug straight in.

Cuts (see IMPROVEMENTS.md, Backtest):
  * No Newey-West / HAC standard errors on the alpha t-stat. Plain OLS
    residual-variance SE — too tight when returns are autocorrelated. A
    `hac_lags` knob is the right follow-up; for now we emit a single t-stat
    and document the caveat in the docstring.
  * No multi-factor model (Fama-French 3F / 5F / Carhart 4F). Market beta only.
  * No risk-decomposed contribution — we attribute *return*, not *variance*.
    The contribution numbers won't add to the strategy's variance even if they
    add to its return.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d(arr: Any, name: str) -> np.ndarray:
    """Coerce to 1-D float numpy array; reject empty / multi-dim."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size < 2:
        raise ValueError(f"{name} needs at least 2 observations")
    return a


# ─── 1. Market attribution (CAPM-style OLS) ────────────────────────────────
def market_attribution(
    strategy_returns: Any,
    market_returns: Any,
    risk_free: float | np.ndarray = 0.0,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """OLS regression of strategy excess returns on market excess returns.

        r_strat,t - rf_t  =  alpha + beta * (r_mkt,t - rf_t)  +  epsilon_t

    Returns
    -------
    dict with:
      * `alpha`              : per-period intercept (raw OLS coefficient)
      * `alpha_annualised`   : `alpha * periods_per_year` (additive convention,
                               matching the literature for small per-period α)
      * `beta`               : OLS slope on market excess returns
      * `alpha_t_stat`       : t-stat of α under the plain-OLS SE (no HAC —
                               see module docstring & IMPROVEMENTS)
      * `r_squared`          : in-sample fit of the regression
      * `residual_returns`   : np.ndarray, per-bar (alpha + epsilon)
      * `systematic_returns` : np.ndarray, per-bar (beta * market_excess)
      * `n_obs`              : sample size

    Notes
    -----
    `risk_free` can be a scalar (per-period rate) or an array matching the
    returns; defaults to 0 because the test harness pumps in synthetic
    excess-of-cash series. For real-world use, pass the daily 3-month T-bill
    rate divided by `periods_per_year`.
    """
    s = _as_1d(strategy_returns, "strategy_returns")
    m = _as_1d(market_returns, "market_returns")
    if s.size != m.size:
        raise ValueError(
            f"strategy_returns / market_returns length mismatch: {s.size} vs {m.size}"
        )

    if np.isscalar(risk_free):
        rf = np.full(s.size, float(risk_free))
    else:
        rf = np.asarray(risk_free, dtype=float)
        if rf.shape != s.shape:
            raise ValueError(
                f"risk_free shape {rf.shape} != returns shape {s.shape}"
            )

    y = s - rf
    x = m - rf
    n = y.size

    # Design matrix [1, x]; OLS via np.linalg.lstsq. rcond=None silences the
    # numpy deprecation warning and uses machine-precision rank cutoff.
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])

    y_hat = X @ coef
    residuals = y - y_hat
    rss = float((residuals**2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss if tss > 0 else float("nan")

    # Plain-OLS standard error of alpha: σ²_resid * (X'X)^{-1}[0,0].
    # σ² uses n - k = n - 2 degrees of freedom.
    if n > 2:
        sigma2 = rss / (n - 2)
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
            se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
            t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")
        except np.linalg.LinAlgError:
            t_alpha = float("nan")
    else:
        t_alpha = float("nan")

    systematic = beta * x
    # residual_returns = alpha + epsilon; matches the "what's left after market
    # beta is removed" intuition the UI's stacked-area chart wants.
    residual_returns = alpha + residuals

    return {
        "alpha": alpha,
        "alpha_annualised": float(alpha * periods_per_year),
        "beta": beta,
        "alpha_t_stat": float(t_alpha),
        "r_squared": float(r_squared),
        "residual_returns": residual_returns,
        "systematic_returns": systematic,
        "n_obs": int(n),
    }


# ─── 2. Child-signal attribution ───────────────────────────────────────────
def child_signal_attribution(
    combined_returns: Any,
    child_returns: dict[str, Any],
    weights: dict[str, Any],
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, float]:
    """Decompose `combined_returns` into per-child contributions.

    For each child `i` we compute `contribution_i = Σ_t w_{i,t} * r_{i,t}` and
    return the dict `{name: contribution, ..., "residual": gap}`. The residual
    is `total_combined - Σ_i contribution_i`; when the weights & returns are
    self-consistent it should be float-precision-zero.

    Parameters
    ----------
    combined_returns : 1-D array
        Per-bar returns of the combined / aggregated strategy.
    child_returns : dict[name, 1-D array]
        Per-bar returns of each child signal, all aligned to `combined_returns`.
    weights : dict[name, 1-D array | float]
        Weight assigned to each child at each bar. A scalar is broadcast to the
        full series (useful for fixed-weight portfolios).
    rtol, atol : float
        Tolerance for the consistency check. We do NOT raise on a mismatch —
        the caller might pass a synthetic case where the residual is the whole
        point — but we surface the gap as `residual` so they can decide.

    Notes
    -----
    Names of `child_returns` and `weights` must match exactly. Both can be
    in any order; we iterate over `child_returns.keys()` to fix the order.
    """
    combined = _as_1d(combined_returns, "combined_returns")
    n = combined.size

    if set(child_returns.keys()) != set(weights.keys()):
        missing = set(child_returns).symmetric_difference(weights)
        raise ValueError(f"child_returns / weights key mismatch: {sorted(missing)}")
    if not child_returns:
        raise ValueError("child_returns must contain at least one signal")

    contributions: dict[str, float] = {}
    for name in child_returns.keys():
        r = _as_1d(child_returns[name], f"child_returns[{name!r}]")
        if r.size != n:
            raise ValueError(
                f"child_returns[{name!r}] length {r.size} != combined length {n}"
            )
        w_raw = weights[name]
        if np.isscalar(w_raw):
            w = np.full(n, float(w_raw))
        else:
            w = np.asarray(w_raw, dtype=float)
            if w.shape != (n,):
                raise ValueError(
                    f"weights[{name!r}] shape {w.shape} != ({n},)"
                )
        contributions[name] = float((w * r).sum())

    explained = sum(contributions.values())
    total = float(combined.sum())
    residual = total - explained

    # Soft consistency check: relative gap vs |total|. We don't raise; the
    # `residual` key tells the caller how big the unexplained chunk is.
    _ = rtol, atol  # accepted for future hard-mode behaviour (see IMPROVEMENTS)

    out = dict(contributions)
    out["residual"] = float(residual)
    return out
