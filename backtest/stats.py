"""Significance statistics for backtest Sharpe ratios (BRIEF Task 7b).

Half the published bugs in this corner of finance come from mixing log/simple
returns or annualising twice. The conventions used here are:

  * `returns` is always a 1-D array of **simple periodic returns** — i.e. the
    output of `prices.pct_change()`. Not log returns, not equity, not cumulated.
  * `periods_per_year` is the annualisation factor the caller chose for their
    bar interval. Daily = 252; hourly RTH = 252*7 ≈ 1764; 5-minute RTH = 252*78
    = 19656. The library will not guess; pass it explicitly.

Citations (the entire module is a faithful implementation of these papers — read
them before changing the formulas):

  * Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient
    Frontier." Journal of Risk, 15(2), 13–44.
    PSR derivation, eq. (4) — adjusts for skew & kurtosis of the return series.
  * Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    Journal of Portfolio Management, 40(5), 94–107.
    DSR uses the expected maximum of N independent normals with std σ_SR as the
    benchmark threshold for PSR.
  * Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
    Journal of the American Statistical Association, 89(428), 1303–1313.
    Block lengths are geometrically distributed (mean = `block_length`) — this
    keeps the bootstrapped series stationary, unlike the fixed-length block
    bootstrap of Künsch (1989).
  * Hansen, P. R. (2005). "A Test for Superior Predictive Ability."
    Journal of Business & Economic Statistics, 23(4), 365–380.
    The full SPA test re-centres the bootstrap distribution at zero for any
    competitor with negative sample mean. We ship **White's (2000) Reality
    Check** — the simpler ancestor — and flag it explicitly below. See the
    `spa_test` docstring and IMPROVEMENTS.md (Backtest) for the cut rationale.
  * White, H. (2000). "A Reality Check for Data Snooping." Econometrica, 68(5),
    1097–1126. The shipped fallback.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl
from scipy import stats as _sps

ArrayLike = np.ndarray | pl.Series | list


# ─── coercion helpers ──────────────────────────────────────────────────────
def _as_array(returns: ArrayLike) -> np.ndarray:
    """Coerce input to a 1-D float64 numpy array. Rejects multi-dim input."""
    if isinstance(returns, pl.Series):
        arr = returns.to_numpy()
    else:
        arr = np.asarray(returns)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D returns, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError("need at least 2 return observations")
    return arr


# ─── helper metrics ────────────────────────────────────────────────────────
def annualised_return(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Geometric annualised return from a series of simple periodic returns.

    `(1 + r̄_g)^periods_per_year - 1` where r̄_g is the geometric per-period
    return. Returns NaN if any factor is non-positive (i.e. a -100% return)."""
    r = _as_array(returns)
    growth = 1.0 + r
    if np.any(growth <= 0):
        return float("nan")
    geo_mean = np.exp(np.log(growth).mean())
    return float(geo_mean**periods_per_year - 1.0)


def annualised_sharpe(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Annualised Sharpe (zero risk-free): `mean / std * sqrt(periods_per_year)`.

    Uses sample std (ddof=1). Returns 0.0 if std is 0 (degenerate series)."""
    r = _as_array(returns)
    sigma = r.std(ddof=1)
    if sigma <= 0:
        return 0.0
    return float(r.mean() / sigma * np.sqrt(periods_per_year))


def max_drawdown(equity_or_returns: ArrayLike) -> float:
    """Maximum peak-to-trough drawdown as a negative float (or 0.0).

    Heuristic: if every value is positive and the series is monotone-like
    (typical equity-curve shape with values ~ initial capital), treat the input
    as an equity curve. Otherwise treat it as periodic returns and cumulate
    `(1 + r).cumprod()` first. To avoid the ambiguity for hand-built test
    series, callers are encouraged to pass equity curves explicitly."""
    arr = _as_array(equity_or_returns)
    # Heuristic: an equity curve is strictly positive throughout; returns
    # straddle 0. If all values >= ~0.5 we treat it as equity; else as returns.
    if (arr > 0).all() and arr.min() > 0.5:
        equity = arr
    else:
        equity = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


# ─── PSR / DSR ─────────────────────────────────────────────────────────────
def _sample_sharpe_non_annualised(r: np.ndarray) -> float:
    """Per-period (non-annualised) Sharpe — the SR̂ used in PSR/DSR formulas."""
    sigma = r.std(ddof=1)
    if sigma <= 0:
        return 0.0
    return float(r.mean() / sigma)


def probabilistic_sharpe_ratio(
    returns: ArrayLike,
    sr_benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probability that the true Sharpe exceeds `sr_benchmark`.

    Bailey & López de Prado (2012), eq. (4):

        PSR(SR*) = Φ( (SR̂ - SR*) * sqrt(n - 1) /
                       sqrt(1 - γ₃ * SR̂ + (γ₄ - 1)/4 * SR̂²) )

    where SR̂ is the per-period sample Sharpe, n is the sample size, γ₃ is
    skewness, and γ₄ is kurtosis (4th standardised moment, NOT excess
    kurtosis — Bailey uses the raw 4th moment so the Gaussian baseline gives
    γ₄ = 3 and the bracketed term collapses to `1 + SR̂²/2`).

    `sr_benchmark` is supplied in **annualised** units to match the rest of the
    library; we deflate it to per-period units internally.
    """
    r = _as_array(returns)
    n = r.size
    sr_hat = _sample_sharpe_non_annualised(r)
    sr_star = sr_benchmark / np.sqrt(periods_per_year)

    # Bailey uses bias-corrected sample skew/kurtosis (the Fisher-Pearson form
    # with default bias=True is what is referenced in the paper).
    skew = float(_sps.skew(r, bias=True))
    kurt = float(_sps.kurtosis(r, fisher=False, bias=True))  # raw, not excess

    denom_sq = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    # Numerical guard: under heavy tails + extreme SR̂ the variance estimate
    # can go negative. Clamp to a small positive epsilon so PSR stays defined.
    if denom_sq <= 0:
        denom_sq = 1e-12
    z = (sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(_sps.norm.cdf(z))


# Euler-Mascheroni constant (EM ≈ 0.57721) — appears in the expected maximum
# of N i.i.d. standard normals, which Bailey & López de Prado (2014) use to
# build the DSR benchmark threshold.
_EULER_MASCHERONI = 0.5772156649015329


def _expected_max_iid_normal(n_trials: int) -> float:
    """E[max] of `n_trials` i.i.d. N(0, 1) draws — Bailey & López de Prado 2014.

        E[max] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N·e))

    where γ is the Euler-Mascheroni constant. Exact only in the limit; the
    paper uses this finite-sample approximation throughout."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    em = _EULER_MASCHERONI
    inv_n = 1.0 / n_trials
    inv_ne = 1.0 / (n_trials * np.e)
    q1 = _sps.norm.ppf(1.0 - inv_n)
    q2 = _sps.norm.ppf(1.0 - inv_ne)
    return float((1.0 - em) * q1 + em * q2)


def deflated_sharpe_ratio(
    returns: ArrayLike,
    n_trials: int,
    sr_trials_std: float,
    periods_per_year: int = 252,
) -> float:
    """Probability that the true Sharpe exceeds the **deflated** benchmark.

    Bailey & López de Prado (2014). The benchmark Sharpe threshold is

        SR₀ = σ_SR * E[max of N i.i.d. N(0,1)]

    where σ_SR is the cross-trial std of (annualised) Sharpes from the sweep
    and N is the number of independent trials. DSR := PSR(SR₀).

    Edge case: `n_trials == 1` and `sr_trials_std == 0` reduce to
    `probabilistic_sharpe_ratio(returns, sr_benchmark=0)` exactly.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_trials_std < 0:
        raise ValueError("sr_trials_std must be non-negative")
    # Annualised threshold; PSR will deflate to per-period internally.
    sr_threshold = sr_trials_std * _expected_max_iid_normal(n_trials)
    return probabilistic_sharpe_ratio(
        returns, sr_benchmark=sr_threshold, periods_per_year=periods_per_year
    )


def deflated_sharpe_ratio_from_sweep(
    best_returns: ArrayLike,
    sweep_sharpes: ArrayLike,
    periods_per_year: int = 252,
) -> float:
    """Convenience: derive `n_trials` and `sr_trials_std` from a sweep array.

    `sweep_sharpes` is the full vector of (annualised) Sharpes from every trial
    in the sweep — including the chosen best. n_trials := len(sweep_sharpes),
    sr_trials_std := sample std (ddof=1). Non-finite values are filtered out
    before computing the std."""
    s = np.asarray(sweep_sharpes, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 1:
        raise ValueError("sweep_sharpes has no finite entries")
    n_trials = int(s.size)
    sr_trials_std = float(s.std(ddof=1)) if s.size > 1 else 0.0
    return deflated_sharpe_ratio(
        best_returns,
        n_trials=n_trials,
        sr_trials_std=sr_trials_std,
        periods_per_year=periods_per_year,
    )


# ─── Stationary block bootstrap (Politis & Romano 1994) ────────────────────
def _stationary_bootstrap_indices(
    n: int, block_length: int, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Index matrix of shape (n_resamples, n) for the stationary bootstrap.

    At each position the block continues with probability `1 - 1/L` (where L is
    the expected block length); with probability `1/L` we draw a fresh start
    uniformly from [0, n). Wraps around via modulo n so blocks near the end
    don't get truncated."""
    p = 1.0 / max(block_length, 1)
    # idx[:, 0] uniform starts; subsequent columns either continue (prev+1) or
    # jump to a fresh uniform start, decided per-cell by a Bernoulli(p) draw.
    starts = rng.integers(0, n, size=(n_resamples, n))
    jumps = rng.random((n_resamples, n)) < p
    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(jumps[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_ci(
    returns: ArrayLike,
    metric_fn: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    block_length: int | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Stationary-bootstrap confidence interval for an arbitrary metric.

    Returns `(point_estimate, ci_low, ci_high)` at the (1 - alpha) level using
    the percentile interval (no BCa correction — see IMPROVEMENTS).

    `block_length` defaults to `max(1, floor(sqrt(n)))`, the conventional
    rule-of-thumb for serially-correlated daily-returns data."""
    r = _as_array(returns)
    n = r.size
    if block_length is None:
        block_length = max(1, int(np.sqrt(n)))
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(n, block_length, n_resamples, rng)
    samples = r[idx]  # (n_resamples, n)
    stats_arr = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        stats_arr[i] = metric_fn(samples[i])

    point = float(metric_fn(r))
    lo = float(np.quantile(stats_arr, alpha / 2.0))
    hi = float(np.quantile(stats_arr, 1.0 - alpha / 2.0))
    return point, lo, hi


def sharpe_ci(
    returns: ArrayLike,
    periods_per_year: int = 252,
    n_resamples: int = 1000,
    block_length: int | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the annualised Sharpe ratio."""
    return bootstrap_ci(
        returns,
        metric_fn=lambda x: annualised_sharpe(x, periods_per_year=periods_per_year),
        n_resamples=n_resamples,
        block_length=block_length,
        alpha=alpha,
        seed=seed,
    )


def total_return_ci(
    returns: ArrayLike,
    n_resamples: int = 1000,
    block_length: int | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for cumulative total return `prod(1+r) - 1`."""
    return bootstrap_ci(
        returns,
        metric_fn=lambda x: float(np.prod(1.0 + x) - 1.0),
        n_resamples=n_resamples,
        block_length=block_length,
        alpha=alpha,
        seed=seed,
    )


def max_drawdown_ci(
    returns: ArrayLike,
    n_resamples: int = 1000,
    block_length: int | None = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for max drawdown. Input must be **returns**, not equity
    (we re-cumulate on each resample, which only makes sense from returns)."""
    return bootstrap_ci(
        returns,
        metric_fn=lambda x: max_drawdown(np.cumprod(1.0 + x)),
        n_resamples=n_resamples,
        block_length=block_length,
        alpha=alpha,
        seed=seed,
    )


# ─── White's Reality Check (SPA-lite) ──────────────────────────────────────
def spa_test(
    strategy_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    competing_returns: np.ndarray,
    n_resamples: int = 1000,
    block_length: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Test whether the best of `competing_returns` beats the benchmark.

    **What we shipped: White's (2000) Reality Check**, NOT the full Hansen
    (2005) SPA. The difference: Hansen re-centres the bootstrap distribution
    at zero for any competitor whose sample mean is negative (so the test
    statistic isn't artificially deflated by clearly-inferior competitors).
    The Reality Check uses the raw bootstrap distribution without that step.
    For sweeps where every competitor is at least roughly comparable, the two
    tests agree closely; SPA is strictly more powerful when many competitors
    are obviously bad. Logged as a cut in IMPROVEMENTS.md (Backtest).

    Procedure
    ---------
    1. Build the loss-differential matrix d_{k,t} = comp_k(t) - benchmark(t).
    2. Test statistic: V = sqrt(n) * max_k mean_t(d_{k,t}).
    3. Resample d via stationary bootstrap (shared time-axis across competitors
       so cross-correlation is preserved); recompute V*_b for each of B draws.
    4. p_value = (1/B) * Σ_b 1{V*_b - V_centre > V} — re-centred at the
       sample mean so the null distribution is correctly anchored at zero.

    `strategy_returns` is accepted for API completeness but is NOT consumed —
    the test asks "does the best competitor beat the benchmark?", which is a
    universal statement over the competitors. (If a caller wants to test a
    single, pre-specified strategy, that's a simpler one-sided test on
    `strategy - benchmark` — not what SPA / Reality Check is for.)
    """
    _ = _as_array(strategy_returns)  # input validation only
    bench = _as_array(benchmark_returns)
    if competing_returns.ndim != 2:
        raise ValueError(
            f"competing_returns must be 2-D (n_obs, n_trials), got {competing_returns.shape}"
        )
    n_obs, n_trials = competing_returns.shape
    if n_obs != bench.size:
        raise ValueError(
            f"benchmark length {bench.size} != competing length {n_obs}"
        )
    if n_trials < 1:
        raise ValueError("need at least 1 competing series")

    d = competing_returns.astype(float) - bench[:, None]  # (n_obs, n_trials)
    d_mean = d.mean(axis=0)  # (n_trials,)
    v = float(np.sqrt(n_obs) * d_mean.max())

    if block_length is None:
        block_length = max(1, int(np.sqrt(n_obs)))
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(n_obs, block_length, n_resamples, rng)

    # Vectorised: for each bootstrap, compute mean over time of d[idx, :].
    # Shape gymnastics: d[idx] is (n_resamples, n_obs, n_trials). Mean axis=1
    # collapses to (n_resamples, n_trials).
    boot_means = d[idx].mean(axis=1)
    # Re-centre at sample mean so the bootstrapped null is anchored at zero.
    centred = boot_means - d_mean[None, :]
    v_boot = np.sqrt(n_obs) * centred.max(axis=1)

    # One-sided p-value: P(V* > V) under the (re-centred) null.
    p_value = float(np.mean(v_boot > v))
    return {"p_value": p_value, "test_statistic": v}
