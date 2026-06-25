"""arxiv:2605.20636 — Xiong (2026) growth/defensive macro timing.

Continuous smooth-score allocation between a growth/tech ETF basket and a
defensive income basket, driven by four direction-normalised macro signals
(paper §5, equations 3–24):

  r_t  = -z(ΔTNX_21)             rate-relief    (negative 21d change in T-bill yld)
  d_t  = -z(SPY drawdown)        drawdown depth (negative = deeper DD)
  vh_t =  z(VIX percentile_756)  high-VIX regime
  vr_t = -z(ΔVIX_21)             VIX-stress relief
  g126 =  z(GD_trailing_126)     growth-vs-defensive crowding (growth - defensive)

The smooth components use softplus(τ=1) (eq. 8); raw score combines:
  CoreScore   = α·r_t + (1-α)·d_t                                (eq. 18)
  StressScore = 0.5·z(r·vh) + 0.5·z(HighVIX·VIXRelief)           (eq. 19)
  CrowdedScore= 0.5·z(g·LowVIX) + 0.5·z(g·LowVIX·RateQuiet)      (eq. 20)
  RawScore    = CoreScore + λ_s·StressScore − λ_c·CrowdedScore   (eq. 21)
  Score       = expanding z-score of RawScore
  w_G_target  = 0.5 + MaxTilt·tanh(Score / τ_w)                  (eq. 22)
  w_G_t       = (1-η)·w_G_{t-1} + η·w_G_target                   (eq. 23)

Paper defaults (selected config eq. 31):
  α=0.50, λ_s=0.50, λ_c=0.05, MaxTilt=0.50, τ_w=0.75, η=0.05.

CUTS / honest notes:
  * The paper allocates continuous weights to two ETF baskets; the
    ``Strategy`` ABC emits boolean entries/exits per ticker. We translate:
    when w_G_target > 0.5 the growth basket gets entries, the defensive
    basket gets exits, and vice versa. Concretely the strategy turns into
    a binary regime switcher on top of the smooth signal. The realised
    Sharpe will differ from the paper's continuous-weight version — this
    is logged in IMPROVEMENTS.
  * Required input columns on ``data['close']``: SPY, ^VIX, ^IRX
    (the FRED FEDFUNDS series isn't available through yfinance — ^IRX
    13-week T-bill is the documented proxy), plus every ticker in
    ``growth_tickers`` and ``defensive_tickers``. Missing any of SPY /
    ^VIX / ^IRX collapses the strategy to zero-signal so the gauntlet
    can show the cut rather than silently lie.
  * The growth-extension term g126 uses the mean of growth-basket vs
    defensive-basket trailing 126d returns. Paper §5 defines it on the
    G vs D portfolios; we mirror that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signals, Strategy


def _safe_z(series: pd.Series, expanding: bool = True) -> pd.Series:
    """Expanding z-score (paper §5 uses expanding standardisation)."""
    if expanding:
        mu = series.expanding(min_periods=20).mean()
        sd = series.expanding(min_periods=20).std()
    else:
        mu = series.mean()
        sd = series.std()
    return (series - mu) / sd.replace(0, np.nan)


def _softplus(x: pd.Series, tau: float = 1.0) -> pd.Series:
    # softplus_τ(x) = τ * log(1 + exp(x/τ))  (paper eq. 8)
    # Clamp x/τ to avoid overflow on extreme z-scores.
    z = np.clip(x.to_numpy() / tau, -50.0, 50.0)
    return pd.Series(tau * np.log1p(np.exp(z)), index=x.index)


class MacroTimingXiong(Strategy):
    """Growth/defensive style timing via smooth macro signals (Xiong 2026)."""

    name = "macro_timing"

    # ─── strategy-specific config (not user-facing params) -----------------
    DEFAULT_GROWTH = ("QQQ", "XLK", "VGT", "SPYG", "VUG")
    DEFAULT_DEFENSIVE = ("SCHD", "XLP", "XLU", "VTV", "VYM")

    def __init__(
        self,
        growth_tickers: tuple[str, ...] | list[str] | None = None,
        defensive_tickers: tuple[str, ...] | list[str] | None = None,
        **params,
    ):
        super().__init__(**params)
        self.growth_tickers = tuple(growth_tickers or self.DEFAULT_GROWTH)
        self.defensive_tickers = tuple(defensive_tickers or self.DEFAULT_DEFENSIVE)

    @classmethod
    def default_params(cls) -> dict:
        # Selected config eq. (31) of Xiong (2026), 50% MaxTilt branch.
        return {
            "alpha": 0.50,
            "lambda_s": 0.50,
            "lambda_c": 0.05,
            "max_tilt": 0.50,
            "tau_w": 0.75,
            "eta": 0.05,
            "rate_window": 21,        # paper: ΔTNX_21
            "drawdown_window": 252,   # rolling 1y high for the DD calc
            "vix_pct_window": 756,    # 3-year VIX percentile
            "vix_window": 21,         # ΔVIX_21
            "growth_window": 126,     # GD_trailing_126
        }

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        # Paper expanded local grid (eqs. 25–30), trimmed to keep sweeps cheap.
        return {
            "alpha": [0.50, 0.67],
            "lambda_s": [0.25, 0.50],
            "lambda_c": [0.05, 0.15],
            "max_tilt": [0.30, 0.50],
            "tau_w": [0.75, 1.0],
            "eta": [0.05, 0.10],
        }

    # ─── core signal -------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close: pd.DataFrame = data["close"]
        all_tickers = list(close.columns)
        empty = pd.DataFrame(False, index=close.index, columns=all_tickers)

        required = {"SPY", "^VIX", "^IRX"}
        if not required.issubset(set(all_tickers)):
            # Honest degradation — caller didn't supply the macro inputs.
            return Signals(entries=empty.copy(), exits=empty.copy())

        spy = close["SPY"].ffill()
        vix = close["^VIX"].ffill()
        irx = close["^IRX"].ffill()

        # ─── signal components (paper eqs. 3-7) -----------------------------
        d_tnx = irx.diff(self.params["rate_window"])
        r_t = -_safe_z(d_tnx)

        rolling_max = spy.rolling(self.params["drawdown_window"], min_periods=20).max()
        spy_dd = spy / rolling_max - 1.0
        d_t = -_safe_z(spy_dd)

        vix_pct = (
            vix.rolling(self.params["vix_pct_window"], min_periods=60)
            .rank(pct=True)
        )
        vh_t = _safe_z(vix_pct)

        d_vix = vix.diff(self.params["vix_window"])
        vr_t = -_safe_z(d_vix)

        # Growth vs defensive trailing 126d return (paper eq. 7).
        growth_present = [t for t in self.growth_tickers if t in all_tickers]
        defensive_present = [t for t in self.defensive_tickers if t in all_tickers]
        if not growth_present or not defensive_present:
            return Signals(entries=empty.copy(), exits=empty.copy())

        gw = self.params["growth_window"]
        g_ret = (
            close[growth_present].pct_change(gw).mean(axis=1)
        )
        d_ret = (
            close[defensive_present].pct_change(gw).mean(axis=1)
        )
        g126_t = _safe_z(g_ret - d_ret)

        # ─── smooth components (paper eqs. 9-13) ----------------------------
        high_vix = _softplus(vh_t.fillna(0.0))
        vix_relief = _softplus(vr_t.fillna(0.0))
        low_vix = _softplus(-vh_t.fillna(0.0))
        growth_ext = _softplus(g126_t.fillna(0.0))
        rate_quiet = pd.Series(
            np.exp(-0.5 * r_t.fillna(0.0).to_numpy() ** 2),
            index=close.index,
        )

        # ─── interaction terms (paper eqs. 14-17) ---------------------------
        i1 = r_t.fillna(0.0) * vh_t.fillna(0.0)
        i2 = high_vix * vix_relief
        i3 = growth_ext * low_vix
        i4 = growth_ext * low_vix * rate_quiet

        # ─── scores (paper eqs. 18-21) --------------------------------------
        alpha = float(self.params["alpha"])
        lam_s = float(self.params["lambda_s"])
        lam_c = float(self.params["lambda_c"])
        core = alpha * r_t.fillna(0.0) + (1.0 - alpha) * d_t.fillna(0.0)
        stress = 0.5 * _safe_z(i1).fillna(0.0) + 0.5 * _safe_z(i2).fillna(0.0)
        crowded = 0.5 * _safe_z(i3).fillna(0.0) + 0.5 * _safe_z(i4).fillna(0.0)
        raw = core + lam_s * stress - lam_c * crowded
        score = _safe_z(raw).fillna(0.0)

        # ─── target weights and EWMA smoothing (paper eqs. 22-23) -----------
        tau_w = float(self.params["tau_w"])
        max_tilt = float(self.params["max_tilt"])
        eta = float(self.params["eta"])
        w_target = 0.5 + max_tilt * np.tanh(score / tau_w)

        # EWMA smoothing applied vectorised.
        w_arr = np.empty(len(w_target))
        w_arr[0] = 0.5
        target_vals = w_target.to_numpy()
        for t in range(1, len(w_arr)):
            w_arr[t] = (1 - eta) * w_arr[t - 1] + eta * target_vals[t]
        wG = pd.Series(w_arr, index=close.index)

        # Cache for inspection / tests / reporting.
        self.last_target_weight_ = wG

        # ─── translate continuous weight to per-ticker entries/exits --------
        # Growth basket: long when wG > 0.55 (a small dead-band around the
        # 0.5 neutral so we don't churn on noise); defensive: long when
        # wG < 0.45. Otherwise neither side gets a fresh entry; previous
        # holdings stay open until the inverse threshold flips.
        long_growth = wG > 0.55
        long_defensive = wG < 0.45

        entries = pd.DataFrame(False, index=close.index, columns=all_tickers)
        exits = pd.DataFrame(False, index=close.index, columns=all_tickers)

        long_growth_prev = long_growth.shift(1, fill_value=False)
        long_defensive_prev = long_defensive.shift(1, fill_value=False)
        enter_g = long_growth & ~long_growth_prev
        enter_d = long_defensive & ~long_defensive_prev
        exit_g = ~long_growth & long_growth_prev
        exit_d = ~long_defensive & long_defensive_prev

        for t in growth_present:
            entries[t] = enter_g
            exits[t] = exit_g
        for t in defensive_present:
            entries[t] = enter_d
            exits[t] = exit_d

        return Signals(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
        )
