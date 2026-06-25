"""PickerCloneStrategy — factor-based clone of a public picker (BRIEF Task 3).

Each rebalance the strategy computes every candidate ticker's factor vector
(z-scored within the universe of the bar's data), measures cosine similarity
against a committed picker profile (see ``research/picker_profiles/``), and
goes long the top-N matches until the next rebalance.

Honest cuts (documented in research/pickers.md):

  * Factor inputs are *price-derived only* (log market cap proxied by log
    cumulative dollar volume, momentum_12_1, realised_vol_60d) at runtime.
    The committed profile was built against the BRIEF's full seven-field
    vector (incl. P/E, P/B, ROE, D/E from yfinance.info). At signal time we
    only have ClickHouse OHLCV, so we project both sides onto the price-
    derivable subset before cosine-similarity. The profile still encodes
    the picker's full preference; we're just measuring similarity in a
    weaker subspace at runtime.
  * Long-only. The picker may have short positions or activist-style
    options; we ignore those and clone the long stock book only.
  * Equal-weight on the top-N basket. Real picker portfolios are
    value-weighted by conviction — we don't try to recover their weights.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd

from research.picker_profiles import (
    PickerProfile,
    cosine_similarity,
    load_profile,
)

from .base import Signals, Strategy

log = logging.getLogger(__name__)


# Factor fields the strategy can compute from OHLCV at runtime. Keep this in
# sync with ``_runtime_factor_vector`` below.
RUNTIME_FIELDS: tuple[str, ...] = (
    "log_market_cap",
    "momentum_12_1",
    "realised_vol_60d",
)


class PickerCloneStrategy(Strategy):
    """Clone a picker by factor similarity, not by literal 13F copying."""

    name = "picker_clone"

    def __init__(
        self,
        picker_name: str = "berkshire",
        profile: PickerProfile | None = None,
        **params,
    ):
        super().__init__(**params)
        self.picker_name = picker_name
        # Allow direct injection (tests). Otherwise read the committed JSON.
        if profile is None:
            try:
                profile = load_profile(picker_name)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"no committed profile for picker {picker_name!r}. "
                    f"Run `python -m scripts.build_picker_profiles {picker_name}` first."
                ) from exc
        self.profile = profile

    @classmethod
    def default_params(cls) -> dict:
        return {
            "top_n": 5,
            "rebalance_freq": 21,        # bars between rebalances (~1 month daily)
            "lookback_for_factors": 252, # window for momentum + vol calc
        }

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        return {
            "top_n": [3, 5, 8],
            "rebalance_freq": [21, 63],
            "lookback_for_factors": [126, 252],
        }

    # ─── core signal --------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close: pd.DataFrame = data["close"]
        volume: pd.DataFrame | None = data["volume"] if "volume" in data else None
        n_bars, n_tickers = close.shape
        empty = pd.DataFrame(False, index=close.index, columns=close.columns)
        if n_tickers < 2 or n_bars < int(self.params["lookback_for_factors"]) + 1:
            return Signals(entries=empty.copy(), exits=empty.copy())

        top_n = max(1, min(int(self.params["top_n"]), n_tickers))
        rebal = max(1, int(self.params["rebalance_freq"]))
        lookback = int(self.params["lookback_for_factors"])

        # Profile vector projected onto the runtime subspace.
        target = np.array([self.profile.profile.get(f, 0.0) for f in RUNTIME_FIELDS],
                          dtype=float)

        in_basket = pd.DataFrame(False, index=close.index, columns=close.columns)
        rebal_bars = list(range(lookback, n_bars, rebal))
        for t in rebal_bars:
            window_close = close.iloc[t - lookback:t + 1]
            window_vol = volume.iloc[t - lookback:t + 1] if volume is not None else None
            vectors = self._runtime_universe_vectors(window_close, window_vol)
            if vectors.empty:
                continue
            z = self._zscore_rows(vectors)
            sims = z.apply(lambda row: cosine_similarity(row.to_numpy(), target), axis=1)
            sims = sims.dropna()
            if sims.empty:
                continue
            top = sims.sort_values(ascending=False).head(top_n).index.tolist()
            # Carry the basket forward until the next rebalance.
            next_t = rebal_bars[rebal_bars.index(t) + 1] if t != rebal_bars[-1] else n_bars
            in_basket.iloc[t:next_t, in_basket.columns.get_indexer(top)] = True

        in_basket_prev = in_basket.shift(1, fill_value=False)
        entries = in_basket & ~in_basket_prev
        exits = ~in_basket & in_basket_prev
        return Signals(
            entries=entries.fillna(False).astype(bool),
            exits=exits.fillna(False).astype(bool),
        )

    # ─── helpers ----------------------------------------------------------
    def _runtime_universe_vectors(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Per-ticker factor frame over the trailing window. Columns = RUNTIME_FIELDS."""
        out: dict[str, dict[str, float]] = {}
        for ticker in close.columns:
            c = close[ticker].dropna()
            if c.empty:
                continue
            vec: dict[str, float] = {}
            # Size proxy: log of cumulative dollar volume over the window.
            if volume is not None and ticker in volume.columns:
                v = volume[ticker].reindex(c.index).fillna(0.0)
                dollar_vol = float((c * v).sum())
                vec["log_market_cap"] = math.log(dollar_vol) if dollar_vol > 0 else float("nan")
            else:
                vec["log_market_cap"] = float("nan")
            # Momentum 12-1 (ret from t-252 to t-21) within the window. The
            # window is exactly 252+1 bars so this is the trailing-year value.
            if len(c) >= 252 + 1:
                p_then = c.iloc[-252]
                p_now = c.iloc[-21]
                if p_then > 0 and math.isfinite(p_now):
                    vec["momentum_12_1"] = float(p_now / p_then - 1.0)
                else:
                    vec["momentum_12_1"] = float("nan")
            else:
                # Shorter window — use whatever lookback exists, skipping 21d for skip-month.
                if len(c) >= 22:
                    p_then = c.iloc[0]
                    p_now = c.iloc[-21]
                    vec["momentum_12_1"] = float(p_now / p_then - 1.0) if p_then > 0 else float("nan")
                else:
                    vec["momentum_12_1"] = float("nan")
            # Realised vol 60d.
            if len(c) >= 61:
                rets = np.log(c).diff().dropna().tail(60)
                sd = float(rets.std(ddof=1))
                vec["realised_vol_60d"] = sd * math.sqrt(252.0) if math.isfinite(sd) else float("nan")
            else:
                vec["realised_vol_60d"] = float("nan")
            out[ticker] = vec
        return pd.DataFrame.from_dict(out, orient="index", columns=list(RUNTIME_FIELDS))

    @staticmethod
    def _zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
        """Column-wise z-score, NaN-tolerant. Rows with all NaN become zero."""
        mu = frame.mean(axis=0, skipna=True)
        sd = frame.std(axis=0, ddof=1, skipna=True).replace(0, np.nan)
        z = (frame - mu) / sd
        return z.fillna(0.0)


# ─── registry entries (one per shipped picker) ─────────────────────────────
def _make_picker_clone_subclass(picker_name: str) -> type[PickerCloneStrategy]:
    """Generate a Strategy subclass bound to a specific picker so each one
    has a distinct ``name`` registry key and shows up in /strategies."""
    class _Bound(PickerCloneStrategy):
        name = f"picker_clone_{picker_name}"

        def __init__(self, **params):
            super().__init__(picker_name=picker_name, **params)

    _Bound.__name__ = f"PickerClone_{picker_name.title()}"
    _Bound.__qualname__ = _Bound.__name__
    return _Bound


PickerCloneBerkshire = _make_picker_clone_subclass("berkshire")
PickerClonePershingSquare = _make_picker_clone_subclass("pershing_square")
PickerCloneAppaloosa = _make_picker_clone_subclass("appaloosa")
PickerCloneScion = _make_picker_clone_subclass("scion")


PICKER_CLONE_REGISTRY: dict[str, type[PickerCloneStrategy]] = {
    "picker_clone_berkshire": PickerCloneBerkshire,
    "picker_clone_pershing_square": PickerClonePershingSquare,
    "picker_clone_appaloosa": PickerCloneAppaloosa,
    "picker_clone_scion": PickerCloneScion,
}
