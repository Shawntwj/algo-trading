"""CombinedExplainableStrategy — Task 4a.

Blends the arxiv strategies (Task 2b) with picker-clones (Task 3) and the
two MVP control strategies into a single tradeable signal that knows
*why* it fired on any given bar.

Construction
------------
Per bar we ask each child strategy for its raw long-side signal interest.
We do **not** re-implement child signal logic; we instantiate the registered
strategies and call their ``generate_signals`` method, then translate the
boolean entry/exit pair into a continuous "interest" series per ticker:

    state_{i,t} = 1 if last action ≤ t was an entry, 0 if last action was an exit.

That state is then z-scored over a rolling window (default 252 bars) and
clipped to ±3, matching the z-score conventions used throughout the
project (`backtest.regimes._safe_z` style and `_safe_z` in
`strategies/macro_timing.py`).

The combined signal for ticker `i` at bar `t` is:

    combined_score_{i,t} = Σ_k w_{k,t} * normalised_signal_{k, i, t}

where `w_{k,t}` are the per-child weights. By default we use equal weights;
``fit_weights_walk_forward`` (added in step 2) learns convex-combination
Sharpe-maximising weights via walk-forward and writes them back onto the
instance.

A long entry fires when (a) at least ``min_active_children`` children agree
on a positive normalised signal at that bar AND (b) the combined score
crosses above zero. The mirror condition produces exits.

Per-bar explanation log
-----------------------
Every (ticker, timestamp) where we emit an entry OR an exit gets an
``explanation`` dict written into ``self.explanation_log``:

    {
        "ticker": str,
        "timestamp": pd.Timestamp,
        "direction": "long_entry" | "long_exit",
        "weights": {child_name: float, ...},   # weights at that bar
        "child_signals": {child_name: float, ...},  # normalised signal value
        "summary": str,  # plain-English one-liner
    }

The log lives on the instance — the explainability module
(``backtest/explainability.py``) reads it via ``get_explanation_log()``
after the backtest has run. No globals.

Honest cuts
-----------
* Long-only — the project's Strategy ABC `Signals` carries entry/exit
  booleans (no first-class shorts; see IMPROVEMENTS).
* Children that emit zero signal on the provided universe (e.g.
  ``macro_timing`` without SPY/^VIX/^IRX columns; ``pca_stat_arb`` on a
  ≤2-ticker frame) silently contribute zero — their weight is still part
  of the sum, the explanation just records ``child_signals[k]=0`` and the
  summary calls them out so the user can see *why* they're not voting.
* Default weights are equal across children; ``fit_weights_walk_forward``
  must be called explicitly to learn weights (kept off the hot path so
  the strategy works as a plain combined-signal generator without an
  expensive sweep at every backtest).
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .base import Signals, Strategy

log = logging.getLogger(__name__)


# Default child line-up per the BRIEF task 4a deliverable:
#   - arxiv: pca_stat_arb, macro_timing, drift_regime
#   - picker-clone: appaloosa (largest A-vs-B gap in Task 3)
#   - MVP controls: ma_crossover, rsi_mean_reversion
DEFAULT_CHILDREN: tuple[str, ...] = (
    "pca_stat_arb",
    "macro_timing",
    "drift_regime",
    "picker_clone_appaloosa",
    "ma_crossover",
    "rsi_mean_reversion",
)


def _zscore_clip(
    series: pd.Series, window: int, clip: float = 3.0
) -> pd.Series:
    """Rolling z-score → clipped to ±`clip`.

    Matches the project's normalisation style: rolling expanding-friendly
    standardisation (`min_periods=window // 4`) with the same NaN-tolerant
    behaviour as `strategies.macro_timing._safe_z`.
    """
    min_p = max(2, window // 4)
    mu = series.rolling(window, min_periods=min_p).mean()
    sd = series.rolling(window, min_periods=min_p).std()
    sd = sd.replace(0, np.nan)
    z = (series - mu) / sd
    return z.clip(lower=-clip, upper=clip).fillna(0.0)


class CombinedExplainableStrategy(Strategy):
    """Weighted ensemble of arxiv + picker + MVP signals with per-trade explanations."""

    name = "combined_explainable"

    def __init__(
        self,
        children: Sequence[str] | None = None,
        child_params: dict[str, dict[str, Any]] | None = None,
        weights: dict[str, float] | None = None,
        **params: Any,
    ):
        super().__init__(**params)
        self.children: tuple[str, ...] = tuple(children or DEFAULT_CHILDREN)
        # Per-child override params (e.g. picker_clone needs picker_name baked in).
        self.child_params: dict[str, dict[str, Any]] = dict(child_params or {})
        # Default to equal weights; can be set explicitly or learned via
        # walk-forward (step 2 below).
        if weights is None:
            n = len(self.children)
            self.weights: dict[str, float] = {c: 1.0 / n for c in self.children}
        else:
            self.weights = dict(weights)
            self._validate_weights(self.weights)

        # Per-bar explanation stream populated by generate_signals. The
        # explainability module reads this via get_explanation_log().
        # Keyed by (ticker, timestamp) for O(1) join.
        self.explanation_log: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
        # Per-bar weight history (for the explainability module — if the
        # caller switches to time-varying weights, this records what was
        # used at every bar).
        self.weight_history_: pd.DataFrame | None = None
        # Per-child normalised signal cache (last call). Useful for tests
        # and for the explainability module.
        self.child_signal_cache_: dict[str, pd.DataFrame] = {}
        # Fallback marker set by the weight fitter if scipy degrades to
        # inverse-vol. The explanation summary surfaces this so the user
        # knows the weight assignment isn't fully optimised.
        self.weight_fit_fallback_: bool = False

    # ─── ABC plumbing ─────────────────────────────────────────────────────
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "norm_window": 252,          # rolling z-score window for child signal normalisation
            "norm_clip": 3.0,            # ±clip after z-score
            "weight_fit_window": 504,    # ~2y of bars for walk-forward weight learning
            "rebalance_freq": 21,        # ~monthly weight refresh during fit
            "min_active_children": 2,    # require ≥N children to agree on sign for an entry
            "entry_threshold": 0.0,      # combined score crosses above this → long entry
        }

    @classmethod
    def param_grid(cls) -> dict[str, list[Any]]:
        return {
            "norm_window": [126, 252],
            "min_active_children": [1, 2, 3],
            "entry_threshold": [0.0, 0.25, 0.5],
        }

    # ─── helpers ──────────────────────────────────────────────────────────
    def _validate_weights(self, w: dict[str, float]) -> None:
        if not w:
            raise ValueError("weights must be non-empty")
        if any(v < -1e-9 for v in w.values()):
            raise ValueError(f"weights must be non-negative; got {w}")
        s = sum(w.values())
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"weights must sum to 1 (±1e-3); got {s} for {w}")

    def _instantiate_children(self) -> dict[str, Strategy]:
        """Build one Strategy instance per child name.

        Imported lazily to avoid an import cycle (`strategies/__init__.py`
        imports this module via REGISTRY)."""
        from . import REGISTRY  # local import, breaks the cycle

        out: dict[str, Strategy] = {}
        for name in self.children:
            if name not in REGISTRY:
                raise KeyError(
                    f"CombinedExplainableStrategy: unknown child {name!r}. "
                    f"Known: {sorted(REGISTRY.keys())}"
                )
            params = self.child_params.get(name, {})
            out[name] = REGISTRY[name](**params)
        return out

    def _child_state(self, signals: Signals) -> pd.DataFrame:
        """Convert (entries, exits) booleans → per-bar long-state {0, 1}.

        state_{i,t} = 1 if last action ≤ t was an entry, 0 if it was an exit.
        We forward-fill the +1/-1 'action' stream to get a step function.
        """
        # action = +1 on entry, -1 on exit, NaN otherwise → ffill → +1 means
        # currently long, -1 means flat. Convert to {0, 1}.
        action = pd.DataFrame(np.nan, index=signals.entries.index, columns=signals.entries.columns)
        action = action.mask(signals.entries.astype(bool), 1.0)
        action = action.mask(signals.exits.astype(bool), -1.0)
        action = action.ffill().fillna(-1.0)  # before first entry → flat
        state = (action > 0).astype(float)
        return state

    def _normalised_child_signals(
        self, data: pd.DataFrame, children: dict[str, Strategy]
    ) -> dict[str, pd.DataFrame]:
        """Per-child wide DataFrame (rows=ts, cols=tickers) of normalised signal strengths."""
        close = data["close"]
        window = int(self.params["norm_window"])
        clip = float(self.params["norm_clip"])
        out: dict[str, pd.DataFrame] = {}
        for name, child in children.items():
            try:
                sig = child.generate_signals(data)
            except Exception as exc:
                log.warning("child %s.generate_signals raised %s — treating as zero signal", name, exc)
                out[name] = pd.DataFrame(0.0, index=close.index, columns=close.columns)
                continue
            state = self._child_state(sig)
            # Reindex defensively in case the child returned a sub-frame.
            state = state.reindex(index=close.index, columns=close.columns).fillna(0.0)
            normed = state.apply(lambda s: _zscore_clip(s, window=window, clip=clip), axis=0)
            out[name] = normed
        return out

    # ─── core signal ──────────────────────────────────────────────────────
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close = data["close"]
        children = self._instantiate_children()
        child_signals = self._normalised_child_signals(data, children)
        self.child_signal_cache_ = child_signals

        # Per-bar weights — for now the same weights every bar (set either
        # in __init__ or by fit_weights_walk_forward via self.weights). We
        # still record a per-bar history so the explainability module can
        # treat the time axis generically.
        weights_series = pd.DataFrame(
            {c: self.weights.get(c, 0.0) for c in self.children},
            index=close.index,
        )
        self.weight_history_ = weights_series

        # Weighted sum across children → combined score per ticker.
        combined = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        # Count of children with a positive normalised signal (used for the
        # min_active_children gate).
        positive_votes = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=int)
        for name in self.children:
            w = self.weights.get(name, 0.0)
            sig = child_signals[name]
            combined = combined + w * sig
            positive_votes = positive_votes + (sig > 0).astype(int)

        # State machine: enter long when combined > threshold AND
        # min_active_children agree on a positive sign; exit when combined
        # falls back below 0.
        thresh = float(self.params["entry_threshold"])
        min_active = int(self.params["min_active_children"])

        wants_long = (combined > thresh) & (positive_votes >= min_active)
        wants_flat = combined <= 0.0

        # Translate wants_long / wants_flat → entry/exit booleans with the
        # same shape and dtype the rest of the codebase uses.
        wl_prev = wants_long.shift(1, fill_value=False)
        wf_prev = wants_flat.shift(1, fill_value=False)
        entries = wants_long & ~wl_prev
        exits = wants_flat & ~wf_prev

        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        # ─── persist per-trade explanations ───────────────────────────────
        self.explanation_log = {}
        for direction, mask in (("long_entry", entries), ("long_exit", exits)):
            # For every True cell we build one explanation entry.
            stacked = mask.stack()
            for (ts, ticker), flag in stacked.items():
                if not bool(flag):
                    continue
                weights_at_t = {c: float(weights_series.at[ts, c]) for c in self.children}
                child_vals = {
                    c: float(child_signals[c].at[ts, ticker]) for c in self.children
                }
                summary = self._summarise(direction, ticker, weights_at_t, child_vals)
                self.explanation_log[(ticker, ts)] = {
                    "ticker": ticker,
                    "timestamp": ts,
                    "direction": direction,
                    "weights": weights_at_t,
                    "child_signals": child_vals,
                    "summary": summary,
                }

        return Signals(entries=entries, exits=exits)

    def _summarise(
        self,
        direction: str,
        ticker: str,
        weights: dict[str, float],
        child_vals: dict[str, float],
    ) -> str:
        """One-line plain-English summary listing each child's contribution."""
        # Top contributors by absolute weighted signal magnitude.
        contributions = sorted(
            ((c, weights[c] * child_vals[c], child_vals[c]) for c in self.children),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        verb = "Long" if direction == "long_entry" else "Exit"
        # Take the top 3 contributors for the headline.
        head_parts: list[str] = []
        for name, contrib, raw in contributions[:3]:
            sign = "+" if raw >= 0 else "−"
            head_parts.append(f"{name} z={sign}{abs(raw):.2f}")
        head = ", ".join(head_parts)
        tail = ""
        if self.weight_fit_fallback_:
            tail = " (weights: fallback inverse-vol)"
        # Note inactive children so the explanation tells the truth about
        # which arxiv signals aren't voting today.
        inactive = [c for c in self.children if abs(child_vals[c]) < 1e-9]
        if inactive:
            tail += f" [inactive: {', '.join(inactive)}]"
        return f"{verb} {ticker} because: {head}.{tail}"

    # ─── weight learning (walk-forward) ───────────────────────────────────
    def fit_weights_walk_forward(
        self,
        prices_wide: pd.DataFrame,
        train_size: int | None = None,
        test_size: int | None = None,
        mode: str = "expanding",
        periods_per_year: int = 252,
        backtest_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """Learn convex-combination Sharpe-maximising weights via walk-forward.

        Thin wrapper around ``backtest.walkforward.fit_combined_weights_walk_forward``
        — kept here so the strategy presents a single import surface to
        callers. After this returns, ``self.weights`` carries the
        most-recent-fold weights and ``self.weight_fit_fallback_`` tells you
        whether any fold fell back to inverse-vol.
        """
        # Local import to avoid an import cycle at module-load time.
        from backtest.walkforward import (
            WalkForwardConfig,
            fit_combined_weights_walk_forward,
        )

        train = int(train_size or self.params["weight_fit_window"])
        test = int(test_size or self.params["rebalance_freq"])
        cfg = WalkForwardConfig(
            train_size=train,
            test_size=test,
            mode=mode,  # type: ignore[arg-type]
        )
        return fit_combined_weights_walk_forward(
            self,
            prices_wide,
            cfg,
            periods_per_year=periods_per_year,
            backtest_kwargs=backtest_kwargs,
        )

    # ─── public read-only accessors ───────────────────────────────────────
    def get_explanation_log(self) -> dict[tuple[str, pd.Timestamp], dict[str, Any]]:
        """Return the per-(ticker, timestamp) explanation map.

        Contract: keys are (ticker, pd.Timestamp); values are dicts with the
        shape consumed by ``backtest.explainability.TradeExplanation``.
        """
        return dict(self.explanation_log)
