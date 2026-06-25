"""Matplotlib chart helpers for the evaluation report (BRIEF Task 7e).

Every chart is rendered with the ``Agg`` backend and returned as a base64-encoded
PNG payload so it can be embedded directly into the HTML via
``<img src="data:image/png;base64,...">``. No external assets, no network — the
report stays a single self-contained file.

Colour palette is sourced from ``RUNBOOK.html`` so the chart styling matches
the rest of the report:

    --bg:        #0e1116    page background
    --panel:     #161b22    surface
    --ink:       #e6edf3    primary text
    --mute:      #8b949e    secondary text
    --accent:    #79c0ff    series 1 / links
    --good:      #3fb950    series 2 / passing badge
    --bad:       #f85149    series 3 / failing badge
    --warn:      #d29922    series 4 / warning

Cuts (see IMPROVEMENTS.md, Reports):
  * No retina (figsize-only) — DPI is fixed at 110, which is fine for a desktop
    report but coarse on a 4K display. Callers wanting print quality can pass
    ``dpi=`` once we plumb it through.
  * Charts that take a benchmark series re-index by row position, not timestamp.
    For walk-forward / sweep reports where every series shares the same bar
    grid this is fine; for ragged calendars we'd need a proper outer join.
"""
from __future__ import annotations

import base64
import io
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend — must come before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ─── palette (mirrors RUNBOOK.html :root variables) ────────────────────────
PALETTE = {
    "bg": "#0e1116",
    "panel": "#161b22",
    "ink": "#e6edf3",
    "mute": "#8b949e",
    "accent": "#79c0ff",
    "good": "#3fb950",
    "bad": "#f85149",
    "warn": "#d29922",
    "border": "#30363d",
    "grid": "#22272e",
}

# Cycle used when multiple series share a chart (entries 1..4 of the palette).
_SERIES_COLOURS = [
    PALETTE["accent"],
    PALETTE["good"],
    PALETTE["warn"],
    PALETTE["bad"],
    "#a371f7",  # purple — keeps 5+ series readable
    "#e3b341",  # amber — falls back to warn tone
]


# ─── plumbing ──────────────────────────────────────────────────────────────
def _new_axes(figsize: tuple[float, float] = (8.5, 3.6)) -> tuple[plt.Figure, plt.Axes]:
    """Build a figure/axes pair pre-styled to the RUNBOOK dark theme."""
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    ax.tick_params(colors=PALETTE["mute"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.title.set_color(PALETTE["ink"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.7)
    return fig, ax


def _to_base64_png(fig: plt.Figure) -> str:
    """Encode the figure as a base64 PNG payload ready for a ``data:`` URI."""
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─── public chart helpers ──────────────────────────────────────────────────
def equity_curve_chart(
    timestamps: Sequence,
    series: dict[str, Sequence[float]],
    title: str = "Equity curves",
) -> str:
    """Multi-line equity curve chart.

    ``series`` is an ordered mapping ``label -> equity_values`` aligned to
    ``timestamps``. Returns a base64-encoded PNG payload (no ``data:`` prefix).
    """
    fig, ax = _new_axes(figsize=(9.2, 3.8))
    xs = list(timestamps)
    for i, (label, values) in enumerate(series.items()):
        colour = _SERIES_COLOURS[i % len(_SERIES_COLOURS)]
        ax.plot(xs, list(values), label=label, color=colour, linewidth=1.6)
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.legend(
        loc="upper left", facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
        labelcolor=PALETTE["ink"], fontsize=9,
    )
    fig.autofmt_xdate(rotation=20, ha="right")
    return _to_base64_png(fig)


def price_with_markers_chart(
    timestamps: Sequence,
    close: Sequence[float],
    entry_ts: Sequence,
    exit_ts: Sequence,
    ticker: str,
) -> str:
    """Close price with entry (^) and exit (v) triangle markers."""
    fig, ax = _new_axes(figsize=(9.2, 3.4))
    xs = list(timestamps)
    ax.plot(xs, list(close), color=PALETTE["accent"], linewidth=1.4, label=ticker)
    # Look up prices at marker timestamps. Use an index map so we don't depend
    # on pandas / a particular timestamp type.
    ts_to_price = dict(zip(xs, close))
    e_xs = [t for t in entry_ts if t in ts_to_price]
    e_ys = [ts_to_price[t] for t in e_xs]
    x_xs = [t for t in exit_ts if t in ts_to_price]
    x_ys = [ts_to_price[t] for t in x_xs]
    if e_xs:
        ax.scatter(e_xs, e_ys, marker="^", color=PALETTE["good"], s=42,
                   zorder=3, label="entry")
    if x_xs:
        ax.scatter(x_xs, x_ys, marker="v", color=PALETTE["bad"], s=42,
                   zorder=3, label="exit")
    ax.set_title(f"{ticker} price with entries / exits")
    ax.set_ylabel("Close")
    ax.legend(
        loc="upper left", facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
        labelcolor=PALETTE["ink"], fontsize=9,
    )
    fig.autofmt_xdate(rotation=20, ha="right")
    return _to_base64_png(fig)


def regime_sharpe_chart(
    dimension: str,
    regimes: Sequence[str],
    sharpes: Sequence[float],
) -> str:
    """Bar chart of Sharpe by regime label (one chart per dimension)."""
    fig, ax = _new_axes(figsize=(6.8, 3.0))
    xs = list(range(len(regimes)))
    # Colour bars by sign: green for positive, red for negative, grey for NaN.
    colours = []
    for s in sharpes:
        if s is None or not np.isfinite(s):
            colours.append(PALETTE["mute"])
        elif s >= 0:
            colours.append(PALETTE["good"])
        else:
            colours.append(PALETTE["bad"])
    ax.bar(xs, [0 if (s is None or not np.isfinite(s)) else s for s in sharpes],
           color=colours, edgecolor=PALETTE["border"])
    ax.set_xticks(xs)
    ax.set_xticklabels(list(regimes), color=PALETTE["ink"])
    ax.axhline(0, color=PALETTE["mute"], linewidth=0.8)
    ax.set_title(f"Sharpe by {dimension} regime")
    ax.set_ylabel("Annualised Sharpe")
    return _to_base64_png(fig)


def walkforward_decay_chart(
    is_sharpes: Sequence[float],
    oos_sharpes: Sequence[float],
    slope: float,
) -> str:
    """IS vs OOS Sharpe scatter with y=x reference line and slope annotation."""
    fig, ax = _new_axes(figsize=(6.8, 4.4))
    is_arr = np.asarray(list(is_sharpes), dtype=float)
    oos_arr = np.asarray(list(oos_sharpes), dtype=float)
    finite = np.isfinite(is_arr) & np.isfinite(oos_arr)
    if finite.any():
        ax.scatter(
            is_arr[finite], oos_arr[finite],
            color=PALETTE["accent"], s=55, edgecolor=PALETTE["ink"], linewidth=0.6,
        )
        lo = float(min(is_arr[finite].min(), oos_arr[finite].min()))
        hi = float(max(is_arr[finite].max(), oos_arr[finite].max()))
        pad = max(0.2, 0.1 * (hi - lo))
        line_lo, line_hi = lo - pad, hi + pad
    else:
        line_lo, line_hi = -1.0, 1.0
    ax.plot(
        [line_lo, line_hi], [line_lo, line_hi],
        color=PALETTE["mute"], linestyle="--", linewidth=1.0, label="y = x",
    )
    ax.set_xlim(line_lo, line_hi)
    ax.set_ylim(line_lo, line_hi)
    ax.set_xlabel("In-sample Sharpe")
    ax.set_ylabel("Out-of-sample Sharpe")
    title_slope = f"{slope:.2f}" if np.isfinite(slope) else "n/a"
    ax.set_title(f"OOS decay (slope = {title_slope})")
    ax.legend(
        loc="upper left", facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
        labelcolor=PALETTE["ink"], fontsize=9,
    )
    return _to_base64_png(fig)
