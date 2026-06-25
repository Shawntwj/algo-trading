"""Explainability layer for the CombinedExplainableStrategy (BRIEF Task 4a).

This module is read-only: it consumes the per-bar explanation stream a
``CombinedExplainableStrategy`` persists during ``generate_signals`` and turns
it into:

  * ``TradeExplanation`` records — one per actual entry / exit timestamp
    in the backtested portfolio,
  * a renderable trade journal (markdown / json / text).

Contract with the strategy
--------------------------
The strategy keeps a dict on the instance:

    self.explanation_log: dict[(ticker, timestamp), {
        "ticker": str,
        "timestamp": pd.Timestamp,
        "direction": "long_entry" | "long_exit",
        "weights": {child_name: float},
        "child_signals": {child_name: float},
        "summary": str,
    }]

We read it via ``strategy.get_explanation_log()`` and join on the entry /
exit timestamps surfaced by the vectorbt backtest's recorded trades. If a
trade's timestamp doesn't appear in the log (e.g. the strategy was
re-fitted between generate_signals and run_backtest), we skip it with a
warning rather than fabricating an explanation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

log = logging.getLogger(__name__)


Direction = Literal["long_entry", "long_exit", "short_entry", "short_exit"]


@dataclass
class TradeExplanation:
    """One entry/exit observation with the contributing children listed."""

    ticker: str
    timestamp: pd.Timestamp
    direction: Direction
    weights: dict[str, float] = field(default_factory=dict)
    child_signals: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe dict (timestamp → ISO string)."""
        d = asdict(self)
        # `weights` / `child_signals` are already floats; just ISO the timestamp.
        ts = self.timestamp
        d["timestamp"] = (
            ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts)
        )
        return d


# ─── extraction ────────────────────────────────────────────────────────────
def explain_trades(backtest_result, strategy) -> list[TradeExplanation]:
    """Join a backtest's trades with the strategy's per-bar explanation log.

    Parameters
    ----------
    backtest_result : backtest.engine.BacktestResult
        The result of running ``run_backtest`` on a CombinedExplainableStrategy.
    strategy : CombinedExplainableStrategy
        The same instance that was passed into ``run_backtest`` — must carry
        the populated ``explanation_log``.

    Returns
    -------
    list[TradeExplanation]
        One entry per actual entry+exit timestamp recorded in the
        backtest's portfolio.trades.records_readable frame. The list is
        sorted by (ticker, timestamp).

    Notes
    -----
    vectorbt's ``records_readable`` exposes columns ``Column`` (ticker),
    ``Entry Timestamp``, ``Exit Timestamp``. We do not infer the direction
    from the trade's sign because the strategy is long-only; we mark every
    entry as ``long_entry`` and every exit as ``long_exit``. If the strategy
    grows shorts (see IMPROVEMENTS — "Strategy ABC has no first-class
    long-short") this function needs to read the trade's ``direction``
    field directly.
    """
    if not hasattr(strategy, "get_explanation_log"):
        raise TypeError(
            "explain_trades expects a strategy with get_explanation_log() "
            "— pass a CombinedExplainableStrategy instance."
        )
    log_map = strategy.get_explanation_log()
    if not log_map:
        return []

    try:
        records = backtest_result.portfolio.trades.records_readable
    except Exception as exc:
        log.warning("could not read backtest trades: %s — returning empty list", exc)
        return []

    out: list[TradeExplanation] = []

    if records is None or len(records) == 0:
        return out

    # Column names from vectorbt: ``Column`` (ticker for multi-asset), ``Entry
    # Timestamp``, ``Exit Timestamp``. Single-ticker portfolios may omit
    # ``Column``; fall back to result.tickers[0] in that case.
    cols = set(records.columns)
    ticker_col = "Column" if "Column" in cols else None
    entry_col = "Entry Timestamp" if "Entry Timestamp" in cols else "Entry Index"
    exit_col = "Exit Timestamp" if "Exit Timestamp" in cols else "Exit Index"
    default_ticker = (
        backtest_result.tickers[0] if backtest_result.tickers else "UNKNOWN"
    )

    for _, row in records.iterrows():
        ticker = (
            str(row[ticker_col]) if ticker_col is not None else default_ticker
        )
        entry_ts = pd.Timestamp(row[entry_col])
        exit_ts = pd.Timestamp(row[exit_col]) if pd.notna(row[exit_col]) else None

        # entry side
        rec_entry = log_map.get((ticker, entry_ts))
        if rec_entry is not None:
            out.append(_record_to_explanation(rec_entry))
        else:
            log.debug("no explanation for entry %s @ %s", ticker, entry_ts)

        # exit side (some trades are still open at the end of the window)
        if exit_ts is not None:
            rec_exit = log_map.get((ticker, exit_ts))
            if rec_exit is not None:
                out.append(_record_to_explanation(rec_exit))
            else:
                log.debug("no explanation for exit %s @ %s", ticker, exit_ts)

    out.sort(key=lambda e: (e.ticker, e.timestamp))
    return out


def _record_to_explanation(rec: dict[str, Any]) -> TradeExplanation:
    return TradeExplanation(
        ticker=str(rec["ticker"]),
        timestamp=pd.Timestamp(rec["timestamp"]),
        direction=rec["direction"],
        weights=dict(rec.get("weights", {})),
        child_signals=dict(rec.get("child_signals", {})),
        summary=str(rec.get("summary", "")),
    )


# ─── rendering ─────────────────────────────────────────────────────────────
def to_journal(
    explanations: list[TradeExplanation],
    fmt: Literal["text", "json", "markdown"] = "markdown",
) -> str:
    """Render an explanation list to a human-readable trade journal.

    * ``markdown``: groups by ticker, one ``## TICKER`` block per ticker,
      each trade as a sub-section with the summary line + a bullet list of
      contributing children.
    * ``text``: same structure, plain text (no markdown markers).
    * ``json``: a JSON array of dicts — convenient for downstream code or
      the React UI (Task 4b).
    """
    if fmt == "json":
        return json.dumps([e.as_dict() for e in explanations], indent=2, default=str)
    if fmt not in {"markdown", "text"}:
        raise ValueError(f"unknown fmt {fmt!r}; want 'markdown' | 'text' | 'json'")

    if not explanations:
        return "_No trades to explain._" if fmt == "markdown" else "No trades to explain."

    # Group by ticker, preserving timestamp order within a ticker.
    by_ticker: dict[str, list[TradeExplanation]] = {}
    for e in sorted(explanations, key=lambda x: (x.ticker, x.timestamp)):
        by_ticker.setdefault(e.ticker, []).append(e)

    lines: list[str] = []
    h1 = "# " if fmt == "markdown" else ""
    h2 = "## " if fmt == "markdown" else ""
    h3 = "### " if fmt == "markdown" else ""
    bullet = "- " if fmt == "markdown" else "  * "

    lines.append(f"{h1}Trade Journal")
    lines.append("")
    for ticker, items in by_ticker.items():
        lines.append(f"{h2}{ticker}")
        lines.append("")
        for e in items:
            ts_str = (
                e.timestamp.isoformat()
                if isinstance(e.timestamp, pd.Timestamp)
                else str(e.timestamp)
            )
            lines.append(f"{h3}{ts_str} — {e.direction}")
            lines.append("")
            lines.append(e.summary)
            lines.append("")
            # Contributing children (sorted by |weight*signal| desc)
            contribs = sorted(
                e.weights.keys(),
                key=lambda k: abs(e.weights.get(k, 0.0) * e.child_signals.get(k, 0.0)),
                reverse=True,
            )
            for c in contribs:
                w = e.weights.get(c, 0.0)
                s = e.child_signals.get(c, 0.0)
                lines.append(f"{bullet}{c}: weight={w:.3f}, signal={s:+.3f}, contribution={w*s:+.3f}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def journal_to_file(
    explanations: list[TradeExplanation],
    path: str | Path,
    fmt: Literal["text", "json", "markdown"] = "markdown",
) -> None:
    """Render the journal and persist to ``path``. Creates parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_journal(explanations, fmt=fmt), encoding="utf-8")
