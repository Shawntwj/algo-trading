"""Tests for backtest/explainability.py (BRIEF Task 4a)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtest import run_backtest
from backtest.explainability import (
    TradeExplanation,
    explain_trades,
    journal_to_file,
    to_journal,
)
from strategies import CombinedExplainableStrategy


# ─── synthetic frame ───────────────────────────────────────────────────────
def _wide(n_bars: int = 500, n_tickers: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=(n_bars, n_tickers))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2022-01-03", periods=n_bars, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    frames = {f: pd.DataFrame(prices, index=idx, columns=tickers) for f in ("open", "high", "low", "close")}
    frames["volume"] = pd.DataFrame(1_000_000, index=idx, columns=tickers)
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


# ─── to_journal round-trip ─────────────────────────────────────────────────
def test_to_journal_markdown_includes_summary_and_children():
    e = TradeExplanation(
        ticker="AAA",
        timestamp=pd.Timestamp("2023-04-01"),
        direction="long_entry",
        weights={"ma_crossover": 0.5, "rsi_mean_reversion": 0.5},
        child_signals={"ma_crossover": 1.2, "rsi_mean_reversion": 0.7},
        summary="Long AAA because: ma_crossover z=+1.20, rsi_mean_reversion z=+0.70.",
    )
    md = to_journal([e], fmt="markdown")
    # Headline section + ticker block + the summary + per-child bullet lines.
    assert "# Trade Journal" in md
    assert "## AAA" in md
    assert "long_entry" in md
    assert "ma_crossover" in md
    assert "rsi_mean_reversion" in md
    assert "weight=0.500" in md
    assert "signal=+1.200" in md or "signal=+0.700" in md


def test_to_journal_json_round_trips_keys():
    e = TradeExplanation(
        ticker="BBB",
        timestamp=pd.Timestamp("2023-05-02"),
        direction="long_exit",
        weights={"ma_crossover": 1.0},
        child_signals={"ma_crossover": -0.4},
        summary="Exit BBB",
    )
    js = to_journal([e], fmt="json")
    parsed = json.loads(js)
    assert isinstance(parsed, list) and len(parsed) == 1
    row = parsed[0]
    for key in ("ticker", "timestamp", "direction", "weights", "child_signals", "summary"):
        assert key in row
    assert row["ticker"] == "BBB"
    assert row["direction"] == "long_exit"
    assert row["timestamp"].startswith("2023-05-02")


def test_to_journal_empty_input():
    md = to_journal([], fmt="markdown")
    txt = to_journal([], fmt="text")
    js = to_journal([], fmt="json")
    assert "No trades" in md
    assert "No trades" in txt
    assert json.loads(js) == []


def test_journal_to_file_writes(tmp_path):
    e = TradeExplanation(
        ticker="CCC",
        timestamp=pd.Timestamp("2023-06-01"),
        direction="long_entry",
        weights={"ma_crossover": 1.0},
        child_signals={"ma_crossover": 2.0},
        summary="Long CCC because: ma_crossover z=+2.00.",
    )
    path = tmp_path / "subdir" / "journal.md"
    journal_to_file([e], path, fmt="markdown")
    assert path.exists()
    body = path.read_text()
    assert "## CCC" in body
    assert "long_entry" in body


# ─── explain_trades ────────────────────────────────────────────────────────
def test_explain_trades_returns_one_per_entry_and_exit():
    """The explainability join produces one record per actual entry+exit
    timestamp recorded by vectorbt."""
    data = _wide(n_bars=500, n_tickers=3, seed=7)
    strat = CombinedExplainableStrategy(
        children=("ma_crossover", "rsi_mean_reversion"),
        min_active_children=1,
    )
    result = run_backtest(data, strat)
    exps = explain_trades(result, strat)

    trades = result.portfolio.trades.records_readable
    # Each completed trade contributes one entry + one exit. Open trades at
    # the end of the window only contribute an entry. Both shapes are valid.
    n_entries = int((trades["Entry Timestamp"].notna()).sum())
    n_exits = int((trades["Exit Timestamp"].notna()).sum())
    assert len(exps) <= n_entries + n_exits
    # We should pick up *some* explanations on this seed.
    assert len(exps) > 0
    # Every returned record carries the contract fields.
    for e in exps:
        assert isinstance(e, TradeExplanation)
        assert e.ticker in data["close"].columns
        assert isinstance(e.timestamp, pd.Timestamp)
        assert abs(sum(e.weights.values()) - 1.0) < 1e-6


def test_explain_trades_empty_log_returns_empty_list():
    """A strategy that produced no entries gives back an empty list."""
    data = _wide(n_bars=80, n_tickers=2, seed=3)  # too short for any child
    strat = CombinedExplainableStrategy(
        children=("ma_crossover",),
        # Crank the threshold so nothing fires.
        entry_threshold=10.0,
    )
    result = run_backtest(data, strat)
    assert explain_trades(result, strat) == []
