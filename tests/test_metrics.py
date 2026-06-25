from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

vbt = pytest.importorskip("vectorbt")

from backtest.engine import BacktestResult, run_backtest
from backtest.metrics import summarize
from strategies import MACrossover


def _toy_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.01, size=(n, 2))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    tickers = ["AAA", "BBB"]
    frames = {}
    for field in ["open", "high", "low", "close"]:
        frames[field] = pd.DataFrame(prices, index=idx, columns=tickers)
    frames["volume"] = pd.DataFrame(1_000_000, index=idx, columns=tickers)
    wide = pd.concat(frames, axis=1)
    wide.columns.set_names(["field", "ticker"], inplace=True)
    return wide


def test_run_backtest_returns_portfolio_and_metrics():
    data = _toy_data()
    strat = MACrossover(fast=10, slow=30)
    result = run_backtest(data, strat, commission=0.0, slippage=0.0)

    assert isinstance(result, BacktestResult)
    metrics = summarize(result)

    # Required fields are present.
    for key in ("total_return", "sharpe", "max_drawdown", "win_rate", "n_trades", "exposure"):
        assert key in metrics

    # Sanity ranges.
    assert metrics["exposure"] >= 0.0
    assert metrics["exposure"] <= 1.0
    assert metrics["n_trades"] >= 0


def test_zero_signal_yields_zero_trades():
    data = _toy_data()

    class NoTrade(MACrossover):
        def generate_signals(self, data):
            sig = super().generate_signals(data)
            sig.entries.iloc[:, :] = False
            sig.exits.iloc[:, :] = False
            return sig

    result = run_backtest(data, NoTrade(fast=10, slow=30), commission=0.0, slippage=0.0)
    metrics = summarize(result)
    assert metrics["n_trades"] == 0
