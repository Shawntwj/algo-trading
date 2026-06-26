from __future__ import annotations

import pandas as pd

from .base import Signals, Strategy


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


class RSIMeanReversion(Strategy):
    """Long when RSI dips below `oversold`, exit when it crosses back above `exit_level`."""

    name = "rsi_mean_reversion"

    @classmethod
    def default_params(cls) -> dict:
        return {"period": 14, "oversold": 30, "exit_level": 55}

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        return {
            "period": [7, 14, 21],
            "oversold": [20, 25, 30, 35],
            "exit_level": [50, 55, 60],
        }

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close = data["close"]
        period = self.params["period"]

        rsi = close.apply(lambda s: _wilder_rsi(s, period))

        entries = (rsi < self.params["oversold"]) & (rsi.shift(1) >= self.params["oversold"])
        exits = (rsi > self.params["exit_level"]) & (rsi.shift(1) <= self.params["exit_level"])
        return Signals(entries=entries.fillna(False), exits=exits.fillna(False))
