from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from .base import Signals, Strategy


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

        rsi = close.apply(lambda s: ta.rsi(s, length=period))

        entries = (rsi < self.params["oversold"]) & (rsi.shift(1) >= self.params["oversold"])
        exits = (rsi > self.params["exit_level"]) & (rsi.shift(1) <= self.params["exit_level"])
        return Signals(entries=entries.fillna(False), exits=exits.fillna(False))
