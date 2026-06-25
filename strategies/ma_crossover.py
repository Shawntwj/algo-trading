from __future__ import annotations

import pandas as pd

from .base import Signals, Strategy


class MACrossover(Strategy):
    """Go long when fast SMA crosses above slow SMA; exit on the opposite cross."""

    name = "ma_crossover"

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 20, "slow": 50}

    @classmethod
    def param_grid(cls) -> dict[str, list]:
        return {
            "fast": [5, 10, 20, 30],
            "slow": [50, 100, 150, 200],
        }

    def generate_signals(self, data: pd.DataFrame) -> Signals:
        close = data["close"]
        fast = close.rolling(self.params["fast"]).mean()
        slow = close.rolling(self.params["slow"]).mean()

        long_state = (fast > slow).fillna(False)
        entries = long_state & ~long_state.shift(1, fill_value=False)
        exits = ~long_state & long_state.shift(1, fill_value=False)
        return Signals(entries=entries, exits=exits)
