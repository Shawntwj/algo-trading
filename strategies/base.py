from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Signals:
    """Boolean entry/exit masks aligned to the price index. For multi-ticker
    backtests both frames are wide (columns = tickers)."""
    entries: pd.DataFrame
    exits: pd.DataFrame


class Strategy(ABC):
    """Implement one strategy per file. Engine only depends on this interface.

    Conventions:
      - `data` is a wide pandas DataFrame with a MultiIndex on columns:
        level 0 = field (open/high/low/close/volume), level 1 = ticker.
      - generate_signals returns boolean DataFrames indexed identically to
        data['close'] (rows=timestamps, columns=tickers).
    """

    name: str = "strategy"

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = {**self.default_params(), **params}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def param_grid(cls) -> dict[str, list[Any]]:
        """Override to expose tunable params for parameter sweeps."""
        return {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        ...

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"
