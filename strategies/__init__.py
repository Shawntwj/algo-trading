from .base import Strategy, Signals
from .ma_crossover import MACrossover
from .rsi_mean_reversion import RSIMeanReversion

REGISTRY: dict[str, type[Strategy]] = {
    "ma_crossover": MACrossover,
    "rsi_mean_reversion": RSIMeanReversion,
}

__all__ = ["Strategy", "Signals", "MACrossover", "RSIMeanReversion", "REGISTRY"]
