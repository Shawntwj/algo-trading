from .base import Strategy, Signals
from .ma_crossover import MACrossover
from .rsi_mean_reversion import RSIMeanReversion
from .pca_stat_arb import PCAStatArb
from .macro_timing import MacroTimingXiong

REGISTRY: dict[str, type[Strategy]] = {
    "ma_crossover": MACrossover,
    "rsi_mean_reversion": RSIMeanReversion,
    "pca_stat_arb": PCAStatArb,
    "macro_timing": MacroTimingXiong,
}

__all__ = [
    "Strategy",
    "Signals",
    "MACrossover",
    "RSIMeanReversion",
    "PCAStatArb",
    "MacroTimingXiong",
    "REGISTRY",
]
