from .base import Strategy, Signals
from .ma_crossover import MACrossover
from .rsi_mean_reversion import RSIMeanReversion
from .pca_stat_arb import PCAStatArb
from .macro_timing import MacroTimingXiong
from .drift_regime import DriftRegimeSingha
from .picker_clone import (
    PICKER_CLONE_REGISTRY,
    PickerCloneAppaloosa,
    PickerCloneBerkshire,
    PickerClonePershingSquare,
    PickerCloneScion,
    PickerCloneStrategy,
)
from .combined_explainable import CombinedExplainableStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "ma_crossover": MACrossover,
    "rsi_mean_reversion": RSIMeanReversion,
    "pca_stat_arb": PCAStatArb,
    "macro_timing": MacroTimingXiong,
    "drift_regime": DriftRegimeSingha,
    **PICKER_CLONE_REGISTRY,
    "combined_explainable": CombinedExplainableStrategy,
}

__all__ = [
    "Strategy",
    "Signals",
    "MACrossover",
    "RSIMeanReversion",
    "PCAStatArb",
    "MacroTimingXiong",
    "DriftRegimeSingha",
    "PickerCloneStrategy",
    "PickerCloneBerkshire",
    "PickerClonePershingSquare",
    "PickerCloneAppaloosa",
    "PickerCloneScion",
    "PICKER_CLONE_REGISTRY",
    "CombinedExplainableStrategy",
    "REGISTRY",
]
