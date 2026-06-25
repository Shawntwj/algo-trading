"""
STUB — paper / live execution interface.

This module is intentionally NOT WIRED UP. It exists so the rest of the codebase
has a stable shape to evolve toward live paper trading. Do not send real orders
from this file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    ticker: str
    side: OrderSide
    quantity: float
    limit_price: float | None = None


class Broker(ABC):
    @abstractmethod
    def submit(self, order: Order) -> str: ...

    @abstractmethod
    def positions(self) -> dict[str, float]: ...

    @abstractmethod
    def cash(self) -> float: ...


class PaperBrokerStub(Broker):
    """Placeholder. Raises on every method so accidental wiring fails loudly."""

    def submit(self, order: Order) -> str:
        raise NotImplementedError("Paper broker not implemented — see execution/broker.py")

    def positions(self) -> dict[str, float]:
        raise NotImplementedError

    def cash(self) -> float:
        raise NotImplementedError
