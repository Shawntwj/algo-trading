"""Broker abstraction shared by paper / live adapters.

The ABC is intentionally minimal and synchronous. Concrete adapters live in
``execution/ibkr.py`` (IBKR via ``ib_insync``) and ``execution/alpaca.py``
(``alpaca-py``). The legacy ``PaperBrokerStub`` is kept for backward
compatibility — any code that imports it still gets a loud
``NotImplementedError`` when it touches a method.

Task 5a expanded the original surface (``submit`` / ``positions`` / ``cash``)
to cover the rest of the lifecycle real adapters need: ``cancel``,
``order_status``, ``get_open_orders``, and a richer ``positions()`` return
shape. ``OrderType`` and ``OrderStatus`` enums are introduced so adapters can
map vendor enums onto a single vocabulary. The live runner (Task 5b) consumes
this surface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BrokerError(RuntimeError):
    """Uniform exception wrapper for broker adapters.

    All concrete adapters (``IBKRBroker``, ``AlpacaBroker``) MUST translate
    vendor-specific errors (``ib_insync`` exceptions, ``alpaca.common.exceptions``,
    HTTP errors, missing credentials, connection drops) into ``BrokerError`` so
    the live runner has a single ``except`` clause to write against.
    """


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Normalised lifecycle states. Adapters MUST map their vendor enum
    onto these (anything not recognised → ``UNKNOWN``).
    """

    PENDING = "pending"           # accepted by adapter / awaiting venue ack
    SUBMITTED = "submitted"       # live at the venue, no fills yet
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class Order:
    """Adapter-agnostic order request."""

    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None


# Adapters return ``str`` order ids (IBKR permId stringified, Alpaca uuid str).
OrderId = str


@dataclass
class Position:
    """Adapter-agnostic open position. ``quantity`` is signed (short < 0)."""

    ticker: str
    quantity: float
    avg_entry_price: float
    market_value: float | None = None
    unrealised_pl: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Broker(ABC):
    """Synchronous broker contract.

    The live runner (Task 5b) calls these from a single event-loop thread.
    Adapters MUST be safe to construct without making any network calls — IO
    happens in ``connect()`` (where applicable) or on the first method call.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish a session with the venue. Idempotent — calling twice on
        a live connection MUST be a no-op."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the session. Idempotent."""

    @abstractmethod
    def is_connected(self) -> bool:
        """``True`` iff the adapter currently holds a live session."""

    @abstractmethod
    def submit(self, order: Order) -> OrderId:
        """Submit an order; return the venue's order id (string)."""

    @abstractmethod
    def cancel(self, order_id: OrderId) -> bool:
        """Best-effort cancel. Returns ``True`` if the venue accepted the
        cancel request; ``False`` if the order was already terminal."""

    @abstractmethod
    def order_status(self, order_id: OrderId) -> OrderStatus:
        """Current lifecycle state for ``order_id``."""

    @abstractmethod
    def get_open_orders(self) -> list[OrderId]:
        """All non-terminal order ids the venue currently knows about."""

    @abstractmethod
    def positions(self) -> dict[str, Position]:
        """Open positions keyed by ticker. Empty dict if flat."""

    @abstractmethod
    def cash(self) -> float:
        """USD cash balance. Multi-currency adapters MUST sum to USD using
        the venue-reported FX (document the conversion in their docstring)."""


class PaperBrokerStub(Broker):
    """Legacy placeholder. Every method raises so accidental wiring fails
    loudly. Real paper trading goes through ``IBKRBroker(paper=True)`` or
    ``AlpacaBroker(paper=True)``.
    """

    def connect(self) -> None:
        raise NotImplementedError(
            "Paper broker not implemented — use IBKRBroker(paper=True) or "
            "AlpacaBroker(paper=True) instead. See execution/ibkr.py / alpaca.py."
        )

    def disconnect(self) -> None:
        raise NotImplementedError

    def is_connected(self) -> bool:  # pragma: no cover — stub
        return False

    def submit(self, order: Order) -> OrderId:
        raise NotImplementedError

    def cancel(self, order_id: OrderId) -> bool:
        raise NotImplementedError

    def order_status(self, order_id: OrderId) -> OrderStatus:
        raise NotImplementedError

    def get_open_orders(self) -> list[OrderId]:
        raise NotImplementedError

    def positions(self) -> dict[str, Position]:
        raise NotImplementedError

    def cash(self) -> float:
        raise NotImplementedError
