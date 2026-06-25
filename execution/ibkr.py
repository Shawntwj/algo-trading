"""IBKR adapter implementing the :class:`Broker` ABC via ``ib_insync``.

Canonical IBKR ports (TWS = Trader Workstation desktop app; Gateway = headless):

    TWS paper      7497
    TWS live       7496
    Gateway paper  4002
    Gateway live   4001

The constructor defaults to TWS ports because that's what a fresh install of
the desktop client uses; pass an explicit ``port=`` if you're running Gateway.

Scope (Task 5a):
    * US equities + ETFs only — ``Stock(symbol, 'SMART', 'USD')``.
    * Futures / options / FX are out of scope and logged as a cut.
    * No market-data subscription wiring (the live runner pulls bars from
      ClickHouse, not IBKR).

All ``ib_insync`` errors are wrapped in :class:`BrokerError`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .broker import (
    Broker,
    BrokerError,
    Order,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

if TYPE_CHECKING:  # pragma: no cover — type hints only
    from ib_insync import IB, Trade as IBTrade


_LOGGER = logging.getLogger(__name__)


# Default ports per IBKR convention (see module docstring).
_DEFAULT_PORTS = {
    # (paper, gateway) -> port
    (True, False): 7497,   # TWS paper
    (False, False): 7496,  # TWS live
    (True, True): 4002,    # Gateway paper
    (False, True): 4001,   # Gateway live
}


# Map normalised OrderStatus from IBKR's textual statuses
# (see https://interactivebrokers.github.io/tws-api/order_submission.html).
_IBKR_STATUS_MAP: dict[str, OrderStatus] = {
    "PendingSubmit": OrderStatus.PENDING,
    "PendingCancel": OrderStatus.SUBMITTED,
    "PreSubmitted": OrderStatus.PENDING,
    "Submitted": OrderStatus.SUBMITTED,
    "ApiPending": OrderStatus.PENDING,
    "ApiCancelled": OrderStatus.CANCELLED,
    "Cancelled": OrderStatus.CANCELLED,
    "Filled": OrderStatus.FILLED,
    "Inactive": OrderStatus.REJECTED,
}


def _default_port(paper: bool, gateway: bool = False) -> int:
    return _DEFAULT_PORTS[(paper, gateway)]


class IBKRBroker(Broker):
    """Broker adapter for Interactive Brokers via TWS or IB Gateway.

    Parameters
    ----------
    host : str
        Hostname running TWS / Gateway. Default ``127.0.0.1``.
    port : int | None
        TCP port. ``None`` picks the canonical port based on ``paper`` and
        ``gateway`` (see module docstring).
    client_id : int
        Per-session client id. Must be unique against the TWS/Gateway instance.
    paper : bool
        ``True`` (default) targets the paper-trading endpoint.
    gateway : bool
        ``False`` (default) means TWS; ``True`` means IB Gateway.
    ib : IB | None
        Inject a pre-built ``ib_insync.IB`` instance — used in tests to
        substitute a ``unittest.mock.Mock``. Real callers leave this ``None``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        client_id: int = 1,
        paper: bool = True,
        gateway: bool = False,
        ib: "IB | None" = None,
    ) -> None:
        self.host = host
        self.port = port if port is not None else _default_port(paper, gateway)
        self.client_id = client_id
        self.paper = paper
        self.gateway = gateway

        if ib is None:
            # Lazy import so the module is importable without ib_insync.
            try:
                from ib_insync import IB  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover — install hint only
                raise BrokerError(
                    "ib_insync is not installed — `pip install ib_insync>=0.9`."
                ) from exc
            self._ib = IB()
        else:
            self._ib = ib

    # ── connection lifecycle ─────────────────────────────────────────────
    def connect(self) -> None:
        if self.is_connected():
            return
        try:
            self._ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
            )
        except Exception as exc:  # noqa: BLE001 — wrap ib_insync surface
            raise BrokerError(
                f"IBKR connect failed (host={self.host}, port={self.port}, "
                f"client_id={self.client_id}): {exc}"
            ) from exc
        _LOGGER.info(
            "IBKR connected host=%s port=%s client_id=%s paper=%s gateway=%s",
            self.host, self.port, self.client_id, self.paper, self.gateway,
        )

    def disconnect(self) -> None:
        if not self.is_connected():
            return
        try:
            self._ib.disconnect()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"IBKR disconnect failed: {exc}") from exc
        _LOGGER.info("IBKR disconnected")

    def is_connected(self) -> bool:
        try:
            return bool(self._ib.isConnected())
        except Exception:  # noqa: BLE001 — defensive
            return False

    # ── orders ───────────────────────────────────────────────────────────
    def submit(self, order: Order) -> OrderId:
        contract = self._stock_contract(order.ticker)
        ib_order = self._to_ib_order(order)
        try:
            trade = self._ib.placeOrder(contract, ib_order)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(
                f"IBKR submit failed for {order.ticker} {order.side} "
                f"{order.quantity}: {exc}"
            ) from exc
        order_id = self._trade_id(trade)
        _LOGGER.info(
            "IBKR submit ticker=%s side=%s qty=%s type=%s id=%s",
            order.ticker, order.side.value, order.quantity,
            order.order_type.value, order_id,
        )
        return order_id

    def cancel(self, order_id: OrderId) -> bool:
        trade = self._find_trade(order_id)
        if trade is None:
            _LOGGER.info("IBKR cancel no-op (order %s not open)", order_id)
            return False
        try:
            self._ib.cancelOrder(trade.order)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"IBKR cancel failed for {order_id}: {exc}") from exc
        _LOGGER.info("IBKR cancel requested id=%s", order_id)
        return True

    def order_status(self, order_id: OrderId) -> OrderStatus:
        trade = self._find_trade(order_id, include_terminal=True)
        if trade is None:
            return OrderStatus.UNKNOWN
        status = getattr(trade.orderStatus, "status", None)
        normalised = _IBKR_STATUS_MAP.get(status, OrderStatus.UNKNOWN)
        _LOGGER.info("IBKR status id=%s raw=%s normalised=%s",
                     order_id, status, normalised.value)
        return normalised

    def get_open_orders(self) -> list[OrderId]:
        try:
            trades = self._ib.openTrades()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"IBKR get_open_orders failed: {exc}") from exc
        return [self._trade_id(t) for t in trades]

    # ── account snapshot ─────────────────────────────────────────────────
    def positions(self) -> dict[str, Position]:
        try:
            raw = self._ib.positions()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"IBKR positions failed: {exc}") from exc
        out: dict[str, Position] = {}
        for p in raw:
            symbol = getattr(p.contract, "symbol", None) or getattr(p.contract, "localSymbol", None)
            if not symbol:
                continue
            out[symbol] = Position(
                ticker=symbol,
                quantity=float(p.position),
                avg_entry_price=float(p.avgCost),
                extra={"account": getattr(p, "account", None)},
            )
        return out

    def cash(self) -> float:
        try:
            values = self._ib.accountSummary()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"IBKR cash failed: {exc}") from exc
        total = 0.0
        for v in values:
            # IBKR reports TotalCashValue per currency plus a "BASE" rollup
            # already converted to the account base currency. Prefer BASE.
            if getattr(v, "tag", None) != "TotalCashValue":
                continue
            if getattr(v, "currency", "") == "BASE":
                try:
                    return float(v.value)
                except (TypeError, ValueError) as exc:
                    raise BrokerError(
                        f"IBKR cash: non-numeric BASE total {v.value!r}"
                    ) from exc
            # Multi-currency fallback: sum reported USD-denominated balances.
            # We deliberately do not synthesise FX here — IBKR already gives
            # us BASE; if BASE is absent, sum USD lines.
            if getattr(v, "currency", "") == "USD":
                try:
                    total += float(v.value)
                except (TypeError, ValueError):
                    continue
        return total

    # ── internals ────────────────────────────────────────────────────────
    @staticmethod
    def _stock_contract(symbol: str):
        from ib_insync import Stock  # type: ignore[import-not-found]
        return Stock(symbol, "SMART", "USD")

    @staticmethod
    def _to_ib_order(order: Order):
        """Translate an :class:`Order` into the appropriate ib_insync order."""
        from ib_insync import (  # type: ignore[import-not-found]
            LimitOrder, MarketOrder, StopLimitOrder, StopOrder,
        )
        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        qty = float(order.quantity)
        if order.order_type == OrderType.MARKET:
            return MarketOrder(action, qty)
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise BrokerError("LIMIT order requires limit_price")
            return LimitOrder(action, qty, float(order.limit_price))
        if order.order_type == OrderType.STOP:
            if order.stop_price is None:
                raise BrokerError("STOP order requires stop_price")
            return StopOrder(action, qty, float(order.stop_price))
        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                raise BrokerError("STOP_LIMIT order requires stop_price and limit_price")
            return StopLimitOrder(action, qty,
                                  float(order.limit_price),
                                  float(order.stop_price))
        raise BrokerError(f"Unsupported order_type: {order.order_type}")

    @staticmethod
    def _trade_id(trade: "IBTrade") -> OrderId:
        # ib_insync.Trade.order.orderId is the local id; permId is the
        # server-assigned id. permId is preferable — survives reconnects.
        perm = getattr(trade.order, "permId", 0)
        if perm:
            return str(perm)
        return str(getattr(trade.order, "orderId", ""))

    def _find_trade(self, order_id: OrderId, include_terminal: bool = False):
        try:
            trades = self._ib.trades() if include_terminal else self._ib.openTrades()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"IBKR trade lookup failed: {exc}") from exc
        for t in trades:
            if self._trade_id(t) == str(order_id):
                return t
        return None
