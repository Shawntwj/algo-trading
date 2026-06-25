"""Alpaca adapter implementing the :class:`Broker` ABC via ``alpaca-py``.

Credentials are read from the environment only — never from a committed file.
Pass them in directly only if you're calling the constructor from your own
process bootstrap; the smoke script and tests both rely on env vars.

    ALPACA_API_KEY     paper or live API key id
    ALPACA_SECRET_KEY  matching secret

Paper vs live URL is selected by the ``paper`` flag:

    paper=True   https://paper-api.alpaca.markets
    paper=False  https://api.alpaca.markets

Scope (Task 5a):
    * US equities only — Alpaca crypto and OTC are out of scope.
    * MARKET / LIMIT / STOP / STOP_LIMIT order types.
    * No streaming (the live runner consumes ClickHouse bars).

``alpaca-py`` doesn't expose a stateful "connection" — every call is a fresh
HTTP request — so :meth:`connect` / :meth:`disconnect` are effectively
markers for the live runner's lifecycle and ``is_connected`` reports whether
the ``TradingClient`` was constructed without error.
"""
from __future__ import annotations

import logging
import os
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
    from alpaca.trading.client import TradingClient


_LOGGER = logging.getLogger(__name__)


PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


# Map Alpaca's terminal/non-terminal statuses onto our normalised enum.
# See https://docs.alpaca.markets/docs/orders-at-alpaca#order-lifecycle.
_ALPACA_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "pending_review": OrderStatus.PENDING,
    "held": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.SUBMITTED,
    "canceled": OrderStatus.CANCELLED,
    "pending_cancel": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "stopped": OrderStatus.FILLED,
    "replaced": OrderStatus.CANCELLED,
    "pending_replace": OrderStatus.SUBMITTED,
    "calculated": OrderStatus.SUBMITTED,
}


class AlpacaBroker(Broker):
    """Broker adapter for Alpaca via the ``alpaca-py`` ``TradingClient``.

    Parameters
    ----------
    api_key : str | None
        API key id. ``None`` falls back to ``$ALPACA_API_KEY``.
    secret_key : str | None
        API secret. ``None`` falls back to ``$ALPACA_SECRET_KEY``.
    paper : bool
        ``True`` (default) targets the paper endpoint.
    client : TradingClient | None
        Inject a pre-built client (used in tests with VCR cassettes).
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        client: "TradingClient | None" = None,
    ) -> None:
        self.paper = paper
        self.base_url = PAPER_URL if paper else LIVE_URL

        # Resolve credentials. Allow client-injection to bypass the env-var
        # requirement (tests and embedded usage).
        if client is None:
            api_key = api_key or os.environ.get("ALPACA_API_KEY")
            secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
            if not api_key or not secret_key:
                raise BrokerError(
                    "AlpacaBroker requires credentials — set ALPACA_API_KEY "
                    "and ALPACA_SECRET_KEY env vars (or pass them explicitly)."
                )
            try:
                from alpaca.trading.client import TradingClient  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover — install hint only
                raise BrokerError(
                    "alpaca-py is not installed — `pip install alpaca-py>=0.20`."
                ) from exc
            try:
                self._client = TradingClient(
                    api_key=api_key,
                    secret_key=secret_key,
                    paper=paper,
                )
            except Exception as exc:  # noqa: BLE001
                raise BrokerError(f"Alpaca client init failed: {exc}") from exc
        else:
            self._client = client

        self._connected = False

    # ── connection lifecycle ─────────────────────────────────────────────
    def connect(self) -> None:
        # No persistent socket — just sanity-probe the account.
        if self._connected:
            return
        try:
            self._client.get_account()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"Alpaca connect failed: {exc}") from exc
        self._connected = True
        _LOGGER.info("Alpaca connected paper=%s base_url=%s",
                     self.paper, self.base_url)

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        _LOGGER.info("Alpaca disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ── orders ───────────────────────────────────────────────────────────
    def submit(self, order: Order) -> OrderId:
        request = self._build_request(order)
        try:
            resp = self._client.submit_order(order_data=request)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(
                f"Alpaca submit failed for {order.ticker} {order.side} "
                f"{order.quantity}: {exc}"
            ) from exc
        order_id = str(getattr(resp, "id", resp))
        _LOGGER.info(
            "Alpaca submit ticker=%s side=%s qty=%s type=%s id=%s",
            order.ticker, order.side.value, order.quantity,
            order.order_type.value, order_id,
        )
        return order_id

    def cancel(self, order_id: OrderId) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"Alpaca cancel failed for {order_id}: {exc}") from exc
        _LOGGER.info("Alpaca cancel requested id=%s", order_id)
        return True

    def order_status(self, order_id: OrderId) -> OrderStatus:
        try:
            order = self._client.get_order_by_id(order_id)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"Alpaca order_status failed for {order_id}: {exc}") from exc
        raw = getattr(order, "status", None)
        raw_str = raw.value if hasattr(raw, "value") else str(raw)
        normalised = _ALPACA_STATUS_MAP.get(raw_str, OrderStatus.UNKNOWN)
        _LOGGER.info("Alpaca status id=%s raw=%s normalised=%s",
                     order_id, raw_str, normalised.value)
        return normalised

    def get_open_orders(self) -> list[OrderId]:
        try:
            from alpaca.trading.enums import QueryOrderStatus  # type: ignore[import-not-found]
            from alpaca.trading.requests import GetOrdersRequest  # type: ignore[import-not-found]
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._client.get_orders(filter=req)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"Alpaca get_open_orders failed: {exc}") from exc
        return [str(o.id) for o in orders]

    # ── account snapshot ─────────────────────────────────────────────────
    def positions(self) -> dict[str, Position]:
        try:
            raw = self._client.get_all_positions()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"Alpaca positions failed: {exc}") from exc
        out: dict[str, Position] = {}
        for p in raw:
            qty = float(p.qty)
            # Alpaca returns positive qty + a 'side' field; sign it here.
            side = getattr(p.side, "value", str(p.side))
            if side == "short":
                qty = -qty
            out[p.symbol] = Position(
                ticker=p.symbol,
                quantity=qty,
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value) if p.market_value is not None else None,
                unrealised_pl=float(p.unrealized_pl) if p.unrealized_pl is not None else None,
                extra={"asset_class": getattr(p.asset_class, "value", str(p.asset_class))},
            )
        return out

    def cash(self) -> float:
        try:
            account = self._client.get_account()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"Alpaca cash failed: {exc}") from exc
        try:
            return float(account.cash)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"Alpaca cash: non-numeric balance {account.cash!r}") from exc

    # ── internals ────────────────────────────────────────────────────────
    @staticmethod
    def _build_request(order: Order):
        from alpaca.trading.enums import (  # type: ignore[import-not-found]
            OrderSide as AlpacaOrderSide,
            TimeInForce,
        )
        from alpaca.trading.requests import (  # type: ignore[import-not-found]
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
        common = dict(
            symbol=order.ticker,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(**common)
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise BrokerError("LIMIT order requires limit_price")
            return LimitOrderRequest(limit_price=float(order.limit_price), **common)
        if order.order_type == OrderType.STOP:
            if order.stop_price is None:
                raise BrokerError("STOP order requires stop_price")
            return StopOrderRequest(stop_price=float(order.stop_price), **common)
        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                raise BrokerError("STOP_LIMIT order requires stop_price and limit_price")
            return StopLimitOrderRequest(
                stop_price=float(order.stop_price),
                limit_price=float(order.limit_price),
                **common,
            )
        raise BrokerError(f"Unsupported order_type: {order.order_type}")
