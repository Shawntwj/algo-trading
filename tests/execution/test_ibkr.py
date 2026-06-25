"""IBKRBroker tests — all mocked (ib_insync uses a socket protocol, not HTTP,
so VCR doesn't apply). We inject a ``unittest.mock.Mock`` via the broker's
``ib=`` constructor kwarg.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from execution import BrokerError, Order, OrderSide, OrderStatus, OrderType
from execution.ibkr import IBKRBroker, _default_port


# ── port selection ────────────────────────────────────────────────────────
def test_default_ports_canonical():
    assert _default_port(paper=True) == 7497
    assert _default_port(paper=False) == 7496
    assert _default_port(paper=True, gateway=True) == 4002
    assert _default_port(paper=False, gateway=True) == 4001


def test_constructor_paper_picks_7497():
    ib = MagicMock()
    b = IBKRBroker(paper=True, ib=ib)
    assert b.port == 7497
    assert b.paper is True


def test_constructor_live_picks_7496():
    ib = MagicMock()
    b = IBKRBroker(paper=False, ib=ib)
    assert b.port == 7496


def test_constructor_explicit_port_wins():
    ib = MagicMock()
    b = IBKRBroker(paper=True, port=4002, ib=ib)
    assert b.port == 4002


# ── lifecycle ─────────────────────────────────────────────────────────────
def test_connect_calls_underlying_and_logs():
    ib = MagicMock()
    ib.isConnected.return_value = False
    b = IBKRBroker(ib=ib, client_id=42)
    b.connect()
    ib.connect.assert_called_once_with(host="127.0.0.1", port=7497, clientId=42)


def test_connect_is_idempotent():
    ib = MagicMock()
    ib.isConnected.return_value = True
    b = IBKRBroker(ib=ib)
    b.connect()
    ib.connect.assert_not_called()


def test_disconnect_is_idempotent():
    ib = MagicMock()
    ib.isConnected.return_value = False
    b = IBKRBroker(ib=ib)
    b.disconnect()
    ib.disconnect.assert_not_called()


def test_connect_wraps_error_in_broker_error():
    ib = MagicMock()
    ib.isConnected.return_value = False
    ib.connect.side_effect = ConnectionRefusedError("no TWS on 7497")
    b = IBKRBroker(ib=ib)
    with pytest.raises(BrokerError, match="IBKR connect failed"):
        b.connect()


# ── submit ────────────────────────────────────────────────────────────────
def _trade(perm_id: int = 1234, status: str = "Submitted") -> SimpleNamespace:
    """Build a stand-in for ``ib_insync.Trade``."""
    return SimpleNamespace(
        order=SimpleNamespace(permId=perm_id, orderId=perm_id),
        orderStatus=SimpleNamespace(status=status),
    )


def test_submit_market_returns_order_id():
    ib = MagicMock()
    ib.placeOrder.return_value = _trade(perm_id=987)
    b = IBKRBroker(ib=ib)
    order = Order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
    assert b.submit(order) == "987"
    ib.placeOrder.assert_called_once()


def test_submit_limit_requires_price():
    ib = MagicMock()
    b = IBKRBroker(ib=ib)
    order = Order("AAPL", OrderSide.BUY, 10, OrderType.LIMIT)
    with pytest.raises(BrokerError, match="LIMIT order requires"):
        b.submit(order)


def test_submit_wraps_vendor_error():
    ib = MagicMock()
    ib.placeOrder.side_effect = RuntimeError("TWS rejected")
    b = IBKRBroker(ib=ib)
    with pytest.raises(BrokerError, match="IBKR submit failed"):
        b.submit(Order("AAPL", OrderSide.BUY, 1, OrderType.MARKET))


# ── cancel ────────────────────────────────────────────────────────────────
def test_cancel_open_order_returns_true():
    ib = MagicMock()
    ib.openTrades.return_value = [_trade(perm_id=42)]
    b = IBKRBroker(ib=ib)
    assert b.cancel("42") is True
    ib.cancelOrder.assert_called_once()


def test_cancel_missing_order_returns_false():
    ib = MagicMock()
    ib.openTrades.return_value = []
    b = IBKRBroker(ib=ib)
    assert b.cancel("999") is False
    ib.cancelOrder.assert_not_called()


# ── order_status ──────────────────────────────────────────────────────────
def test_order_status_maps_filled():
    ib = MagicMock()
    ib.trades.return_value = [_trade(perm_id=7, status="Filled")]
    b = IBKRBroker(ib=ib)
    assert b.order_status("7") == OrderStatus.FILLED


def test_order_status_unknown_when_missing():
    ib = MagicMock()
    ib.trades.return_value = []
    b = IBKRBroker(ib=ib)
    assert b.order_status("404") == OrderStatus.UNKNOWN


def test_get_open_orders_returns_ids():
    ib = MagicMock()
    ib.openTrades.return_value = [_trade(perm_id=1), _trade(perm_id=2)]
    b = IBKRBroker(ib=ib)
    assert b.get_open_orders() == ["1", "2"]


# ── positions / cash ──────────────────────────────────────────────────────
def test_positions_dict_shape():
    ib = MagicMock()
    pos = SimpleNamespace(
        contract=SimpleNamespace(symbol="MSFT", localSymbol="MSFT"),
        position=100.0,
        avgCost=350.5,
        account="DU123",
    )
    ib.positions.return_value = [pos]
    b = IBKRBroker(ib=ib)
    out = b.positions()
    assert "MSFT" in out
    assert out["MSFT"].quantity == 100.0
    assert out["MSFT"].avg_entry_price == 350.5


def test_positions_empty_dict():
    ib = MagicMock()
    ib.positions.return_value = []
    b = IBKRBroker(ib=ib)
    assert b.positions() == {}


def test_cash_prefers_base_total():
    ib = MagicMock()
    ib.accountSummary.return_value = [
        SimpleNamespace(tag="TotalCashValue", currency="USD", value="50000"),
        SimpleNamespace(tag="TotalCashValue", currency="BASE", value="48500"),
        SimpleNamespace(tag="NetLiquidation", currency="BASE", value="123"),
    ]
    b = IBKRBroker(ib=ib)
    assert b.cash() == 48500.0


def test_cash_sums_usd_when_base_missing():
    ib = MagicMock()
    ib.accountSummary.return_value = [
        SimpleNamespace(tag="TotalCashValue", currency="USD", value="1000"),
        SimpleNamespace(tag="TotalCashValue", currency="USD", value="2500"),
    ]
    b = IBKRBroker(ib=ib)
    assert b.cash() == 3500.0


def test_cash_wraps_error():
    ib = MagicMock()
    ib.accountSummary.side_effect = RuntimeError("ack timeout")
    b = IBKRBroker(ib=ib)
    with pytest.raises(BrokerError, match="IBKR cash failed"):
        b.cash()
