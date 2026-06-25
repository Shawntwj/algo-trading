"""Tests for ``execution.risk`` — RiskLimits + RiskGate.

Pure-Python tests; no broker or DB I/O. The Broker reference is mocked
because :meth:`RiskGate.check_order` doesn't call it (kept on the instance
per the BRIEF signature so future caps can use it).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from execution.broker import Order, OrderSide, OrderType, Position
from execution.risk import RiskGate, RiskLimits


def _limits(**over):
    base = dict(
        max_position_usd=25_000.0,
        max_daily_loss_usd=1_000.0,
        max_gross_exposure_usd=100_000.0,
        max_order_notional_usd=5_000.0,
    )
    base.update(over)
    return RiskLimits(**base)


def _gate(daily_pnl: float = 0.0, **over):
    return RiskGate(limits=_limits(**over), broker=MagicMock(), daily_pnl=daily_pnl)


def _order(qty: float = 10.0, side: OrderSide = OrderSide.BUY, ticker: str = "AAPL"):
    return Order(ticker=ticker, side=side, quantity=qty, order_type=OrderType.MARKET)


# ─── RiskLimits validation ────────────────────────────────────────────────
def test_risk_limits_rejects_non_positive_caps():
    for bad in ("max_position_usd", "max_daily_loss_usd", "max_gross_exposure_usd"):
        with pytest.raises(ValueError, match=bad):
            RiskLimits(
                max_position_usd=1.0 if bad != "max_position_usd" else 0,
                max_daily_loss_usd=1.0 if bad != "max_daily_loss_usd" else 0,
                max_gross_exposure_usd=1.0 if bad != "max_gross_exposure_usd" else 0,
            )


def test_risk_limits_rejects_negative_order_notional():
    with pytest.raises(ValueError, match="max_order_notional_usd"):
        RiskLimits(
            max_position_usd=1.0,
            max_daily_loss_usd=1.0,
            max_gross_exposure_usd=1.0,
            max_order_notional_usd=-1,
        )


# ─── Happy path ───────────────────────────────────────────────────────────
def test_happy_path_allows_modest_order():
    gate = _gate()
    order = _order(qty=10)
    projected = {"AAPL": Position("AAPL", quantity=10, avg_entry_price=200.0)}
    ok, reason = gate.check_order(order, projected, price_hints={"AAPL": 200.0})
    assert ok is True
    assert reason is None


# ─── Each cap individually ────────────────────────────────────────────────
def test_blocks_when_no_price_hint_available():
    gate = _gate()
    order = _order()
    ok, reason = gate.check_order(order, {}, price_hints={})
    assert ok is False
    assert "no price hint" in reason


def test_max_order_notional_breach():
    gate = _gate()
    # 30 * 200 = 6000 > 5000 cap
    order = _order(qty=30)
    projected = {"AAPL": Position("AAPL", quantity=30, avg_entry_price=200.0)}
    ok, reason = gate.check_order(order, projected, price_hints={"AAPL": 200.0})
    assert ok is False
    assert "max_order_notional_usd" in reason


def test_max_position_breach():
    # Order itself fits notional cap (1 share at $200 = $200) but the
    # projected position is huge.
    gate = _gate(max_order_notional_usd=1_000_000.0)
    order = _order(qty=1)
    projected = {"AAPL": Position("AAPL", quantity=200, avg_entry_price=200.0)}  # 40k > 25k
    ok, reason = gate.check_order(order, projected, price_hints={"AAPL": 200.0})
    assert ok is False
    assert "max_position_usd" in reason


def test_max_gross_exposure_breach():
    # Single small order, multiple existing positions push gross over.
    gate = _gate(max_order_notional_usd=1_000_000.0, max_position_usd=1_000_000.0)
    order = _order(qty=1)
    projected = {
        "AAPL": Position("AAPL", 200, 200.0),   # 40k
        "MSFT": Position("MSFT", 100, 400.0),   # 40k
        "GOOGL": Position("GOOGL", 200, 150.0), # 30k → gross 110k > 100k
    }
    ok, reason = gate.check_order(
        order, projected, price_hints={"AAPL": 200.0, "MSFT": 400.0, "GOOGL": 150.0},
    )
    assert ok is False
    assert "max_gross_exposure_usd" in reason


def test_daily_loss_breach_blocks_all_orders():
    # Daily loss of -1500 exceeds the 1000 cap → block any order regardless
    # of size.
    gate = _gate(daily_pnl=-1500.0)
    order = _order(qty=1)
    projected = {"AAPL": Position("AAPL", 1, 200.0)}
    ok, reason = gate.check_order(order, projected, price_hints={"AAPL": 200.0})
    assert ok is False
    assert "max_daily_loss_usd" in reason


def test_daily_loss_just_under_cap_still_allows():
    gate = _gate(daily_pnl=-999.0)
    order = _order(qty=1)
    projected = {"AAPL": Position("AAPL", 1, 200.0)}
    ok, reason = gate.check_order(order, projected, price_hints={"AAPL": 200.0})
    assert ok is True
    assert reason is None


# ─── Combo / ordering ─────────────────────────────────────────────────────
def test_daily_loss_takes_priority_over_other_caps():
    """If daily loss is breached, we should NOT need price hints to refuse."""
    gate = _gate(daily_pnl=-5000.0)
    order = _order(qty=10)
    ok, reason = gate.check_order(order, {}, price_hints={})
    assert ok is False
    assert "daily_loss" in reason


def test_combo_passes_when_no_cap_breached():
    gate = _gate()
    order = _order(qty=10)
    projected = {
        "AAPL": Position("AAPL", 10, 200.0),    # 2000
        "MSFT": Position("MSFT", 50, 400.0),    # 20000 → gross 22000
    }
    ok, reason = gate.check_order(
        order, projected, price_hints={"AAPL": 200.0, "MSFT": 400.0},
    )
    assert ok is True
    assert reason is None


# ─── project_positions helper ─────────────────────────────────────────────
def test_project_positions_buy_into_empty_book():
    current: dict[str, Position] = {}
    order = _order(qty=5, side=OrderSide.BUY)
    proj = RiskGate.project_positions(current, order)
    assert proj["AAPL"].quantity == 5.0


def test_project_positions_sell_reduces_existing():
    current = {"AAPL": Position("AAPL", quantity=10, avg_entry_price=180.0)}
    order = _order(qty=4, side=OrderSide.SELL)
    proj = RiskGate.project_positions(current, order)
    assert proj["AAPL"].quantity == 6.0
    # Doesn't mutate the input.
    assert current["AAPL"].quantity == 10.0


def test_record_pnl_updates_snapshot():
    gate = _gate()
    assert gate.daily_pnl == 0.0
    gate.record_pnl(-250.0)
    assert gate.daily_pnl == -250.0
