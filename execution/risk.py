"""Risk gate consumed by the live runner before every order submission.

The gate is a pure function — no broker calls, no DB writes. The runner
fetches current positions + a price hint (last close) once per iteration and
passes them in along with the projected post-fill positions. The gate
returns ``(allowed, reason_if_blocked)`` so the runner can either submit or
log a ``risk_blocked`` decision row.

Caps implemented:
    * ``max_position_usd`` — absolute dollar exposure per ticker.
    * ``max_daily_loss_usd`` — once realised+unrealised intraday loss
      exceeds this, all new orders are refused.
    * ``max_gross_exposure_usd`` — Σ |position_usd| across portfolio.
    * ``max_order_notional_usd`` — optional sanity cap on a single order.

The runner is expected to call :meth:`RiskGate.record_pnl` (or pass a
``daily_pnl`` snapshot) so the daily-loss check has something real to test
against. The default constructor seeds ``daily_pnl=0``; the runner refreshes
it each iteration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .broker import Broker, Order, OrderSide, Position

log = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Per-account risk caps. All values in USD. Daily-loss is a negative
    threshold magnitude (we compare ``daily_pnl < -max_daily_loss_usd``).
    """

    max_position_usd: float
    max_daily_loss_usd: float
    max_gross_exposure_usd: float
    max_order_notional_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_position_usd", "max_daily_loss_usd", "max_gross_exposure_usd"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be > 0; got {v}")
        if self.max_order_notional_usd is not None and self.max_order_notional_usd <= 0:
            raise ValueError(
                f"max_order_notional_usd must be > 0 if set; got {self.max_order_notional_usd}"
            )


@dataclass
class RiskGate:
    """Pre-trade checker.

    Parameters
    ----------
    limits : RiskLimits
        The cap bundle.
    broker : Broker
        Held for live cash / multi-asset queries the gate might want later.
        Not currently called from :meth:`check_order` — kept on the instance
        per the BRIEF signature so future caps (e.g. cash floor) can use it
        without changing the call site.
    daily_pnl : float
        Snapshot of realised + unrealised P&L since the start of the
        trading day. The runner sets this each iteration. Default ``0``.
    """

    limits: RiskLimits
    broker: Broker
    daily_pnl: float = 0.0
    # Internal: track the price hint used in the last check (for debug logging).
    _last_price_hint: dict[str, float] = field(default_factory=dict)

    def record_pnl(self, pnl: float) -> None:
        """Update the daily-PnL snapshot. Idempotent."""
        self.daily_pnl = float(pnl)

    def check_order(
        self,
        order: Order,
        projected_positions: dict[str, Position],
        price_hints: dict[str, float] | None = None,
    ) -> tuple[bool, str | None]:
        """Returns ``(allowed, reason_if_blocked)``. Pure — no side effects
        beyond updating ``self._last_price_hint`` for debug.

        ``projected_positions`` must be the portfolio state AS IF the order
        were filled (the runner computes this by applying the diff to the
        current snapshot). ``price_hints[ticker]`` is the most recent close
        for that ticker — used to translate share quantities into USD.
        """
        # 1. Daily-loss kill: applies regardless of order details.
        if self.daily_pnl < -abs(self.limits.max_daily_loss_usd):
            return (
                False,
                f"max_daily_loss_usd breached: pnl={self.daily_pnl:.2f} "
                f"< -{self.limits.max_daily_loss_usd:.2f}",
            )

        price_hints = price_hints or {}
        self._last_price_hint = dict(price_hints)

        # 2. Per-order notional cap.
        order_price = price_hints.get(order.ticker) or order.limit_price or 0.0
        if order_price <= 0:
            return (False, f"no price hint for {order.ticker} — cannot enforce caps")
        order_notional = abs(float(order.quantity)) * float(order_price)
        if (
            self.limits.max_order_notional_usd is not None
            and order_notional > self.limits.max_order_notional_usd
        ):
            return (
                False,
                f"max_order_notional_usd breached: "
                f"{order_notional:.2f} > {self.limits.max_order_notional_usd:.2f}",
            )

        # 3. Per-ticker absolute exposure cap on the projected position.
        proj = projected_positions.get(order.ticker)
        if proj is not None:
            proj_price = price_hints.get(order.ticker, 0.0)
            proj_notional = abs(float(proj.quantity)) * float(proj_price)
            if proj_notional > self.limits.max_position_usd:
                return (
                    False,
                    f"max_position_usd breached for {order.ticker}: "
                    f"{proj_notional:.2f} > {self.limits.max_position_usd:.2f}",
                )

        # 4. Gross exposure cap across portfolio.
        gross = 0.0
        for tkr, pos in projected_positions.items():
            px = price_hints.get(tkr, 0.0)
            gross += abs(float(pos.quantity)) * float(px)
        if gross > self.limits.max_gross_exposure_usd:
            return (
                False,
                f"max_gross_exposure_usd breached: "
                f"{gross:.2f} > {self.limits.max_gross_exposure_usd:.2f}",
            )

        return (True, None)

    @staticmethod
    def project_positions(
        current: dict[str, Position],
        order: Order,
    ) -> dict[str, Position]:
        """Apply ``order`` to ``current`` and return the post-fill snapshot.

        Helper for the runner: keeps the projection logic in one place so
        ``check_order`` can be tested without re-coding it.
        """
        out = {k: Position(
            ticker=v.ticker,
            quantity=v.quantity,
            avg_entry_price=v.avg_entry_price,
            market_value=v.market_value,
            unrealised_pl=v.unrealised_pl,
            extra=dict(v.extra),
        ) for k, v in current.items()}
        signed = float(order.quantity) if order.side == OrderSide.BUY else -float(order.quantity)
        existing = out.get(order.ticker)
        if existing is None:
            out[order.ticker] = Position(
                ticker=order.ticker,
                quantity=signed,
                avg_entry_price=0.0,
            )
        else:
            existing.quantity = float(existing.quantity) + signed
        return out
