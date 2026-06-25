"""Manual verification CLI for broker adapters (Task 5a).

Examples
--------

    # Read-only — never touches orders, no confirmation needed.
    python scripts/broker_smoke.py --broker alpaca --action cash
    python scripts/broker_smoke.py --broker alpaca --action positions
    python scripts/broker_smoke.py --broker ibkr  --action cash

    # Submit a single MARKET buy. --confirm is required; without
    # --allow-large we cap qty=1 on tickers > $5 (defensive default).
    python scripts/broker_smoke.py --broker alpaca --action submit \\
        --ticker AAPL --qty 1 --confirm

Credentials come from the environment (ALPACA_API_KEY, ALPACA_SECRET_KEY).
Paper is the default for both brokers; pass --live to flip them.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution import BrokerError, Order, OrderSide, OrderType  # noqa: E402
from execution.alpaca import AlpacaBroker  # noqa: E402
from execution.ibkr import IBKRBroker  # noqa: E402

_LOGGER = logging.getLogger("broker_smoke")


def _build_broker(args: argparse.Namespace):
    paper = not args.live
    if args.broker == "alpaca":
        return AlpacaBroker(paper=paper)
    if args.broker == "ibkr":
        return IBKRBroker(
            host=args.ibkr_host,
            port=args.ibkr_port,
            client_id=args.ibkr_client_id,
            paper=paper,
            gateway=args.ibkr_gateway,
        )
    raise ValueError(f"unknown broker: {args.broker}")


def _do_cash(broker) -> int:
    print(f"cash = {broker.cash():.2f} USD")
    return 0


def _do_positions(broker) -> int:
    positions = broker.positions()
    if not positions:
        print("(no open positions)")
        return 0
    for ticker, pos in positions.items():
        print(f"{ticker:8s} qty={pos.quantity:>12.4f} avg={pos.avg_entry_price:>10.2f}"
              f" mv={pos.market_value!r}")
    return 0


def _do_open_orders(broker) -> int:
    ids = broker.get_open_orders()
    if not ids:
        print("(no open orders)")
        return 0
    for oid in ids:
        print(oid)
    return 0


def _do_submit(broker, args: argparse.Namespace) -> int:
    if not args.confirm:
        print("submit requires --confirm (defensive default — paper only "
              "without explicit confirmation).", file=sys.stderr)
        return 2
    if not args.ticker or args.qty is None:
        print("submit requires --ticker and --qty", file=sys.stderr)
        return 2

    # Defensive cap unless --allow-large is set.
    qty = float(args.qty)
    if not args.allow_large and qty > 1:
        print(f"--allow-large not set; capping qty={qty} -> 1 share",
              file=sys.stderr)
        qty = 1.0

    order = Order(
        ticker=args.ticker,
        side=OrderSide.BUY if args.side == "buy" else OrderSide.SELL,
        quantity=qty,
        order_type=OrderType.MARKET,
    )
    order_id = broker.submit(order)
    print(f"submitted id={order_id}")
    print(f"status = {broker.order_status(order_id).value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="broker_smoke",
        description="Manual smoke against a paper / live broker account.",
    )
    parser.add_argument("--broker", required=True, choices=["alpaca", "ibkr"])
    parser.add_argument("--action", required=True,
                        choices=["cash", "positions", "open_orders", "submit"])
    parser.add_argument("--live", action="store_true",
                        help="Hit the live endpoint instead of paper. "
                             "Disabled by default.")
    # Submit-only.
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--qty", type=float, default=None)
    parser.add_argument("--side", default="buy", choices=["buy", "sell"])
    parser.add_argument("--confirm", action="store_true",
                        help="Required for --action submit.")
    parser.add_argument("--allow-large", action="store_true",
                        help="Lift the defensive 1-share cap on --action submit.")
    # IBKR-only.
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=None,
                        help="Defaults to TWS paper 7497 / live 7496 "
                             "(or Gateway 4002/4001 with --ibkr-gateway).")
    parser.add_argument("--ibkr-client-id", type=int, default=1)
    parser.add_argument("--ibkr-gateway", action="store_true",
                        help="Target IB Gateway ports instead of TWS.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        broker = _build_broker(args)
        broker.connect()
        try:
            if args.action == "cash":
                return _do_cash(broker)
            if args.action == "positions":
                return _do_positions(broker)
            if args.action == "open_orders":
                return _do_open_orders(broker)
            if args.action == "submit":
                return _do_submit(broker, args)
            raise AssertionError("unreachable")
        finally:
            broker.disconnect()
    except BrokerError as exc:
        print(f"broker error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
