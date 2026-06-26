"""Live runner — long-running process that turns strategy signals into orders.

Workflow per iteration:

1. Consult :func:`execution.kill_switch.is_killed`. If tripped, log a
   ``decisions`` row with ``risk_blocked=1, risk_reason='killed: …'`` for
   each ticker in the universe and sleep / exit.
2. Pull the latest bars for the configured tickers from ClickHouse (the
   project's existing data layer — never reimplemented here).
3. Run the strategy's :meth:`generate_signals` to get entries/exits, derive a
   "target position" per ticker from the most recent bar's signal
   (target = 1 share if signal long, 0 if flat — sized by the per-ticker
   ``max_position_usd`` cap divided by last close).
4. Fetch the broker's current positions, compute the per-ticker diff in
   shares.
5. For each diff, build a MARKET :class:`Order`, run it through
   :class:`RiskGate`; on block log a ``decisions`` row and skip the order,
   on allow submit + log both ``orders`` + ``decisions`` rows.
6. Sleep ``rebalance_seconds`` until the next iteration (skipped under
   ``--once``).

The ``--once`` flag runs exactly one iteration end-to-end and exits — this
is the BRIEF's acceptance test (one paper order placed, visible in the
client portal).

Live-trading guardrail: without ``--paper``, the runner refuses to start
unless the env var ``ALGO_LIVE_CONFIRMED=yes`` is set; otherwise it prints a
loud warning and exits with code 2. This honours the BRIEF principle "Never
run live orders without my explicit 'go live' confirmation."
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .broker import Broker, BrokerError, Order, OrderSide, OrderType, Position
from .kill_switch import is_killed
from .risk import RiskGate, RiskLimits

log = logging.getLogger(__name__)


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "live.yaml"


# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class LiveConfig:
    broker: str
    paper: bool
    strategy: str
    tickers: list[str]
    rebalance_seconds: int
    risk: RiskLimits
    broker_credentials: dict[str, dict[str, Any]] = field(default_factory=dict)
    kill_switch_file: str | None = None
    lookback_days: int = 365      # how much history the strategy gets per iteration
    interval: str = "1d"


def load_config(path: Path | str = DEFAULT_CONFIG) -> LiveConfig:
    """Load + validate ``config/live.yaml``. Credentials remain as env-var
    NAMES — we resolve them at broker-construction time so they never sit in
    config objects in plaintext.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    risk_raw = raw.get("risk") or {}
    risk = RiskLimits(
        max_position_usd=float(risk_raw["max_position_usd"]),
        max_daily_loss_usd=float(risk_raw["max_daily_loss_usd"]),
        max_gross_exposure_usd=float(risk_raw["max_gross_exposure_usd"]),
        max_order_notional_usd=(
            float(risk_raw["max_order_notional_usd"])
            if risk_raw.get("max_order_notional_usd") is not None
            else None
        ),
    )
    return LiveConfig(
        broker=str(raw.get("broker", "alpaca")),
        paper=bool(raw.get("paper", True)),
        strategy=str(raw.get("strategy", "combined_explainable")),
        tickers=list(raw.get("tickers", [])),
        rebalance_seconds=int(raw.get("rebalance_seconds", 300)),
        risk=risk,
        broker_credentials=dict(raw.get("broker_credentials", {})),
        kill_switch_file=raw.get("kill_switch_file"),
        lookback_days=int(raw.get("lookback_days", 365)),
        interval=str(raw.get("interval", "1d")),
    )


# ─── Broker construction (env-var creds) ─────────────────────────────────────
def build_broker(cfg: LiveConfig) -> Broker:
    creds = cfg.broker_credentials.get(cfg.broker, {})
    if cfg.broker == "ibkr":
        from .ibkr import IBKRBroker
        return IBKRBroker(
            host=creds.get("host", "127.0.0.1"),
            port=creds.get("port"),
            client_id=int(creds.get("client_id", 1)),
            paper=cfg.paper,
            gateway=bool(creds.get("gateway", False)),
        )
    if cfg.broker == "alpaca":
        from .alpaca import AlpacaBroker
        api_key_env = creds.get("api_key_env", "ALPACA_API_KEY")
        secret_key_env = creds.get("secret_key_env", "ALPACA_SECRET_KEY")
        return AlpacaBroker(
            api_key=os.environ.get(api_key_env),
            secret_key=os.environ.get(secret_key_env),
            paper=cfg.paper,
        )
    raise ValueError(f"Unknown broker: {cfg.broker!r}")


# ─── Strategy I/O ────────────────────────────────────────────────────────────
def _strategy_for(name: str):
    """Look up + instantiate a strategy from the registry."""
    from strategies import REGISTRY
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Known: {sorted(REGISTRY.keys())}")
    return REGISTRY[name]()


def load_bars_wide(tickers: list[str], lookback_days: int, interval: str) -> pd.DataFrame:
    """Pull recent bars and convert to the wide MultiIndex frame strategies expect."""
    from backtest.engine import polars_to_wide
    from data import load_bars
    end = datetime.now(timezone.utc).date()
    start = end.replace(day=1)
    # Subtract whole days via timedelta to handle month rollovers.
    from datetime import timedelta
    start = end - timedelta(days=lookback_days)
    df = load_bars(tickers, start=start.isoformat(), end=end.isoformat(), interval=interval)
    if df.is_empty():
        raise RuntimeError(f"No bars in ClickHouse for {tickers} {start}..{end}")
    return polars_to_wide(df)


# ─── Decision / order plumbing ───────────────────────────────────────────────
@dataclass
class IterationResult:
    """Bookkeeping returned by :func:`run_once` — keeps tests assertion-friendly."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    killed: bool = False
    kill_reason: str | None = None


def _target_position_shares(
    *,
    signal_long: bool,
    last_close: float,
    max_position_usd: float,
) -> float:
    """Translate the strategy's boolean long-state into a target share count.

    Long → fill the per-ticker dollar cap (rounded DOWN to whole shares to
    avoid fractional-share oddities on IBKR; Alpaca supports fractions but we
    keep it simple here — see IMPROVEMENTS for the fractional-share entry).
    Flat → 0 shares.
    """
    if not signal_long or last_close <= 0:
        return 0.0
    return float(int(max_position_usd / last_close))


def _signals_to_targets(
    strategy,
    data: pd.DataFrame,
    max_position_usd: float,
) -> tuple[dict[str, float], dict[str, float], pd.Timestamp]:
    """Run the strategy on ``data``, derive target shares per ticker.

    Returns ``(targets, last_close_by_ticker, bar_timestamp)``.
    """
    signals = strategy.generate_signals(data)
    close = data["close"]
    last_bar = close.index[-1]
    # State derivation matches CombinedExplainableStrategy._child_state.
    # entry → +1, exit → -1, ffill, > 0 means long.
    action = pd.DataFrame(
        float("nan"), index=signals.entries.index, columns=signals.entries.columns
    )
    action = action.mask(signals.entries.astype(bool), 1.0)
    action = action.mask(signals.exits.astype(bool), -1.0)
    action = action.ffill().fillna(-1.0)
    long_state = (action.iloc[-1] > 0).to_dict()

    targets: dict[str, float] = {}
    last_close: dict[str, float] = {}
    for ticker in close.columns:
        px = float(close.iloc[-1][ticker])
        last_close[ticker] = px
        targets[ticker] = _target_position_shares(
            signal_long=bool(long_state.get(ticker, False)),
            last_close=px,
            max_position_usd=max_position_usd,
        )
    return targets, last_close, last_bar


def _explanation_for(strategy, ticker: str, bar_ts: pd.Timestamp) -> str:
    """Pull the per-(ticker, ts) explanation dict from the combined strategy
    and serialise to JSON. Other strategies return ``""``."""
    getter = getattr(strategy, "get_explanation_log", None)
    if getter is None:
        return ""
    log_map = getter() or {}
    payload = log_map.get((ticker, bar_ts))
    if payload is None:
        return ""
    # Make timestamps and numpy values JSON-safe.
    safe = json.loads(json.dumps(payload, default=str))
    return json.dumps(safe)


# ─── ClickHouse writers ──────────────────────────────────────────────────────
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def write_decision(ch_client, row: dict[str, Any]) -> None:
    """Insert a single ``decisions`` row."""
    ch_client.insert(
        "decisions",
        [[
            row["decision_id"],
            row["decided_at"],
            row["strategy"],
            row["ticker"],
            row["bar_timestamp"],
            float(row["target_position"]),
            float(row["current_position"]),
            float(row["diff_qty"]),
            row.get("explanation", ""),
            int(row.get("risk_blocked", 0)),
            row.get("risk_reason"),
        ]],
        column_names=[
            "decision_id", "decided_at", "strategy", "ticker", "bar_timestamp",
            "target_position", "current_position", "diff_qty", "explanation",
            "risk_blocked", "risk_reason",
        ],
    )


def write_order(ch_client, row: dict[str, Any]) -> None:
    ch_client.insert(
        "orders",
        [[
            row["order_id"],
            row["submitted_at"],
            row["strategy"],
            row["ticker"],
            row["side"],
            float(row["qty"]),
            row["order_type"],
            row.get("limit_price"),
            row.get("stop_price"),
            row["broker"],
            row.get("broker_order_id"),
            row.get("status", "submitted"),
            float(row.get("filled_qty", 0.0)),
            row.get("avg_fill_price"),
            row["last_updated"],
        ]],
        column_names=[
            "order_id", "submitted_at", "strategy", "ticker", "side", "qty",
            "order_type", "limit_price", "stop_price", "broker", "broker_order_id",
            "status", "filled_qty", "avg_fill_price", "last_updated",
        ],
    )


# ─── Core iteration ──────────────────────────────────────────────────────────
def run_once(
    cfg: LiveConfig,
    broker: Broker,
    *,
    ch_client=None,
    data_loader=None,
    strategy_factory=None,
) -> IterationResult:
    """Single decision → submit → log cycle. Returns counts for tests.

    Parameters
    ----------
    cfg : LiveConfig
        Loaded config.
    broker : Broker
        Already-connected broker adapter.
    ch_client : optional ClickHouse client
        Defaults to ``data.clickhouse_client.get_client()``. Pass ``None``
        with ``write_to_ch=False`` to skip DB writes (tests).
    data_loader : callable(tickers, lookback_days, interval) -> wide pd.DataFrame
        Defaults to :func:`load_bars_wide`. Tests inject a synthetic frame.
    strategy_factory : callable() -> Strategy
        Defaults to looking up ``cfg.strategy`` in the registry.
    """
    result = IterationResult()

    # 1. Kill switch.
    killed, reason = is_killed(kill_flag_path=cfg.kill_switch_file)
    if killed:
        log.warning("kill switch tripped: %s", reason)
        result.killed = True
        result.kill_reason = reason
        # Log one decision row per ticker so the halt is auditable.
        for ticker in cfg.tickers:
            row = {
                "decision_id": str(uuid.uuid4()),
                "decided_at": _now_utc(),
                "strategy": cfg.strategy,
                "ticker": ticker,
                "bar_timestamp": _now_utc(),
                "target_position": 0.0,
                "current_position": 0.0,
                "diff_qty": 0.0,
                "explanation": "",
                "risk_blocked": 1,
                "risk_reason": f"killed: {reason}",
            }
            result.decisions.append(row)
            if ch_client is not None:
                write_decision(ch_client, row)
        return result

    # 2. Bars.
    loader = data_loader or load_bars_wide
    data = loader(cfg.tickers, cfg.lookback_days, cfg.interval)

    # 3. Strategy → target positions.
    factory = strategy_factory or (lambda: _strategy_for(cfg.strategy))
    strategy = factory()
    targets, last_close, bar_ts = _signals_to_targets(
        strategy, data, max_position_usd=cfg.risk.max_position_usd
    )

    # 4. Current positions.
    try:
        current_positions = broker.positions()
    except BrokerError as exc:
        log.error("broker.positions() failed: %s", exc)
        raise

    # 5. Diff + risk-check + submit.
    gate = RiskGate(limits=cfg.risk, broker=broker)
    for ticker in cfg.tickers:
        current_qty = float(
            current_positions[ticker].quantity if ticker in current_positions else 0.0
        )
        target_qty = float(targets.get(ticker, 0.0))
        diff = target_qty - current_qty
        decision_id = str(uuid.uuid4())
        explanation = _explanation_for(strategy, ticker, bar_ts)

        if abs(diff) < 1e-9:
            # No-op decision row so the log carries the "we considered this
            # bar and chose not to trade" evidence.
            row = {
                "decision_id": decision_id,
                "decided_at": _now_utc(),
                "strategy": cfg.strategy,
                "ticker": ticker,
                "bar_timestamp": bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts,
                "target_position": target_qty,
                "current_position": current_qty,
                "diff_qty": 0.0,
                "explanation": explanation,
                "risk_blocked": 0,
                "risk_reason": None,
            }
            result.decisions.append(row)
            if ch_client is not None:
                write_decision(ch_client, row)
            continue

        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        qty = abs(diff)
        order = Order(ticker=ticker, side=side, quantity=qty, order_type=OrderType.MARKET)

        # Projected positions (current + this order).
        projected = RiskGate.project_positions(current_positions, order)
        allowed, reason = gate.check_order(
            order, projected, price_hints=last_close
        )

        decision_row = {
            "decision_id": decision_id,
            "decided_at": _now_utc(),
            "strategy": cfg.strategy,
            "ticker": ticker,
            "bar_timestamp": bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts,
            "target_position": target_qty,
            "current_position": current_qty,
            "diff_qty": diff,
            "explanation": explanation,
            "risk_blocked": 0 if allowed else 1,
            "risk_reason": None if allowed else reason,
        }

        if not allowed:
            log.warning("risk blocked %s %s %s: %s", side.value, qty, ticker, reason)
            result.decisions.append(decision_row)
            if ch_client is not None:
                write_decision(ch_client, decision_row)
            continue

        # Submit.
        try:
            broker_order_id = broker.submit(order)
            log.info("submitted %s %s %s id=%s", side.value, qty, ticker, broker_order_id)
        except BrokerError as exc:
            log.error("broker submit failed for %s: %s", ticker, exc)
            decision_row["risk_blocked"] = 1
            decision_row["risk_reason"] = f"broker_error: {exc}"
            result.decisions.append(decision_row)
            if ch_client is not None:
                write_decision(ch_client, decision_row)
            continue

        now = _now_utc()
        order_row = {
            "order_id": decision_id,           # one-to-one with the decision
            "submitted_at": now,
            "strategy": cfg.strategy,
            "ticker": ticker,
            "side": side.value,
            "qty": qty,
            "order_type": OrderType.MARKET.value,
            "limit_price": None,
            "stop_price": None,
            "broker": cfg.broker,
            "broker_order_id": broker_order_id,
            "status": "submitted",
            "filled_qty": 0.0,
            "avg_fill_price": None,
            "last_updated": now,
        }
        result.decisions.append(decision_row)
        result.orders.append(order_row)
        if ch_client is not None:
            write_decision(ch_client, decision_row)
            write_order(ch_client, order_row)

    return result


# ─── Long-running loop ───────────────────────────────────────────────────────
class _ShutdownFlag:
    """Tiny mutable bool flipped by signal handlers."""

    def __init__(self) -> None:
        self.stop = False

    def request_stop(self, *_a) -> None:  # pragma: no cover — signal path
        log.warning("shutdown signal received — finishing in-flight iteration and exiting")
        self.stop = True


def run_loop(
    cfg: LiveConfig,
    broker: Broker,
    *,
    ch_client=None,
    data_loader=None,
    strategy_factory=None,
) -> None:
    """Block in a poll loop until SIGINT/SIGTERM."""
    flag = _ShutdownFlag()
    signal.signal(signal.SIGINT, flag.request_stop)
    signal.signal(signal.SIGTERM, flag.request_stop)
    log.info("live runner loop started — poll every %s s", cfg.rebalance_seconds)
    while not flag.stop:
        try:
            run_once(
                cfg, broker,
                ch_client=ch_client,
                data_loader=data_loader,
                strategy_factory=strategy_factory,
            )
        except Exception as exc:  # noqa: BLE001 — keep loop alive on transient errors
            log.exception("iteration failed: %s", exc)
        # Sleep in small chunks so SIGTERM is responsive.
        slept = 0
        while not flag.stop and slept < cfg.rebalance_seconds:
            time.sleep(1)
            slept += 1
    log.info("live runner loop exited cleanly")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m execution.live_runner",
        description="Long-running live/paper trading runner.",
    )
    p.add_argument("--strategy", default=None, help="Override strategy name from config.")
    p.add_argument(
        "--broker", choices=["ibkr", "alpaca"], default=None,
        help="Override broker from config.",
    )
    p.add_argument("--paper", action="store_true", help="Force paper mode.")
    p.add_argument("--live", action="store_true", help="Force live mode (requires ALGO_LIVE_CONFIRMED=yes).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to live.yaml.")
    p.add_argument("--once", action="store_true", help="Run one iteration then exit.")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging.")
    return p


def _enforce_live_confirmation(cfg: LiveConfig) -> None:
    """If running live, require an explicit env-var confirmation. Exits 2 otherwise."""
    if cfg.paper:
        return
    if os.environ.get("ALGO_LIVE_CONFIRMED", "").lower() != "yes":
        sys.stderr.write(
            "\n========================================================================\n"
            "  REFUSING TO RUN LIVE: ALGO_LIVE_CONFIRMED=yes not set.\n"
            "\n"
            "  The runner is configured for LIVE trading (paper=false).\n"
            "  To proceed, you must explicitly confirm by setting:\n"
            "      export ALGO_LIVE_CONFIRMED=yes\n"
            "\n"
            "  Otherwise, set paper=true in your config (or pass --paper) and re-run.\n"
            "========================================================================\n"
        )
        sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load_config(args.config)
    if args.strategy:
        cfg.strategy = args.strategy
    if args.broker:
        cfg.broker = args.broker
    if args.paper and args.live:
        sys.stderr.write("cannot pass both --paper and --live\n")
        return 2
    if args.paper:
        cfg.paper = True
    if args.live:
        cfg.paper = False

    _enforce_live_confirmation(cfg)

    broker = build_broker(cfg)
    broker.connect()

    # ClickHouse client — fall back to None on connection failure so the
    # runner can still log decisions to stdout. We log a clear warning so the
    # user knows the audit table isn't being written.
    ch_client = None
    try:
        from data.clickhouse_client import get_client
        ch_client = get_client()
    except Exception as exc:  # noqa: BLE001
        log.warning("ClickHouse unreachable — running without audit DB writes: %s", exc)

    try:
        if args.once:
            log.info("running ONE iteration (--once)")
            res = run_once(cfg, broker, ch_client=ch_client)
            log.info(
                "iteration complete: decisions=%d orders=%d killed=%s",
                len(res.decisions), len(res.orders), res.killed,
            )
        else:
            run_loop(cfg, broker, ch_client=ch_client)
    finally:
        try:
            broker.disconnect()
        except BrokerError as exc:
            log.warning("broker disconnect raised: %s", exc)
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI
    raise SystemExit(main())
