"""Tests for ``execution.live_runner``.

Mocks the broker (the ABC is small) and injects a synthetic price frame so
no network / no ClickHouse / no real strategy registry is required for the
behavioural tests. DB writes use a real tmp ClickHouse schema per the
no-mock-DB rule; skipped with a clear reason if ClickHouse is unreachable.
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from execution.broker import Order, OrderSide, OrderStatus, Position
from execution.live_runner import (
    DEFAULT_CONFIG,
    LiveConfig,
    _enforce_live_confirmation,
    _signals_to_targets,
    _target_position_shares,
    load_config,
    run_once,
)
from execution.risk import RiskLimits


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─── Fixtures ─────────────────────────────────────────────────────────────
def _wide_frame(tickers=("AAPL", "MSFT"), n_bars: int = 60, seed: int = 0):
    """Synthetic OHLCV wide frame the strategy can run on."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="B")
    fields = ["open", "high", "low", "close", "volume"]
    cols = pd.MultiIndex.from_product([fields, list(tickers)], names=["field", "ticker"])
    base = 100 + np.cumsum(rng.normal(0, 1, size=(n_bars, len(tickers))), axis=0)
    data = np.zeros((n_bars, len(fields) * len(tickers)))
    for fi, _ in enumerate(fields):
        for ti, _ in enumerate(tickers):
            col_idx = fi * len(tickers) + ti
            if fields[fi] == "volume":
                data[:, col_idx] = 1_000_000.0
            else:
                data[:, col_idx] = base[:, ti] + rng.normal(0, 0.1, n_bars)
    return pd.DataFrame(data, index=dates, columns=cols)


class _StubStrategy:
    """Minimal Strategy stand-in — emits one long entry on every ticker at the
    last bar, no exits. Lets us exercise the diff/order plumbing."""

    name = "stub"

    def generate_signals(self, data):
        from strategies.base import Signals
        close = data["close"]
        entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        exits = pd.DataFrame(False, index=close.index, columns=close.columns)
        # Fire an entry on the very last bar for every ticker.
        entries.iloc[-1, :] = True
        return Signals(entries=entries, exits=exits)


def _stub_data_loader(tickers, lookback_days, interval):
    return _wide_frame(tuple(tickers))


def _make_cfg(tmp_path, **risk_over):
    kill_path = tmp_path / "kill"
    # Defaults chosen so the stub strategy's ~25k notional × 2 tickers fits.
    base_risk = dict(
        max_position_usd=25_000.0,
        max_daily_loss_usd=1_000.0,
        max_gross_exposure_usd=100_000.0,
        max_order_notional_usd=30_000.0,
    )
    base_risk.update(risk_over)
    return LiveConfig(
        broker="alpaca",
        paper=True,
        strategy="stub",
        tickers=["AAPL", "MSFT"],
        rebalance_seconds=1,
        risk=RiskLimits(**base_risk),
        broker_credentials={},
        kill_switch_file=str(kill_path),
    )


def _flat_broker():
    """Mock broker reporting flat positions and accepting any submit."""
    b = MagicMock()
    b.positions.return_value = {}
    b.submit.side_effect = lambda o: f"id-{uuid.uuid4().hex[:8]}"
    b.order_status.return_value = OrderStatus.SUBMITTED
    return b


# ─── target sizing helper ─────────────────────────────────────────────────
def test_target_position_zero_when_flat():
    assert _target_position_shares(signal_long=False, last_close=100.0, max_position_usd=10000) == 0


def test_target_position_sized_by_dollar_cap():
    # 25000 / 250 = 100 shares
    assert _target_position_shares(signal_long=True, last_close=250.0, max_position_usd=25000) == 100


def test_target_position_floors_to_whole_shares():
    # 25000 / 333 = 75.07 → 75
    assert _target_position_shares(signal_long=True, last_close=333.0, max_position_usd=25000) == 75


# ─── _signals_to_targets ──────────────────────────────────────────────────
def test_signals_to_targets_long_state():
    data = _wide_frame()
    targets, last_close, bar_ts = _signals_to_targets(
        _StubStrategy(), data, max_position_usd=10_000.0
    )
    assert set(targets.keys()) == {"AAPL", "MSFT"}
    # All targets > 0 because the stub fires entries on the last bar.
    assert all(v > 0 for v in targets.values())
    assert bar_ts == data.index[-1]


# ─── run_once: happy path → one order placed ──────────────────────────────
def test_run_once_emits_decision_and_order_when_diff_nonzero(tmp_path):
    cfg = _make_cfg(tmp_path)
    broker = _flat_broker()
    res = run_once(
        cfg, broker,
        ch_client=None,
        data_loader=_stub_data_loader,
        strategy_factory=lambda: _StubStrategy(),
    )
    # Two tickers, both diff > 0 → two decisions, two orders.
    assert len(res.decisions) == 2
    assert len(res.orders) == 2
    assert all(d["risk_blocked"] == 0 for d in res.decisions)
    assert broker.submit.call_count == 2


def test_run_once_no_op_when_already_at_target(tmp_path):
    """If positions match the target, decisions are logged but no submit."""
    cfg = _make_cfg(tmp_path)
    # The stub strategy with max_position_usd=25000 at price~100 → ~250 shares.
    # Make the broker report exactly that.
    broker = MagicMock()

    def _signals_then_positions(*_a, **_kw):
        return {}
    broker.positions.return_value = {}

    # Quick computation of expected target so we can preload positions.
    data = _wide_frame()
    targets, _last, _ts = _signals_to_targets(
        _StubStrategy(), data, max_position_usd=cfg.risk.max_position_usd,
    )
    broker.positions.return_value = {
        t: Position(t, quantity=q, avg_entry_price=100.0)
        for t, q in targets.items()
    }
    broker.submit = MagicMock()

    res = run_once(
        cfg, broker,
        ch_client=None,
        data_loader=_stub_data_loader,
        strategy_factory=lambda: _StubStrategy(),
    )
    assert broker.submit.call_count == 0
    assert len(res.decisions) == 2
    assert all(d["diff_qty"] == 0.0 for d in res.decisions)


# ─── run_once: risk gate blocks oversized order ───────────────────────────
def test_run_once_blocks_oversized_order(tmp_path):
    """Set a tiny order-notional cap so every diff is blocked."""
    cfg = _make_cfg(tmp_path, max_order_notional_usd=1.0)
    broker = _flat_broker()
    res = run_once(
        cfg, broker,
        ch_client=None,
        data_loader=_stub_data_loader,
        strategy_factory=lambda: _StubStrategy(),
    )
    assert broker.submit.call_count == 0
    assert len(res.decisions) == 2
    assert all(d["risk_blocked"] == 1 for d in res.decisions)
    assert all("max_order_notional" in d["risk_reason"] for d in res.decisions)


# ─── run_once: kill switch halts cleanly ──────────────────────────────────
def test_run_once_kill_switch_halts_orders(tmp_path):
    cfg = _make_cfg(tmp_path)
    # Trip the file flag.
    Path(cfg.kill_switch_file).write_text("")
    broker = _flat_broker()
    res = run_once(
        cfg, broker,
        ch_client=None,
        data_loader=_stub_data_loader,
        strategy_factory=lambda: _StubStrategy(),
    )
    assert res.killed is True
    assert broker.submit.call_count == 0
    # One decision per ticker, each marked risk_blocked=1.
    assert len(res.decisions) == 2
    assert all(d["risk_blocked"] == 1 for d in res.decisions)
    assert all(d["risk_reason"].startswith("killed:") for d in res.decisions)


def test_run_once_kill_switch_env_var(tmp_path, monkeypatch):
    """Env-var kill source should also halt."""
    cfg = _make_cfg(tmp_path)
    monkeypatch.setenv("ALGO_KILL", "1")
    broker = _flat_broker()
    res = run_once(
        cfg, broker,
        ch_client=None,
        data_loader=_stub_data_loader,
        strategy_factory=lambda: _StubStrategy(),
    )
    assert res.killed is True
    assert broker.submit.call_count == 0


# ─── Live-mode confirmation gate ──────────────────────────────────────────
def test_live_mode_exits_2_without_confirmation(monkeypatch, tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.paper = False
    monkeypatch.delenv("ALGO_LIVE_CONFIRMED", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        _enforce_live_confirmation(cfg)
    assert exc_info.value.code == 2


def test_live_mode_allowed_with_confirmation(monkeypatch, tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.paper = False
    monkeypatch.setenv("ALGO_LIVE_CONFIRMED", "yes")
    # Should not raise / exit.
    _enforce_live_confirmation(cfg)


def test_paper_mode_skips_confirmation(monkeypatch, tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.paper = True
    monkeypatch.delenv("ALGO_LIVE_CONFIRMED", raising=False)
    _enforce_live_confirmation(cfg)


# ─── Config loader ────────────────────────────────────────────────────────
def test_load_config_parses_committed_sample():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.strategy == "combined_explainable"
    assert cfg.paper is True
    assert cfg.broker in {"ibkr", "alpaca"}
    assert cfg.risk.max_position_usd > 0
    assert cfg.tickers


# ─── CLI smoke ────────────────────────────────────────────────────────────
def test_cli_help_runs_cleanly():
    """`python -m execution.live_runner --help` must exit 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "execution.live_runner", "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "live/paper trading" in proc.stdout.lower()


def test_cli_live_without_confirmation_exits_2(tmp_path, monkeypatch):
    """End-to-end: live mode without ALGO_LIVE_CONFIRMED exits 2."""
    # Build a temp config with paper: false.
    tmp_cfg = tmp_path / "live.yaml"
    tmp_cfg.write_text(
        """
broker: alpaca
paper: false
strategy: combined_explainable
tickers: [AAPL]
rebalance_seconds: 1
risk:
  max_position_usd: 1000
  max_daily_loss_usd: 100
  max_gross_exposure_usd: 1000
  max_order_notional_usd: 100
broker_credentials:
  alpaca:
    api_key_env: ALPACA_API_KEY
    secret_key_env: ALPACA_SECRET_KEY
kill_switch_file: /tmp/no-such-flag
"""
    )
    env = os.environ.copy()
    env.pop("ALGO_LIVE_CONFIRMED", None)
    proc = subprocess.run(
        [
            sys.executable, "-m", "execution.live_runner",
            "--config", str(tmp_cfg), "--once",
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=30,
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "REFUSING TO RUN LIVE" in proc.stderr


# ─── ClickHouse integration (tmp DB) ──────────────────────────────────────
def _ch_setup():
    """Create a tmp DB with the audit schema; return (client, db_name) or
    (None, None) if ClickHouse is unreachable."""
    try:
        import clickhouse_connect
        from config import load_settings
        from execution.migrate import apply_schema
        cfg = load_settings().clickhouse
        admin = clickhouse_connect.get_client(
            host=cfg.host, port=cfg.port,
            username=cfg.user, password=cfg.password,
        )
        admin.command("SELECT 1")
        db = f"algo_test_{uuid.uuid4().hex[:8]}"
        admin.command(f"CREATE DATABASE {db}")
        scoped = clickhouse_connect.get_client(
            host=cfg.host, port=cfg.port,
            username=cfg.user, password=cfg.password,
            database=db,
        )
        apply_schema(client=scoped)
        return scoped, db, admin
    except Exception:  # noqa: BLE001
        return None, None, None


CH_CLIENT, CH_DB, CH_ADMIN = _ch_setup()
ch_required = pytest.mark.skipif(
    CH_CLIENT is None,
    reason="ClickHouse unreachable — skipping audit-DB tests.",
)


@ch_required
def test_run_once_writes_audit_rows(tmp_path):
    cfg = _make_cfg(tmp_path)
    broker = _flat_broker()
    res = run_once(
        cfg, broker,
        ch_client=CH_CLIENT,
        data_loader=_stub_data_loader,
        strategy_factory=lambda: _StubStrategy(),
    )
    assert len(res.orders) == 2

    rows = CH_CLIENT.query(f"SELECT count() FROM {CH_DB}.orders").result_rows
    assert rows[0][0] == 2
    rows = CH_CLIENT.query(f"SELECT count() FROM {CH_DB}.decisions").result_rows
    assert rows[0][0] == 2


@ch_required
def test_run_once_blocked_order_logs_decision_only(tmp_path):
    cfg = _make_cfg(tmp_path, max_order_notional_usd=1.0)
    broker = _flat_broker()
    run_once(
        cfg, broker,
        ch_client=CH_CLIENT,
        data_loader=_stub_data_loader,
        strategy_factory=lambda: _StubStrategy(),
    )
    rows = CH_CLIENT.query(
        f"SELECT count() FROM {CH_DB}.decisions WHERE risk_blocked = 1"
    ).result_rows
    # 2 new blocked decisions (plus the 2 from the previous test on same DB).
    assert rows[0][0] >= 2


def teardown_module(module):  # noqa: D401 — pytest hook
    """Drop the tmp DB at the end of the module run."""
    if CH_ADMIN is not None and CH_DB is not None:
        try:
            CH_ADMIN.command(f"DROP DATABASE IF EXISTS {CH_DB}")
        except Exception:  # noqa: BLE001
            pass
