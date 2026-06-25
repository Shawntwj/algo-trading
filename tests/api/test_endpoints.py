"""FastAPI endpoint tests.

Strategy / health / schema endpoints are always exercised (pure introspection).
Data-dependent endpoints (`/tickers`, `/backtest`, `/sweep`) require a reachable
ClickHouse with the `bars` table populated; they are skipped otherwise with a
clear reason (see IMPROVEMENTS.md → Tests). We do NOT mock ClickHouse per the
BRIEF's "Working principles".
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app
from data import list_tickers


client = TestClient(app)


# ─── Probe ClickHouse once; gate data-dependent tests on it ─────────────────
def _ch_available() -> tuple[bool, list[str]]:
    try:
        tickers = list_tickers()
        return (bool(tickers), tickers)
    except Exception:
        return (False, [])


CH_OK, CH_TICKERS = _ch_available()
ch_required = pytest.mark.skipif(
    not CH_OK,
    reason="ClickHouse unreachable or `bars` table empty — skipping data-backed tests.",
)


# ─── Always-on tests ────────────────────────────────────────────────────────
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["clickhouse"] in {"ok", "down"}


def test_strategies_lists_registry():
    r = client.get("/strategies")
    assert r.status_code == 200
    body = r.json()
    names = {s["name"] for s in body}
    assert {"ma_crossover", "rsi_mean_reversion"}.issubset(names)
    for s in body:
        assert isinstance(s["default_params"], dict)
        assert isinstance(s["param_grid"], dict)


def test_backtest_unknown_strategy_returns_404():
    r = client.post(
        "/backtest",
        json={
            "tickers": ["AAPL"],
            "start": "2024-01-01",
            "end": "2024-06-01",
            "interval": "1d",
            "strategy": "does_not_exist",
            "params": {},
        },
    )
    assert r.status_code == 404


def test_backtest_request_validation_rejects_empty_tickers():
    r = client.post(
        "/backtest",
        json={
            "tickers": [],
            "start": "2024-01-01",
            "end": "2024-06-01",
            "interval": "1d",
            "strategy": "ma_crossover",
            "params": {},
        },
    )
    assert r.status_code == 422


# ─── Data-backed tests ──────────────────────────────────────────────────────
@ch_required
def test_tickers_returns_list():
    r = client.get("/tickers")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert all(isinstance(t, str) for t in body)
    assert body == sorted(body)


@ch_required
def test_backtest_returns_metrics_and_equity_curve():
    ticker = CH_TICKERS[0]
    r = client.post(
        "/backtest",
        json={
            "tickers": [ticker],
            "start": "2022-01-01",
            "end": "2023-01-01",
            "interval": "1d",
            "strategy": "ma_crossover",
            "params": {"fast": 10, "slow": 30},
            "commission": 0.0005,
            "slippage": 0.0005,
        },
    )
    # 422 is acceptable only if there are literally zero bars for that range.
    if r.status_code == 422:
        pytest.skip(f"No bars for {ticker} in the requested range: {r.json()}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == "ma_crossover"
    assert body["params"] == {"fast": 10, "slow": 30}
    assert "portfolio_metrics" in body
    assert "sharpe" in body["portfolio_metrics"]
    assert len(body["results"]) == 1
    block = body["results"][0]
    assert block["ticker"] == ticker
    assert len(block["equity_curve"]) > 0
    pt = block["equity_curve"][0]
    assert "timestamp" in pt and "value" in pt
    assert isinstance(block["entries"], list)
    assert isinstance(block["exits"], list)


@ch_required
def test_sweep_ranks_by_sharpe_desc():
    ticker = CH_TICKERS[0]
    r = client.post(
        "/sweep",
        json={
            "tickers": [ticker],
            "start": "2022-01-01",
            "end": "2023-01-01",
            "interval": "1d",
            "strategy": "ma_crossover",
            "grid": {"fast": [5, 10], "slow": [50, 100]},
            "commission": 0.0005,
            "slippage": 0.0005,
        },
    )
    if r.status_code == 422:
        pytest.skip(f"No bars for {ticker} in the requested range: {r.json()}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == "ma_crossover"
    assert len(body["results"]) >= 1
    sharpes = [
        e["metrics"].get("sharpe")
        for e in body["results"]
        if e["metrics"].get("sharpe") is not None
    ]
    # Must be sorted descending where defined.
    assert sharpes == sorted(sharpes, reverse=True)
    for entry in body["results"]:
        assert "params" in entry
        assert "label" in entry
