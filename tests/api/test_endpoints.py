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
def test_benchmarks_returns_equity_curve():
    ticker = CH_TICKERS[0]
    r = client.post(
        "/benchmarks",
        json={
            "tickers": [ticker],
            "start": "2022-01-01",
            "end": "2023-01-01",
            "interval": "1d",
            "weights": "equal",
        },
    )
    if r.status_code == 422:
        pytest.skip(f"No bars for {ticker} in the requested range: {r.json()}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["weights"] == "equal"
    assert isinstance(body["curves"], list) and body["curves"]
    curve = body["curves"][0]
    assert curve["name"] == "buy_and_hold_equal"
    assert len(curve["equity_curve"]) > 0
    pt = curve["equity_curve"][0]
    assert "timestamp" in pt and "value" in pt


def test_stats_returns_metrics_with_cis():
    """`/stats` is pure compute — no ClickHouse dependency, never skipped."""
    import numpy as np

    rng = np.random.default_rng(0)
    # 1 year of daily returns with a positive drift.
    returns = rng.normal(0.0008, 0.01, size=252).tolist()
    r = client.post(
        "/stats",
        json={
            "returns": returns,
            "sr_benchmark": 0.0,
            "periods_per_year": 252,
            "n_resamples": 200,
            "alpha": 0.05,
            "seed": 7,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Top-level scalars present and finite (PSR is a probability in [0, 1]).
    assert "sharpe" in body and isinstance(body["sharpe"], float)
    assert "psr" in body and 0.0 <= body["psr"] <= 1.0
    assert "max_dd" in body and body["max_dd"] <= 0.0
    assert "total_return" in body and isinstance(body["total_return"], float)
    # CI blocks have the right shape and brackets are well-ordered.
    for key in ("sharpe_ci", "max_dd_ci", "total_return_ci"):
        block = body[key]
        assert set(block) == {"point", "low", "high"}, key
        assert block["low"] <= block["point"] <= block["high"], key


def test_stats_rejects_too_few_returns():
    r = client.post(
        "/stats",
        json={"returns": [0.01], "periods_per_year": 252},
    )
    assert r.status_code == 422


def test_benchmarks_rejects_bad_weights():
    r = client.post(
        "/benchmarks",
        json={
            "tickers": ["AAPL"],
            "start": "2022-01-01",
            "end": "2023-01-01",
            "interval": "1d",
            "weights": "garbage",
        },
    )
    # Either 422 from our validation or 422 from a downstream ValueError —
    # both surface as 422 by design.
    assert r.status_code == 422


def test_attribution_recovers_beta_and_returns_metrics():
    """`/attribution` is pure compute — synthetic arrays, never skipped."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 600
    market = rng.normal(0.0, 0.01, size=n).tolist()
    # strat = 0.8 * market + alpha + noise
    strat = (
        0.8 * np.array(market)
        + 0.0004
        + rng.normal(0.0, 0.004, size=n)
    ).tolist()
    r = client.post(
        "/attribution",
        json={
            "strategy_returns": strat,
            "market_returns": market,
            "risk_free": 0.0,
            "periods_per_year": 252,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["beta"] == pytest.approx(0.8, abs=0.05)
    assert body["alpha_t_stat"] > 0
    assert 0.0 <= body["r_squared"] <= 1.0
    assert body["n_obs"] == n


def test_attribution_rejects_length_mismatch():
    r = client.post(
        "/attribution",
        json={
            "strategy_returns": [0.0, 0.0, 0.0],
            "market_returns": [0.0, 0.0],
        },
    )
    assert r.status_code == 422


@ch_required
def test_walkforward_returns_folds_and_aggregate():
    """Walk-forward over a real bars window; needs ~600+ bars for the default
    train/test sizes, so we ask for a multi-year range. Falls back to skip if
    that range is empty for the chosen ticker."""
    ticker = CH_TICKERS[0]
    r = client.post(
        "/walkforward",
        json={
            "tickers": [ticker],
            "start": "2020-01-01",
            "end": "2024-01-01",
            "interval": "1d",
            "strategy": "ma_crossover",
            "grid": {"fast": [5, 10], "slow": [50, 100]},
            "train_size": 250,
            "test_size": 60,
            "mode": "expanding",
            "commission": 0.0005,
            "slippage": 0.0005,
            "n_resamples": 200,
            "seed": 7,
        },
    )
    if r.status_code == 422:
        pytest.skip(f"Not enough bars for {ticker} in the requested range: {r.json()}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_folds"] >= 1
    assert len(body["folds"]) == body["n_folds"]
    assert len(body["is_vs_oos"]) == body["n_folds"]
    for fold in body["folds"]:
        assert {"fold_idx", "selected_params", "train_start", "test_end"}.issubset(fold)
        # Each is_vs_oos pair is [IS, OOS] — None or float.
    for pair in body["is_vs_oos"]:
        assert len(pair) == 2


@ch_required
def test_regimes_split_returns_per_regime_stats():
    """Per-regime stats need both the strategy bars *and* SPY/VIX bars in
    ClickHouse. Skips cleanly if any are missing."""
    ticker = CH_TICKERS[0]
    r = client.post(
        "/regimes/split",
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
    if r.status_code == 422:
        pytest.skip(f"Missing SPY/VIX or no overlap: {r.json()}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == "ma_crossover"
    assert isinstance(body["regimes"], list)
    if body["regimes"]:
        row = body["regimes"][0]
        assert {"dimension", "regime", "n_bars", "sharpe"}.issubset(row)


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
