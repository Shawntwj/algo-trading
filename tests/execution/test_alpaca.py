"""AlpacaBroker tests — replay recorded HTTP via ``pytest-vcr`` cassettes.

The cassettes live in ``tests/execution/cassettes/`` and are name-matched by
test function (``pytest-vcr`` default). Credentials are scrubbed on record via
``filter_headers`` — the cassettes contain placeholder ``test_*`` values only.

Tests requiring a live Alpaca paper account (no cassette recorded yet) are
marked ``@pytest.mark.live`` and skipped in CI. To record additional
cassettes, set ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` and run with
``--vcr-record=once`` (after removing the existing cassette file).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from execution import BrokerError
from execution.alpaca import LIVE_URL, PAPER_URL, AlpacaBroker


# ── VCR config ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def vcr_config():
    """Scrub credentials from recorded cassettes."""
    return {
        "filter_headers": [
            "authorization",
            "apca-api-key-id",
            "apca-api-secret-key",
        ],
        "record_mode": "none",  # never hit the network in CI
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    return os.path.join(os.path.dirname(__file__), "cassettes")


# ── env hygiene ───────────────────────────────────────────────────────────
@pytest.fixture
def alpaca_env(monkeypatch):
    """Stamp placeholder creds so the constructor reaches client init."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")


@pytest.fixture
def no_alpaca_env(monkeypatch):
    """Strip Alpaca creds for the missing-credential test."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


# ── constructor wiring ────────────────────────────────────────────────────
def test_constructor_paper_uses_paper_url(alpaca_env):
    b = AlpacaBroker(paper=True)
    assert b.paper is True
    assert b.base_url == PAPER_URL
    assert b.base_url == "https://paper-api.alpaca.markets"


def test_constructor_live_uses_live_url(alpaca_env):
    b = AlpacaBroker(paper=False)
    assert b.paper is False
    assert b.base_url == LIVE_URL
    assert b.base_url == "https://api.alpaca.markets"


def test_constructor_accepts_injected_client_without_creds(no_alpaca_env):
    """Injecting a client must bypass the env-var requirement."""
    fake = MagicMock()
    b = AlpacaBroker(client=fake, paper=True)
    assert b._client is fake
    assert b.base_url == PAPER_URL


# ── missing credentials path (no network) ─────────────────────────────────
def test_missing_credentials_raises_broker_error(no_alpaca_env):
    with pytest.raises(BrokerError, match="requires credentials"):
        AlpacaBroker(paper=True)


def test_missing_only_secret_raises(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(BrokerError, match="requires credentials"):
        AlpacaBroker(paper=True)


# ── cash() — VCR replay of get_account ────────────────────────────────────
# pytest-vcr defaults the cassette filename to the test function name. The
# ``_account_cassette`` fixture below pins both tests below to the single
# recorded ``get_account.yaml`` cassette.
@pytest.fixture
def _account_cassette(vcr, alpaca_env):
    with vcr.use_cassette("get_account.yaml"):
        yield


def test_cash_happy_path(_account_cassette):
    b = AlpacaBroker(paper=True)
    # The recorded account cassette reports cash=100000.
    assert b.cash() == 100000.0


def test_connect_uses_get_account_probe(_account_cassette):
    """connect() should call get_account; the same cassette covers it."""
    b = AlpacaBroker(paper=True)
    assert b.is_connected() is False
    b.connect()
    assert b.is_connected() is True
    # Idempotent — second call is a no-op (no extra HTTP).
    b.connect()
    assert b.is_connected() is True


# ── cash error wrapping (no network — stubbed client) ─────────────────────
def test_cash_wraps_vendor_error(no_alpaca_env):
    fake = MagicMock()
    fake.get_account.side_effect = RuntimeError("HTTP 401")
    b = AlpacaBroker(client=fake, paper=True)
    with pytest.raises(BrokerError, match="Alpaca cash failed"):
        b.cash()


def test_cash_non_numeric_balance_raises(no_alpaca_env):
    fake = MagicMock()
    fake.get_account.return_value = MagicMock(cash="not-a-number")
    b = AlpacaBroker(client=fake, paper=True)
    with pytest.raises(BrokerError, match="non-numeric"):
        b.cash()


# ── positions() — needs a live cassette we haven't recorded ───────────────
@pytest.mark.live
@pytest.mark.skip(
    reason="positions() requires a live paper account or a recorded "
    "get_all_positions.yaml cassette — see test docstring for recording steps."
)
def test_positions_live():  # pragma: no cover — live-only
    b = AlpacaBroker(paper=True)
    out = b.positions()
    assert isinstance(out, dict)
