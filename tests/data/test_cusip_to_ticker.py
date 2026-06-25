"""Tests for the CUSIP→ticker mapper. The OpenFIGI path is not hit in CI;
we exercise the CSV cache layer only."""
from __future__ import annotations

from data import cusip_to_ticker


def test_cache_resolves_known_cusip():
    out = cusip_to_ticker.resolve(["037833100"], use_openfigi=False)
    assert out["037833100"] == "AAPL"


def test_cache_handles_case_and_whitespace():
    out = cusip_to_ticker.resolve([" 594918104 "], use_openfigi=False)
    assert out["594918104"] == "MSFT"


def test_cache_returns_none_for_unknown_when_offline():
    out = cusip_to_ticker.resolve(["ZZZZZZZZZ"], use_openfigi=False)
    assert out["ZZZZZZZZZ"] is None


def test_empty_input_returns_empty_dict():
    assert cusip_to_ticker.resolve([], use_openfigi=False) == {}


def test_load_cache_non_empty():
    cache = cusip_to_ticker.load_cache()
    # The committed bootstrap covers at least the core mega-caps.
    assert "037833100" in cache
    assert cache["037833100"] == "AAPL"
