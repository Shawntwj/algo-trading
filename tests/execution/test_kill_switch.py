"""Tests for ``execution.kill_switch``.

All tests use ``monkeypatch`` to mutate env, and ``tmp_path`` for the flag
file — we never touch the real ``config/.kill_switch`` or the real
``ALGO_KILL`` env var in CI.
"""
from __future__ import annotations

import pytest

from execution.kill_switch import is_killed


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip kill-switch env vars before every test."""
    monkeypatch.delenv("ALGO_KILL", raising=False)
    monkeypatch.delenv("ALGO_KILL_FLAG_PATH", raising=False)


def test_default_no_kill(tmp_path):
    flag = tmp_path / "kill"  # absent
    killed, reason = is_killed(kill_flag_path=flag)
    assert killed is False
    assert reason is None


# ─── env-var source ───────────────────────────────────────────────────────
@pytest.mark.parametrize("value", ["1", "yes", "true", "on", "Y", "True", "TRUE"])
def test_env_var_truthy_trips_kill(monkeypatch, tmp_path, value):
    monkeypatch.setenv("ALGO_KILL", value)
    flag = tmp_path / "kill"
    killed, reason = is_killed(kill_flag_path=flag)
    assert killed is True
    assert "ALGO_KILL" in reason


@pytest.mark.parametrize("value", ["", "0", "no", "false", "off"])
def test_env_var_falsy_does_not_trip(monkeypatch, tmp_path, value):
    monkeypatch.setenv("ALGO_KILL", value)
    flag = tmp_path / "kill"
    killed, reason = is_killed(kill_flag_path=flag)
    assert killed is False
    assert reason is None


# ─── file-flag source ─────────────────────────────────────────────────────
def test_file_flag_presence_trips_kill(tmp_path):
    flag = tmp_path / ".kill_switch"
    flag.write_text("")  # empty contents are fine — only presence matters
    killed, reason = is_killed(kill_flag_path=flag)
    assert killed is True
    assert ".kill_switch" in reason


def test_file_flag_with_arbitrary_content_trips(tmp_path):
    flag = tmp_path / ".kill_switch"
    flag.write_text("halt now please")
    killed, reason = is_killed(kill_flag_path=flag)
    assert killed is True


# ─── resolution precedence ────────────────────────────────────────────────
def test_explicit_path_overrides_env_path(monkeypatch, tmp_path):
    decoy = tmp_path / "decoy"
    decoy.write_text("")
    monkeypatch.setenv("ALGO_KILL_FLAG_PATH", str(decoy))
    real = tmp_path / "real"   # absent → no kill via explicit path
    killed, _ = is_killed(kill_flag_path=real)
    assert killed is False


def test_env_path_used_when_no_explicit(monkeypatch, tmp_path):
    flag = tmp_path / "envflag"
    flag.write_text("")
    monkeypatch.setenv("ALGO_KILL_FLAG_PATH", str(flag))
    killed, reason = is_killed()
    assert killed is True
    assert "envflag" in reason


def test_either_source_trips_kill(monkeypatch, tmp_path):
    """Both env-var and file flag together — kill fires (any source is enough)."""
    monkeypatch.setenv("ALGO_KILL", "1")
    flag = tmp_path / ".kill_switch"
    flag.write_text("")
    killed, reason = is_killed(kill_flag_path=flag)
    assert killed is True
    # Env check runs first.
    assert "ALGO_KILL" in reason
