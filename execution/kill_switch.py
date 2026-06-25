"""Process-wide kill switch the live runner consults at the top of every iteration.

Two sources, OR'd together so either can halt trading:

1. Environment variable ``ALGO_KILL=1`` (or any truthy value: ``yes``, ``true``, ``on``).
2. A flag file at ``KILL_FLAG_PATH`` (default ``config/.kill_switch``) —
   presence alone halts; the file's content is ignored.

The file path is configurable via the ``ALGO_KILL_FLAG_PATH`` env var, with
the YAML config's ``kill_switch_file`` as the runner-supplied default. Tests
must use ``tmp_path`` + ``monkeypatch.setenv`` to avoid touching the real env
or filesystem.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


DEFAULT_KILL_FLAG_PATH = Path("config/.kill_switch")

# Tokens accepted as "kill" by the env-var check.
_TRUTHY = {"1", "yes", "true", "on", "y", "t"}


def _env_killed() -> tuple[bool, str | None]:
    raw = os.environ.get("ALGO_KILL", "").strip().lower()
    if raw and raw in _TRUTHY:
        return (True, f"ALGO_KILL={raw} env var set")
    return (False, None)


def _file_killed(path: Path) -> tuple[bool, str | None]:
    try:
        if path.exists():
            return (True, f"kill flag file present: {path}")
    except OSError as exc:  # defensive — perms, FS errors
        log.warning("kill-switch file probe failed for %s: %s", path, exc)
    return (False, None)


def _resolve_path(override: Path | str | None) -> Path:
    if override is not None:
        return Path(override)
    env_path = os.environ.get("ALGO_KILL_FLAG_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_KILL_FLAG_PATH


def is_killed(kill_flag_path: Path | str | None = None) -> tuple[bool, str | None]:
    """Returns ``(is_killed, reason)``. Either source firing trips the switch.

    Parameters
    ----------
    kill_flag_path : Path | str | None
        Explicit override for the flag file. Falls back to the
        ``ALGO_KILL_FLAG_PATH`` env var, then :data:`DEFAULT_KILL_FLAG_PATH`.
    """
    env_kill, env_reason = _env_killed()
    if env_kill:
        return (True, env_reason)
    path = _resolve_path(kill_flag_path)
    file_kill, file_reason = _file_killed(path)
    if file_kill:
        return (True, file_reason)
    return (False, None)
