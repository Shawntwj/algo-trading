"""CUSIP → ticker resolver.

OpenFIGI (https://www.openfigi.com/api) is the canonical public free mapper
and works without an API key at low volume (25 requests / 6s per IP, batch
size 100 per request). For everything outside that envelope we fall back
to a committed CSV cache at ``data/cusip_to_ticker.csv`` (manually curated
mappings for the picker universe so the offline tests + the strategy run
deterministically even if OpenFIGI is unreachable).

Public contract:

    resolve(["594918104", "037833100"]) -> {"594918104": "MSFT", "037833100": "AAPL"}

Unknown CUSIPs map to ``None`` (caller decides whether to drop / warn).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable
from urllib import error as urlerror
from urllib import request as urlrequest

log = logging.getLogger(__name__)

# Repo-root anchored so tests + scripts find the same file regardless of cwd.
_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cusip_to_ticker.csv"
_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
_OPENFIGI_BATCH = 100
_OPENFIGI_RATE_PAUSE = 6.0 / 25.0  # ≈ 0.24s between requests (25/6s rate limit)


# ─── CSV cache layer ───────────────────────────────────────────────────────
def _read_cache() -> dict[str, str]:
    if not _CACHE_PATH.exists():
        return {}
    out: dict[str, str] = {}
    with _CACHE_PATH.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cusip = (row.get("cusip") or "").strip().upper()
            ticker = (row.get("ticker") or "").strip().upper()
            if cusip and ticker:
                out[cusip] = ticker
    return out


def _write_cache(mapping: dict[str, str]) -> None:
    """Atomic-ish overwrite — sort by CUSIP for deterministic diffs."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cusip", "ticker"])
        for cusip in sorted(mapping.keys()):
            writer.writerow([cusip, mapping[cusip]])
    tmp.replace(_CACHE_PATH)


# ─── OpenFIGI ──────────────────────────────────────────────────────────────
def _openfigi_lookup(cusips: list[str]) -> dict[str, str]:
    """Batch-lookup ``cusips`` against OpenFIGI. Silent-skip on transport error.

    OpenFIGI returns an array shaped 1:1 with the request body; each entry is
    either ``{"data": [...]}`` (one or more matches) or ``{"warning": "..."}``
    (no match / ambiguous). We take the first match's ``ticker``. Multi-class
    issuers (BRK has BRK.A + BRK.B) are deterministically the alphabetical
    first match — the picker universe rarely cares which share class.
    """
    if not cusips:
        return {}
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENFIGI_API_KEY")
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key

    out: dict[str, str] = {}
    for chunk_start in range(0, len(cusips), _OPENFIGI_BATCH):
        chunk = cusips[chunk_start:chunk_start + _OPENFIGI_BATCH]
        payload = [{"idType": "ID_CUSIP", "idValue": c} for c in chunk]
        body = json.dumps(payload).encode()
        req = urlrequest.Request(_OPENFIGI_URL, data=body, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=20) as resp:
                results = json.loads(resp.read())
        except urlerror.HTTPError as exc:
            log.warning("OpenFIGI HTTP %s on batch starting %s — falling back to cache only",
                        exc.code, chunk[0])
            return out
        except urlerror.URLError as exc:
            log.warning("OpenFIGI URLError %s — falling back to cache only", exc)
            return out

        for cusip, entry in zip(chunk, results):
            matches = entry.get("data") or []
            if not matches:
                continue
            # Prefer common-stock entries when present (filters out warrants/
            # preferreds for multi-class issuers).
            common = [m for m in matches if (m.get("securityType2") or "").lower() == "common stock"]
            preferred = sorted(common or matches, key=lambda m: (m.get("ticker") or ""))
            ticker = (preferred[0].get("ticker") or "").upper()
            if ticker:
                out[cusip] = ticker
        time.sleep(_OPENFIGI_RATE_PAUSE)
    return out


# ─── Public API ────────────────────────────────────────────────────────────
def resolve(
    cusips: Iterable[str],
    *,
    use_openfigi: bool = True,
    persist: bool = True,
) -> dict[str, str | None]:
    """Resolve ``cusips`` to tickers using cache → OpenFIGI in that order.

    Unknown CUSIPs map to ``None``. When ``persist=True`` (default) any new
    mappings learned from OpenFIGI are written back to the committed CSV.
    """
    cusips_clean = [c.strip().upper() for c in cusips if c and c.strip()]
    if not cusips_clean:
        return {}

    cache = _read_cache()
    out: dict[str, str | None] = {c: cache.get(c) for c in cusips_clean}
    missing = [c for c in cusips_clean if out[c] is None]

    if missing and use_openfigi:
        learned = _openfigi_lookup(missing)
        for c, t in learned.items():
            out[c] = t
        if persist and learned:
            cache.update(learned)
            _write_cache(cache)

    return out


def load_cache() -> dict[str, str]:
    """Read-only view of the committed CSV cache. Used by tests."""
    return _read_cache()


def update_cache(mapping: dict[str, str]) -> None:
    """Public hook for the curated bootstrap script. Adds new CUSIP→ticker
    rows to the committed CSV (existing rows kept; new rows merged)."""
    existing = _read_cache()
    existing.update({k.upper(): v.upper() for k, v in mapping.items() if k and v})
    _write_cache(existing)
