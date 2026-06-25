"""Build & load picker factor profiles.

A **factor profile** is the picker's average factor-vector minus the S&P 500's
average, expressed as z-scores against the S&P 500 distribution (BRIEF Task 3
step 3). Positive entries = the picker over-weights that factor; negative =
under-weight.

This module gives three entry points:

  * :func:`build_profile_from_holdings`  pure function — given holdings + an
    S&P 500 reference frame, returns a profile dict. Used by the build
    script and the offline tests.
  * :func:`build_profile_live`           pulls the picker's latest 13F from
    EDGAR, resolves CUSIPs, fetches fundamentals, and writes JSON. Slow.
  * :func:`load_profile`                 reads the committed JSON. Used by
    PickerCloneStrategy at runtime — no network needed.

Hard-coded sample holdings are kept under ``FALLBACK_HOLDINGS`` so a build
without network access still produces a usable, reproducible profile. Each
fallback list is sourced from the most recent publicly-known 13F snapshot
(documented per-picker in research/pickers.md).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from data import edgar
from data.cusip_to_ticker import resolve as resolve_cusips
from data.fundamentals import (
    FACTOR_FIELDS,
    FactorVector,
    fundamentals_for,
    vectors_to_frame,
)

log = logging.getLogger(__name__)

# Where the committed profile JSONs live.
PROFILES_DIR = Path(__file__).resolve().parent / "picker_profiles"


# ─── reference universe & fallback holdings ─────────────────────────────
# A liquid 30-name proxy for the S&P 500. Full 500 is overkill for a profile
# difference (cosine sim is robust to universe size > ~20); 30 keeps the
# build cheap and deterministic.
SP500_PROXY: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "TSLA",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX",
    "LLY", "AVGO", "ABBV", "MRK", "KO", "PEP", "BAC", "COST", "TMO",
    "MCD", "ADBE", "CRM",
)


# Picker-specific recent top-15 holdings (publicly reported via 13F as of
# 2024 Q4 / 2025 Q1, depending on filer). These act as the offline fallback
# so build_profile() always returns *something* even without network.
# Sources: SEC EDGAR (each filer's most recent 13F-HR at time of writing).
FALLBACK_HOLDINGS: dict[str, tuple[str, ...]] = {
    "berkshire": (
        # Buffett: high-quality value, megacap, holds forever.
        "AAPL", "BAC", "AXP", "KO", "CVX", "OXY", "MCO", "KHC",
        "DVA", "C", "VRSN", "AMZN", "V", "MA", "AON",
    ),
    "pershing_square": (
        # Ackman: concentrated, activist large-cap.
        "CMG", "HLT", "QSR", "LOW", "GOOG", "GOOGL", "CP", "BN",
        "UBER", "NKE", "HHH",
    ),
    "appaloosa": (
        # Tepper: macro-driven, tech-heavy, some China ADRs.
        "BABA", "AMZN", "META", "MSFT", "GOOGL", "GOOG", "ORCL", "FXI",
        "PDD", "KWEB", "UBER", "BIDU", "JD", "LRCX", "MU",
    ),
    "scion": (
        # Burry: small, concentrated, contrarian. Holdings churn quarterly.
        "BABA", "JD", "PDD", "BAC", "REAL", "MOH", "OLPX",
    ),
}


@dataclass(frozen=True)
class PickerProfile:
    """Computed factor profile for a picker."""

    name: str
    holdings: tuple[str, ...]
    profile: dict[str, float]                # factor → z-score diff vs S&P 500
    picker_means: dict[str, float]           # raw means across holdings
    benchmark_means: dict[str, float]        # raw means across S&P 500 proxy
    benchmark_stds: dict[str, float]         # std across S&P 500 proxy
    as_of: str                               # ISO date the profile was built
    n_holdings_with_data: int                # how many holdings had usable factor data

    def vector(self) -> np.ndarray:
        return np.array([self.profile[f] for f in FACTOR_FIELDS], dtype=float)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "holdings": list(self.holdings),
            "profile": self.profile,
            "picker_means": self.picker_means,
            "benchmark_means": self.benchmark_means,
            "benchmark_stds": self.benchmark_stds,
            "as_of": self.as_of,
            "n_holdings_with_data": self.n_holdings_with_data,
        }

    @staticmethod
    def from_json(payload: dict) -> "PickerProfile":
        return PickerProfile(
            name=payload["name"],
            holdings=tuple(payload["holdings"]),
            profile={k: float(v) for k, v in payload["profile"].items()},
            picker_means={k: float(v) for k, v in payload["picker_means"].items()},
            benchmark_means={k: float(v) for k, v in payload["benchmark_means"].items()},
            benchmark_stds={k: float(v) for k, v in payload["benchmark_stds"].items()},
            as_of=payload["as_of"],
            n_holdings_with_data=int(payload["n_holdings_with_data"]),
        )


# ─── core builders ──────────────────────────────────────────────────────
def build_profile_from_frame(
    name: str,
    picker_vectors: pd.DataFrame,    # rows = picker holdings, cols = FACTOR_FIELDS
    benchmark_vectors: pd.DataFrame, # rows = S&P 500 proxy, cols = FACTOR_FIELDS
    *,
    holdings: Sequence[str],
    as_of: str,
) -> PickerProfile:
    """Compute the profile diff (z-scored against the benchmark).

    Picker mean is the column-wise mean of ``picker_vectors``; benchmark mean
    and std come from ``benchmark_vectors``. Profile = (picker - benchmark) /
    benchmark_std. NaNs are ignored when averaging.
    """
    if list(picker_vectors.columns) != list(FACTOR_FIELDS):
        raise ValueError("picker_vectors columns must equal FACTOR_FIELDS")
    if list(benchmark_vectors.columns) != list(FACTOR_FIELDS):
        raise ValueError("benchmark_vectors columns must equal FACTOR_FIELDS")

    picker_means = picker_vectors.mean(axis=0, skipna=True)
    bench_means = benchmark_vectors.mean(axis=0, skipna=True)
    bench_stds = benchmark_vectors.std(axis=0, ddof=1, skipna=True)

    profile = {}
    for f in FACTOR_FIELDS:
        mu_b = bench_means[f]
        sd_b = bench_stds[f]
        mu_p = picker_means[f]
        if not math.isfinite(mu_b) or not math.isfinite(sd_b) or sd_b == 0 or not math.isfinite(mu_p):
            profile[f] = 0.0
        else:
            profile[f] = float((mu_p - mu_b) / sd_b)

    return PickerProfile(
        name=name,
        holdings=tuple(holdings),
        profile=profile,
        picker_means={k: (float(v) if math.isfinite(v) else float("nan"))
                      for k, v in picker_means.items()},
        benchmark_means={k: (float(v) if math.isfinite(v) else float("nan"))
                         for k, v in bench_means.items()},
        benchmark_stds={k: (float(v) if math.isfinite(v) else float("nan"))
                        for k, v in bench_stds.items()},
        as_of=as_of,
        n_holdings_with_data=int(picker_vectors.notna().any(axis=1).sum()),
    )


def build_profile_from_holdings(
    name: str,
    holdings: Sequence[str],
    benchmark: Sequence[str] = SP500_PROXY,
    *,
    as_of: str,
    fundamentals_fn=fundamentals_for,
) -> PickerProfile:
    """Fetch fundamentals for ``holdings`` + ``benchmark`` then build the
    profile. ``fundamentals_fn`` is injectable so tests can stub yfinance.
    """
    picker_dict: dict[str, FactorVector] = {}
    for t in holdings:
        try:
            picker_dict[t] = fundamentals_fn(t, as_of)
        except Exception as exc:  # noqa: BLE001
            log.warning("fundamentals failed for picker holding %s: %s", t, exc)

    bench_dict: dict[str, FactorVector] = {}
    for t in benchmark:
        try:
            bench_dict[t] = fundamentals_fn(t, as_of)
        except Exception as exc:  # noqa: BLE001
            log.warning("fundamentals failed for benchmark %s: %s", t, exc)

    picker_frame = vectors_to_frame(picker_dict)
    bench_frame = vectors_to_frame(bench_dict)
    return build_profile_from_frame(
        name, picker_frame, bench_frame,
        holdings=holdings, as_of=as_of,
    )


def build_profile_live(
    picker_name: str,
    *,
    top_n: int = 15,
    benchmark: Sequence[str] = SP500_PROXY,
    as_of: str | None = None,
) -> PickerProfile:
    """End-to-end live build: EDGAR → CUSIP map → fundamentals → profile.

    Slow (hits SEC + OpenFIGI + yfinance). Use ``build_profile_from_holdings``
    with the offline ``FALLBACK_HOLDINGS`` for deterministic / fast builds.
    """
    cik = edgar.picker_cik(picker_name)
    filings = edgar.list_13f_filings(cik)
    if not filings:
        raise RuntimeError(f"no 13F filings for {picker_name}")

    latest = filings[0]
    holdings_raw = edgar.fetch_13f_holdings(cik, latest)
    # Equity-only (drop puts/calls so we're profiling the picker's long stock
    # book — BRIEF spec).
    equity = [h for h in holdings_raw if (h.put_call is None or h.put_call.lower() not in ("put", "call"))]
    # Aggregate by CUSIP (multi-class issuers) and rank by value_usd.
    by_cusip: dict[str, float] = {}
    for h in equity:
        by_cusip[h.cusip] = by_cusip.get(h.cusip, 0.0) + h.value_usd
    top_cusips = [c for c, _ in sorted(by_cusip.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    resolved = resolve_cusips(top_cusips)
    tickers = [t for t in (resolved.get(c) for c in top_cusips) if t]

    as_of_str = as_of or latest.report_date
    return build_profile_from_holdings(
        picker_name, tickers, benchmark, as_of=as_of_str,
    )


# ─── disk I/O ───────────────────────────────────────────────────────────
def profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{name}.json"


def save_profile(profile: PickerProfile) -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = profile_path(profile.name)
    path.write_text(json.dumps(profile.to_json(), indent=2, sort_keys=True))
    return path


def load_profile(name: str) -> PickerProfile:
    path = profile_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no committed profile for {name!r} at {path}")
    payload = json.loads(path.read_text())
    return PickerProfile.from_json(payload)


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


# ─── similarity helpers (used by PickerCloneStrategy) ───────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Numpy cosine sim with zero-vector guard. -1..1; higher = more similar."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0 or not math.isfinite(na) or not math.isfinite(nb):
        return 0.0
    return float(np.dot(a, b) / (na * nb))
