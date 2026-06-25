"""Build (or refresh) the committed picker factor profile JSONs.

Usage:
    python -m scripts.build_picker_profiles            # all pickers, offline fallback holdings
    python -m scripts.build_picker_profiles --live     # pull holdings from EDGAR live
    python -m scripts.build_picker_profiles berkshire  # just one

The default mode reads ``research.picker_profiles.FALLBACK_HOLDINGS`` for the
holdings list and only hits yfinance for the per-ticker factor data. This
is deterministic at the holdings level (factor values still drift day-to-day
since yfinance.info is live) and runs in <2 minutes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.picker_profiles import (  # noqa: E402
    FALLBACK_HOLDINGS,
    SP500_PROXY,
    build_profile_from_holdings,
    build_profile_live,
    save_profile,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_picker_profiles")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pickers", nargs="*",
        help="Picker names to build (default: all under FALLBACK_HOLDINGS).",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Pull holdings from SEC EDGAR live (slower; respects 10 req/s).",
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument(
        "--as-of", default=dt.date.today().isoformat(),
        help="Snapshot date (ISO). Defaults to today.",
    )
    args = parser.parse_args(argv)

    targets = args.pickers or sorted(FALLBACK_HOLDINGS.keys())
    for picker in targets:
        log.info("building profile for %s", picker)
        if args.live:
            profile = build_profile_live(
                picker, top_n=args.top_n, benchmark=SP500_PROXY, as_of=args.as_of,
            )
        else:
            holdings = FALLBACK_HOLDINGS.get(picker)
            if not holdings:
                log.error("no FALLBACK_HOLDINGS for %s — pass --live", picker)
                continue
            profile = build_profile_from_holdings(
                picker, holdings, SP500_PROXY, as_of=args.as_of,
            )
        path = save_profile(profile)
        log.info("wrote %s (%d holdings with data)", path, profile.n_holdings_with_data)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
