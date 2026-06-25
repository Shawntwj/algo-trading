"""CLI: backfill the configured universe directly (without Dagster).

Usage:
    python scripts/backfill.py
    python scripts/backfill.py --tickers AAPL,MSFT --start 2020-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_settings
from data import backfill_universe


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()

    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default=None, help="Comma-separated; defaults to config universe")
    p.add_argument("--start", default=settings.backfill_start)
    p.add_argument("--end", default=settings.end_date)
    p.add_argument("--interval", default=settings.intervals[0])
    args = p.parse_args()

    tickers = args.tickers.split(",") if args.tickers else settings.universe
    counts = backfill_universe(tickers, start=args.start, end=args.end, interval=args.interval)
    for t, n in counts.items():
        print(f"  {t:6s}  {n:>8d} rows")


if __name__ == "__main__":
    main()
