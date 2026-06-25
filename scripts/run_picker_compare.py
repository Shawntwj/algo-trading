"""One-shot driver: run picker_compare A-vs-B for each shipped picker against
the ClickHouse universe. Prints a small summary table and writes a JSON dump
for the writeup.

Usage:
    python -m scripts.run_picker_compare \\
        --tickers AAPL,MSFT,JPM,JNJ,V,XOM,AMZN,META,NVDA,GOOGL \\
        --start 2020-01-01 --end 2024-12-31

The default tickers are the largest-cap S&P 500 names that ClickHouse has
backfilled at task time. Holdings_history is pulled live from EDGAR (one
call per 13F filing in the window, polite back-off per data/edgar.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import polars_to_wide  # noqa: E402
from backtest.picker_compare import (  # noqa: E402
    build_holdings_history_live,
    picker_compare,
)
from data import load_bars  # noqa: E402
from research.picker_profiles import list_profiles, load_profile  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("picker_compare")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickers", default=None,
        help="Comma-separated picker names; default = all committed.",
    )
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--rebalance-freq", type=int, default=21)
    parser.add_argument("--lookback-for-factors", type=int, default=252)
    parser.add_argument("--filing-delay-days", type=int, default=45)
    parser.add_argument("--out-json", default="reports/output/picker_compare_summary.json")
    args = parser.parse_args(argv)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    pickers = (
        [p.strip() for p in args.pickers.split(",") if p.strip()]
        if args.pickers else list_profiles()
    )

    log.info("loading %d tickers from ClickHouse: %s", len(tickers), tickers)
    df = load_bars(tickers, start=args.start, end=args.end)
    if df.is_empty():
        log.error("no bars returned — backfill the universe first")
        return 1
    prices_wide = polars_to_wide(df)

    summary: dict[str, dict] = {}
    for picker in pickers:
        log.info("running picker_compare for %s", picker)
        try:
            profile = load_profile(picker)
        except FileNotFoundError:
            log.warning("no profile for %s — skipping", picker)
            continue

        log.info("  pulling live holdings_history from EDGAR…")
        try:
            history = build_holdings_history_live(
                picker, args.start, args.end,
                top_n=args.top_n, filing_delay_days=args.filing_delay_days,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("EDGAR fetch failed for %s: %s — skipping", picker, exc)
            continue

        try:
            out = picker_compare(
                picker, prices_wide,
                holdings_history=history, profile=profile,
                top_n=args.top_n, rebalance_freq=args.rebalance_freq,
                lookback_for_factors=args.lookback_for_factors,
                filing_delay_days=args.filing_delay_days,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("picker_compare failed for %s: %s", picker, exc)
            continue

        summary[picker] = {
            "sharpe_a_factor_clone": out.sharpe_a,
            "sharpe_b_literal_follow": out.sharpe_b,
            "sharpe_gap": out.sharpe_gap,
            "n_filings_in_window": len(out.holdings_history),
            "first_filing": min(out.holdings_history.keys(), default=None),
            "last_filing": max(out.holdings_history.keys(), default=None),
        }
        log.info(
            "  picker=%s sharpe_A=%.3f sharpe_B=%.3f gap=%.3f (filings=%d)",
            picker, out.sharpe_a, out.sharpe_b, out.sharpe_gap,
            len(out.holdings_history),
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    log.info("wrote %s", out_path)

    # Pretty print.
    print("\nPicker compare summary (A=factor clone, B=literal 13F follow):")
    print(f"  universe = {len(tickers)} tickers; window = {args.start}..{args.end}")
    print(f"  top_n = {args.top_n}; filing_delay_days = {args.filing_delay_days}")
    print(f"  {'picker':<25} {'Sharpe A':>10} {'Sharpe B':>10} {'A-B gap':>10} {'filings':>10}")
    for k, v in summary.items():
        print(f"  {k:<25} {v['sharpe_a_factor_clone']:>10.3f} {v['sharpe_b_literal_follow']:>10.3f}"
              f" {v['sharpe_gap']:>10.3f} {v['n_filings_in_window']:>10d}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
