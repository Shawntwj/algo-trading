"""Overfit demo (BRIEF Task 7e §2).

Run a generous in-sample sweep on `ma_crossover`, promote the winner, then
generate a report for it via ``reports.evaluate``. The whole point is to show
that the **deflated** Sharpe ratio recognises selection bias — even when the
in-sample Sharpe looks fantastic, DSR < 0.95 because we tried many trials.

CLI:

    python -m reports.overfit_demo

Outputs a report alongside the regular ones and prints either

    ✅ overfit gauntlet works    (DSR < 0.95, the gate fires)

or

    ⚠️ DSR > 0.95 — sweep wasn't overfit enough (try a wider grid)

The demo *does not raise* on the warning case — it just notes the demo data
wasn't extreme enough.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import polars_to_wide, run_backtest
from backtest.stats import deflated_sharpe_ratio_from_sweep
from backtest.sweep import sweep
from strategies import REGISTRY
from strategies.base import Strategy

from .evaluate import (
    ReportConfig,
    _composite_returns,
    build_report,
)

_LOGGER = logging.getLogger(__name__)


# ─── sweep + winner ────────────────────────────────────────────────────────
def _wide_sweep_grid(strategy_cls: type[Strategy]) -> dict[str, list]:
    """A deliberately generous grid — wider than the registered ``param_grid``
    so the implied DSR penalty is heavy."""
    if strategy_cls.name == "ma_crossover":
        return {
            "fast": [3, 5, 8, 10, 15, 20, 25, 30, 40],
            "slow": [40, 60, 80, 100, 120, 150, 180, 200],
        }
    if strategy_cls.name == "rsi_mean_reversion":
        return {
            "period": [7, 10, 14, 20, 28],
            "oversold": [10, 20, 25, 30, 35],
            "exit_level": [45, 50, 55, 60, 65, 70],
        }
    # Fall back to the strategy's own grid (still better than nothing).
    return strategy_cls.param_grid()


@dataclass
class OverfitDemoResult:
    out_path: Path
    is_sharpe: float
    dsr: float
    n_trials: int
    sr_trials_std: float
    winner_params: dict[str, Any]


def _sharpe_of(result) -> float:
    """Single-number Sharpe from a vbt BacktestResult: cross-ticker mean of
    `portfolio.sharpe_ratio()`. NaN-safe."""
    s = result.portfolio.sharpe_ratio()
    if hasattr(s, "mean"):
        s = float(s.mean())
    else:
        s = float(s)
    return s if math.isfinite(s) else float("nan")


def run_overfit_demo(
    strategy_name: str = "ma_crossover",
    prices_wide: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
    out_path: Path | None = None,
    n_resamples: int = 1000,
    n_random_paths: int = 200,
) -> OverfitDemoResult:
    """Sweep a wide grid in-sample, promote the winner, run the gauntlet.

    Returns an ``OverfitDemoResult`` so tests can assert on ``dsr``. The HTML
    report is also written to ``out_path`` (default: alongside the regular
    reports under ``reports/output/``).
    """
    strategy_cls = REGISTRY[strategy_name]

    # 1. Resolve prices.
    if prices_wide is None:
        from data import load_bars  # noqa: PLC0415

        if not tickers:
            from data import list_tickers  # noqa: PLC0415

            tickers = list_tickers()[:5] or ["AAPL"]
        df = load_bars(tickers, start=start, end=end, interval=interval)
        if df.is_empty():
            raise RuntimeError(
                f"no bars for {tickers} between {start} and {end} — backfill first"
            )
        prices_wide = polars_to_wide(df)
        tickers_in = list(prices_wide["close"].columns)
    else:
        tickers_in = list(prices_wide["close"].columns)

    # 2. Big sweep on the full in-sample window.
    grid = _wide_sweep_grid(strategy_cls)
    _LOGGER.info("sweeping %s combos on %d bars", len(_combos(grid)), len(prices_wide))
    sweep_results = sweep(prices_wide, strategy_cls, grid=grid)
    if not sweep_results:
        raise RuntimeError("sweep returned zero results — grid is degenerate")

    # 3. Pick the winner.
    sharpes = np.array([_sharpe_of(r) for r in sweep_results], dtype=float)
    if not np.isfinite(sharpes).any():
        raise RuntimeError("every sweep config produced a non-finite Sharpe")
    winner_idx = int(np.nanargmax(sharpes))
    winner = sweep_results[winner_idx]
    winner_sharpe = float(sharpes[winner_idx])

    # 4. Compute DSR with the full sweep std deflating it.
    winner_returns = _composite_returns(winner)
    dsr = deflated_sharpe_ratio_from_sweep(
        best_returns=winner_returns,
        sweep_sharpes=sharpes,
    )
    finite_sharpes = sharpes[np.isfinite(sharpes)]
    sr_trials_std = float(finite_sharpes.std(ddof=1)) if finite_sharpes.size > 1 else 0.0

    # 5. Render a regular report so the artefact looks identical to the live one.
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{strategy_name}_overfit_{ts}.html"

    cfg = ReportConfig(
        strategy=strategy_name,
        tickers=tickers_in,
        start=start, end=end, interval=interval,
        params=dict(winner.params),
        walk_forward=False,
        n_resamples=n_resamples,
        n_random_paths=n_random_paths,
        prices_wide=prices_wide,
        out_path=out_path,
    )
    final_path = build_report(cfg)

    # Patch the rendered HTML's significance block to reflect the *real* sweep
    # statistics — the standard report has n_trials=1 because the single-config
    # entry point doesn't see the sweep. The overfit demo is the exception.
    html = final_path.read_text(encoding="utf-8")
    html = _patch_significance(html, n_trials=int(finite_sharpes.size),
                               sr_trials_std=sr_trials_std, dsr=dsr)
    final_path.write_text(html, encoding="utf-8")

    return OverfitDemoResult(
        out_path=final_path,
        is_sharpe=winner_sharpe,
        dsr=float(dsr),
        n_trials=int(finite_sharpes.size),
        sr_trials_std=sr_trials_std,
        winner_params=dict(winner.params),
    )


def _combos(grid: dict[str, list]) -> list[dict]:
    """Mirror of ``sweep._grid`` for log-only count purposes."""
    from itertools import product

    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, c)) for c in product(*[grid[k] for k in keys])]


def _patch_significance(html: str, n_trials: int, sr_trials_std: float, dsr: float) -> str:
    """Rewrite the DSR row + badge to reflect the post-sweep numbers.

    Cheap string find/replace — the template emits a known shape so the
    targeted edits are stable. If the template changes, this needs to follow.
    """
    # Replace the n_trials text inside the DSR row.
    html = html.replace(
        "across 1 trial(s) (σ = 0.000)",
        f"across {n_trials} trial(s) (σ = {sr_trials_std:.3f})",
    )
    # Replace the DSR row value. The template uses %.3f formatting.
    import re

    html = re.sub(
        r"<td>Deflated Sharpe Ratio</td>\s*<td>[0-9.]+</td>",
        f"<td>Deflated Sharpe Ratio</td><td>{dsr:.3f}</td>",
        html,
        count=1,
    )
    # Pick the matching badge wording for the post-sweep DSR.
    if dsr > 0.95:
        new_badge = (
            '<span class="badge pass">DSR pass — '
            "true Sharpe likely beats the deflated benchmark</span>"
        )
    elif dsr > 0.5:
        new_badge = (
            '<span class="badge warn">DSR borderline — '
            "not significant at 95% but above chance</span>"
        )
    else:
        new_badge = (
            '<span class="badge fail">DSR fail — '
            "selection bias swallows this result</span>"
        )
    html = re.sub(
        r'<span class="badge (pass|warn|fail)">[^<]+</span>',
        new_badge,
        html,
        count=1,
    )
    return html


# ─── CLI ───────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reports.overfit_demo",
        description="Run the in-sample sweep winner through the gauntlet and "
        "assert the DSR < 0.95 honesty check fires.",
    )
    parser.add_argument("--strategy", default="ma_crossover")
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else None
    )
    res = run_overfit_demo(
        strategy_name=args.strategy,
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        out_path=Path(args.out) if args.out else None,
        n_resamples=args.n_resamples,
    )
    print(f"wrote {res.out_path}")
    print(
        f"  in-sample winner: {json.dumps(res.winner_params, sort_keys=True)}  "
        f"is_sharpe={res.is_sharpe:.3f}"
    )
    print(
        f"  n_trials={res.n_trials}  σ(Sharpe)={res.sr_trials_std:.3f}  "
        f"DSR={res.dsr:.3f}"
    )
    if res.dsr < 0.95:
        print("\N{WHITE HEAVY CHECK MARK} overfit gauntlet works "
              "(DSR < 0.95 — the gate fires)")
    else:
        print(
            "\N{WARNING SIGN}️  DSR > 0.95 — sweep wasn't overfit enough. "
            "Try a wider grid (more params, more values per param, "
            "or a longer in-sample window)."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
