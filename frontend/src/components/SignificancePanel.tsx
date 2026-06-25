import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import type { RunResult } from "../App";
import { runStats } from "../api/client";
import type { StatsResponse, TickerBacktest } from "../api/types";

interface SignificancePanelProps {
  lastResult: RunResult | null;
}

// Bar-over-bar simple returns derived from an equity curve.
function equityToReturns(curve: TickerBacktest["equity_curve"]): number[] {
  const out: number[] = [];
  for (let i = 1; i < curve.length; i++) {
    const prev = curve[i - 1].value;
    const cur = curve[i].value;
    if (!Number.isFinite(prev) || !Number.isFinite(cur) || prev === 0) continue;
    out.push(cur / prev - 1);
  }
  return out;
}

// Aggregate-portfolio returns: average per-bar across tickers in the response.
// Coarser than a true portfolio-equity series but the BRIEF only asks for a
// significance summary, not a full risk-attribution.
function aggregateReturns(results: TickerBacktest[]): number[] {
  if (results.length === 0) return [];
  const perTicker = results.map((r) => equityToReturns(r.equity_curve));
  const minLen = Math.min(...perTicker.map((a) => a.length));
  if (!Number.isFinite(minLen) || minLen <= 1) return [];
  const out: number[] = [];
  for (let i = 0; i < minLen; i++) {
    let s = 0;
    for (const arr of perTicker) s += arr[arr.length - minLen + i];
    out.push(s / perTicker.length);
  }
  return out;
}

// Inverse standard normal CDF, Acklam's approximation.
// Used to compute E[max] of n iid N(0,1) for the DSR threshold.
function normalInv(p: number): number {
  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
  ];
  const pl_low = 0.02425;
  const pl_high = 1 - pl_low;
  let q: number;
  let r: number;
  if (p < pl_low) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p <= pl_high) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

const EULER_MASCHERONI = 0.5772156649015329;

function expectedMaxIidNormal(n: number): number {
  if (n < 1) return 0;
  if (n === 1) return 0;
  const invN = 1 / n;
  const invNe = 1 / (n * Math.E);
  const q1 = normalInv(1 - invN);
  const q2 = normalInv(1 - invNe);
  return (1 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2;
}

interface DsrInputs {
  n_trials: number;
  sr_trials_std: number;
  sr_threshold: number;
}

function deriveDsrInputs(result: RunResult | null): DsrInputs {
  // If the latest run was a sweep, derive trial count + Sharpe std from it.
  if (result?.mode === "sweep") {
    const sharpes = result.response.results
      .map((e) => {
        const v = e.metrics.sharpe;
        return typeof v === "number" && Number.isFinite(v) ? v : null;
      })
      .filter((v): v is number => v !== null);
    const n = sharpes.length;
    if (n > 1) {
      const mean = sharpes.reduce((a, b) => a + b, 0) / n;
      const variance =
        sharpes.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
      const std = Math.sqrt(variance);
      return {
        n_trials: n,
        sr_trials_std: std,
        sr_threshold: std * expectedMaxIidNormal(n),
      };
    }
  }
  return { n_trials: 1, sr_trials_std: 0, sr_threshold: 0 };
}

function badgeColor(dsr: number | null): { bg: string; label: string } {
  if (dsr === null) return { bg: "bg-slate-200 text-slate-600", label: "n/a" };
  if (dsr > 0.95) return { bg: "bg-emerald-100 text-emerald-800", label: "strong" };
  if (dsr > 0.5) return { bg: "bg-amber-100 text-amber-800", label: "weak" };
  return { bg: "bg-red-100 text-red-800", label: "not significant" };
}

function fmt(v: number | null | undefined, d = 2): string {
  return v === null || v === undefined || !Number.isFinite(v)
    ? "—"
    : v.toFixed(d);
}

export default function SignificancePanel({ lastResult }: SignificancePanelProps) {
  // For single-backtest mode, also accept sweep-mode results (we'll pick the
  // best Sharpe entry for the deflated stats input but we don't have its
  // returns curve from /sweep — so the panel requires a single backtest).
  const results = useMemo(() => {
    if (!lastResult) return null;
    if (lastResult.mode === "single") return lastResult.response.results;
    return null;
  }, [lastResult]);

  const aggReturns = useMemo(() => (results ? aggregateReturns(results) : []), [results]);
  const dsrInputs = useMemo(() => deriveDsrInputs(lastResult), [lastResult]);

  const baseQuery = useQuery({
    queryKey: ["sig-base", aggReturns.length, dsrInputs.sr_threshold],
    queryFn: () =>
      runStats({
        returns: aggReturns,
        sr_benchmark: 0.0,
        n_resamples: 500,
      }),
    enabled: aggReturns.length > 2,
    retry: false,
    staleTime: 5 * 60_000,
  });

  // DSR := PSR with sr_benchmark = sr_threshold.
  const dsrQuery = useQuery({
    queryKey: ["sig-dsr", aggReturns.length, dsrInputs.sr_threshold],
    queryFn: () =>
      runStats({
        returns: aggReturns,
        sr_benchmark: dsrInputs.sr_threshold,
        n_resamples: 500,
      }),
    enabled: aggReturns.length > 2,
    retry: false,
    staleTime: 5 * 60_000,
  });

  if (!lastResult) {
    return (
      <div className="text-sm text-slate-500 italic">Run a backtest first.</div>
    );
  }
  if (!results) {
    return (
      <div className="text-sm text-slate-500 italic">
        Significance summary requires a single-mode backtest result.
      </div>
    );
  }

  const base: StatsResponse | undefined = baseQuery.data;
  const dsrStats: StatsResponse | undefined = dsrQuery.data;
  const psr = base?.psr ?? null;
  const dsr = dsrStats?.psr ?? null;
  const badge = badgeColor(dsr);

  return (
    <div className="space-y-4">
      <div
        className={`inline-flex flex-col items-start rounded-lg px-4 py-3 ${badge.bg}`}
      >
        <div className="text-xs uppercase tracking-wide">DSR</div>
        <div className="text-3xl font-mono font-bold">
          {fmt(dsr)} <span className="text-base font-normal">({badge.label})</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
        <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
          <div className="text-xs text-slate-500">PSR (vs 0)</div>
          <div className="font-mono">{fmt(psr)}</div>
        </div>
        <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
          <div className="text-xs text-slate-500">Sharpe</div>
          <div className="font-mono">{fmt(base?.sharpe ?? null, 3)}</div>
        </div>
        <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
          <div className="text-xs text-slate-500"># Trials (DSR)</div>
          <div className="font-mono">{dsrInputs.n_trials}</div>
        </div>
        <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
          <div className="text-xs text-slate-500">σ(Sharpe) trials</div>
          <div className="font-mono">{fmt(dsrInputs.sr_trials_std, 3)}</div>
        </div>
      </div>

      <p className="text-xs text-slate-500 max-w-prose">
        DSR (Deflated Sharpe Ratio, Bailey & López de Prado 2014) is the
        probability that the strategy's true Sharpe exceeds a benchmark inflated
        for the number of trials you searched over —{" "}
        <span className="font-mono">DSR &gt; 0.95</span> is the conventional bar
        for "real". Random-entry null p-value is on the backlog (see
        IMPROVEMENTS).
      </p>

      {baseQuery.isError && (
        <div className="text-xs text-red-600">
          Failed to compute base stats — check console.
        </div>
      )}
    </div>
  );
}
