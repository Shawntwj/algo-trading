import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  CartesianGrid,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { RunResult } from "../App";
import { runWalkforward } from "../api/client";
import type { WalkForwardRequest, WalkForwardResponse } from "../api/types";

interface OOSDecayChartProps {
  lastResult: RunResult | null;
}

function fmt(v: number | null | undefined, d = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(d);
}

export default function OOSDecayChart({ lastResult }: OOSDecayChartProps) {
  const [trainSize, setTrainSize] = useState(250);
  const [testSize, setTestSize] = useState(60);
  const [mode, setMode] = useState<"expanding" | "rolling">("expanding");

  const baseReq = useMemo(() => {
    if (!lastResult) return null;
    // For sweep mode use the sweep's grid; otherwise build a 1-cell grid from
    // the single-backtest params so the harness has something to optimise over.
    let grid: Record<string, unknown[]>;
    let r: { tickers: string[]; start: string; end: string; interval?: string; strategy: string };
    if (lastResult.mode === "sweep") {
      grid = lastResult.request.grid ?? {};
      r = lastResult.request;
    } else {
      grid = {};
      for (const [k, v] of Object.entries(lastResult.request.params ?? {})) {
        grid[k] = [v];
      }
      r = lastResult.request;
    }
    return {
      tickers: r.tickers,
      start: r.start,
      end: r.end,
      interval: r.interval,
      strategy: r.strategy,
      grid,
    };
  }, [lastResult]);

  const mutation = useMutation({
    mutationFn: (req: WalkForwardRequest) => runWalkforward(req),
  });

  function onRun() {
    if (!baseReq) return;
    mutation.mutate({
      ...baseReq,
      train_size: trainSize,
      test_size: testSize,
      mode,
      n_resamples: 300,
    });
  }

  const data: WalkForwardResponse | undefined = mutation.data;

  // Build scatter data + a diagonal line spanning the visible range.
  const points = useMemo(() => {
    if (!data) return [];
    return data.is_vs_oos
      .map(([is, oos]) => ({ is, oos }))
      .filter(
        (p): p is { is: number; oos: number } =>
          typeof p.is === "number" &&
          typeof p.oos === "number" &&
          Number.isFinite(p.is) &&
          Number.isFinite(p.oos),
      );
  }, [data]);

  const { domainMin, domainMax, diagData } = useMemo(() => {
    if (points.length === 0) return { domainMin: -1, domainMax: 1, diagData: [] };
    const vals = points.flatMap((p) => [p.is, p.oos]);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = Math.max(0.1, (hi - lo) * 0.1);
    const dmin = lo - pad;
    const dmax = hi + pad;
    return {
      domainMin: dmin,
      domainMax: dmax,
      diagData: [
        { is: dmin, diag: dmin },
        { is: dmax, diag: dmax },
      ],
    };
  }, [points]);

  if (!lastResult) {
    return (
      <div className="text-sm text-slate-500 italic">Run a backtest first.</div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3 max-w-lg">
        <label className="text-xs text-slate-600">
          train_size
          <input
            type="number"
            value={trainSize}
            min={2}
            onChange={(e) => setTrainSize(Number(e.target.value))}
            className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-sm"
          />
        </label>
        <label className="text-xs text-slate-600">
          test_size
          <input
            type="number"
            value={testSize}
            min={1}
            onChange={(e) => setTestSize(Number(e.target.value))}
            className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-sm"
          />
        </label>
        <label className="text-xs text-slate-600">
          mode
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as "expanding" | "rolling")}
            className="mt-1 w-full border border-slate-300 rounded px-2 py-1 text-sm bg-white"
          >
            <option value="expanding">expanding</option>
            <option value="rolling">rolling</option>
          </select>
        </label>
      </div>

      <button
        onClick={onRun}
        disabled={!baseReq || mutation.isPending}
        className="bg-indigo-600 text-white text-sm font-medium rounded px-3 py-1.5 hover:bg-indigo-700 disabled:opacity-50"
      >
        {mutation.isPending ? "Running…" : "Run walk-forward"}
      </button>

      {mutation.isError && (
        <div className="text-xs text-red-600">
          Walk-forward failed — check console (may need more bars).
        </div>
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
              <div className="text-xs text-slate-500"># Folds</div>
              <div className="font-mono">{data.n_folds}</div>
            </div>
            <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
              <div className="text-xs text-slate-500">OOS Sharpe (mean)</div>
              <div className="font-mono">{fmt(data.oos_sharpe_mean)}</div>
            </div>
            <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
              <div className="text-xs text-slate-500">OOS Sharpe 95% CI</div>
              <div className="font-mono text-xs">
                [{fmt(data.oos_sharpe_ci.low)}, {fmt(data.oos_sharpe_ci.high)}]
              </div>
            </div>
            <div className="border border-slate-200 rounded px-3 py-2 bg-slate-50">
              <div className="text-xs text-slate-500">Decay slope</div>
              <div className="font-mono">{fmt(data.decay_slope)}</div>
            </div>
          </div>

          {typeof data.decay_slope === "number" && data.decay_slope < 0.3 && (
            <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
              Warning: decay slope {fmt(data.decay_slope)} &lt; 0.3 — IS Sharpe
              isn't predicting OOS Sharpe well.
            </div>
          )}

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  type="number"
                  dataKey="is"
                  name="IS Sharpe"
                  domain={[domainMin, domainMax]}
                  stroke="#64748b"
                  fontSize={11}
                />
                <YAxis
                  type="number"
                  dataKey="oos"
                  name="OOS Sharpe"
                  domain={[domainMin, domainMax]}
                  stroke="#64748b"
                  fontSize={11}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(v) =>
                    typeof v === "number" ? v.toFixed(3) : String(v)
                  }
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine
                  segment={[
                    { x: domainMin, y: domainMin },
                    { x: domainMax, y: domainMax },
                  ]}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  ifOverflow="extendDomain"
                  label={{ value: "y = x", position: "insideTopLeft", fontSize: 10, fill: "#64748b" }}
                />
                <Scatter
                  name="Folds"
                  data={points}
                  fill="#4f46e5"
                />
                {/* Hidden line gives us a y=x diagonal even if ReferenceLine
                    can't infer the cartesian. Not always visible, harmless. */}
                <Line
                  data={diagData}
                  dataKey="diag"
                  stroke="transparent"
                  isAnimationActive={false}
                  dot={false}
                  legendType="none"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
