import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TickerBacktest } from "../api/types";

interface EquityChartProps {
  results: TickerBacktest[];
}

// Stable-ish palette; cycles for >8 tickers.
const COLORS = [
  "#4f46e5",
  "#059669",
  "#dc2626",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#65a30d",
];

interface MergedPoint {
  ts: number;
  [ticker: string]: number;
}

export default function EquityChart({ results }: EquityChartProps) {
  const { data, tickers } = useMemo(() => {
    const byTs = new Map<number, MergedPoint>();
    const symbols: string[] = [];
    for (const r of results) {
      symbols.push(r.ticker);
      for (const p of r.equity_curve) {
        const ts = new Date(p.timestamp).getTime();
        if (!Number.isFinite(ts)) continue;
        let row = byTs.get(ts);
        if (!row) {
          row = { ts } as MergedPoint;
          byTs.set(ts, row);
        }
        row[r.ticker] = p.value;
      }
    }
    const merged = Array.from(byTs.values()).sort((a, b) => a.ts - b.ts);
    return { data: merged, tickers: symbols };
  }, [results]);

  if (data.length === 0) {
    return (
      <div className="text-sm text-slate-400 italic">No equity curve data.</div>
    );
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="ts"
            type="number"
            domain={["dataMin", "dataMax"]}
            scale="time"
            tickFormatter={(v) => new Date(v).toISOString().slice(0, 10)}
            stroke="#64748b"
            fontSize={11}
          />
          <YAxis stroke="#64748b" fontSize={11} />
          <Tooltip
            labelFormatter={(v) => new Date(v as number).toISOString().slice(0, 10)}
            formatter={(value) => (typeof value === "number" ? value.toFixed(2) : String(value))}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {tickers.map((t, i) => (
            <Line
              key={t}
              type="monotone"
              dataKey={t}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={1.5}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
