import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { SelectedTrade } from "../App";
import type { TickerBacktest } from "../api/types";

interface PriceChartProps {
  results: TickerBacktest[];
  // Optional: when provided, entry/exit markers become clickable and emit the
  // matching trade key. Without it, the chart behaves exactly as before.
  selectedTrade?: SelectedTrade | null;
  onSelectTrade?: (t: SelectedTrade | null) => void;
  // When true, marker clicks are no-ops (used for non-explainable strategies
  // — we still want the chart, just without the affordance).
  explainable?: boolean;
}

interface LinePoint {
  ts: number;
  value: number;
}

interface MarkerPoint {
  ts: number;
  value: number;
  // Raw ISO timestamp from the backtest payload — preserved so click
  // handlers can emit it back up unchanged (the chart converts to ms for
  // recharts' time scale).
  iso: string;
  direction: "long_entry" | "long_exit";
}

// Backend currently returns equity_curve per ticker (not raw price). We plot
// that as the line and overlay entry/exit markers at the closest equity
// timestamp. See IMPROVEMENTS.md — a /price endpoint would let this show real
// price action.
function nearestValue(curve: LinePoint[], targetTs: number): number | null {
  if (curve.length === 0) return null;
  let lo = 0;
  let hi = curve.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (curve[mid].ts < targetTs) lo = mid + 1;
    else hi = mid;
  }
  // Pick closer of lo and lo-1
  if (lo > 0 && Math.abs(curve[lo - 1].ts - targetTs) < Math.abs(curve[lo].ts - targetTs)) {
    return curve[lo - 1].value;
  }
  return curve[lo].value;
}

// Up-pointing triangle for entries (green), down-pointing for exits (red).
// Recharts passes the full marker payload to a custom shape, so we can read
// the trade's iso timestamp + direction off `props.payload` and fire onClick
// up to the parent. Selected markers get a thicker ring + brighter fill so
// the user can see what they picked even after the side panel opens.
interface TriangleProps {
  cx?: number;
  cy?: number;
  payload?: MarkerPoint;
  ticker?: string;
  selectedTrade?: SelectedTrade | null;
  onSelectTrade?: (t: SelectedTrade | null) => void;
  explainable?: boolean;
}

function isSelected(
  ticker: string | undefined,
  marker: MarkerPoint | undefined,
  selected: SelectedTrade | null | undefined,
): boolean {
  if (!ticker || !marker || !selected) return false;
  return (
    selected.ticker === ticker &&
    selected.timestamp === marker.iso &&
    selected.direction === marker.direction
  );
}

function UpTriangle(props: TriangleProps) {
  const {
    cx = 0,
    cy = 0,
    payload,
    ticker,
    selectedTrade,
    onSelectTrade,
    explainable,
  } = props;
  const sel = isSelected(ticker, payload, selectedTrade);
  const interactive = explainable && payload && ticker && onSelectTrade;
  return (
    <g
      style={{ cursor: interactive ? "pointer" : "default" }}
      onClick={
        interactive
          ? (e) => {
              e.stopPropagation();
              onSelectTrade!({
                ticker: ticker!,
                timestamp: payload!.iso,
                direction: payload!.direction,
              });
            }
          : undefined
      }
    >
      {sel && (
        <circle cx={cx} cy={cy} r={9} fill="none" stroke="#059669" strokeWidth={2} />
      )}
      <polygon
        points={`${cx},${cy - 6} ${cx - 5},${cy + 4} ${cx + 5},${cy + 4}`}
        fill={sel ? "#10b981" : "#059669"}
        stroke="#065f46"
        strokeWidth={sel ? 1.5 : 1}
      />
    </g>
  );
}

function DownTriangle(props: TriangleProps) {
  const {
    cx = 0,
    cy = 0,
    payload,
    ticker,
    selectedTrade,
    onSelectTrade,
    explainable,
  } = props;
  const sel = isSelected(ticker, payload, selectedTrade);
  const interactive = explainable && payload && ticker && onSelectTrade;
  return (
    <g
      style={{ cursor: interactive ? "pointer" : "default" }}
      onClick={
        interactive
          ? (e) => {
              e.stopPropagation();
              onSelectTrade!({
                ticker: ticker!,
                timestamp: payload!.iso,
                direction: payload!.direction,
              });
            }
          : undefined
      }
    >
      {sel && (
        <circle cx={cx} cy={cy} r={9} fill="none" stroke="#dc2626" strokeWidth={2} />
      )}
      <polygon
        points={`${cx},${cy + 6} ${cx - 5},${cy - 4} ${cx + 5},${cy - 4}`}
        fill={sel ? "#ef4444" : "#dc2626"}
        stroke="#7f1d1d"
        strokeWidth={sel ? 1.5 : 1}
      />
    </g>
  );
}

export default function PriceChart({
  results,
  selectedTrade,
  onSelectTrade,
  explainable = false,
}: PriceChartProps) {
  const [activeTicker, setActiveTicker] = useState<string>(
    results[0]?.ticker ?? "",
  );

  const active = results.find((r) => r.ticker === activeTicker) ?? results[0];

  const { line, entries, exits } = useMemo(() => {
    if (!active) return { line: [], entries: [], exits: [] };
    const linePts: LinePoint[] = active.equity_curve
      .map((p) => ({ ts: new Date(p.timestamp).getTime(), value: p.value }))
      .filter((p) => Number.isFinite(p.ts) && Number.isFinite(p.value));

    const toMarker = (
      ts: string,
      direction: "long_entry" | "long_exit",
    ): MarkerPoint | null => {
      const t = new Date(ts).getTime();
      if (!Number.isFinite(t)) return null;
      const v = nearestValue(linePts, t);
      if (v === null) return null;
      return { ts: t, value: v, iso: ts, direction };
    };

    const ent = active.entries
      .map((ts) => toMarker(ts, "long_entry"))
      .filter((x): x is MarkerPoint => x !== null);
    const ext = active.exits
      .map((ts) => toMarker(ts, "long_exit"))
      .filter((x): x is MarkerPoint => x !== null);
    return { line: linePts, entries: ent, exits: ext };
  }, [active]);

  if (!active) {
    return (
      <div className="text-sm text-slate-400 italic">
        No ticker data to chart.
      </div>
    );
  }

  return (
    <div>
      {results.length > 1 && (
        <div className="mb-2 flex flex-wrap gap-1">
          {results.map((r) => (
            <button
              key={r.ticker}
              onClick={() => setActiveTicker(r.ticker)}
              className={`px-2 py-1 text-xs rounded border font-mono ${
                r.ticker === active.ticker
                  ? "bg-indigo-600 text-white border-indigo-600"
                  : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50"
              }`}
            >
              {r.ticker}
            </button>
          ))}
        </div>
      )}

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={line}
            margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
          >
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
            <YAxis
              dataKey="value"
              stroke="#64748b"
              fontSize={11}
              domain={["auto", "auto"]}
            />
            <Tooltip
              labelFormatter={(v) =>
                new Date(v as number).toISOString().slice(0, 10)
              }
              formatter={(value) => (typeof value === "number" ? value.toFixed(2) : String(value))}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="value"
              name={active.ticker}
              stroke="#4f46e5"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
            <Scatter
              name="entries"
              data={entries}
              dataKey="value"
              shape={(props: unknown) => (
                <UpTriangle
                  {...(props as TriangleProps)}
                  ticker={active.ticker}
                  selectedTrade={selectedTrade}
                  onSelectTrade={onSelectTrade}
                  explainable={explainable}
                />
              )}
            />
            <Scatter
              name="exits"
              data={exits}
              dataKey="value"
              shape={(props: unknown) => (
                <DownTriangle
                  {...(props as TriangleProps)}
                  ticker={active.ticker}
                  selectedTrade={selectedTrade}
                  onSelectTrade={onSelectTrade}
                  explainable={explainable}
                />
              )}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
