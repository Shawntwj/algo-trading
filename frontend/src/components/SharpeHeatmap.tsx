import { useMemo } from "react";

import type { SweepEntry } from "../api/types";

interface SharpeHeatmapProps {
  results: SweepEntry[];
  paramKeys: string[]; // exactly 2 swept param names
}

function sharpeOf(e: SweepEntry): number | null {
  const v = e.metrics.sharpe;
  if (v === null || v === undefined) return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

// Linear interpolation in HSL hue between red (0) and green (120).
function colorFor(value: number, vMin: number, vMax: number): string {
  if (vMax === vMin) return "hsl(60, 70%, 80%)";
  const t = (value - vMin) / (vMax - vMin); // 0..1
  const hue = t * 120; // red->green
  return `hsl(${hue}, 70%, 55%)`;
}

export default function SharpeHeatmap({ results, paramKeys }: SharpeHeatmapProps) {
  const { xVals, yVals, cells, vMin, vMax } = useMemo(() => {
    if (paramKeys.length !== 2) {
      return {
        xVals: [] as string[],
        yVals: [] as string[],
        cells: new Map<string, number>(),
        vMin: 0,
        vMax: 0,
      };
    }
    const [xKey, yKey] = paramKeys;
    const xSet = new Set<string>();
    const ySet = new Set<string>();
    const cellMap = new Map<string, number>();
    let vMin = Infinity;
    let vMax = -Infinity;
    for (const r of results) {
      const x = String(r.params[xKey]);
      const y = String(r.params[yKey]);
      xSet.add(x);
      ySet.add(y);
      const s = sharpeOf(r);
      if (s === null) continue;
      cellMap.set(`${x}|${y}`, s);
      if (s < vMin) vMin = s;
      if (s > vMax) vMax = s;
    }
    // Sort numerically if possible, else lexically.
    const sortVals = (vals: string[]) =>
      vals.slice().sort((a, b) => {
        const na = Number(a);
        const nb = Number(b);
        if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
        return a.localeCompare(b);
      });

    return {
      xVals: sortVals([...xSet]),
      yVals: sortVals([...ySet]),
      cells: cellMap,
      vMin: vMin === Infinity ? 0 : vMin,
      vMax: vMax === -Infinity ? 0 : vMax,
    };
  }, [results, paramKeys]);

  if (paramKeys.length !== 2) {
    return (
      <p className="text-xs text-slate-500 italic">
        Heatmap shown when exactly 2 params are swept.
      </p>
    );
  }

  if (xVals.length === 0 || yVals.length === 0) {
    return <p className="text-xs text-slate-500 italic">No sweep cells.</p>;
  }

  const [xKey, yKey] = paramKeys;

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse">
        <thead>
          <tr>
            <th className="px-2 py-1 text-xs text-slate-500 font-mono">
              {yKey} \ {xKey}
            </th>
            {xVals.map((x) => (
              <th
                key={x}
                className="px-2 py-1 text-xs text-slate-600 font-mono font-medium"
              >
                {x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {yVals.map((y) => (
            <tr key={y}>
              <th className="px-2 py-1 text-xs text-slate-600 font-mono font-medium text-right">
                {y}
              </th>
              {xVals.map((x) => {
                const v = cells.get(`${x}|${y}`);
                if (v === undefined) {
                  return (
                    <td
                      key={x}
                      className="border border-slate-200 px-3 py-2 text-xs text-slate-300 text-center"
                    >
                      —
                    </td>
                  );
                }
                return (
                  <td
                    key={x}
                    className="border border-slate-200 px-3 py-2 text-xs text-center font-mono"
                    style={{
                      backgroundColor: colorFor(v, vMin, vMax),
                      color: "#0f172a",
                    }}
                    title={`Sharpe ${v.toFixed(3)}`}
                  >
                    {v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 text-xs text-slate-500">
        Sharpe range: {vMin.toFixed(2)} → {vMax.toFixed(2)} (red = low, green = high)
      </div>
    </div>
  );
}
