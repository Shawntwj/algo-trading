import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";

import { runRegimesSplit } from "../api/client";
import type { RegimeSplitRequest, RegimeStat } from "../api/types";

interface RegimeBreakdownProps {
  request: RegimeSplitRequest | null;
}

const DIM_COLORS: Record<string, string> = {
  trend: "#4f46e5",
  vol: "#059669",
  drawdown: "#dc2626",
};

const helper = createColumnHelper<RegimeStat>();

function fmtPct(v: number | null): string {
  return v === null || v === undefined ? "—" : `${(v * 100).toFixed(2)}%`;
}
function fmtNum(v: number | null, d = 3): string {
  return v === null || v === undefined ? "—" : v.toFixed(d);
}

export default function RegimeBreakdown({ request }: RegimeBreakdownProps) {
  const [activeDim, setActiveDim] = useState<string>("trend");
  const mutation = useMutation({
    mutationFn: (req: RegimeSplitRequest) => runRegimesSplit(req),
  });

  const rows = mutation.data?.regimes ?? [];
  const dims = useMemo(
    () => Array.from(new Set(rows.map((r) => r.dimension))),
    [rows],
  );
  const chartData = useMemo(
    () => rows.filter((r) => r.dimension === activeDim),
    [rows, activeDim],
  );

  const columns = useMemo(
    () => [
      helper.accessor("dimension", { header: "Dimension" }),
      helper.accessor("regime", {
        header: "Regime",
        cell: (info) => (
          <span className="font-mono">{info.getValue()}</span>
        ),
      }),
      helper.accessor("n_bars", { header: "# Bars" }),
      helper.accessor("total_return", {
        header: "Total Return",
        cell: (info) => fmtPct(info.getValue()),
      }),
      helper.accessor("sharpe", {
        header: "Sharpe",
        cell: (info) => fmtNum(info.getValue()),
      }),
      helper.accessor("max_drawdown", {
        header: "Max Drawdown",
        cell: (info) => fmtPct(info.getValue()),
      }),
      helper.accessor("exposure", {
        header: "Exposure",
        cell: (info) => fmtPct(info.getValue()),
      }),
    ],
    [],
  );

  const table = useReactTable({
    data: rows,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  function onRun() {
    if (!request) return;
    mutation.mutate(request);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={onRun}
          disabled={!request || mutation.isPending}
          className="bg-indigo-600 text-white text-sm font-medium rounded px-3 py-1.5 hover:bg-indigo-700 disabled:opacity-50"
        >
          {mutation.isPending ? "Loading…" : "Run regime split"}
        </button>
        {!request && (
          <span className="text-xs text-slate-500 italic">
            Run a single backtest first.
          </span>
        )}
        {mutation.isError && (
          <span className="text-xs text-red-600">
            Failed — check console / ensure SPY & VIX are backfilled.
          </span>
        )}
      </div>

      {rows.length > 0 && (
        <>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-600">Bar chart dimension:</span>
            {dims.map((d) => (
              <button
                key={d}
                onClick={() => setActiveDim(d)}
                className={`px-2 py-0.5 rounded border text-xs font-mono ${
                  activeDim === d
                    ? "bg-slate-800 text-white border-slate-800"
                    : "bg-white text-slate-700 border-slate-300"
                }`}
              >
                {d}
              </button>
            ))}
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="regime" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="sharpe" name="Sharpe">
                  {chartData.map((row, i) => (
                    <Cell key={i} fill={DIM_COLORS[row.dimension] ?? "#4f46e5"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="overflow-x-auto border border-slate-200 rounded">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-100 text-slate-700">
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id}>
                    {hg.headers.map((header) => (
                      <th key={header.id} className="px-3 py-2 text-left font-medium">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody>
                {table.getRowModel().rows.map((row) => (
                  <tr key={row.id} className="border-t border-slate-100">
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="px-3 py-2 text-slate-700">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
