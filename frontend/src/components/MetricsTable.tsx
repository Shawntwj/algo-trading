import { useMemo } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";

import type { TickerBacktest } from "../api/types";

interface MetricsTableProps {
  results: TickerBacktest[];
}

interface Row {
  ticker: string;
  total_return: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
}

function num(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtPct(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(2)}%`;
}

function fmtNum(v: number | null, digits = 3): string {
  return v === null ? "—" : v.toFixed(digits);
}

const helper = createColumnHelper<Row>();

export default function MetricsTable({ results }: MetricsTableProps) {
  const data = useMemo<Row[]>(
    () =>
      results.map((r) => ({
        ticker: r.ticker,
        total_return: num(r.metrics.total_return),
        sharpe: num(r.metrics.sharpe),
        max_drawdown: num(r.metrics.max_drawdown),
        win_rate: num(r.metrics.win_rate),
      })),
    [results],
  );

  const columns = useMemo(
    () => [
      helper.accessor("ticker", {
        header: "Ticker",
        cell: (info) => (
          <span className="font-mono text-slate-800">{info.getValue()}</span>
        ),
      }),
      helper.accessor("total_return", {
        header: "Total Return",
        cell: (info) => fmtPct(info.getValue()),
        sortingFn: "basic",
      }),
      helper.accessor("sharpe", {
        header: "Sharpe",
        cell: (info) => fmtNum(info.getValue()),
        sortingFn: "basic",
      }),
      helper.accessor("max_drawdown", {
        header: "Max Drawdown",
        cell: (info) => fmtPct(info.getValue()),
        sortingFn: "basic",
      }),
      helper.accessor("win_rate", {
        header: "Win Rate",
        cell: (info) => fmtPct(info.getValue()),
        sortingFn: "basic",
      }),
    ],
    [],
  );

  const [sorting, setSorting] = useState<SortingState>([
    { id: "sharpe", desc: true },
  ]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (data.length === 0) {
    return (
      <div className="text-sm text-slate-400 italic">No per-ticker metrics.</div>
    );
  }

  return (
    <div className="overflow-x-auto border border-slate-200 rounded">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-100 text-slate-700">
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((header) => {
                const sortDir = header.column.getIsSorted();
                return (
                  <th
                    key={header.id}
                    className="px-3 py-2 text-left font-medium cursor-pointer select-none"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext(),
                    )}
                    {sortDir === "asc" && " ▲"}
                    {sortDir === "desc" && " ▼"}
                  </th>
                );
              })}
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
  );
}
