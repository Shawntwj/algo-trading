import { useMemo, useState } from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";

import type { SweepEntry } from "../api/types";

interface SweepTableProps {
  results: SweepEntry[];
  paramKeys: string[];
}

interface Row {
  label: string;
  sharpe: number | null;
  total_return: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  n_trades: number | null;
  exposure: number | null;
  params: Record<string, unknown>;
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

export default function SweepTable({ results, paramKeys }: SweepTableProps) {
  const data = useMemo<Row[]>(
    () =>
      results.map((r) => ({
        label: r.label,
        sharpe: num(r.metrics.sharpe),
        total_return: num(r.metrics.total_return),
        max_drawdown: num(r.metrics.max_drawdown),
        win_rate: num(r.metrics.win_rate),
        n_trades: num(r.metrics.n_trades),
        exposure: num(r.metrics.exposure),
        params: r.params,
      })),
    [results],
  );

  const columns = useMemo<ColumnDef<Row, unknown>[]>(() => {
    const paramCols: ColumnDef<Row, unknown>[] = paramKeys.map((k) => ({
      id: `param_${k}`,
      header: k,
      accessorFn: (r) => r.params[k],
      cell: (info) => (
        <span className="font-mono text-slate-700">{String(info.getValue() ?? "—")}</span>
      ),
      sortingFn: "alphanumeric",
    }));
    const metricCols: ColumnDef<Row, unknown>[] = [
      {
        id: "sharpe",
        header: "Sharpe",
        accessorFn: (r) => r.sharpe,
        cell: (info) => fmtNum(info.getValue() as number | null),
        sortingFn: "basic",
      },
      {
        id: "total_return",
        header: "Total Return",
        accessorFn: (r) => r.total_return,
        cell: (info) => fmtPct(info.getValue() as number | null),
        sortingFn: "basic",
      },
      {
        id: "max_drawdown",
        header: "Max Drawdown",
        accessorFn: (r) => r.max_drawdown,
        cell: (info) => fmtPct(info.getValue() as number | null),
        sortingFn: "basic",
      },
      {
        id: "win_rate",
        header: "Win Rate",
        accessorFn: (r) => r.win_rate,
        cell: (info) => fmtPct(info.getValue() as number | null),
        sortingFn: "basic",
      },
      {
        id: "n_trades",
        header: "# Trades",
        accessorFn: (r) => r.n_trades,
        cell: (info) => {
          const v = info.getValue() as number | null;
          return v === null ? "—" : String(Math.round(v));
        },
        sortingFn: "basic",
      },
      {
        id: "exposure",
        header: "Exposure",
        accessorFn: (r) => r.exposure,
        cell: (info) => fmtPct(info.getValue() as number | null),
        sortingFn: "basic",
      },
    ];
    return [...paramCols, ...metricCols];
  }, [paramKeys]);

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
    return <div className="text-sm text-slate-400 italic">No sweep results.</div>;
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
