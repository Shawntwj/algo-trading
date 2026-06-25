import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";

import { runStats } from "../api/client";
import type { CIBlock, StatsResponse, TickerBacktest } from "../api/types";

interface MetricsTableProps {
  results: TickerBacktest[];
}

interface Row {
  ticker: string;
  total_return: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  win_rate: number | null;
  ci?: StatsResponse;
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

function fmtCI(block: CIBlock | undefined, opts: { pct?: boolean }): string {
  if (!block) return "";
  const f = opts.pct ? fmtPct : (v: number | null) => fmtNum(v);
  return `95% CI [${f(num(block.low))}, ${f(num(block.high))}]`;
}

interface HoverCellProps {
  display: string;
  tooltip: string | null;
}

function HoverCell({ display, tooltip }: HoverCellProps) {
  if (!tooltip) return <>{display}</>;
  return (
    <span className="relative inline-flex items-center gap-1 group cursor-help">
      <span>{display}</span>
      <span className="text-[10px] text-slate-400 border border-slate-300 rounded-full w-3 h-3 inline-flex items-center justify-center leading-none">
        ?
      </span>
      <span className="pointer-events-none absolute z-10 left-0 top-full mt-1 hidden group-hover:block whitespace-nowrap rounded bg-slate-800 px-2 py-1 text-xs text-white shadow-md">
        {tooltip}
      </span>
    </span>
  );
}

const helper = createColumnHelper<Row>();

export default function MetricsTable({ results }: MetricsTableProps) {
  // Fire /stats per ticker in parallel; on failure, render the point estimate
  // without a "?" badge. Don't block initial paint.
  const statsQueries = useQueries({
    queries: results.map((r) => ({
      queryKey: ["stats", r.ticker, r.equity_curve.length],
      queryFn: () =>
        runStats({
          returns: equityToReturns(r.equity_curve),
          n_resamples: 500,
          periods_per_year: 252,
        }),
      enabled: r.equity_curve.length > 2,
      retry: false,
      staleTime: 5 * 60_000,
    })),
  });

  const data = useMemo<Row[]>(
    () =>
      results.map((r, i) => ({
        ticker: r.ticker,
        total_return: num(r.metrics.total_return),
        sharpe: num(r.metrics.sharpe),
        max_drawdown: num(r.metrics.max_drawdown),
        win_rate: num(r.metrics.win_rate),
        ci: statsQueries[i]?.data,
      })),
    [results, statsQueries],
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
        cell: (info) => (
          <HoverCell
            display={fmtPct(info.getValue())}
            tooltip={
              info.row.original.ci
                ? fmtCI(info.row.original.ci.total_return_ci, { pct: true })
                : null
            }
          />
        ),
        sortingFn: "basic",
      }),
      helper.accessor("sharpe", {
        header: "Sharpe",
        cell: (info) => (
          <HoverCell
            display={fmtNum(info.getValue())}
            tooltip={
              info.row.original.ci
                ? fmtCI(info.row.original.ci.sharpe_ci, {})
                : null
            }
          />
        ),
        sortingFn: "basic",
      }),
      helper.accessor("max_drawdown", {
        header: "Max Drawdown",
        cell: (info) => (
          <HoverCell
            display={fmtPct(info.getValue())}
            tooltip={
              info.row.original.ci
                ? fmtCI(info.row.original.ci.max_dd_ci, { pct: true })
                : null
            }
          />
        ),
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
                <td key={cell.id} className="px-3 py-2 text-slate-700 align-top">
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
