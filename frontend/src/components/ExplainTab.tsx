import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";

import type { SelectedTrade } from "../App";
import type { TradeExplanation } from "../api/types";

interface ExplainTabProps {
  explanations: TradeExplanation[] | undefined;
  selectedTrade: SelectedTrade | null;
  onSelectTrade: (t: SelectedTrade | null) => void;
}

interface TradeRow {
  ticker: string;
  timestamp: string;
  direction: string;
  childCount: number;
  raw: TradeExplanation;
}

const helper = createColumnHelper<TradeRow>();

// One trade rendered as a single journal entry (mirrors backtest/
// explainability.py::to_journal's "markdown" branch for one explanation).
// Kept inline so the Copy-as-Markdown button has no extra dependencies.
export function tradeToMarkdown(e: TradeExplanation): string {
  const lines: string[] = [];
  lines.push(`# Trade Journal`);
  lines.push("");
  lines.push(`## ${e.ticker}`);
  lines.push("");
  lines.push(`### ${e.timestamp} — ${e.direction}`);
  lines.push("");
  lines.push(e.summary || "_(no summary)_");
  lines.push("");
  const childNames = Object.keys(e.weights);
  const sorted = [...childNames].sort((a, b) => {
    const aw = (e.weights[a] ?? 0) * (e.child_signals[a] ?? 0);
    const bw = (e.weights[b] ?? 0) * (e.child_signals[b] ?? 0);
    return Math.abs(bw) - Math.abs(aw);
  });
  for (const c of sorted) {
    const w = e.weights[c] ?? 0;
    const s = e.child_signals[c] ?? 0;
    const wStr = w.toFixed(3);
    const sStr = (s >= 0 ? "+" : "") + s.toFixed(3);
    const cStr = (w * s >= 0 ? "+" : "") + (w * s).toFixed(3);
    lines.push(`- ${c}: weight=${wStr}, signal=${sStr}, contribution=${cStr}`);
  }
  return lines.join("\n") + "\n";
}

function tradeKey(t: { ticker: string; timestamp: string; direction: string }): string {
  return `${t.ticker}|${t.timestamp}|${t.direction}`;
}

function findExplanation(
  all: TradeExplanation[],
  sel: SelectedTrade | null,
): TradeExplanation | null {
  if (!sel) return null;
  const k = tradeKey(sel);
  return all.find((e) => tradeKey(e) === k) ?? null;
}

interface ContribBar {
  name: string;
  contribution: number;
  weight: number;
  signal: number;
}

function contribRows(e: TradeExplanation): ContribBar[] {
  const out: ContribBar[] = [];
  for (const c of Object.keys(e.weights)) {
    const w = e.weights[c] ?? 0;
    const s = e.child_signals[c] ?? 0;
    out.push({ name: c, contribution: w * s, weight: w, signal: s });
  }
  out.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return out;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        try {
          // navigator.clipboard is not available in jsdom by default; fall back
          // to a no-op so tests don't blow up. Real browsers always have it.
          if (
            typeof navigator !== "undefined" &&
            navigator.clipboard &&
            typeof navigator.clipboard.writeText === "function"
          ) {
            await navigator.clipboard.writeText(text);
          }
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // ignore
        }
      }}
      className="px-2 py-1 text-xs rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-50"
    >
      {copied ? "Copied!" : "Copy as Markdown"}
    </button>
  );
}

export default function ExplainTab({
  explanations,
  selectedTrade,
  onSelectTrade,
}: ExplainTabProps) {
  const data = useMemo<TradeRow[]>(
    () =>
      (explanations ?? []).map((e) => ({
        ticker: e.ticker,
        timestamp: e.timestamp,
        direction: e.direction,
        childCount: Object.keys(e.weights).length,
        raw: e,
      })),
    [explanations],
  );

  const columns = useMemo(
    () => [
      helper.accessor("ticker", {
        header: "Ticker",
        cell: (info) => (
          <span className="font-mono text-slate-800">{info.getValue()}</span>
        ),
      }),
      helper.accessor("timestamp", {
        header: "Timestamp",
        cell: (info) => (
          <span className="font-mono text-xs text-slate-700">
            {info.getValue()}
          </span>
        ),
      }),
      helper.accessor("direction", {
        header: "Direction",
        cell: (info) => (
          <span
            className={
              info.getValue().endsWith("_entry")
                ? "text-emerald-700"
                : "text-rose-700"
            }
          >
            {info.getValue()}
          </span>
        ),
      }),
      helper.accessor("childCount", {
        header: "# Children",
        cell: (info) => info.getValue(),
      }),
    ],
    [],
  );

  const [sorting, setSorting] = useState<SortingState>([
    { id: "timestamp", desc: false },
  ]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const current = findExplanation(explanations ?? [], selectedTrade);

  if (!explanations || explanations.length === 0) {
    return (
      <div className="text-sm text-slate-500 italic">
        No explanations available — run a{" "}
        <span className="font-mono">combined_explainable</span> backtest first.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <section>
        <h3 className="text-sm font-medium text-slate-700 mb-2">
          Trades ({data.length})
        </h3>
        <div
          className="overflow-x-auto border border-slate-200 rounded"
          data-testid="explain-trade-table"
        >
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
              {table.getRowModel().rows.map((row) => {
                const sel =
                  selectedTrade &&
                  tradeKey(row.original) === tradeKey(selectedTrade);
                return (
                  <tr
                    key={row.id}
                    onClick={() =>
                      onSelectTrade({
                        ticker: row.original.ticker,
                        timestamp: row.original.timestamp,
                        direction: row.original.direction,
                      })
                    }
                    className={`border-t border-slate-100 cursor-pointer ${
                      sel ? "bg-indigo-50" : "hover:bg-slate-50"
                    }`}
                    data-testid="explain-trade-row"
                  >
                    {row.getVisibleCells().map((cell) => (
                      <td
                        key={cell.id}
                        className="px-3 py-2 text-slate-700 align-top"
                      >
                        {flexRender(
                          cell.column.columnDef.cell,
                          cell.getContext(),
                        )}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h3 className="text-sm font-medium text-slate-700 mb-2">
          Selected trade
        </h3>
        {!current && (
          <div className="text-sm text-slate-500 italic">
            Select a trade marker on the price chart to see its explanation.
          </div>
        )}
        {current && (
          <div className="space-y-4" data-testid="explain-detail">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p
                  className="text-base text-slate-900"
                  data-testid="explain-summary"
                >
                  {current.summary || "(no summary)"}
                </p>
                <p className="text-xs text-slate-500 mt-1 font-mono">
                  {current.ticker} · {current.timestamp} · {current.direction}
                </p>
              </div>
              <CopyButton text={tradeToMarkdown(current)} />
            </div>

            <div className="h-56 w-full" data-testid="explain-bar-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={contribRows(current)}
                  margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="name"
                    stroke="#64748b"
                    fontSize={11}
                    interval={0}
                    angle={-20}
                    height={50}
                    textAnchor="end"
                  />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip
                    formatter={(v) =>
                      typeof v === "number" ? v.toFixed(3) : String(v)
                    }
                  />
                  <Bar dataKey="contribution" isAnimationActive={false}>
                    {contribRows(current).map((r) => (
                      <Cell
                        key={r.name}
                        fill={r.contribution >= 0 ? "#059669" : "#dc2626"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="overflow-x-auto border border-slate-200 rounded">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-100 text-slate-700">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Child</th>
                    <th className="px-3 py-2 text-right font-medium">Weight</th>
                    <th className="px-3 py-2 text-right font-medium">Signal</th>
                    <th className="px-3 py-2 text-right font-medium">
                      Contribution
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {contribRows(current).map((r) => (
                    <tr key={r.name} className="border-t border-slate-100">
                      <td className="px-3 py-2 font-mono text-slate-800">
                        {r.name}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-700 font-mono">
                        {r.weight.toFixed(3)}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-700 font-mono">
                        {(r.signal >= 0 ? "+" : "") + r.signal.toFixed(3)}
                      </td>
                      <td
                        className={`px-3 py-2 text-right font-mono ${
                          r.contribution >= 0
                            ? "text-emerald-700"
                            : "text-rose-700"
                        }`}
                      >
                        {(r.contribution >= 0 ? "+" : "") +
                          r.contribution.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
