import { useMemo, useState } from "react";

import type { RunResult, SelectedTrade } from "../App";
import type { RegimeSplitRequest } from "../api/types";
import EquityChart from "./EquityChart";
import ExplainTab from "./ExplainTab";
import MetricsTable from "./MetricsTable";
import OOSDecayChart from "./OOSDecayChart";
import PriceChart from "./PriceChart";
import RegimeBreakdown from "./RegimeBreakdown";
import SharpeHeatmap from "./SharpeHeatmap";
import SignificancePanel from "./SignificancePanel";
import SweepTable from "./SweepTable";

interface MainProps {
  lastResult: RunResult | null;
  selectedTrade: SelectedTrade | null;
  onSelectTrade: (t: SelectedTrade | null) => void;
}

type TabKey =
  | "backtest"
  | "sweep"
  | "regimes"
  | "significance"
  | "walkforward"
  | "explain";

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: "backtest", label: "Backtest" },
  { key: "sweep", label: "Sweep" },
  { key: "regimes", label: "Regimes" },
  { key: "significance", label: "Significance" },
  { key: "walkforward", label: "Walk-forward" },
  { key: "explain", label: "Explain" },
];

function fmtPct(v: unknown): string {
  if (v === null || v === undefined) return "—";
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : "—";
}

function fmtNum(v: unknown, digits = 3): string {
  if (v === null || v === undefined) return "—";
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function PortfolioSummary({ metrics }: { metrics: Record<string, unknown> }) {
  const cards: Array<[string, string]> = [
    ["Sharpe", fmtNum(metrics.sharpe)],
    ["Total Return", fmtPct(metrics.total_return)],
    ["Max Drawdown", fmtPct(metrics.max_drawdown)],
    ["Win Rate", fmtPct(metrics.win_rate)],
    ["# Trades", String(metrics.n_trades ?? "—")],
    ["Exposure", fmtPct(metrics.exposure)],
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
      {cards.map(([k, v]) => (
        <div
          key={k}
          className="border border-slate-200 rounded px-3 py-2 bg-slate-50"
        >
          <div className="text-xs text-slate-500">{k}</div>
          <div className="text-base font-mono text-slate-800">{v}</div>
        </div>
      ))}
    </div>
  );
}

export default function Main({ lastResult, selectedTrade, onSelectTrade }: MainProps) {
  // Convenience: derive the explainable flag once. The Explain tab + the
  // PriceChart marker affordance both key off it.
  const explainable =
    lastResult?.mode === "single" &&
    (lastResult.explanations !== undefined ||
      lastResult.request.strategy === "combined_explainable");
  const [tab, setTab] = useState<TabKey>("backtest");

  // Build a RegimeSplit request from the latest single backtest (if any).
  const regimeReq: RegimeSplitRequest | null = useMemo(() => {
    if (!lastResult || lastResult.mode !== "single") return null;
    const r = lastResult.request;
    return {
      tickers: r.tickers,
      start: r.start,
      end: r.end,
      interval: r.interval,
      strategy: r.strategy,
      params: r.params ?? {},
      commission: r.commission,
      slippage: r.slippage,
    };
  }, [lastResult]);

  return (
    <main className="flex-1 p-6 overflow-auto bg-white">
      <h1 className="text-xl font-semibold text-slate-800 mb-4">
        Algo Trading Research
      </h1>

      <div className="border-b border-slate-200 mb-4 flex gap-1">
        {TABS.map((t) => {
          // The Explain tab is only meaningful after a combined_explainable
          // run lands — disable it otherwise so the user isn't tempted to
          // click into an empty state.
          const disabled = t.key === "explain" && !explainable;
          return (
            <button
              key={t.key}
              onClick={() => !disabled && setTab(t.key)}
              disabled={disabled}
              className={`px-3 py-1.5 text-sm border-b-2 -mb-px transition-colors ${
                tab === t.key
                  ? "border-indigo-600 text-indigo-700 font-medium"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              } ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}
              title={
                disabled
                  ? "Run combined_explainable to enable per-trade explanations"
                  : undefined
              }
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {!lastResult && (
        <div className="text-sm text-slate-400 italic">
          No run yet — pick tickers and a strategy, then hit Run.
        </div>
      )}

      {tab === "backtest" && lastResult?.mode === "single" && (
        <div className="space-y-6">
          <section>
            <h2 className="text-sm font-medium text-slate-700 mb-2">
              Portfolio summary —{" "}
              <span className="font-mono">{lastResult.response.label}</span>
            </h2>
            <PortfolioSummary metrics={lastResult.response.portfolio_metrics} />
          </section>

          <section>
            <h2 className="text-sm font-medium text-slate-700 mb-2">
              Per-ticker metrics
            </h2>
            <MetricsTable results={lastResult.response.results} />
          </section>

          <section>
            <h2 className="text-sm font-medium text-slate-700 mb-2">
              Equity curves
            </h2>
            <EquityChart results={lastResult.response.results} />
          </section>

          <section>
            <h2 className="text-sm font-medium text-slate-700 mb-2">
              Price + entry/exit markers
            </h2>
            <PriceChart
              results={lastResult.response.results}
              selectedTrade={selectedTrade}
              onSelectTrade={onSelectTrade}
              explainable={Boolean(explainable)}
            />
            {explainable && (
              <p className="text-xs text-slate-500 mt-1">
                Click a triangle marker to open its explanation in the
                <span className="font-medium"> Explain </span>
                tab.
              </p>
            )}
          </section>
        </div>
      )}

      {tab === "backtest" && lastResult?.mode === "sweep" && (
        <div className="text-sm text-slate-500 italic">
          The latest run was a sweep — switch to the "Sweep" tab.
        </div>
      )}

      {tab === "sweep" && lastResult?.mode === "sweep" && (
        <div className="space-y-6">
          <section>
            <h2 className="text-sm font-medium text-slate-700 mb-2">
              Sweep results — {lastResult.response.results.length} configs
            </h2>
            <SweepTable
              results={lastResult.response.results}
              paramKeys={lastResult.sweptKeys}
            />
          </section>

          <section>
            <h2 className="text-sm font-medium text-slate-700 mb-2">
              Sharpe heatmap
            </h2>
            {lastResult.sweptKeys.length === 2 ? (
              <SharpeHeatmap
                results={lastResult.response.results}
                paramKeys={lastResult.sweptKeys}
              />
            ) : (
              <p className="text-xs text-slate-500 italic">
                Heatmap shows when exactly 2 params are swept (got{" "}
                {lastResult.sweptKeys.length}).
              </p>
            )}
          </section>
        </div>
      )}

      {tab === "sweep" && lastResult?.mode !== "sweep" && (
        <div className="text-sm text-slate-500 italic">
          The latest run wasn't a sweep — run one in Sweep mode to populate this
          tab.
        </div>
      )}

      {tab === "regimes" && (
        <section>
          <h2 className="text-sm font-medium text-slate-700 mb-2">
            Regime breakdown
          </h2>
          <p className="text-xs text-slate-500 mb-3">
            Splits the latest single-backtest returns by trend / volatility /
            drawdown regimes (derived from SPY + VIX). Requires both symbols to
            be backfilled.
          </p>
          <RegimeBreakdown request={regimeReq} />
        </section>
      )}

      {tab === "significance" && (
        <section>
          <h2 className="text-sm font-medium text-slate-700 mb-2">
            Significance
          </h2>
          <SignificancePanel lastResult={lastResult} />
        </section>
      )}

      {tab === "walkforward" && (
        <section>
          <h2 className="text-sm font-medium text-slate-700 mb-2">
            Walk-forward
          </h2>
          <OOSDecayChart lastResult={lastResult} />
        </section>
      )}

      {tab === "explain" && (
        <section>
          <h2 className="text-sm font-medium text-slate-700 mb-2">
            Trade explanations
          </h2>
          <ExplainTab
            explanations={
              lastResult?.mode === "single"
                ? lastResult.explanations
                : undefined
            }
            selectedTrade={selectedTrade}
            onSelectTrade={onSelectTrade}
          />
        </section>
      )}
    </main>
  );
}
