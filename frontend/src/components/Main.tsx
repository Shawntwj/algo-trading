import type { RunResult } from "../App";
import EquityChart from "./EquityChart";
import MetricsTable from "./MetricsTable";
import PriceChart from "./PriceChart";
import SharpeHeatmap from "./SharpeHeatmap";
import SweepTable from "./SweepTable";

interface MainProps {
  lastResult: RunResult | null;
}

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

export default function Main({ lastResult }: MainProps) {
  return (
    <main className="flex-1 p-6 overflow-auto bg-white">
      <h1 className="text-xl font-semibold text-slate-800 mb-4">
        Algo Trading Research
      </h1>

      {!lastResult && (
        <div className="text-sm text-slate-400 italic">
          No run yet — pick tickers and a strategy, then hit Run.
        </div>
      )}

      {lastResult?.mode === "single" && (
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
            <PriceChart results={lastResult.response.results} />
          </section>
        </div>
      )}

      {lastResult?.mode === "sweep" && (
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
    </main>
  );
}
