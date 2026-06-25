import type { BacktestResponse } from "../api/types";

interface MainProps {
  lastResponse: BacktestResponse | null;
}

export default function Main({ lastResponse }: MainProps) {
  return (
    <main className="flex-1 p-6 overflow-auto bg-white">
      <h1 className="text-xl font-semibold text-slate-800 mb-2">
        Algo Trading Research
      </h1>
      <p className="text-sm text-slate-500 mb-4">
        Charts and metrics land in Task 1c. For now, the latest `/backtest`
        response is dumped below so we can confirm the wiring.
      </p>

      {!lastResponse && (
        <div className="text-sm text-slate-400 italic">
          No response yet — pick tickers and a strategy, then hit Run.
        </div>
      )}

      {lastResponse && (
        <pre className="text-xs bg-slate-900 text-slate-100 p-4 rounded overflow-auto max-h-[80vh]">
          {JSON.stringify(lastResponse, null, 2)}
        </pre>
      )}
    </main>
  );
}
