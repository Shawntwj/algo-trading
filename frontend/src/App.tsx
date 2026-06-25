import { useCallback, useState } from "react";

import Main from "./components/Main";
import Sidebar from "./components/Sidebar";
import type {
  BacktestRequest,
  BacktestResponse,
  SweepRequest,
  SweepResponse,
  TradeExplanation,
} from "./api/types";

export type RunResult =
  | {
      mode: "single";
      response: BacktestResponse;
      request: BacktestRequest;
      // Optional per-trade explanations — only populated when the run was
      // routed through /backtest/explain (i.e. strategy === combined_explainable).
      explanations?: TradeExplanation[];
    }
  | {
      mode: "sweep";
      response: SweepResponse;
      request: SweepRequest;
      sweptKeys: string[];
    };

// A trade is keyed by (ticker, timestamp, direction) — the same composite key
// used in the explanation list — so the click on a price-chart marker resolves
// to exactly one TradeExplanation.
export interface SelectedTrade {
  ticker: string;
  timestamp: string;
  direction: string;
}

export default function App() {
  const [lastResult, setLastResult] = useState<RunResult | null>(null);
  const [selectedTrade, setSelectedTrade] = useState<SelectedTrade | null>(null);

  // Reset the trade selection whenever a new run lands — stale selections from
  // a previous strategy/window would point at trades that no longer exist.
  const onResult = useCallback((r: RunResult) => {
    setLastResult(r);
    setSelectedTrade(null);
  }, []);

  return (
    <div className="flex h-screen w-screen bg-white text-slate-900">
      <Sidebar onResult={onResult} />
      <Main
        lastResult={lastResult}
        selectedTrade={selectedTrade}
        onSelectTrade={setSelectedTrade}
      />
    </div>
  );
}
