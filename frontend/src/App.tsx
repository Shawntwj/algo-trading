import { useState } from "react";

import Main from "./components/Main";
import Sidebar from "./components/Sidebar";
import type { BacktestResponse, SweepResponse } from "./api/types";

export type RunResult =
  | { mode: "single"; response: BacktestResponse }
  | { mode: "sweep"; response: SweepResponse; sweptKeys: string[] };

export default function App() {
  const [lastResult, setLastResult] = useState<RunResult | null>(null);

  return (
    <div className="flex h-screen w-screen bg-white text-slate-900">
      <Sidebar onResult={setLastResult} />
      <Main lastResult={lastResult} />
    </div>
  );
}
