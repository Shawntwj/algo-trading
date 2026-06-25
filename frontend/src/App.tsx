import { useState } from "react";

import Main from "./components/Main";
import Sidebar from "./components/Sidebar";
import type {
  BacktestRequest,
  BacktestResponse,
  SweepRequest,
  SweepResponse,
} from "./api/types";

export type RunResult =
  | {
      mode: "single";
      response: BacktestResponse;
      request: BacktestRequest;
    }
  | {
      mode: "sweep";
      response: SweepResponse;
      request: SweepRequest;
      sweptKeys: string[];
    };

export default function App() {
  const [lastResult, setLastResult] = useState<RunResult | null>(null);

  return (
    <div className="flex h-screen w-screen bg-white text-slate-900">
      <Sidebar onResult={setLastResult} />
      <Main lastResult={lastResult} />
    </div>
  );
}
