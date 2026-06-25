import type { RunResult } from "../App";

interface SignificancePanelProps {
  lastResult: RunResult | null;
}

export default function SignificancePanel({ lastResult }: SignificancePanelProps) {
  if (!lastResult) {
    return (
      <div className="text-sm text-slate-500 italic">
        Run a backtest first.
      </div>
    );
  }
  return (
    <div className="text-sm text-slate-500 italic">Coming up in Step E.</div>
  );
}
