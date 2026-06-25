import type { RunResult } from "../App";

interface OOSDecayChartProps {
  lastResult: RunResult | null;
}

export default function OOSDecayChart({ lastResult }: OOSDecayChartProps) {
  if (!lastResult) {
    return (
      <div className="text-sm text-slate-500 italic">
        Run a backtest first.
      </div>
    );
  }
  return (
    <div className="text-sm text-slate-500 italic">Coming up in Step F.</div>
  );
}
