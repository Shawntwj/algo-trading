import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import SignificancePanel from "../components/SignificancePanel";
import type { RunResult } from "../App";

function renderWithClient(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function makeSingleRun(): RunResult {
  return {
    mode: "single",
    request: {
      tickers: ["AAPL"],
      start: "2022-01-01",
      end: "2023-01-01",
      strategy: "ma_crossover",
      params: {},
    },
    response: {
      strategy: "ma_crossover",
      params: {},
      label: "ma_crossover(fast=10,slow=30)",
      portfolio_metrics: {},
      results: [
        {
          ticker: "AAPL",
          metrics: {},
          equity_curve: Array.from({ length: 20 }, (_, i) => ({
            timestamp: `2022-01-${String(i + 1).padStart(2, "0")}`,
            value: 100 + i * 0.5,
          })),
          entries: [],
          exits: [],
        },
      ],
    },
  };
}

describe("SignificancePanel", () => {
  it("renders DSR badge once /stats resolves", async () => {
    renderWithClient(<SignificancePanel lastResult={makeSingleRun()} />);

    // PSR card with the mock value (0.97 → "0.97") arrives after /stats resolves.
    expect(await screen.findAllByText("0.97")).not.toHaveLength(0);
    expect(screen.getAllByText(/DSR/).length).toBeGreaterThan(0);
  });

  it("prompts user to run a backtest when none exists", () => {
    renderWithClient(<SignificancePanel lastResult={null} />);
    expect(screen.getByText(/run a backtest first/i)).toBeInTheDocument();
  });
});
