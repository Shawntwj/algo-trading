import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import OOSDecayChart from "../components/OOSDecayChart";
import type { RunResult } from "../App";

function renderWithClient(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function makeRun(): RunResult {
  return {
    mode: "single",
    request: {
      tickers: ["AAPL"],
      start: "2022-01-01",
      end: "2023-01-01",
      strategy: "ma_crossover",
      params: { fast: 10, slow: 30 },
    },
    response: {
      strategy: "ma_crossover",
      params: { fast: 10, slow: 30 },
      label: "ma_crossover(fast=10,slow=30)",
      portfolio_metrics: {},
      results: [],
    },
  };
}

describe("OOSDecayChart", () => {
  it("runs walk-forward and shows aggregate stats", async () => {
    renderWithClient(<OOSDecayChart lastResult={makeRun()} />);
    await userEvent.click(screen.getByRole("button", { name: /run walk-forward/i }));

    // n_folds card shows the mock value.
    expect(await screen.findByText("3")).toBeInTheDocument();
    expect(screen.getByText(/decay slope/i)).toBeInTheDocument();
  });

  it("prompts the user when no backtest has been run", () => {
    renderWithClient(<OOSDecayChart lastResult={null} />);
    expect(screen.getByText(/run a backtest first/i)).toBeInTheDocument();
  });
});
