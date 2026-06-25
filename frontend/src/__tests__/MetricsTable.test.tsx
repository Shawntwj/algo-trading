import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import MetricsTable from "../components/MetricsTable";
import type { TickerBacktest } from "../api/types";

function renderWithClient(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

const sample: TickerBacktest[] = [
  {
    ticker: "AAPL",
    metrics: { total_return: 0.21, sharpe: 1.42, max_drawdown: -0.12, win_rate: 0.55 },
    equity_curve: Array.from({ length: 12 }, (_, i) => ({
      timestamp: `2024-01-${String(i + 1).padStart(2, "0")}`,
      value: 100 + i,
    })),
    entries: [],
    exits: [],
  },
];

describe("MetricsTable", () => {
  it("renders point estimates immediately and exposes a CI hover badge", async () => {
    renderWithClient(<MetricsTable results={sample} />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("1.420")).toBeInTheDocument();
    // The "?" hover indicators arrive after /stats resolves via MSW.
    expect(await screen.findAllByText("?")).not.toHaveLength(0);
  });
});
