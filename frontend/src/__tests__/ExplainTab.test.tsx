import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

import ExplainTab, { tradeToMarkdown } from "../components/ExplainTab";
import type { TradeExplanation } from "../api/types";
import type { SelectedTrade } from "../App";

function renderWithClient(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
  );
}

// Two trades, three child signals — the smallest payload that exercises the
// table render, the row-click → detail-panel selection, and the bar chart
// sort by |weight × signal|.
const EXPLANATIONS: TradeExplanation[] = [
  {
    ticker: "AAPL",
    timestamp: "2023-03-15T00:00:00",
    direction: "long_entry",
    weights: { momentum: 0.4, mean_rev: 0.3, breakout: 0.3 },
    child_signals: { momentum: 0.9, mean_rev: -0.2, breakout: 0.6 },
    summary:
      "Entered AAPL long: momentum + breakout fired together, mean-rev mildly disagreed.",
  },
  {
    ticker: "AAPL",
    timestamp: "2023-04-20T00:00:00",
    direction: "long_exit",
    weights: { momentum: 0.4, mean_rev: 0.3, breakout: 0.3 },
    child_signals: { momentum: -0.5, mean_rev: 0.8, breakout: -0.1 },
    summary: "Exited AAPL: mean-reversion flipped positive, momentum decayed.",
  },
];

describe("ExplainTab", () => {
  it("renders one row per trade and shows the selected trade's detail", async () => {
    const user = userEvent.setup();
    const onSelectTrade = vi.fn();
    let selected: SelectedTrade | null = null;

    const { rerender } = renderWithClient(
      <ExplainTab
        explanations={EXPLANATIONS}
        selectedTrade={selected}
        onSelectTrade={(t) => {
          selected = t;
          onSelectTrade(t);
        }}
      />,
    );

    // Two rows in the trade list.
    const rows = screen.getAllByTestId("explain-trade-row");
    expect(rows).toHaveLength(2);

    // Empty-state copy in the right pane.
    expect(
      screen.getByText(/select a trade marker on the price chart/i),
    ).toBeInTheDocument();

    // Click the first row → onSelectTrade fires with the matching key.
    await user.click(rows[0]);
    expect(onSelectTrade).toHaveBeenCalledTimes(1);
    expect(onSelectTrade).toHaveBeenCalledWith({
      ticker: "AAPL",
      timestamp: "2023-03-15T00:00:00",
      direction: "long_entry",
    });

    // Re-render with the now-selected trade and assert the detail panel +
    // bar chart show up.
    rerender(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ExplainTab
          explanations={EXPLANATIONS}
          selectedTrade={selected}
          onSelectTrade={onSelectTrade}
        />
      </QueryClientProvider>,
    );

    const detail = await screen.findByTestId("explain-detail");
    expect(
      within(detail).getByTestId("explain-summary").textContent,
    ).toContain("Entered AAPL long");
    expect(screen.getByTestId("explain-bar-chart")).toBeInTheDocument();
    // The child names appear in the key-value table.
    expect(within(detail).getAllByText("momentum").length).toBeGreaterThan(0);
    expect(within(detail).getAllByText("mean_rev").length).toBeGreaterThan(0);
    expect(within(detail).getAllByText("breakout").length).toBeGreaterThan(0);
  });

  it("falls back to the empty state when no explanations are provided", () => {
    renderWithClient(
      <ExplainTab
        explanations={undefined}
        selectedTrade={null}
        onSelectTrade={() => undefined}
      />,
    );
    // Empty-state copy is split across a <span>; assert each fragment.
    expect(screen.getByText(/no explanations available/i)).toBeInTheDocument();
    expect(screen.getByText("combined_explainable")).toBeInTheDocument();
  });

  it("tradeToMarkdown renders the headline summary + sorted child bullets", () => {
    const md = tradeToMarkdown(EXPLANATIONS[0]);
    expect(md).toContain("# Trade Journal");
    expect(md).toContain("## AAPL");
    expect(md).toContain("### 2023-03-15T00:00:00 — long_entry");
    expect(md).toContain("Entered AAPL long");
    // Highest absolute contribution should appear first (momentum: 0.4*0.9).
    const lines = md.split("\n").filter((l) => l.startsWith("- "));
    expect(lines[0]).toContain("momentum");
  });
});
