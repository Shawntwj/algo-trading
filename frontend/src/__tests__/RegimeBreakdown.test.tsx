import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";

import RegimeBreakdown from "../components/RegimeBreakdown";

function renderWithClient(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("RegimeBreakdown", () => {
  it("fetches and renders per-regime rows on click", async () => {
    const req = {
      tickers: ["AAPL"],
      start: "2022-01-01",
      end: "2023-01-01",
      strategy: "ma_crossover",
      params: { fast: 10, slow: 30 },
    };
    renderWithClient(<RegimeBreakdown request={req} />);

    await userEvent.click(screen.getByRole("button", { name: /run regime split/i }));

    expect(await screen.findByText("bull")).toBeInTheDocument();
    expect(screen.getByText("bear")).toBeInTheDocument();
  });
});
