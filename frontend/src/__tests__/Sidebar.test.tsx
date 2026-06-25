import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

import Sidebar from "../components/Sidebar";

function renderWithClient(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
  );
}

describe("Sidebar", () => {
  it("loads tickers + strategies and renders the params form", async () => {
    const onResult = vi.fn();
    renderWithClient(<Sidebar onResult={onResult} />);

    // Tickers from the MSW mock arrive asynchronously.
    expect(await screen.findByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("MSFT")).toBeInTheDocument();
    expect(screen.getByText("NVDA")).toBeInTheDocument();

    // Strategy auto-selects the first one and its params form renders.
    await waitFor(() => {
      expect(screen.getByTestId("params-form")).toBeInTheDocument();
    });
    expect(screen.getByText("fast")).toBeInTheDocument();
    expect(screen.getByText("slow")).toBeInTheDocument();
  });
});
