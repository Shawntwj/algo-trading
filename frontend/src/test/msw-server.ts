import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";

const BASE = "http://localhost:8000";

export const handlers = [
  http.get(`${BASE}/tickers`, () =>
    HttpResponse.json(["AAPL", "MSFT", "NVDA"]),
  ),
  http.get(`${BASE}/strategies`, () =>
    HttpResponse.json([
      {
        name: "ma_crossover",
        default_params: { fast: 10, slow: 30 },
        param_grid: { fast: [5, 10, 20], slow: [30, 50, 100] },
      },
    ]),
  ),
];

export const server = setupServer(...handlers);
