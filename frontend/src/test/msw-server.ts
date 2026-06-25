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
  http.post(`${BASE}/stats`, () =>
    HttpResponse.json({
      sharpe: 1.42,
      sharpe_ci: { point: 1.42, low: 0.91, high: 1.93 },
      psr: 0.97,
      max_dd: -0.12,
      max_dd_ci: { point: -0.12, low: -0.18, high: -0.07 },
      total_return: 0.21,
      total_return_ci: { point: 0.21, low: 0.12, high: 0.29 },
    }),
  ),
  http.post(`${BASE}/regimes/split`, () =>
    HttpResponse.json({
      strategy: "ma_crossover",
      regimes: [
        {
          dimension: "trend",
          regime: "bull",
          n_bars: 120,
          total_return: 0.08,
          sharpe: 1.1,
          max_drawdown: -0.05,
          exposure: 0.6,
        },
        {
          dimension: "trend",
          regime: "bear",
          n_bars: 60,
          total_return: -0.03,
          sharpe: -0.4,
          max_drawdown: -0.12,
          exposure: 0.55,
        },
        {
          dimension: "vol",
          regime: "low",
          n_bars: 50,
          total_return: 0.04,
          sharpe: 1.5,
          max_drawdown: -0.02,
          exposure: 0.5,
        },
      ],
    }),
  ),
  http.post(`${BASE}/walkforward`, () =>
    HttpResponse.json({
      n_folds: 3,
      oos_sharpe_mean: 0.8,
      oos_sharpe_ci: { point: 0.8, low: 0.3, high: 1.3 },
      decay_slope: 0.45,
      is_vs_oos: [
        [1.2, 0.9],
        [1.5, 0.7],
        [0.8, 0.8],
      ],
      folds: [
        {
          fold_idx: 0,
          train_start: 0,
          train_end: 250,
          test_start: 250,
          test_end: 310,
          in_sample_sharpe: 1.2,
          out_of_sample_sharpe: 0.9,
          selected_params: { fast: 10, slow: 30 },
        },
      ],
    }),
  ),
];

export const server = setupServer(...handlers);
