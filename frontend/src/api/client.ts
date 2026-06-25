import axios from "axios";

import type {
  AttributionRequest,
  AttributionResponse,
  BacktestExplainResponse,
  BacktestRequest,
  BacktestResponse,
  BenchmarkRequest,
  BenchmarkResponse,
  ExplanationSchema,
  HealthResponse,
  RegimeSplitRequest,
  RegimeSplitResponse,
  StatsRequest,
  StatsResponse,
  StrategyInfo,
  SweepRequest,
  SweepResponse,
  WalkForwardRequest,
  WalkForwardResponse,
} from "./types";

const BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export const http = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 60_000,
});

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>("/health");
  return data;
}

export async function getTickers(): Promise<string[]> {
  const { data } = await http.get<string[]>("/tickers");
  return data;
}

export async function getStrategies(): Promise<StrategyInfo[]> {
  const { data } = await http.get<StrategyInfo[]>("/strategies");
  return data;
}

export async function runBacktest(req: BacktestRequest): Promise<BacktestResponse> {
  const { data } = await http.post<BacktestResponse>("/backtest", req);
  return data;
}

export async function runSweep(req: SweepRequest): Promise<SweepResponse> {
  const { data } = await http.post<SweepResponse>("/sweep", req);
  return data;
}

export async function runBenchmarks(
  req: BenchmarkRequest,
): Promise<BenchmarkResponse> {
  const { data } = await http.post<BenchmarkResponse>("/benchmarks", req);
  return data;
}

export async function runStats(req: StatsRequest): Promise<StatsResponse> {
  const { data } = await http.post<StatsResponse>("/stats", req);
  return data;
}

export async function runWalkforward(
  req: WalkForwardRequest,
): Promise<WalkForwardResponse> {
  const { data } = await http.post<WalkForwardResponse>("/walkforward", req);
  return data;
}

export async function runAttribution(
  req: AttributionRequest,
): Promise<AttributionResponse> {
  const { data } = await http.post<AttributionResponse>("/attribution", req);
  return data;
}

export async function runRegimesSplit(
  req: RegimeSplitRequest,
): Promise<RegimeSplitResponse> {
  const { data } = await http.post<RegimeSplitResponse>("/regimes/split", req);
  return data;
}

// ─── Explainable backtest (Task 4) ────────────────────────────────────────
// POST /backtest/explain — same body as /backtest, only valid for
// `combined_explainable`. Returns the standard backtest payload plus a
// per-trade `explanations` array.
export async function backtestExplain(
  req: BacktestRequest,
): Promise<BacktestExplainResponse> {
  const { data } = await http.post<BacktestExplainResponse>(
    "/backtest/explain",
    req,
  );
  return data;
}

// GET /strategies/{name}/explanation_schema — 404 for strategies that don't
// expose an explanation contract. We swallow the 404 to `null` so callers can
// treat "no schema" as the common case instead of an error.
export async function strategyExplanationSchema(
  name: string,
): Promise<ExplanationSchema | null> {
  try {
    const { data } = await http.get<ExplanationSchema>(
      `/strategies/${encodeURIComponent(name)}/explanation_schema`,
    );
    return data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err) && err.response?.status === 404) {
      return null;
    }
    throw err;
  }
}
