import axios from "axios";

import type {
  BacktestRequest,
  BacktestResponse,
  HealthResponse,
  StrategyInfo,
  SweepRequest,
  SweepResponse,
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
