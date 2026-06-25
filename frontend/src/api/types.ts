// Hand-typed mirrors of api/schemas.py (Pydantic v2). Keep these in sync manually;
// see IMPROVEMENTS.md for a follow-up to share types via codegen.

export interface HealthResponse {
  status: string;
  clickhouse: string;
}

export interface StrategyInfo {
  name: string;
  default_params: Record<string, unknown>;
  param_grid: Record<string, unknown[]>;
}

export interface BacktestRequest {
  tickers: string[];
  start: string;
  end: string;
  interval?: string;
  strategy: string;
  params?: Record<string, unknown>;
  commission?: number;
  slippage?: number;
}

export interface EquityPoint {
  timestamp: string;
  value: number;
}

export interface TickerBacktest {
  ticker: string;
  metrics: Record<string, unknown>;
  equity_curve: EquityPoint[];
  entries: string[];
  exits: string[];
}

export interface BacktestResponse {
  strategy: string;
  params: Record<string, unknown>;
  label: string;
  portfolio_metrics: Record<string, unknown>;
  results: TickerBacktest[];
}

export interface SweepRequest {
  tickers: string[];
  start: string;
  end: string;
  interval?: string;
  strategy: string;
  grid?: Record<string, unknown[]>;
  commission?: number;
  slippage?: number;
}

export interface SweepEntry {
  params: Record<string, unknown>;
  label: string;
  metrics: Record<string, unknown>;
}

export interface SweepResponse {
  strategy: string;
  results: SweepEntry[];
}
