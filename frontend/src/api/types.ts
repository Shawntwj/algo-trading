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

// ─── Benchmarks ───────────────────────────────────────────────────────────
export interface BenchmarkRequest {
  tickers: string[];
  start: string;
  end: string;
  interval?: string;
  weights?: "equal" | "cap";
  caps?: Record<string, number> | null;
  init_cash?: number;
  include_spy?: boolean;
}

export interface BenchmarkCurve {
  name: string;
  equity_curve: EquityPoint[];
}

export interface BenchmarkResponse {
  weights: string;
  tickers: string[];
  curves: BenchmarkCurve[];
}

// ─── Stats (PSR / Sharpe / max-DD / total-return CIs) ─────────────────────
export interface StatsRequest {
  returns: number[];
  sr_benchmark?: number;
  periods_per_year?: number;
  n_resamples?: number;
  alpha?: number;
  seed?: number;
}

export interface CIBlock {
  point: number | null;
  low: number | null;
  high: number | null;
}

export interface StatsResponse {
  sharpe: number | null;
  sharpe_ci: CIBlock;
  psr: number | null;
  max_dd: number | null;
  max_dd_ci: CIBlock;
  total_return: number | null;
  total_return_ci: CIBlock;
}

// ─── Walk-forward ─────────────────────────────────────────────────────────
export interface WalkForwardRequest {
  tickers: string[];
  start: string;
  end: string;
  interval?: string;
  strategy: string;
  grid?: Record<string, unknown[]>;
  train_size: number;
  test_size: number;
  step?: number | null;
  mode?: "expanding" | "rolling";
  min_train?: number | null;
  periods_per_year?: number;
  commission?: number;
  slippage?: number;
  n_resamples?: number;
  alpha?: number;
  seed?: number;
}

export interface FoldEntry {
  fold_idx: number;
  train_start: number;
  train_end: number;
  test_start: number;
  test_end: number;
  in_sample_sharpe: number | null;
  out_of_sample_sharpe: number | null;
  selected_params: Record<string, unknown>;
}

export interface WalkForwardResponse {
  n_folds: number;
  oos_sharpe_mean: number | null;
  oos_sharpe_ci: CIBlock;
  decay_slope: number | null;
  is_vs_oos: Array<[number | null, number | null]>;
  folds: FoldEntry[];
}

// ─── Attribution (CAPM) ───────────────────────────────────────────────────
export interface AttributionRequest {
  strategy_returns: number[];
  market_returns: number[];
  risk_free?: number;
  periods_per_year?: number;
}

export interface AttributionResponse {
  alpha: number;
  alpha_annualised: number;
  beta: number;
  alpha_t_stat: number | null;
  r_squared: number | null;
  n_obs: number;
}

// ─── Regime split ─────────────────────────────────────────────────────────
export interface RegimeSplitRequest {
  tickers: string[];
  start: string;
  end: string;
  interval?: string;
  strategy: string;
  params?: Record<string, unknown>;
  commission?: number;
  slippage?: number;
  spy_ticker?: string;
  vix_ticker?: string;
}

export interface RegimeStat {
  dimension: string;
  regime: string;
  n_bars: number;
  total_return: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  exposure: number | null;
}

export interface RegimeSplitResponse {
  strategy: string;
  regimes: RegimeStat[];
}

// ─── Combined-explainable (Task 4a) ───────────────────────────────────────
// Mirrors api/schemas.py::TradeExplanationModel. The `direction` field is a
// stringly-typed enum on the backend (long_entry | long_exit | short_entry |
// short_exit); we narrow it here for autocomplete but keep `string` as the
// escape hatch if the backend grows new directions.
export type TradeDirection =
  | "long_entry"
  | "long_exit"
  | "short_entry"
  | "short_exit";

export interface TradeExplanation {
  ticker: string;
  timestamp: string;
  direction: TradeDirection | string;
  weights: Record<string, number>;
  child_signals: Record<string, number>;
  summary: string;
}

// /backtest/explain returns the standard BacktestResponse fields PLUS the
// `explanations` array.
export interface BacktestExplainResponse extends BacktestResponse {
  explanations: TradeExplanation[];
}

// /strategies/{name}/explanation_schema → JSON Schema describing one
// TradeExplanation entry, plus the child-strategy names the explanation will
// reference. 404 for strategies that don't implement the explanation contract.
export interface ExplanationSchema {
  strategy: string;
  schema: Record<string, unknown>;
  children: string[];
}
