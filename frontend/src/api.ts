import * as authStorage from "./auth";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ?? "";

/**
 * Custom event dispatched on 401 responses so AuthContext can handle
 * navigation via React Router instead of a full-page reload.
 */
export const UNAUTHORIZED_EVENT = "app:unauthorized";

/**
 * Build an API URL relative to the configured frontend base URL.
 *
 * This keeps local/dev and deployed environments aligned without changing
 * call sites throughout the dashboard code.
 */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/**
 * Headers for JWT auth (Bearer token). Used for all protected /api/* requests.
 */
function authHeaders(): HeadersInit {
  const token = authStorage.getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Parse JSON from a response with a descriptive error on failure.
 * Wraps r.json() so a non-JSON body (HTML error page, empty body)
 * produces a clear message instead of a cryptic SyntaxError.
 */
async function safeJson<T>(r: Response): Promise<T> {
  try {
    return (await r.json()) as T;
  } catch {
    throw new Error(`Expected JSON response but got ${r.status} ${r.statusText}`);
  }
}

/**
 * Fetch with auth header and 401 handling.
 * On 401: clears token and dispatches an event so AuthContext navigates
 * to /login via React Router (no full-page reload).
 */
async function fetchWithAuth(url: string, init: RequestInit = {}): Promise<Response> {
  const headers = { ...authHeaders(), ...(init.headers as Record<string, string>) };
  const r = await fetch(url, { ...init, headers });
  if (r.status === 401) {
    authStorage.clearToken();
    window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
    throw new Error("Unauthorized");
  }
  return r;
}


export type RunSnapshotResult = {
  skipped: boolean;
  reason: string | null;
  now_et: string;
  inserted: Array<{
    target_dte: number;
    expiration: string;
    actual_dte_days: number;
    checksum: string;
  }>;
};

/**
 * Trigger a manual snapshot run through the admin endpoint.
 */
export async function runSnapshotNow(): Promise<RunSnapshotResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-snapshot`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<RunSnapshotResult>(r);
}

export type RunQuotesResult = {
  skipped: boolean;
  reason: string | null;
  now_et: string;
  quotes_inserted: number;
};

/**
 * Trigger a manual quote ingestion run through the admin endpoint.
 */
export async function runQuotesNow(): Promise<RunQuotesResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-quotes`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<RunQuotesResult>(r);
}

export type RunDecisionResult = {
  skipped: boolean;
  reason?: string | null;
  now_et: string;
  decisions_created_count: number;
  trades_created_count: number;
  decisions_created: Array<{
    decision_id: number;
    trade_id: number;
    target_dte: number;
    delta_target: number;
    spread_side: "put" | "call";
    score: number | null;
    decision_source: string;
  }>;
  trades_created: number[];
  selection_meta?: {
    candidates_total?: number;
    candidates_ranked?: number;
    candidates_after_dedupe?: number;
    duplicates_removed?: number;
    max_trades_per_run?: number;
    day_remaining_before?: number | null;
    open_remaining_before?: number | null;
    selected_count?: number;
    clipped_by?: string | null;
  } | null;
};

/**
 * Trigger the decision engine immediately through the admin endpoint.
 */
export async function runDecisionNow(): Promise<RunDecisionResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-decision`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<RunDecisionResult>(r);
}

export type RunTradePnlResult = {
  skipped: boolean;
  reason?: string | null;
  now_et: string;
  updated?: number;
  closed?: number;
  marks_written?: number;
};

/**
 * Trigger live trade mark-to-market updates immediately.
 */
export async function runTradePnlNow(): Promise<RunTradePnlResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-trade-pnl`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<RunTradePnlResult>(r);
}

export type GenericAdminRunResult = {
  ok?: boolean;
  skipped?: boolean;
  reason?: string | null;
  status?: string;
  [key: string]: unknown;
};

/**
 * Trigger a manual GEX computation run.
 */
export async function runGexNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-gex`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

// runFeatureBuilderNow / runLabelerNow / runTrainerNow /
// runShadowInferenceNow / runPromotionGatesNow were removed when the
// online ML pipeline was decommissioned.  Their corresponding backend
// routes (/api/admin/run-feature-builder, /run-labeler, /run-trainer,
// /run-shadow-inference, /run-promotion-gates) no longer exist.

/**
 * Trigger a manual performance-analytics refresh.
 */
export async function runPerformanceAnalyticsNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-performance-analytics`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

/**
 * Trigger a manual CBOE GEX run.
 */
export async function runCboeGexNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-cboe-gex`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

export type GexSnapshot = {
  snapshot_id: number;
  ts: string;
  underlying: string;
  source: string;
  spot_price: number | null;
  gex_net: number | null;
  gex_calls: number | null;
  gex_puts: number | null;
  gex_abs: number | null;
  zero_gamma_level: number | null;
  method: string;
};

export type GexCurvePoint = {
  strike: number;
  gex_net: number | null;
  gex_calls: number | null;
  gex_puts: number | null;
};


/**
 * Fetch recent GEX snapshot aggregates for the panel selector.
 *
 * When `underlying` is provided, the backend returns only that symbol's
 * snapshots so the UI can switch between SPX/SPY/VIX cleanly.
 */
export async function fetchGexSnapshots(limit = 20, underlying?: string, source?: string, signal?: AbortSignal): Promise<GexSnapshot[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (underlying && underlying.trim()) {
    params.set("underlying", underlying.trim().toUpperCase());
  }
  if (source && source.trim()) {
    params.set("source", source.trim().toUpperCase());
  }
  const r = await fetchWithAuth(apiUrl(`/api/gex/snapshots?${params.toString()}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: GexSnapshot[] }>(r);
  return data.items;
}

/**
 * Fetch available DTE filters for the selected GEX batch.
 */
export async function fetchGexDtes(snapshotId: number, signal?: AbortSignal): Promise<number[]> {
  const r = await fetchWithAuth(apiUrl(`/api/gex/dtes?snapshot_id=${encodeURIComponent(snapshotId)}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ dte_days: number[] }>(r);
  return data.dte_days;
}

/**
 * Fetch the GEX curve by strike.
 *
 * The query supports:
 * - all expirations in the batch,
 * - one DTE filter, or
 * - a custom expiration set.
 */
export async function fetchGexCurve(snapshotId: number, dteDays?: number, expirations?: string[], signal?: AbortSignal): Promise<GexCurvePoint[]> {
  const params = new URLSearchParams({ snapshot_id: String(snapshotId) });
  if (typeof dteDays === "number") params.set("dte_days", String(dteDays));
  if (expirations && expirations.length > 0) params.set("expirations_csv", expirations.join(","));
  const r = await fetchWithAuth(apiUrl(`/api/gex/curve?${params.toString()}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ points: GexCurvePoint[] }>(r);
  return data.points;
}

export type TradeDecision = {
  decision_id: number;
  ts: string;
  target_dte: number;
  entry_slot: number;
  delta_target: number;
  decision: string;
  reason: string | null;
  score: number | null;
  chain_snapshot_id: number | null;
  decision_source: string;
  ruleset_version: string;
  chosen_legs_json: Record<string, unknown> | null;
  strategy_params_json: Record<string, unknown> | null;
};

/**
 * Fetch recent trade decisions for the decision table.
 */
export async function fetchTradeDecisions(limit = 50, signal?: AbortSignal): Promise<TradeDecision[]> {
  const r = await fetchWithAuth(apiUrl(`/api/trade-decisions?limit=${encodeURIComponent(limit)}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: TradeDecision[] }>(r);
  return data.items;
}

export type TradeLeg = {
  leg_index: number;
  option_symbol: string;
  side: string;
  qty: number;
  entry_price: number | null;
  exit_price: number | null;
  strike: number | null;
  expiration: string | null;
  option_right: string | null;
};

export type TradeRow = {
  trade_id: number;
  decision_id: number | null;
  // `candidate_id` and `feature_snapshot_id` were removed from the response
  // when their backing columns on `trades` were dropped by Track A.7
  // migration 015 (online ML schema decommission).
  status: string;
  trade_source: string;
  strategy_type: string;
  underlying: string;
  entry_time: string;
  exit_time: string | null;
  last_mark_ts: string | null;
  target_dte: number | null;
  expiration: string | null;
  contracts: number;
  contract_multiplier: number;
  spread_width_points: number | null;
  entry_credit: number | null;
  current_exit_cost: number | null;
  current_pnl: number | null;
  realized_pnl: number | null;
  max_profit: number | null;
  max_loss: number | null;
  take_profit_target: number | null;
  stop_loss_target: number | null;
  exit_reason: string | null;
  mark_count: number;
  legs: TradeLeg[];
};

/**
 * Fetch trades with optional status filter for the live PnL table.
 */
export async function fetchTrades(limit = 100, status?: "OPEN" | "CLOSED" | "ROLLED", signal?: AbortSignal): Promise<TradeRow[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set("status", status);
  const r = await fetchWithAuth(apiUrl(`/api/trades?${params.toString()}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: TradeRow[] }>(r);
  return data.items;
}

export type PerformanceAnalyticsMode = "realized" | "combined";

export type PerformanceAnalyticsSummary = {
  trade_count: number;
  win_count: number;
  loss_count: number;
  net_pnl: number;
  realized_net_pnl: number;
  unrealized_net_pnl: number;
  combined_net_pnl: number;
  win_rate: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  avg_pnl: number | null;
  expectancy: number | null;
  profit_factor: number | null;
  max_drawdown: number | null;
};

export type PerformanceAnalyticsCurvePoint = {
  date: string;
  daily_pnl: number;
  cumulative_pnl: number;
  drawdown: number;
  trade_count: number;
  win_count: number;
  loss_count: number;
};

export type PerformanceAnalyticsBreakdownRow = {
  bucket: string;
  trade_count: number;
  win_count: number;
  loss_count: number;
  net_pnl: number;
  win_rate: number | null;
  avg_win: number | null;
  avg_loss: number | null;
  avg_pnl: number | null;
  expectancy: number | null;
  profit_factor: number | null;
};

export type PerformanceAnalyticsBreakdowns = {
  side: PerformanceAnalyticsBreakdownRow[];
  dte_bucket: PerformanceAnalyticsBreakdownRow[];
  delta_bucket: PerformanceAnalyticsBreakdownRow[];
  weekday: PerformanceAnalyticsBreakdownRow[];
  hour: PerformanceAnalyticsBreakdownRow[];
  source: PerformanceAnalyticsBreakdownRow[];
};

export type PerformanceAnalyticsResponse = {
  lookback_days: number;
  mode: PerformanceAnalyticsMode;
  window_start_utc: string | null;
  as_of_utc: string | null;
  snapshot: {
    analytics_snapshot_id: number;
    source_trade_count: number;
    source_closed_count: number;
    source_open_count: number;
  } | null;
  summary: PerformanceAnalyticsSummary | null;
  equity_curve: PerformanceAnalyticsCurvePoint[];
  breakdowns: PerformanceAnalyticsBreakdowns;
};

/**
 * Fetch aggregate trade-performance analytics for the selected mode/lookback.
 */
export async function fetchPerformanceAnalytics(
  lookbackDays = 90,
  mode: PerformanceAnalyticsMode = "combined",
  signal?: AbortSignal,
): Promise<PerformanceAnalyticsResponse> {
  const params = new URLSearchParams({
    lookback_days: String(lookbackDays),
    mode,
  });
  const r = await fetchWithAuth(apiUrl(`/api/performance-analytics?${params.toString()}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<PerformanceAnalyticsResponse>(r);
}

// ModelOpsGate / ModelOpsModelVersion / ModelOpsTrainingRun / ModelOpsResponse
// + fetchModelOps were removed when the online ML pipeline was decommissioned.
// The backend route /api/model-ops no longer exists.  ``model_versions`` is
// preserved on the backend for offline ML re-entry but currently has no
// dedicated frontend surface.

export type AdminPreflightCounts = {
  underlying_quotes: number;
  chain_snapshots: number;
  option_chain_rows: number;
  gex_snapshots: number;
  trade_decisions: number;
  // ``trade_candidates``, ``labeled_candidates``, ``training_runs``,
  // ``model_predictions``, ``feature_snapshots`` were dropped from the
  // admin preflight payload when the online ML pipeline was decommissioned
  // and migration 015 (Track A.7) removed the underlying tables.
  // ``model_versions`` is preserved (offline ML re-entry).
  model_versions: number;
  trades: number;
  open_trades: number;
  closed_trades: number;
};

export type AdminPreflightLatest = {
  quote_ts: string | null;
  snapshot_ts: string | null;
  gex_ts: string | null;
  decision_ts: string | null;
  // ``candidate_ts``, ``training_run_ts``, ``prediction_ts``, ``feature_ts``
  // were dropped alongside their counts above.
  model_version_ts: string | null;
  trade_mark_ts: string | null;
  trade_entry_ts: string | null;
  market_clock_ts: string | null;
};

export type AdminPreflightSnapshot = {
  snapshot_id: number;
  ts: string | null;
  target_dte: number | null;
  expiration: string | null;
};

export type AdminPreflightGex = {
  snapshot_id: number;
  ts: string | null;
  gex_net: number | null;
  zero_gamma_level: number | null;
  method: string | null;
};

export type AdminPreflightDecision = {
  decision_id: number;
  ts: string | null;
  decision: string | null;
  reason: string | null;
  score: number | null;
  target_dte: number | null;
  delta_target: number | null;
  chain_snapshot_id: number | null;
  decision_source: string | null;
};

export type AdminPreflightQuote = {
  symbol: string;
  ts: string | null;
  last: number | null;
};

export type AdminPreflightResponse = {
  now_utc: string;
  counts: AdminPreflightCounts;
  latest: AdminPreflightLatest;
  latest_snapshot: AdminPreflightSnapshot | null;
  latest_gex: AdminPreflightGex | null;
  latest_decision: AdminPreflightDecision | null;
  latest_quotes_by_symbol: AdminPreflightQuote[];
  warnings: string[];
};

/**
 * Fetch admin preflight diagnostics for pipeline freshness and warning state.
 */
export async function fetchAdminPreflight(signal?: AbortSignal): Promise<AdminPreflightResponse> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/preflight`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<AdminPreflightResponse>(r);
}

/** Per-source freshness entry from the pipeline-status endpoint. */
export type PipelineFreshness = {
  age_minutes: number | null;
  threshold_minutes?: number;
  stale: boolean;
};

/** Response from GET /api/pipeline-status (available to all authenticated users). */
export type PipelineStatusResponse = {
  freshness: Record<string, PipelineFreshness>;
  warnings: string[];
};

/**
 * Fetch pipeline freshness for all authenticated users (no admin required).
 */
export async function fetchPipelineStatus(signal?: AbortSignal): Promise<PipelineStatusResponse> {
  const r = await fetchWithAuth(apiUrl(`/api/pipeline-status`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<PipelineStatusResponse>(r);
}


/** Single auth audit log entry (admin-only). */
export type AuthAuditEvent = {
  id: number;
  event_type: string;
  user_id: number | null;
  username: string | null;
  occurred_at: string | null;
  ip_address: string | null;
  user_agent: string | null;
  country: string | null;
  /** Full ip-api.com geo lookup response (continent, country, city, lat, lon, isp, etc.). */
  geo_json: Record<string, unknown> | null;
  details: Record<string, unknown> | null;
};

export type AuthAuditResponse = {
  total: number;
  limit: number;
  offset: number;
  events: AuthAuditEvent[];
};

/**
 * Fetch auth audit log entries. Admin-only; returns 403 for non-admins.
 */
export async function fetchAuthAudit(
  limit = 100,
  offset = 0,
  eventType?: string | null,
  userId?: number | null,
  signal?: AbortSignal,
): Promise<AuthAuditResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (eventType && eventType.trim()) params.set("event_type", eventType.trim());
  if (userId != null) params.set("user_id", String(userId));
  const r = await fetchWithAuth(apiUrl(`/api/admin/auth-audit?${params.toString()}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<AuthAuditResponse>(r);
}

// ---------------------------------------------------------------------------
// Model monitoring endpoints (REMOVED -- Phase 3 deletion)
// ---------------------------------------------------------------------------
// The Phase 3 model-monitoring API surface (ModelPredictionRow,
// ModelPredictionsResponse, AccuracyWindow, ModelAccuracyResponse,
// CalibrationBin, ModelCalibrationResponse, ModelPnlAttributionResponse,
// fetchModelPredictions, fetchModelAccuracy, fetchModelCalibration,
// fetchModelPnlAttribution) was removed when the online ML pipeline was
// decommissioned.  The corresponding backend routes
// (/api/model-predictions, /api/model-accuracy, /api/model-calibration,
// /api/model-pnl-attribution) and the underlying ``model_predictions``
// table no longer exist.

// ---------------------------------------------------------------------------
// Portfolio management endpoints
// ---------------------------------------------------------------------------

/**
 * Live portfolio status snapshot returned by `/api/portfolio/status`.
 *
 * Note: the legacy `portfolio_enabled` field was removed when the
 * `PORTFOLIO_ENABLED` flag was deleted (the live decision job is always
 * portfolio-managed after the online ML decommission).  Consumers that
 * previously branched on `portfolio_enabled` should now treat the
 * presence of a non-null `PortfolioStatus` as "portfolio data is
 * available" and fall back to legacy metrics only on fetch error.
 */
export type PortfolioStatus = {
  date: string;
  equity: number;
  month_start_equity: number;
  drawdown_pct: number;
  lots_per_trade: number;
  trades_today: number;
  max_trades_per_day: number;
  max_trades_per_run: number;
  monthly_stop_active: boolean;
  daily_pnl: number;
  event_signals: string[];
};

export type PortfolioHistoryDay = {
  date: string;
  equity_start: number | null;
  equity_end: number | null;
  trades_placed: number;
  lots_per_trade: number;
  daily_pnl: number | null;
  monthly_stop_active: boolean;
  event_signals: string[] | null;
};

export type PortfolioTrade = {
  id: number;
  trade_id: number;
  date: string;
  trade_source: string;
  event_signal: string | null;
  lots: number;
  margin_committed: number | null;
  realized_pnl: number | null;
  equity_before: number | null;
  equity_after: number | null;
  created_at: string | null;
  strategy_type: string | null;
  entry_credit: number | null;
  trade_status: string | null;
  target_dte: number | null;
  expiration: string | null;
};

export type PortfolioConfig = {
  portfolio: {
    enabled: boolean;
    starting_capital: number;
    max_trades_per_day: number;
    max_trades_per_run: number;
    monthly_drawdown_limit: number;
    lot_per_equity: number;
    max_equity_risk_pct: number;
    max_margin_pct: number;
    calls_only: boolean;
  };
  event: {
    enabled: boolean;
    budget_mode: string;
    max_trades: number;
    spx_drop_threshold: number;
    spx_drop_2d_threshold: number;
    vix_spike_threshold: number;
    vix_elevated_threshold: number;
    term_inversion_threshold: number;
    side_preference: string;
    min_dte: number;
    max_dte: number;
    min_delta: number;
    max_delta: number;
    rally_avoidance: boolean;
    rally_threshold: number;
  };
  decision: {
    entry_times: string;
    dte_targets: string;
    delta_targets: string;
    spread_width_points: number;
  };
};

/**
 * Fetch current portfolio status (equity, lots, budget, signals).
 */
export async function fetchPortfolioStatus(signal?: AbortSignal): Promise<PortfolioStatus> {
  const r = await fetchWithAuth(apiUrl("/api/portfolio/status"), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<PortfolioStatus>(r);
}

/**
 * Fetch daily portfolio state history for equity charting.
 */
export async function fetchPortfolioHistory(days = 90, signal?: AbortSignal): Promise<PortfolioHistoryDay[]> {
  const r = await fetchWithAuth(apiUrl(`/api/portfolio/history?days=${days}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: PortfolioHistoryDay[] }>(r);
  return data.items;
}

/**
 * Fetch portfolio trades with source/signal enrichment.
 */
export async function fetchPortfolioTrades(limit = 100, source?: string, signal?: AbortSignal): Promise<PortfolioTrade[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (source) params.set("source", source);
  const r = await fetchWithAuth(apiUrl(`/api/portfolio/trades?${params.toString()}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: PortfolioTrade[] }>(r);
  return data.items;
}

/**
 * Fetch current portfolio and event configuration.
 */
export async function fetchPortfolioConfig(signal?: AbortSignal): Promise<PortfolioConfig> {
  const r = await fetchWithAuth(apiUrl("/api/portfolio/config"), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<PortfolioConfig>(r);
}


// ── Optimizer Dashboard API ────────────────────────────────────

export type OptimizerRun = {
  run_id: string;
  run_name: string | null;
  git_hash: string | null;
  config_file: string | null;
  optimizer_mode: string;
  started_at: string | null;
  finished_at: string | null;
  num_configs: number | null;
  status: string;
  result_count: number;
  best_sharpe: number | null;
  best_return_pct: number | null;
  pareto_count: number;
};

export type OptimizerResult = {
  id: number;
  run_id: string;
  sharpe: number;
  return_pct: number;
  max_dd_pct: number;
  win_rate: number;
  total_trades: number;
  final_equity: number;
  ann_return_pct: number;
  is_pareto: boolean;
  [key: string]: unknown;
};

export type WalkforwardRow = {
  id: number;
  run_id: string;
  config_key: string | null;
  window_label: string | null;
  train_sharpe: number | null;
  test_sharpe: number | null;
  train_return: number | null;
  test_return: number | null;
  decay_ratio: number | null;
};

/** List all optimizer runs with summary stats. */
export async function fetchOptimizerRuns(
  limit = 50,
  offset = 0,
  signal?: AbortSignal,
): Promise<{ runs: OptimizerRun[]; total: number }> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  const r = await fetchWithAuth(apiUrl(`/api/optimizer/runs?${params}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson(r);
}

/** Paginated results for a run with sorting and filtering. */
export async function fetchOptimizerResults(
  runId: string,
  opts: {
    sortBy?: string;
    sortOrder?: string;
    minSharpe?: number;
    minWinRate?: number;
    minTrades?: number;
    limit?: number;
    offset?: number;
  } = {},
  signal?: AbortSignal,
): Promise<{ results: OptimizerResult[]; total: number }> {
  const params = new URLSearchParams();
  if (opts.sortBy) params.set("sort_by", opts.sortBy);
  if (opts.sortOrder) params.set("sort_order", opts.sortOrder);
  if (opts.minSharpe !== undefined) params.set("min_sharpe", String(opts.minSharpe));
  if (opts.minWinRate !== undefined) params.set("min_win_rate", String(opts.minWinRate));
  if (opts.minTrades !== undefined) params.set("min_trades", String(opts.minTrades));
  params.set("limit", String(opts.limit ?? 100));
  params.set("offset", String(opts.offset ?? 0));
  const r = await fetchWithAuth(apiUrl(`/api/optimizer/runs/${runId}/results?${params}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson(r);
}

/** Pareto frontier configs for a run. */
export async function fetchParetoFrontier(
  runId: string,
  signal?: AbortSignal,
): Promise<{ pareto: OptimizerResult[]; count: number }> {
  const r = await fetchWithAuth(apiUrl(`/api/optimizer/runs/${runId}/pareto`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson(r);
}

/** Side-by-side config comparison. */
export async function fetchConfigComparison(
  ids: number[],
  signal?: AbortSignal,
): Promise<{ configs: OptimizerResult[]; differing_columns: string[] }> {
  const r = await fetchWithAuth(apiUrl(`/api/optimizer/compare?ids=${ids.join(",")}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson(r);
}

/** Walk-forward results for a run. */
export async function fetchWalkforwardResults(
  runId: string,
  signal?: AbortSignal,
): Promise<{ walkforward: WalkforwardRow[]; count: number }> {
  const r = await fetchWithAuth(apiUrl(`/api/optimizer/walkforward/${runId}`), { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson(r);
}


