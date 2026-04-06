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
function apiUrl(path: string): string {
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

/**
 * Trigger a manual feature-builder run.
 */
export async function runFeatureBuilderNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-feature-builder`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

/**
 * Trigger a manual labeler run.
 */
export async function runLabelerNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-labeler`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

/**
 * Trigger a manual trainer run.
 */
export async function runTrainerNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-trainer`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

/**
 * Trigger a manual shadow-inference run.
 */
export async function runShadowInferenceNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-shadow-inference`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

/**
 * Trigger a manual promotion-gate run.
 */
export async function runPromotionGatesNow(): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-promotion-gates`), {
    method: "POST",
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<GenericAdminRunResult>(r);
}

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
export async function fetchGexSnapshots(limit = 20, underlying?: string, source?: string): Promise<GexSnapshot[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (underlying && underlying.trim()) {
    params.set("underlying", underlying.trim().toUpperCase());
  }
  if (source && source.trim()) {
    params.set("source", source.trim().toUpperCase());
  }
  const r = await fetchWithAuth(apiUrl(`/api/gex/snapshots?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: GexSnapshot[] }>(r);
  return data.items;
}

/**
 * Fetch available DTE filters for the selected GEX batch.
 */
export async function fetchGexDtes(snapshotId: number): Promise<number[]> {
  const r = await fetchWithAuth(apiUrl(`/api/gex/dtes?snapshot_id=${encodeURIComponent(snapshotId)}`));
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
export async function fetchGexCurve(snapshotId: number, dteDays?: number, expirations?: string[]): Promise<GexCurvePoint[]> {
  const params = new URLSearchParams({ snapshot_id: String(snapshotId) });
  if (typeof dteDays === "number") params.set("dte_days", String(dteDays));
  if (expirations && expirations.length > 0) params.set("expirations_csv", expirations.join(","));
  const r = await fetchWithAuth(apiUrl(`/api/gex/curve?${params.toString()}`));
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
export async function fetchTradeDecisions(limit = 50): Promise<TradeDecision[]> {
  const r = await fetchWithAuth(apiUrl(`/api/trade-decisions?limit=${encodeURIComponent(limit)}`));
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
  candidate_id: number | null;
  feature_snapshot_id: number | null;
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
export async function fetchTrades(limit = 100, status?: "OPEN" | "CLOSED" | "ROLLED"): Promise<TradeRow[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set("status", status);
  const r = await fetchWithAuth(apiUrl(`/api/trades?${params.toString()}`));
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
  mode: PerformanceAnalyticsMode = "combined"
): Promise<PerformanceAnalyticsResponse> {
  const params = new URLSearchParams({
    lookback_days: String(lookbackDays),
    mode,
  });
  const r = await fetchWithAuth(apiUrl(`/api/performance-analytics?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<PerformanceAnalyticsResponse>(r);
}

export type ModelOpsGate = {
  passed: boolean;
  checks: Record<string, { value: number; threshold: number; pass: boolean }>;
  summary?: Record<string, number>;
};

export type ModelOpsModelVersion = {
  model_version_id: number;
  version: string;
  rollout_status: string;
  is_active: boolean;
  created_at_utc: string | null;
  promoted_at_utc: string | null;
  metrics?: {
    tp50_rate_test?: number | null;
    expectancy_test?: number | null;
    max_drawdown_test?: number | null;
    tail_loss_proxy_test?: number | null;
    avg_margin_usage_test?: number | null;
  };
};

export type ModelOpsTrainingRun = {
  training_run_id: number;
  model_version_id: number | null;
  status: string;
  started_at_utc: string | null;
  finished_at_utc: string | null;
  rows_train: number;
  rows_test: number;
  notes: string | null;
  skip_reason: string | null;
  gate: ModelOpsGate | null;
};

export type ModelOpsResponse = {
  model_name: string;
  counts: {
    model_versions: number;
    training_runs: number;
    model_predictions: number;
    model_predictions_24h: number;
  };
  latest_prediction_ts: string | null;
  latest_model_version: ModelOpsModelVersion | null;
  active_model_version: ModelOpsModelVersion | null;
  latest_training_run: ModelOpsTrainingRun | null;
  warnings: string[];
};

/**
 * Fetch model-ops status for monitoring training/gates/prediction activity.
 */
export async function fetchModelOps(modelName?: string): Promise<ModelOpsResponse> {
  const params = new URLSearchParams();
  if (modelName && modelName.trim()) params.set("model_name", modelName.trim());
  const query = params.toString();
  const r = await fetchWithAuth(apiUrl(`/api/model-ops${query ? `?${query}` : ""}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<ModelOpsResponse>(r);
}

export type AdminPreflightCounts = {
  underlying_quotes: number;
  chain_snapshots: number;
  option_chain_rows: number;
  gex_snapshots: number;
  trade_decisions: number;
  feature_snapshots: number;
  trade_candidates: number;
  labeled_candidates: number;
  model_versions: number;
  training_runs: number;
  model_predictions: number;
  trades: number;
  open_trades: number;
  closed_trades: number;
};

export type AdminPreflightLatest = {
  quote_ts: string | null;
  snapshot_ts: string | null;
  gex_ts: string | null;
  decision_ts: string | null;
  feature_ts: string | null;
  candidate_ts: string | null;
  model_version_ts: string | null;
  training_run_ts: string | null;
  prediction_ts: string | null;
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
export async function fetchAdminPreflight(): Promise<AdminPreflightResponse> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/preflight`));
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
export async function fetchPipelineStatus(): Promise<PipelineStatusResponse> {
  const r = await fetchWithAuth(apiUrl(`/api/pipeline-status`));
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
  userId?: number | null
): Promise<AuthAuditResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (eventType && eventType.trim()) params.set("event_type", eventType.trim());
  if (userId != null) params.set("user_id", String(userId));
  const r = await fetchWithAuth(apiUrl(`/api/admin/auth-audit?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<AuthAuditResponse>(r);
}

// ---------------------------------------------------------------------------
// Model monitoring endpoints (Phase 3)
// ---------------------------------------------------------------------------

export type ModelPredictionRow = {
  prediction_id: number;
  candidate_id: number | null;
  model_version_id: number;
  model_name: string;
  model_version: string;
  probability_win: number | null;
  expected_value: number | null;
  score_raw: number | null;
  decision_hint: string | null;
  created_at: string | null;
  hold_realized_pnl: number | null;
  hold_hit_tp50: string | null;
  hold_exit_reason: string | null;
  realized_pnl: number | null;
  label_status: string | null;
};

export type ModelPredictionsResponse = {
  total: number;
  limit: number;
  offset: number;
  items: ModelPredictionRow[];
};

/**
 * Fetch paginated model predictions with outcome data for the prediction browser.
 */
export async function fetchModelPredictions(
  limit = 50,
  offset = 0,
  modelVersionId?: number,
  decision?: string,
  dateFrom?: string,
  dateTo?: string
): Promise<ModelPredictionsResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (modelVersionId != null) params.set("model_version_id", String(modelVersionId));
  if (decision) params.set("decision", decision);
  if (dateFrom) params.set("date_from", dateFrom);
  if (dateTo) params.set("date_to", dateTo);
  const r = await fetchWithAuth(apiUrl(`/api/model-predictions?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<ModelPredictionsResponse>(r);
}

export type AccuracyWindow = {
  period: string | null;
  total: number;
  true_positive: number;
  false_positive: number;
  true_negative: number;
  false_negative: number;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  avg_pnl_traded: number | null;
  avg_pnl_skipped: number | null;
};

export type ModelAccuracyResponse = {
  model_name: string;
  window: string;
  windows: AccuracyWindow[];
};

/**
 * Fetch accuracy metrics aggregated over time windows (weekly/monthly).
 */
export async function fetchModelAccuracy(modelName?: string, window = "week"): Promise<ModelAccuracyResponse> {
  const params = new URLSearchParams({ window });
  if (modelName) params.set("model_name", modelName);
  const r = await fetchWithAuth(apiUrl(`/api/model-accuracy?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<ModelAccuracyResponse>(r);
}

export type CalibrationBin = {
  bin_lower: number;
  bin_upper: number;
  predicted_avg: number;
  observed_rate: number | null;
  count: number;
};

export type ModelCalibrationResponse = {
  model_name: string;
  bins: CalibrationBin[];
};

/**
 * Fetch calibration curve data for the model monitor.
 */
export async function fetchModelCalibration(modelName?: string, bins = 10): Promise<ModelCalibrationResponse> {
  const params = new URLSearchParams({ bins: String(bins) });
  if (modelName) params.set("model_name", modelName);
  const r = await fetchWithAuth(apiUrl(`/api/model-calibration?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<ModelCalibrationResponse>(r);
}

export type ModelPnlAttributionResponse = {
  model_name: string;
  baseline_pnl: number;
  model_pnl: number;
  saved_pnl: number;
  missed_pnl: number;
  net_impact: number;
  trade_count: number;
  skip_count: number;
  total_candidates: number;
};

/**
 * Fetch PnL attribution comparing model-filtered vs baseline trades.
 */
export async function fetchModelPnlAttribution(modelName?: string): Promise<ModelPnlAttributionResponse> {
  const params = new URLSearchParams();
  if (modelName) params.set("model_name", modelName);
  const query = params.toString();
  const r = await fetchWithAuth(apiUrl(`/api/model-pnl-attribution${query ? `?${query}` : ""}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<ModelPnlAttributionResponse>(r);
}

// ---------------------------------------------------------------------------
// Portfolio management endpoints
// ---------------------------------------------------------------------------

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
  portfolio_enabled: boolean;
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
export async function fetchPortfolioStatus(): Promise<PortfolioStatus> {
  const r = await fetchWithAuth(apiUrl("/api/portfolio/status"));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<PortfolioStatus>(r);
}

/**
 * Fetch daily portfolio state history for equity charting.
 */
export async function fetchPortfolioHistory(days = 90): Promise<PortfolioHistoryDay[]> {
  const r = await fetchWithAuth(apiUrl(`/api/portfolio/history?days=${days}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: PortfolioHistoryDay[] }>(r);
  return data.items;
}

/**
 * Fetch portfolio trades with source/signal enrichment.
 */
export async function fetchPortfolioTrades(limit = 100, source?: string): Promise<PortfolioTrade[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (source) params.set("source", source);
  const r = await fetchWithAuth(apiUrl(`/api/portfolio/trades?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await safeJson<{ items: PortfolioTrade[] }>(r);
  return data.items;
}

/**
 * Fetch current portfolio and event configuration.
 */
export async function fetchPortfolioConfig(): Promise<PortfolioConfig> {
  const r = await fetchWithAuth(apiUrl("/api/portfolio/config"));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<PortfolioConfig>(r);
}

export type BacktestStrategyResult = {
  name: string;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  profit_factor: number | null;
  sharpe: number | null;
  max_drawdown: number;
  equity_curve: { date: string; cumulative_pnl: number }[];
  monthly: { month: string; pnl: number; trades: number }[];
};

export type BacktestResponse = {
  strategies: BacktestStrategyResult[];
  generated_at: string | null;
};

/**
 * Fetch precomputed backtest strategy comparison results.
 */
export async function fetchBacktestResults(): Promise<BacktestResponse> {
  const r = await fetchWithAuth(apiUrl("/api/backtest-results"));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return safeJson<BacktestResponse>(r);
}

