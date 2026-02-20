import * as authStorage from "./auth";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ?? "";

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
 * Fetch with auth header and 401 handling: on 401 clears token and redirects to /login.
 */
async function fetchWithAuth(url: string, init: RequestInit = {}): Promise<Response> {
  const headers = { ...authHeaders(), ...(init.headers as Record<string, string>) };
  const r = await fetch(url, { ...init, headers });
  if (r.status === 401) {
    authStorage.clearToken();
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }
  return r;
}

/**
 * Build admin auth headers when an API key is provided (optional; JWT is primary).
 */
function adminHeaders(apiKey?: string): HeadersInit | undefined {
  if (!apiKey) return undefined;
  const trimmed = apiKey.trim();
  if (!trimmed) return undefined;
  const normalized =
    trimmed.length >= 2 &&
    ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'")))
      ? trimmed.slice(1, -1).trim()
      : trimmed;
  return normalized ? { "X-API-Key": normalized } : undefined;
}

export type ChainSnapshot = {
  snapshot_id: number;
  ts: string;
  underlying: string;
  target_dte: number;
  expiration: string;
  checksum: string;
};

/**
 * Fetch recent chain snapshot metadata for the dashboard table.
 */
export async function fetchChainSnapshots(limit = 50): Promise<ChainSnapshot[]> {
  const r = await fetchWithAuth(apiUrl(`/api/chain-snapshots?limit=${encodeURIComponent(limit)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { items: ChainSnapshot[] };
  return data.items;
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
export async function runSnapshotNow(apiKey?: string): Promise<RunSnapshotResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-snapshot`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as RunSnapshotResult;
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
export async function runQuotesNow(apiKey?: string): Promise<RunQuotesResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-quotes`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as RunQuotesResult;
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
export async function runDecisionNow(apiKey?: string): Promise<RunDecisionResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-decision`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as RunDecisionResult;
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
export async function runTradePnlNow(apiKey?: string): Promise<RunTradePnlResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-trade-pnl`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as RunTradePnlResult;
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
export async function runGexNow(apiKey?: string): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-gex`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as GenericAdminRunResult;
}

/**
 * Trigger a manual feature-builder run.
 */
export async function runFeatureBuilderNow(apiKey?: string): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-feature-builder`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as GenericAdminRunResult;
}

/**
 * Trigger a manual labeler run.
 */
export async function runLabelerNow(apiKey?: string): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-labeler`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as GenericAdminRunResult;
}

/**
 * Trigger a manual trainer run.
 */
export async function runTrainerNow(apiKey?: string): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-trainer`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as GenericAdminRunResult;
}

/**
 * Trigger a manual shadow-inference run.
 */
export async function runShadowInferenceNow(apiKey?: string): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-shadow-inference`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as GenericAdminRunResult;
}

/**
 * Trigger a manual promotion-gate run.
 */
export async function runPromotionGatesNow(apiKey?: string): Promise<GenericAdminRunResult> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/run-promotion-gates`), {
    method: "POST",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as GenericAdminRunResult;
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

export type GexExpirationItem = {
  expiration: string;
  dte_days: number | null;
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
  const data = (await r.json()) as { items: GexSnapshot[] };
  return data.items;
}

/**
 * Fetch available DTE filters for the selected GEX batch.
 */
export async function fetchGexDtes(snapshotId: number): Promise<number[]> {
  const r = await fetchWithAuth(apiUrl(`/api/gex/dtes?snapshot_id=${encodeURIComponent(snapshotId)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { dte_days: number[] };
  return data.dte_days;
}

/**
 * Fetch available expirations for the selected GEX batch.
 */
export async function fetchGexExpirations(snapshotId: number): Promise<GexExpirationItem[]> {
  const r = await fetchWithAuth(apiUrl(`/api/gex/expirations?snapshot_id=${encodeURIComponent(snapshotId)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { items: GexExpirationItem[] };
  return data.items;
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
  const data = (await r.json()) as { points: GexCurvePoint[] };
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
  const data = (await r.json()) as { items: TradeDecision[] };
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
  const data = (await r.json()) as { items: TradeRow[] };
  return data.items;
}

export type LabelMetricsSummary = {
  resolved: number;
  tp50: number;
  tp100_at_expiry: number;
  tp50_rate: number | null;
  tp100_at_expiry_rate: number | null;
  avg_realized_pnl: number | null;
};

export type LabelMetricsBySide = {
  spread_side: string;
  resolved: number;
  tp50: number;
  tp100_at_expiry: number;
  tp50_rate: number | null;
  tp100_at_expiry_rate: number | null;
  avg_realized_pnl: number | null;
};

export type LabelMetricsResponse = {
  lookback_days: number;
  window_start_utc: string;
  summary: LabelMetricsSummary;
  by_side: LabelMetricsBySide[];
};

/**
 * Fetch TP50/TP100 label metrics for the selected lookback window.
 */
export async function fetchLabelMetrics(lookbackDays = 90): Promise<LabelMetricsResponse> {
  const r = await fetchWithAuth(apiUrl(`/api/label-metrics?lookback_days=${encodeURIComponent(lookbackDays)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as LabelMetricsResponse;
}

export type StrategyMetricsSummary = {
  resolved: number;
  tp50: number;
  tp100_at_expiry: number;
  tp50_rate: number | null;
  tp100_at_expiry_rate: number | null;
  expectancy: number | null;
  avg_realized_pnl: number | null;
  max_drawdown: number | null;
  tail_loss_proxy: number | null;
  avg_margin_usage: number | null;
};

export type StrategyMetricsBySide = {
  spread_side: string;
  resolved: number;
  tp50: number;
  tp100_at_expiry: number;
  tp50_rate: number | null;
  tp100_at_expiry_rate: number | null;
  expectancy: number | null;
  avg_realized_pnl: number | null;
  max_drawdown: number | null;
  tail_loss_proxy: number | null;
  avg_margin_usage: number | null;
};

export type StrategyMetricsResponse = {
  lookback_days: number;
  window_start_utc: string;
  summary: StrategyMetricsSummary;
  by_side: StrategyMetricsBySide[];
};

/**
 * Fetch strategy quality/risk metrics (expectancy, drawdown, tail proxy, margin).
 */
export async function fetchStrategyMetrics(lookbackDays = 90): Promise<StrategyMetricsResponse> {
  const r = await fetchWithAuth(apiUrl(`/api/strategy-metrics?lookback_days=${encodeURIComponent(lookbackDays)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as StrategyMetricsResponse;
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
  model_version_id: number;
  status: string;
  started_at_utc: string | null;
  finished_at_utc: string | null;
  rows_train: number;
  rows_test: number;
  notes: string | null;
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
  return (await r.json()) as ModelOpsResponse;
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
export async function fetchAdminPreflight(apiKey?: string): Promise<AdminPreflightResponse> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/preflight`), {
    method: "GET",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as AdminPreflightResponse;
}

/**
 * Delete one persisted trade decision by ID.
 */
export async function deleteTradeDecision(decisionId: number, apiKey?: string): Promise<{ deleted: boolean; decision_id: number }> {
  const r = await fetchWithAuth(apiUrl(`/api/admin/trade-decisions/${encodeURIComponent(decisionId)}`), {
    method: "DELETE",
    headers: { ...authHeaders(), ...adminHeaders(apiKey) },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as { deleted: boolean; decision_id: number };
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
  return (await r.json()) as AuthAuditResponse;
}

