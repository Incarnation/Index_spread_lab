const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ?? "";

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export type ChainSnapshot = {
  snapshot_id: number;
  ts: string;
  underlying: string;
  target_dte: number;
  expiration: string;
  checksum: string;
};

export async function fetchChainSnapshots(limit = 50): Promise<ChainSnapshot[]> {
  const r = await fetch(apiUrl(`/api/chain-snapshots?limit=${encodeURIComponent(limit)}`));
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

export async function runSnapshotNow(apiKey?: string): Promise<RunSnapshotResult> {
  const r = await fetch(apiUrl(`/api/admin/run-snapshot`), {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : undefined,
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

export async function runQuotesNow(apiKey?: string): Promise<RunQuotesResult> {
  const r = await fetch(apiUrl(`/api/admin/run-quotes`), {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : undefined,
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as RunQuotesResult;
}

export type RunDecisionResult = {
  skipped: boolean;
  reason?: string | null;
  now_et: string;
  decision?: string;
  chosen?: Record<string, unknown>;
};

export async function runDecisionNow(apiKey?: string): Promise<RunDecisionResult> {
  const r = await fetch(apiUrl(`/api/admin/run-decision`), {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : undefined,
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

export async function runTradePnlNow(apiKey?: string): Promise<RunTradePnlResult> {
  const r = await fetch(apiUrl(`/api/admin/run-trade-pnl`), {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : undefined,
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as RunTradePnlResult;
}

export type GexSnapshot = {
  snapshot_id: number;
  ts: string;
  underlying: string;
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

export async function fetchGexSnapshots(limit = 20): Promise<GexSnapshot[]> {
  const r = await fetch(apiUrl(`/api/gex/snapshots?limit=${encodeURIComponent(limit)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { items: GexSnapshot[] };
  return data.items;
}

export async function fetchGexDtes(snapshotId: number): Promise<number[]> {
  const r = await fetch(apiUrl(`/api/gex/dtes?snapshot_id=${encodeURIComponent(snapshotId)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { dte_days: number[] };
  return data.dte_days;
}

export async function fetchGexExpirations(snapshotId: number): Promise<GexExpirationItem[]> {
  const r = await fetch(apiUrl(`/api/gex/expirations?snapshot_id=${encodeURIComponent(snapshotId)}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { items: GexExpirationItem[] };
  return data.items;
}

export async function fetchGexCurve(snapshotId: number, dteDays?: number, expirations?: string[]): Promise<GexCurvePoint[]> {
  const params = new URLSearchParams({ snapshot_id: String(snapshotId) });
  if (typeof dteDays === "number") params.set("dte_days", String(dteDays));
  if (expirations && expirations.length > 0) params.set("expirations_csv", expirations.join(","));
  const r = await fetch(apiUrl(`/api/gex/curve?${params.toString()}`));
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

export async function fetchTradeDecisions(limit = 50): Promise<TradeDecision[]> {
  const r = await fetch(apiUrl(`/api/trade-decisions?limit=${encodeURIComponent(limit)}`));
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

export async function fetchTrades(limit = 100, status?: "OPEN" | "CLOSED" | "ROLLED"): Promise<TradeRow[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set("status", status);
  const r = await fetch(apiUrl(`/api/trades?${params.toString()}`));
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { items: TradeRow[] };
  return data.items;
}

export async function deleteTradeDecision(decisionId: number, apiKey?: string): Promise<{ deleted: boolean; decision_id: number }> {
  const r = await fetch(apiUrl(`/api/admin/trade-decisions/${encodeURIComponent(decisionId)}`), {
    method: "DELETE",
    headers: apiKey ? { "X-API-Key": apiKey } : undefined,
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as { deleted: boolean; decision_id: number };
}

