export type ChainSnapshot = {
  snapshot_id: number;
  ts: string;
  underlying: string;
  target_dte: number;
  expiration: string;
  checksum: string;
};

export async function fetchChainSnapshots(limit = 50): Promise<ChainSnapshot[]> {
  const r = await fetch(`/api/chain-snapshots?limit=${encodeURIComponent(limit)}`);
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
  const r = await fetch(`/api/admin/run-snapshot`, {
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
  const r = await fetch(`/api/admin/run-quotes`, {
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
  const r = await fetch(`/api/admin/run-decision`, {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : undefined,
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as RunDecisionResult;
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

export async function fetchGexSnapshots(limit = 20): Promise<GexSnapshot[]> {
  const r = await fetch(`/api/gex/snapshots?limit=${encodeURIComponent(limit)}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { items: GexSnapshot[] };
  return data.items;
}

export async function fetchGexDtes(snapshotId: number): Promise<number[]> {
  const r = await fetch(`/api/gex/dtes?snapshot_id=${encodeURIComponent(snapshotId)}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { dte_days: number[] };
  return data.dte_days;
}

export async function fetchGexCurve(snapshotId: number, dteDays?: number): Promise<GexCurvePoint[]> {
  const params = new URLSearchParams({ snapshot_id: String(snapshotId) });
  if (typeof dteDays === "number") params.set("dte_days", String(dteDays));
  const r = await fetch(`/api/gex/curve?${params.toString()}`);
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
  const r = await fetch(`/api/trade-decisions?limit=${encodeURIComponent(limit)}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = (await r.json()) as { items: TradeDecision[] };
  return data.items;
}

