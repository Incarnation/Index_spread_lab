import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { timeAgo } from "@/lib/utils";
import {
  fetchAdminPreflight,
  type AdminPreflightResponse,
  runSnapshotNow,
  runQuotesNow,
  runDecisionNow,
  runTradePnlNow,
  runGexNow,
  runPerformanceAnalyticsNow,
  runCboeGexNow,
} from "@/api";
import { AlertTriangle, Play } from "lucide-react";

// Online-ML run buttons (Feature Builder, Labeler, Trainer, Shadow
// Inference, Promotion Gate) were removed from this list when the
// online ML pipeline was decommissioned -- the corresponding admin
// endpoints (/api/admin/run-feature-builder, /run-labeler, /run-trainer,
// /run-shadow-inference, /run-promotion-gates) no longer exist.
const JOBS = [
  { label: "Quotes", fn: runQuotesNow },
  { label: "Snapshot", fn: runSnapshotNow },
  { label: "GEX", fn: runGexNow },
  { label: "CBOE GEX", fn: runCboeGexNow },
  { label: "Decision", fn: runDecisionNow },
  { label: "Trade PnL", fn: runTradePnlNow },
  { label: "Perf Analytics", fn: runPerformanceAnalyticsNow },
] as const;

/**
 * Admin / Ops page -- pipeline status, job triggers, and system health.
 */
export function AdminPage() {
  const { tick } = useAutoRefresh(30_000);
  const [preflight, setPreflight] = useState<AdminPreflightResponse | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [result, setResult] = useState<{ job: string; data: unknown } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setError(null);
    fetchAdminPreflight(ac.signal)
      .then((data) => {
        if (!ac.signal.aborted) setPreflight(data);
      })
      .catch((e) => {
        if (!ac.signal.aborted) setError(e.message ?? "Failed to load preflight data");
      });
    return () => ac.abort();
  }, [tick]);

  const runJob = useCallback(async (label: string, fn: () => Promise<unknown>) => {
    setRunning(label);
    setResult(null);
    try {
      const data = await fn();
      setResult({ job: label, data });
    } catch (e) {
      setResult({ job: label, data: { error: e instanceof Error ? e.message : "Failed" } });
    } finally {
      setRunning(null);
    }
  }, []);

  const c = preflight?.counts;
  const l = preflight?.latest;

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-foreground">Admin / Ops</h2>

      {error && (
        <div className="rounded-lg border border-loss/30 bg-loss-bg p-3 text-sm text-loss">{error}</div>
      )}

      {/* Warnings */}
      {preflight && preflight.warnings.length > 0 && (
        <div className="rounded-lg border border-warning/30 bg-warning-bg p-3">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className="h-4 w-4 text-warning" />
            <span className="text-xs font-medium text-warning">Warnings</span>
          </div>
          {preflight.warnings.map((w, i) => (
            <p key={i} className="text-xs text-warning/80">{w}</p>
          ))}
        </div>
      )}

      {/* Pipeline freshness */}
      {l && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[
            { label: "Quotes", ts: l.quote_ts },
            { label: "Snapshots", ts: l.snapshot_ts },
            { label: "GEX", ts: l.gex_ts },
            { label: "Decisions", ts: l.decision_ts },
            { label: "Trade Mark", ts: l.trade_mark_ts },
            { label: "Models", ts: l.model_version_ts },
          ].map(({ label, ts }) => (
            <Card key={label}>
              <CardHeader className="pb-1">
                <CardTitle className="text-xs">{label}</CardTitle>
              </CardHeader>
              <CardContent>
                <span className="text-sm font-medium text-foreground">{timeAgo(ts)}</span>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Row counts */}
      {c && (
        <div className="grid grid-cols-3 gap-3 lg:grid-cols-6">
          {[
            { label: "Trades", val: c.trades },
            { label: "Open", val: c.open_trades },
            { label: "Closed", val: c.closed_trades },
            { label: "Decisions", val: c.trade_decisions },
            { label: "Snapshots", val: c.chain_snapshots },
            { label: "GEX", val: c.gex_snapshots },
            { label: "Models", val: c.model_versions },
          ].map(({ label, val }) => (
            <div key={label} className="rounded-lg border border-border bg-card p-3">
              <span className="text-xs text-muted">{label}</span>
              <div className="text-lg font-semibold text-foreground">{val.toLocaleString()}</div>
            </div>
          ))}
        </div>
      )}

      {/* Job triggers */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Play className="h-4 w-4" /> Manual Job Triggers
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {JOBS.map(({ label, fn }) => (
              <Button
                key={label}
                variant="secondary"
                size="sm"
                disabled={running !== null}
                onClick={() => runJob(label, fn)}
              >
                {running === label ? "Running..." : label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Job result */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Result: {result.job}</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="rounded-md bg-background p-3 text-xs text-foreground-secondary overflow-x-auto max-h-64">
              {JSON.stringify(result.data, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
