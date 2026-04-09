import { useEffect, useState } from "react";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatDateTime } from "@/lib/utils";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { fetchTradeDecisions, type TradeDecision } from "@/api";
import { X } from "lucide-react";

/**
 * Map raw decision_source values to display-friendly badge labels and variants.
 */
function sourceLabel(src: string | undefined): { text: string; variant: "profit" | "warning" | "muted" } {
  if (!src) return { text: "—", variant: "muted" };
  if (src.includes("event")) return { text: "Event", variant: "warning" };
  if (src.includes("scheduled") || src.includes("portfolio")) return { text: "Scheduled", variant: "profit" };
  if (src.includes("hybrid") || src.includes("model")) return { text: "ML Model", variant: "muted" };
  return { text: src, variant: "muted" };
}

/**
 * Decisions page -- trade entry/skip log with reasoning and model scores.
 * Shows decision_source as a badge and portfolio-related metadata in the drawer.
 */
export function DecisionsPage() {
  const { tick } = useAutoRefresh(30_000);
  const [decisions, setDecisions] = useState<TradeDecision[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<TradeDecision | null>(null);
  const [filter, setFilter] = useState<"all" | "TRADE" | "SKIP">("all");

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchTradeDecisions(200, ac.signal)
      .then((data) => {
        if (!ac.signal.aborted) setDecisions(data);
      })
      .catch((e) => {
        if (!ac.signal.aborted) setError(e.message ?? "Failed to load decisions");
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [tick]);

  const filtered = filter === "all" ? decisions : decisions.filter((d) => d.decision === filter);

  const tradeCt = decisions.filter((d) => d.decision === "TRADE").length;
  const skipCt = decisions.filter((d) => d.decision === "SKIP").length;

  return (
    <div className="space-y-4">
      {error && (
        <div className="rounded-lg border border-loss/30 bg-loss-bg p-3 text-sm text-loss">{error}</div>
      )}

      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-foreground">Decisions</h2>
        <div className="flex gap-1">
          {(["all", "TRADE", "SKIP"] as const).map((f) => (
            <Button
              key={f}
              variant={filter === f ? "default" : "ghost"}
              size="sm"
              onClick={() => setFilter(f)}
            >
              {f === "all" ? `All (${decisions.length})` : f === "TRADE" ? `Trade (${tradeCt})` : `Skip (${skipCt})`}
            </Button>
          ))}
        </div>
      </div>

      <DataTable
        columns={[
          { key: "decision_id", header: "ID", className: "w-16" },
          {
            key: "decision",
            header: "Decision",
            render: (r: TradeDecision) => (
              <Badge variant={r.decision === "TRADE" ? "profit" : "loss"}>{r.decision}</Badge>
            ),
          },
          {
            key: "decision_source",
            header: "Source",
            render: (r: TradeDecision) => {
              const { text, variant } = sourceLabel(r.decision_source);
              return <Badge variant={variant}>{text}</Badge>;
            },
          },
          { key: "ts", header: "Time", render: (r: TradeDecision) => formatDateTime(r.ts) },
          { key: "target_dte", header: "DTE" },
          { key: "delta_target", header: "Delta", render: (r: TradeDecision) => r.delta_target?.toFixed(2) ?? "—" },
          { key: "score", header: "Score", render: (r: TradeDecision) => r.score?.toFixed(2) ?? "—" },
          { key: "reason", header: "Reason", render: (r: TradeDecision) => (
            <span className="max-w-[200px] truncate block text-xs text-muted-foreground">{r.reason || "—"}</span>
          )},
        ]}
        data={filtered}
        keyFn={(r) => r.decision_id}
        onRowClick={setSelected}
        emptyMessage={loading ? "Loading..." : "No decisions"}
      />

      {/* Detail drawer */}
      {selected && (
        <div className="fixed inset-y-0 right-0 z-50 w-[480px] border-l border-border bg-card shadow-2xl overflow-y-auto">
          <div className="flex items-center justify-between p-4 border-b border-border">
            <h3 className="text-sm font-medium text-foreground">Decision #{selected.decision_id}</h3>
            <Button variant="ghost" size="icon" onClick={() => setSelected(null)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="p-4 space-y-4">
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-xs text-muted">Decision</span>
                <div><Badge variant={selected.decision === "TRADE" ? "profit" : "loss"}>{selected.decision}</Badge></div>
              </div>
              <div>
                <span className="text-xs text-muted">Source</span>
                <div>
                  {(() => {
                    const { text, variant } = sourceLabel(selected.decision_source);
                    return <Badge variant={variant}>{text}</Badge>;
                  })()}
                </div>
              </div>
              <div>
                <span className="text-xs text-muted">Time</span>
                <div className="text-foreground-secondary">{formatDateTime(selected.ts)}</div>
              </div>
              <div>
                <span className="text-xs text-muted">DTE</span>
                <div className="text-foreground-secondary">{selected.target_dte}</div>
              </div>
              <div>
                <span className="text-xs text-muted">Delta</span>
                <div className="text-foreground-secondary">{selected.delta_target}</div>
              </div>
              <div>
                <span className="text-xs text-muted">Score</span>
                <div className="text-foreground-secondary">{selected.score ?? "—"}</div>
              </div>
            </div>

            {/* Portfolio / selection metadata */}
            {selected.strategy_params_json && (
              <div>
                <span className="text-xs text-muted">Selection Meta</span>
                <div className="mt-1 grid grid-cols-2 gap-2 text-sm">
                  {selected.strategy_params_json.equity != null && (
                    <div>
                      <span className="text-xs text-muted-foreground">Equity</span>
                      <div className="text-foreground-secondary">${Number(selected.strategy_params_json.equity).toLocaleString()}</div>
                    </div>
                  )}
                  {selected.strategy_params_json.lots != null && (
                    <div>
                      <span className="text-xs text-muted-foreground">Lots</span>
                      <div className="text-foreground-secondary">{String(selected.strategy_params_json.lots)}</div>
                    </div>
                  )}
                  {selected.strategy_params_json.event_signals != null && (
                    <div className="col-span-2">
                      <span className="text-xs text-muted-foreground">Event Signals</span>
                      <div className="flex flex-wrap gap-1 mt-0.5">
                        {(Array.isArray(selected.strategy_params_json.event_signals)
                          ? (selected.strategy_params_json.event_signals as string[])
                          : [String(selected.strategy_params_json.event_signals)]
                        ).map((sig) => (
                          <Badge key={sig} variant="warning" className="text-xs">{sig}</Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  {selected.strategy_params_json.mode != null && (
                    <div>
                      <span className="text-xs text-muted-foreground">Mode</span>
                      <div className="text-foreground-secondary">{String(selected.strategy_params_json.mode)}</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {selected.reason && (
              <div>
                <span className="text-xs text-muted">Reason</span>
                <p className="text-sm text-foreground-secondary mt-1">{selected.reason}</p>
              </div>
            )}
            {selected.chosen_legs_json && (
              <div>
                <span className="text-xs text-muted">Chosen Legs</span>
                <pre className="mt-1 rounded-md bg-background p-3 text-xs text-foreground-secondary overflow-x-auto">
                  {JSON.stringify(selected.chosen_legs_json, null, 2)}
                </pre>
              </div>
            )}
            {selected.strategy_params_json && (
              <div>
                <span className="text-xs text-muted">Full Strategy Params</span>
                <pre className="mt-1 rounded-md bg-background p-3 text-xs text-foreground-secondary overflow-x-auto">
                  {JSON.stringify(selected.strategy_params_json, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
