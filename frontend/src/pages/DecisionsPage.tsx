import { useEffect, useState } from "react";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatDateTime } from "@/lib/utils";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { fetchTradeDecisions, type TradeDecision } from "@/api";
import { X } from "lucide-react";

/**
 * Decisions page -- trade entry/skip log with reasoning and model scores.
 */
export function DecisionsPage() {
  const { tick } = useAutoRefresh(30_000);
  const [decisions, setDecisions] = useState<TradeDecision[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<TradeDecision | null>(null);
  const [filter, setFilter] = useState<"all" | "TRADE" | "SKIP">("all");

  useEffect(() => {
    setLoading(true);
    fetchTradeDecisions(200)
      .then(setDecisions)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [tick]);

  const filtered = filter === "all" ? decisions : decisions.filter((d) => d.decision === filter);

  const tradeCt = decisions.filter((d) => d.decision === "TRADE").length;
  const skipCt = decisions.filter((d) => d.decision === "SKIP").length;

  return (
    <div className="space-y-4">
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
          { key: "ts", header: "Time", render: (r: TradeDecision) => formatDateTime(r.ts) },
          { key: "target_dte", header: "DTE" },
          { key: "delta_target", header: "Delta", render: (r: TradeDecision) => r.delta_target?.toFixed(2) ?? "—" },
          { key: "score", header: "Score", render: (r: TradeDecision) => r.score?.toFixed(2) ?? "—" },
          { key: "decision_source", header: "Source" },
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
              <div>
                <span className="text-xs text-muted">Source</span>
                <div className="text-foreground-secondary">{selected.decision_source}</div>
              </div>
            </div>
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
                <span className="text-xs text-muted">Strategy Params</span>
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
