import { useEffect, useState } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatCurrency, formatDateTime, timeAgo } from "@/lib/utils";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { fetchTrades, type TradeRow } from "@/api";

/**
 * Trades page -- open positions and closed trade history with PnL.
 * Includes source-based filtering (All / Scheduled / Event).
 */
export function TradesPage() {
  const { tick } = useAutoRefresh(30_000);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<"all" | "scheduled" | "event">("all");

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchTrades(500, undefined, ac.signal)
      .then((data) => {
        if (!ac.signal.aborted) setTrades(data);
      })
      .catch((e) => {
        if (!ac.signal.aborted) setError(e.message ?? "Failed to load trades");
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [tick]);

  const filteredBySource =
    sourceFilter === "all"
      ? trades
      : trades.filter((t) => t.trade_source?.includes(sourceFilter));

  const openTrades = filteredBySource.filter((t) => t.status === "OPEN");
  const closedTrades = filteredBySource.filter((t) => t.status === "CLOSED");

  const scheduledCount = trades.filter((t) => t.trade_source?.includes("scheduled")).length;
  const eventCount = trades.filter((t) => t.trade_source?.includes("event")).length;

  const columns = [
    { key: "trade_id", header: "ID", className: "w-16" },
    {
      key: "status",
      header: "Status",
      render: (r: TradeRow) => (
        <Badge variant={r.status === "OPEN" ? "profit" : "muted"}>{r.status}</Badge>
      ),
    },
    {
      key: "trade_source",
      header: "Source",
      render: (r: TradeRow) => {
        if (r.trade_source?.includes("event")) return <Badge variant="warning">Event</Badge>;
        if (r.trade_source?.includes("scheduled")) return <Badge variant="profit">Sched</Badge>;
        return <span className="text-xs text-muted-foreground">{r.trade_source || "—"}</span>;
      },
    },
    {
      key: "side",
      header: "Side",
      render: (r: TradeRow) => {
        const side = r.legs?.[0]?.option_right || "—";
        return <span className={side === "put" ? "text-profit" : "text-accent"}>{side}</span>;
      },
    },
    { key: "entry_time", header: "Entered", render: (r: TradeRow) => formatDateTime(r.entry_time) },
    { key: "expiration", header: "Expiry" },
    { key: "target_dte", header: "DTE" },
    { key: "contracts", header: "Lots" },
    {
      key: "entry_credit",
      header: "Credit",
      className: "text-right",
      render: (r: TradeRow) => formatCurrency(r.entry_credit),
    },
    {
      key: "pnl",
      header: "PnL",
      className: "text-right",
      render: (r: TradeRow) => {
        const pnl = r.status === "OPEN" ? r.current_pnl : r.realized_pnl;
        return (
          <span className={pnl != null && pnl >= 0 ? "text-profit font-medium" : "text-loss font-medium"}>
            {formatCurrency(pnl)}
          </span>
        );
      },
    },
    {
      key: "targets",
      header: "TP / SL",
      render: (r: TradeRow) => (
        <span className="text-xs text-muted-foreground">
          {r.take_profit_target != null ? formatCurrency(r.take_profit_target) : "—"}
          {" / "}
          {r.stop_loss_target != null ? formatCurrency(r.stop_loss_target) : "off"}
        </span>
      ),
    },
    { key: "exit_reason", header: "Exit", render: (r: TradeRow) => r.exit_reason || "—" },
    {
      key: "updated",
      header: "Updated",
      render: (r: TradeRow) => (
        <span className="text-xs text-muted-foreground">{timeAgo(r.last_mark_ts ?? r.entry_time)}</span>
      ),
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-foreground">Trades</h2>
        <div className="flex gap-1">
          {(["all", "scheduled", "event"] as const).map((f) => (
            <Button
              key={f}
              variant={sourceFilter === f ? "default" : "ghost"}
              size="sm"
              onClick={() => setSourceFilter(f)}
            >
              {f === "all"
                ? `All (${trades.length})`
                : f === "scheduled"
                  ? `Scheduled (${scheduledCount})`
                  : `Event (${eventCount})`}
            </Button>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-loss/30 bg-loss-bg p-3 text-sm text-loss">{error}</div>
      )}

      <Tabs defaultValue="open">
        <TabsList>
          <TabsTrigger value="open">Open ({openTrades.length})</TabsTrigger>
          <TabsTrigger value="closed">Closed ({closedTrades.length})</TabsTrigger>
          <TabsTrigger value="all">All ({filteredBySource.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="open">
          <DataTable
            columns={columns}
            data={openTrades}
            keyFn={(r) => r.trade_id}
            emptyMessage={loading ? "Loading..." : "No open trades"}
          />
        </TabsContent>

        <TabsContent value="closed">
          <DataTable
            columns={columns}
            data={closedTrades}
            keyFn={(r) => r.trade_id}
            emptyMessage={loading ? "Loading..." : "No closed trades"}
          />
        </TabsContent>

        <TabsContent value="all">
          <DataTable
            columns={columns}
            data={filteredBySource}
            keyFn={(r) => r.trade_id}
            emptyMessage={loading ? "Loading..." : "No trades"}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
