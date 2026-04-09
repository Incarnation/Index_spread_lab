import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/shared/StatCard";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { formatCurrency, formatDateTime } from "@/lib/utils";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import {
  fetchPortfolioStatus,
  fetchPortfolioHistory,
  fetchPortfolioTrades,
  fetchPortfolioConfig,
  type PortfolioStatus,
  type PortfolioHistoryDay,
  type PortfolioTrade,
  type PortfolioConfig,
} from "@/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  Activity,
  ShieldAlert,
  Zap,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

/**
 * Portfolio management dashboard -- equity tracking, event signals,
 * trade history, and active configuration.
 */
export function PortfolioPage() {
  const { tick } = useAutoRefresh(30_000);
  const [status, setStatus] = useState<PortfolioStatus | null>(null);
  const [history, setHistory] = useState<PortfolioHistoryDay[]>([]);
  const [trades, setTrades] = useState<PortfolioTrade[]>([]);
  const [config, setConfig] = useState<PortfolioConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sourceTab, setSourceTab] = useState<"all" | "scheduled" | "event">("all");
  const [configOpen, setConfigOpen] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    Promise.all([
      fetchPortfolioStatus(ac.signal).then((data) => {
        if (!ac.signal.aborted) setStatus(data);
      }),
      fetchPortfolioHistory(90, ac.signal).then((data) => {
        if (!ac.signal.aborted) setHistory(data);
      }),
      fetchPortfolioTrades(200, undefined, ac.signal).then((data) => {
        if (!ac.signal.aborted) setTrades(data);
      }),
      fetchPortfolioConfig(ac.signal).then((data) => {
        if (!ac.signal.aborted) setConfig(data);
      }),
    ])
      .catch((e) => {
        if (!ac.signal.aborted) setError(e.message ?? "Failed to load portfolio data");
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [tick]);

  const filteredTrades =
    sourceTab === "all" ? trades : trades.filter((t) => t.trade_source?.includes(sourceTab));
  const scheduledCount = trades.filter((t) => t.trade_source?.includes("scheduled")).length;
  const eventCount = trades.filter((t) => t.trade_source?.includes("event")).length;

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-foreground">Portfolio</h2>

      {error && (
        <div className="rounded-lg border border-loss/30 bg-loss-bg p-3 text-sm text-loss">
          {error}
        </div>
      )}

      {!status?.portfolio_enabled && !loading && (
        <div className="rounded-lg border border-warning/30 bg-warning-bg p-3 text-sm text-warning">
          Portfolio management is not enabled. Set <code>PORTFOLIO_ENABLED=true</code> to activate.
        </div>
      )}

      {/* KPI stat cards */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard
          title="Current Equity"
          value={status ? formatCurrency(status.equity) : "—"}
          icon={Wallet}
          trend={status && status.equity >= (status.month_start_equity ?? 0) ? "up" : "down"}
        />
        <StatCard
          title="Lots / Trade"
          value={status ? String(status.lots_per_trade) : "—"}
          icon={TrendingUp}
        />
        <StatCard
          title="Trades Today"
          value={
            status
              ? `${status.trades_today} / ${status.max_trades_per_day}`
              : "—"
          }
          icon={Activity}
          subtitle={status ? `max ${status.max_trades_per_run}/run` : undefined}
        />
        <StatCard
          title="Monthly Drawdown"
          value={status ? `${status.drawdown_pct.toFixed(1)}%` : "—"}
          icon={TrendingDown}
          trend={status && status.drawdown_pct > 5 ? "down" : undefined}
        />
        <StatCard
          title="Monthly Stop"
          value={status?.monthly_stop_active ? "ACTIVE" : "Inactive"}
          icon={ShieldAlert}
          trend={status?.monthly_stop_active ? "down" : undefined}
        />
        <StatCard
          title="Event Signals"
          value={
            status && status.event_signals.length > 0
              ? String(status.event_signals.length)
              : "None"
          }
          icon={Zap}
        />
      </div>

      {/* Market signals card */}
      {status && status.event_signals.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-warning" /> Active Market Signals
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {status.event_signals.map((sig) => (
                <Badge key={sig} variant="warning" className="text-xs px-3 py-1">
                  {sig.replace(/_/g, " ")}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Equity history chart */}
      {history.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Equity History (90d)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={history}>
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: "#6b6b80" }}
                  tickLine={false}
                  axisLine={{ stroke: "#1e1e2e" }}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#6b6b80" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#111118",
                    border: "1px solid #1e1e2e",
                    borderRadius: "6px",
                    fontSize: "12px",
                    color: "#e4e4ef",
                  }}
                  formatter={(v: number | undefined) => [formatCurrency(v ?? 0), "Equity"]}
                />
                {status && (
                  <ReferenceLine
                    y={status.month_start_equity}
                    stroke="#6b6b80"
                    strokeDasharray="3 3"
                    label={{ value: "Month Start", fill: "#6b6b80", fontSize: 10, position: "insideTopLeft" }}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="equity_end"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="Equity"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Portfolio trades table */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs
            value={sourceTab}
            onValueChange={(v) => setSourceTab(v as typeof sourceTab)}
          >
            <TabsList>
              <TabsTrigger value="all">All ({trades.length})</TabsTrigger>
              <TabsTrigger value="scheduled">
                Scheduled ({scheduledCount})
              </TabsTrigger>
              <TabsTrigger value="event">Event ({eventCount})</TabsTrigger>
            </TabsList>

            <TabsContent value={sourceTab}>
              <DataTable
                columns={[
                  { key: "trade_id", header: "Trade ID", className: "w-16" },
                  {
                    key: "trade_source",
                    header: "Source",
                    render: (r: PortfolioTrade) => (
                      <Badge
                        variant={
                          r.trade_source?.includes("event") ? "warning" : "profit"
                        }
                      >
                        {r.trade_source?.includes("event") ? "Event" : "Scheduled"}
                      </Badge>
                    ),
                  },
                  {
                    key: "event_signal",
                    header: "Signal",
                    render: (r: PortfolioTrade) => r.event_signal || "—",
                  },
                  { key: "lots", header: "Lots" },
                  {
                    key: "margin_committed",
                    header: "Margin",
                    className: "text-right",
                    render: (r: PortfolioTrade) =>
                      r.margin_committed != null
                        ? formatCurrency(r.margin_committed)
                        : "—",
                  },
                  {
                    key: "realized_pnl",
                    header: "PnL",
                    className: "text-right",
                    render: (r: PortfolioTrade) => {
                      if (r.realized_pnl == null)
                        return (
                          <span className="text-muted-foreground">pending</span>
                        );
                      return (
                        <span
                          className={
                            r.realized_pnl >= 0
                              ? "text-profit font-medium"
                              : "text-loss font-medium"
                          }
                        >
                          {formatCurrency(r.realized_pnl)}
                        </span>
                      );
                    },
                  },
                  {
                    key: "equity_change",
                    header: "Equity",
                    render: (r: PortfolioTrade) =>
                      r.equity_before != null && r.equity_after != null
                        ? `${formatCurrency(r.equity_before)} → ${formatCurrency(r.equity_after)}`
                        : "—",
                  },
                  { key: "date", header: "Date" },
                ]}
                data={filteredTrades}
                keyFn={(r) => r.id}
                emptyMessage={loading ? "Loading..." : "No portfolio trades yet"}
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Config panel */}
      {config && (
        <Card>
          <CardHeader>
            <button
              className="flex w-full items-center justify-between"
              onClick={() => setConfigOpen(!configOpen)}
            >
              <CardTitle>Active Configuration</CardTitle>
              {configOpen ? (
                <ChevronDown className="h-4 w-4 text-muted" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted" />
              )}
            </button>
          </CardHeader>
          {configOpen && (
            <CardContent>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div>
                  <h4 className="text-sm font-medium text-muted mb-3">
                    Portfolio Settings
                  </h4>
                  <dl className="space-y-2 text-sm">
                    {Object.entries(config.portfolio).map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <dt className="text-muted-foreground">
                          {k.replace(/_/g, " ")}
                        </dt>
                        <dd className="font-medium text-foreground">
                          {typeof v === "boolean" ? (v ? "Yes" : "No") : String(v)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-muted mb-3">
                    Event Settings
                  </h4>
                  <dl className="space-y-2 text-sm">
                    {Object.entries(config.event).map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <dt className="text-muted-foreground">
                          {k.replace(/_/g, " ")}
                        </dt>
                        <dd className="font-medium text-foreground">
                          {typeof v === "boolean" ? (v ? "Yes" : "No") : String(v)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>
              <div className="mt-6">
                <h4 className="text-sm font-medium text-muted mb-3">
                  Decision Settings
                </h4>
                <dl className="space-y-2 text-sm">
                  {Object.entries(config.decision).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <dt className="text-muted-foreground">
                        {k.replace(/_/g, " ")}
                      </dt>
                      <dd className="font-medium text-foreground">{String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );
}
