import { useEffect, useState } from "react";
import { StatCard } from "@/components/shared/StatCard";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { formatCurrency, formatDateTime, timeAgo } from "@/lib/utils";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import {
  fetchTrades,
  fetchPerformanceAnalytics,
  fetchAdminPreflight,
  fetchPortfolioStatus,
  fetchPortfolioHistory,
  type TradeRow,
  type PerformanceAnalyticsResponse,
  type AdminPreflightResponse,
  type PortfolioStatus,
  type PortfolioHistoryDay,
} from "@/api";
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Wallet,
  Target,
  Zap,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

/**
 * Overview / Home page -- cockpit view with key stats, mini equity curve, and recent trades.
 * Shows portfolio-aware KPIs when portfolio management is active.
 */
export function OverviewPage() {
  const { tick } = useAutoRefresh(30_000);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [perf, setPerf] = useState<PerformanceAnalyticsResponse | null>(null);
  const [preflight, setPreflight] = useState<AdminPreflightResponse | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioStatus | null>(null);
  const [equityHistory, setEquityHistory] = useState<PortfolioHistoryDay[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetchTrades(100).then(setTrades),
      fetchPerformanceAnalytics(90, "realized").then(setPerf),
      fetchPortfolioStatus().then(setPortfolio).catch(() => {}),
      fetchPortfolioHistory(90).then(setEquityHistory).catch(() => {}),
    ])
      .catch((e) => setError(e.message ?? "Failed to load data"))
      .finally(() => setLoading(false));

    fetchAdminPreflight().then(setPreflight).catch(() => {});
  }, [tick]);

  const openTrades = trades.filter((t) => t.status === "OPEN");
  const closedTrades = trades.filter((t) => t.status === "CLOSED");
  const summary = perf?.summary;
  const portfolioActive = portfolio?.portfolio_enabled === true;

  const todayPnl = (() => {
    if (portfolioActive && portfolio) return portfolio.daily_pnl;
    const today = new Date().toISOString().slice(0, 10);
    const curve = perf?.equity_curve || [];
    const todayPoint = curve.find((p) => p.date === today);
    return todayPoint?.daily_pnl ?? null;
  })();

  const usePortfolioEquity = portfolioActive && equityHistory.length > 0;

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-foreground">Overview</h2>

      {error && (
        <div className="rounded-lg border border-loss/30 bg-loss-bg p-3 text-sm text-loss">{error}</div>
      )}

      {/* KPI stat cards */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        {portfolioActive && portfolio ? (
          <StatCard
            title="Current Equity"
            value={formatCurrency(portfolio.equity)}
            icon={Wallet}
            trend={portfolio.equity >= portfolio.month_start_equity ? "up" : "down"}
          />
        ) : (
          <StatCard
            title="Open Trades"
            value={String(openTrades.length)}
            icon={Activity}
            subtitle={`${closedTrades.length} closed`}
          />
        )}
        <StatCard
          title="Today's PnL"
          value={todayPnl != null ? formatCurrency(todayPnl) : "—"}
          icon={todayPnl != null && todayPnl >= 0 ? TrendingUp : TrendingDown}
          trend={todayPnl != null ? (todayPnl >= 0 ? "up" : "down") : undefined}
        />
        {portfolioActive && portfolio ? (
          <StatCard
            title="Trades Today"
            value={`${portfolio.trades_today} / ${portfolio.max_trades_per_day}`}
            icon={Activity}
            subtitle={`max ${portfolio.max_trades_per_run}/run`}
          />
        ) : (
          <StatCard
            title="Total PnL"
            value={summary ? formatCurrency(summary.net_pnl) : "—"}
            icon={Wallet}
            trend={summary ? (summary.net_pnl >= 0 ? "up" : "down") : undefined}
          />
        )}
        {portfolioActive && portfolio ? (
          <StatCard
            title="Lots / Trade"
            value={String(portfolio.lots_per_trade)}
            icon={TrendingUp}
          />
        ) : (
          <StatCard
            title="Win Rate"
            value={summary?.win_rate != null ? `${(summary.win_rate * 100).toFixed(1)}%` : "—"}
            icon={Target}
            subtitle={summary ? `${summary.trade_count} trades` : undefined}
          />
        )}
        <StatCard
          title={portfolioActive ? "Win Rate" : "Profit Factor"}
          value={
            portfolioActive
              ? summary?.win_rate != null
                ? `${(summary.win_rate * 100).toFixed(1)}%`
                : "—"
              : summary?.profit_factor != null
                ? summary.profit_factor.toFixed(2)
                : "—"
          }
          icon={portfolioActive ? Target : BarChart3}
          subtitle={portfolioActive && summary ? `${summary.trade_count} trades` : undefined}
        />
        <StatCard
          title="Max Drawdown"
          value={summary?.max_drawdown != null ? formatCurrency(summary.max_drawdown) : "—"}
          icon={TrendingDown}
          trend={summary?.max_drawdown != null ? "down" : undefined}
        />
      </div>

      {/* Market signals compact card */}
      {portfolioActive && portfolio && portfolio.event_signals.length > 0 && (
        <div className="rounded-lg border border-warning/30 bg-warning-bg p-3">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="h-4 w-4 text-warning" />
            <span className="text-xs font-medium text-warning">Active Market Signals</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {portfolio.event_signals.map((sig) => (
              <Badge key={sig} variant="warning" className="text-xs">
                {sig.replace(/_/g, " ")}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Mini equity curve -- use portfolio equity when active */}
      {usePortfolioEquity ? (
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 text-sm font-medium text-muted">Portfolio Equity (90d)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={equityHistory}>
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
              {portfolio && (
                <ReferenceLine
                  y={portfolio.month_start_equity}
                  stroke="#6b6b80"
                  strokeDasharray="3 3"
                />
              )}
              <Line
                type="monotone"
                dataKey="equity_end"
                stroke="#3b82f6"
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        perf && perf.equity_curve.length > 0 && (
          <div className="rounded-lg border border-border bg-card p-4">
            <h3 className="mb-3 text-sm font-medium text-muted">Equity Curve (90d)</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={perf.equity_curve}>
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
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#111118",
                    border: "1px solid #1e1e2e",
                    borderRadius: "6px",
                    fontSize: "12px",
                    color: "#e4e4ef",
                  }}
                  formatter={(v: number | undefined) => [formatCurrency(v ?? 0), "Cumulative PnL"]}
                />
                <Line
                  type="monotone"
                  dataKey="cumulative_pnl"
                  stroke="#10b981"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )
      )}

      {/* Recent trades */}
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-muted">Recent Trades</h3>
        <DataTable
          columns={[
            { key: "trade_id", header: "ID", className: "w-16" },
            { key: "status", header: "Status", render: (r: TradeRow) => (
              <Badge variant={r.status === "OPEN" ? "profit" : "muted"}>{r.status}</Badge>
            )},
            {
              key: "trade_source",
              header: "Source",
              render: (r: TradeRow) => {
                if (r.trade_source?.includes("event")) return <Badge variant="warning">Event</Badge>;
                if (r.trade_source?.includes("scheduled")) return <Badge variant="profit">Sched</Badge>;
                return <span className="text-xs text-muted-foreground">{r.trade_source || "—"}</span>;
              },
            },
            { key: "entry_time", header: "Entered", render: (r: TradeRow) => formatDateTime(r.entry_time) },
            { key: "expiration", header: "Expiry" },
            { key: "target_dte", header: "DTE" },
            { key: "contracts", header: "Lots" },
            {
              key: "pnl",
              header: "PnL",
              className: "text-right",
              render: (r: TradeRow) => {
                const pnl = r.status === "OPEN" ? r.current_pnl : r.realized_pnl;
                return (
                  <span className={pnl != null && pnl >= 0 ? "text-profit" : "text-loss"}>
                    {formatCurrency(pnl)}
                  </span>
                );
              },
            },
            { key: "last_mark_ts", header: "Updated", render: (r: TradeRow) => timeAgo(r.last_mark_ts ?? r.entry_time) },
          ]}
          data={trades.slice(0, 15)}
          keyFn={(r) => r.trade_id}
          emptyMessage={loading ? "Loading..." : "No trades"}
        />
      </div>

      {/* Pipeline status */}
      {preflight && preflight.warnings.length > 0 && (
        <div className="rounded-lg border border-warning/30 bg-warning-bg p-3">
          <h4 className="text-xs font-medium text-warning mb-1">Pipeline Warnings</h4>
          <ul className="space-y-0.5">
            {preflight.warnings.map((w, i) => (
              <li key={i} className="text-xs text-warning/80">{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
