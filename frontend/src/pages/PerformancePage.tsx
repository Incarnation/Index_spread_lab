import { useEffect, useState } from "react";
import { StatCard } from "@/components/shared/StatCard";
import { DataTable } from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { formatCurrency, formatPercent } from "@/lib/utils";
import {
  fetchPerformanceAnalytics,
  type PerformanceAnalyticsResponse,
  type PerformanceAnalyticsBreakdownRow,
  type PerformanceAnalyticsMode,
} from "@/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import { TrendingUp, TrendingDown, BarChart3, Target, Wallet, Activity } from "lucide-react";

const LOOKBACKS = [7, 30, 90, 365] as const;

/**
 * Performance page -- equity curve, KPIs, breakdowns, and monthly table.
 */
export function PerformancePage() {
  const [lookback, setLookback] = useState<number>(90);
  const [mode, setMode] = useState<PerformanceAnalyticsMode>("realized");
  const [data, setData] = useState<PerformanceAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchPerformanceAnalytics(lookback, mode)
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [lookback, mode]);

  const s = data?.summary;
  const curve = data?.equity_curve || [];

  const breakdownColumns = [
    { key: "bucket", header: "Bucket" },
    { key: "trade_count", header: "Trades" },
    { key: "win_rate", header: "Win Rate", render: (r: PerformanceAnalyticsBreakdownRow) => formatPercent(r.win_rate) },
    { key: "net_pnl", header: "Net PnL", className: "text-right", render: (r: PerformanceAnalyticsBreakdownRow) => (
      <span className={r.net_pnl >= 0 ? "text-profit" : "text-loss"}>{formatCurrency(r.net_pnl)}</span>
    )},
    { key: "avg_pnl", header: "Avg PnL", className: "text-right", render: (r: PerformanceAnalyticsBreakdownRow) => formatCurrency(r.avg_pnl) },
    { key: "profit_factor", header: "PF", render: (r: PerformanceAnalyticsBreakdownRow) => r.profit_factor?.toFixed(2) ?? "—" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-foreground">Performance</h2>
        <div className="flex gap-2">
          <div className="flex gap-1">
            {(["realized", "combined"] as const).map((m) => (
              <Button key={m} variant={mode === m ? "default" : "ghost"} size="sm" onClick={() => setMode(m)}>
                {m}
              </Button>
            ))}
          </div>
          <div className="flex gap-1">
            {LOOKBACKS.map((lb) => (
              <Button key={lb} variant={lookback === lb ? "default" : "ghost"} size="sm" onClick={() => setLookback(lb)}>
                {lb}d
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        <StatCard title="Net PnL" value={s ? formatCurrency(s.net_pnl) : "—"} icon={Wallet} trend={s ? (s.net_pnl >= 0 ? "up" : "down") : undefined} />
        <StatCard title="Win Rate" value={s?.win_rate != null ? `${(s.win_rate * 100).toFixed(1)}%` : "—"} icon={Target} />
        <StatCard title="Trades" value={s ? String(s.trade_count) : "—"} icon={Activity} subtitle={s ? `${s.win_count}W / ${s.loss_count}L` : undefined} />
        <StatCard title="Profit Factor" value={s?.profit_factor != null ? s.profit_factor.toFixed(2) : "—"} icon={BarChart3} />
        <StatCard title="Avg Win" value={formatCurrency(s?.avg_win)} icon={TrendingUp} trend="up" />
        <StatCard title="Max Drawdown" value={formatCurrency(s?.max_drawdown)} icon={TrendingDown} trend="down" />
      </div>

      {/* Equity curve */}
      {curve.length > 0 && (
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 text-sm font-medium text-muted">Equity Curve</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={curve}>
              <defs>
                <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={{ stroke: "#1e1e2e" }} />
              <YAxis tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip contentStyle={{ backgroundColor: "#111118", border: "1px solid #1e1e2e", borderRadius: "6px", fontSize: "12px", color: "#e4e4ef" }} formatter={(v: number | undefined) => [formatCurrency(v ?? 0), "PnL"]} />
              <Area type="monotone" dataKey="cumulative_pnl" stroke="#10b981" strokeWidth={1.5} fill="url(#pnlGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Drawdown */}
      {curve.length > 0 && (
        <div className="rounded-lg border border-border bg-card p-4">
          <h3 className="mb-3 text-sm font-medium text-muted">Drawdown</h3>
          <ResponsiveContainer width="100%" height={150}>
            <AreaChart data={curve}>
              <defs>
                <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={{ stroke: "#1e1e2e" }} />
              <YAxis tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `$${v.toFixed(0)}`} />
              <Tooltip contentStyle={{ backgroundColor: "#111118", border: "1px solid #1e1e2e", borderRadius: "6px", fontSize: "12px", color: "#e4e4ef" }} formatter={(v: number | undefined) => [formatCurrency(v ?? 0), "Drawdown"]} />
              <Area type="monotone" dataKey="drawdown" stroke="#ef4444" strokeWidth={1} fill="url(#ddGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Breakdowns */}
      {data && (
        <div className="space-y-4">
          {(["side", "dte_bucket", "delta_bucket", "weekday", "hour", "source"] as const).map((dim) => {
            const rows = data.breakdowns[dim];
            if (!rows || rows.length === 0) return null;
            return (
              <div key={dim}>
                <h3 className="mb-2 text-sm font-medium text-muted capitalize">{dim.replace("_", " ")}</h3>
                <DataTable columns={breakdownColumns} data={rows} keyFn={(r) => r.bucket} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
