import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { DataTable } from "@/components/shared/DataTable";
import { StatCard } from "@/components/shared/StatCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatCurrency } from "@/lib/utils";
import { fetchBacktestResults, type BacktestStrategyResult, type BacktestResponse } from "@/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { FlaskConical, TrendingUp, Target, BarChart3, TrendingDown } from "lucide-react";

const CHART_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

/**
 * Backtest results page -- compare strategy performance from offline backtesting.
 * Fetches from GET /api/backtest-results (returns cached/precomputed results).
 */
export function BacktestPage() {
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchBacktestResults()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-lg font-semibold text-foreground">Backtest Results</h2>
        <div className="py-12 text-center text-muted-foreground text-sm">Loading...</div>
      </div>
    );
  }

  if (error || !data || data.strategies.length === 0) {
    return (
      <div className="space-y-6">
        <h2 className="text-lg font-semibold text-foreground">Backtest Results</h2>
        <Card>
          <CardContent className="py-12">
            <div className="text-center text-muted-foreground text-sm">
              <FlaskConical className="mx-auto h-8 w-8 mb-2 opacity-30" />
              <p>{error ? `Error: ${error}` : "No backtest results available."}</p>
              <p className="text-xs mt-1 text-muted">
                Run the backtest script to generate results, then upload or restart the server.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const strategies = data.strategies;
  const best = strategies.reduce((a, b) => ((a.sharpe ?? 0) > (b.sharpe ?? 0) ? a : b));
  const selected = strategies.find((s) => s.name === selectedStrategy) || best;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-foreground">Backtest Results</h2>
        {data.generated_at && (
          <span className="text-xs text-muted-foreground">Generated: {data.generated_at}</span>
        )}
      </div>

      {/* Strategy comparison table */}
      <DataTable
        columns={[
          {
            key: "name",
            header: "Strategy",
            render: (r: BacktestStrategyResult) => (
              <div className="flex items-center gap-2">
                <span className="font-medium text-foreground">{r.name}</span>
                {r.name === best.name && <Badge variant="profit">Best</Badge>}
              </div>
            ),
          },
          { key: "total_trades", header: "Trades" },
          { key: "win_rate", header: "Win Rate", render: (r: BacktestStrategyResult) => `${(r.win_rate * 100).toFixed(1)}%` },
          {
            key: "total_pnl",
            header: "Total PnL",
            className: "text-right",
            render: (r: BacktestStrategyResult) => (
              <span className={r.total_pnl >= 0 ? "text-profit font-medium" : "text-loss font-medium"}>
                {formatCurrency(r.total_pnl)}
              </span>
            ),
          },
          { key: "avg_pnl", header: "Avg PnL", className: "text-right", render: (r: BacktestStrategyResult) => formatCurrency(r.avg_pnl) },
          { key: "profit_factor", header: "PF", render: (r: BacktestStrategyResult) => r.profit_factor?.toFixed(2) ?? "—" },
          { key: "sharpe", header: "Sharpe", render: (r: BacktestStrategyResult) => r.sharpe?.toFixed(2) ?? "—" },
          {
            key: "max_drawdown",
            header: "Max DD",
            className: "text-right",
            render: (r: BacktestStrategyResult) => (
              <span className="text-loss">{formatCurrency(-r.max_drawdown)}</span>
            ),
          },
        ]}
        data={strategies}
        keyFn={(r) => r.name}
        onRowClick={(r) => setSelectedStrategy(r.name)}
      />

      {/* Equity curves overlay */}
      {strategies.some((s) => s.equity_curve.length > 0) && (
        <Card>
          <CardHeader>
            <CardTitle>Equity Curves</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart>
                <XAxis
                  dataKey="date"
                  type="category"
                  allowDuplicatedCategory={false}
                  tick={{ fontSize: 10, fill: "#6b6b80" }}
                  tickLine={false}
                  axisLine={{ stroke: "#1e1e2e" }}
                />
                <YAxis tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip contentStyle={{ backgroundColor: "#111118", border: "1px solid #1e1e2e", borderRadius: "6px", fontSize: "12px", color: "#e4e4ef" }} />
                <Legend wrapperStyle={{ fontSize: "11px", color: "#a0a0b8" }} />
                {strategies.map((s, i) => (
                  <Line
                    key={s.name}
                    data={s.equity_curve}
                    dataKey="cumulative_pnl"
                    name={s.name}
                    stroke={CHART_COLORS[i % CHART_COLORS.length]}
                    strokeWidth={s.name === selected.name ? 2 : 1}
                    opacity={s.name === selected.name ? 1 : 0.5}
                    dot={false}
                    type="monotone"
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Selected strategy detail */}
      <Card>
        <CardHeader>
          <CardTitle>
            Detail: {selected.name}
            <div className="flex gap-1 mt-2">
              {strategies.map((s) => (
                <Button
                  key={s.name}
                  variant={s.name === selected.name ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setSelectedStrategy(s.name)}
                >
                  {s.name}
                </Button>
              ))}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5 mb-4">
            <StatCard title="Total PnL" value={formatCurrency(selected.total_pnl)} trend={selected.total_pnl >= 0 ? "up" : "down"} icon={TrendingUp} />
            <StatCard title="Win Rate" value={`${(selected.win_rate * 100).toFixed(1)}%`} icon={Target} subtitle={`${selected.wins}W / ${selected.losses}L`} />
            <StatCard title="Profit Factor" value={selected.profit_factor?.toFixed(2) ?? "—"} icon={BarChart3} />
            <StatCard title="Sharpe" value={selected.sharpe?.toFixed(2) ?? "—"} icon={TrendingUp} />
            <StatCard title="Max Drawdown" value={formatCurrency(-selected.max_drawdown)} icon={TrendingDown} trend="down" />
          </div>

          {/* Monthly breakdown */}
          {selected.monthly.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-muted mb-2">Monthly Breakdown</h4>
              <DataTable
                columns={[
                  { key: "month", header: "Month" },
                  { key: "trades", header: "Trades" },
                  {
                    key: "pnl",
                    header: "PnL",
                    className: "text-right",
                    render: (r: { month: string; pnl: number; trades: number }) => (
                      <span className={r.pnl >= 0 ? "text-profit" : "text-loss"}>{formatCurrency(r.pnl)}</span>
                    ),
                  },
                ]}
                data={selected.monthly}
                keyFn={(r) => r.month}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
