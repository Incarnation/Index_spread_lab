import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/shared/StatCard";
import { Badge } from "@/components/ui/badge";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { fetchPortfolioConfig, type PortfolioConfig } from "@/api";
import { FlaskConical, TrendingUp, Target, BarChart3, TrendingDown, ShieldCheck } from "lucide-react";

/**
 * Backtest-validated strategy metrics for the C3 configuration
 * (W15, E7, M2, TP50) from the Phase 2 event-only optimizer run.
 * profit_factor capped at 99 (effectively infinite — zero losing days).
 */
const BACKTEST_METRICS = {
  sharpe: 10.78,
  return_pct: 215.8,
  max_dd_pct: 0.0,
  total_trades: 208,
  win_rate: 1.0,
  profit_factor: 99.0,
};

/**
 * Walk-forward validation window summaries for C3 config.
 * 9 rolling windows (3-month train / 1-month test) from 2025-03 to 2026-03.
 */
const WALKFORWARD_WINDOWS = [
  { window: "2025-03 → 2025-07", train_sharpe: 9.18, test_sharpe: 5.08, verdict: "PASS" },
  { window: "2025-04 → 2025-08", train_sharpe: 10.70, test_sharpe: 8.11, verdict: "PASS" },
  { window: "2025-05 → 2025-09", train_sharpe: 7.94, test_sharpe: 6.13, verdict: "PASS" },
  { window: "2025-06 → 2025-10", train_sharpe: 6.59, test_sharpe: 7.65, verdict: "PASS" },
  { window: "2025-07 → 2025-11", train_sharpe: 7.30, test_sharpe: 7.24, verdict: "PASS" },
  { window: "2025-08 → 2025-12", train_sharpe: 6.79, test_sharpe: 3.55, verdict: "PASS" },
  { window: "2025-09 → 2026-01", train_sharpe: 4.75, test_sharpe: 3.92, verdict: "PASS" },
  { window: "2025-10 → 2026-02", train_sharpe: 5.05, test_sharpe: 7.71, verdict: "PASS" },
  { window: "2025-11 → 2026-03", train_sharpe: 5.24, test_sharpe: 2.04, verdict: "PASS" },
];

/**
 * Strategy configuration page -- shows active portfolio/event config and
 * backtest validation summary. Replaces the old BacktestPage.
 */
export function StrategyConfigPage() {
  const { tick } = useAutoRefresh(60_000);
  const [config, setConfig] = useState<PortfolioConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchPortfolioConfig(ac.signal)
      .then((data) => {
        if (!ac.signal.aborted) setConfig(data);
      })
      .catch((e) => {
        if (!ac.signal.aborted) setError(e.message ?? "Failed to load config");
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [tick]);

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-lg font-semibold text-foreground">Strategy</h2>
        <div className="py-12 text-center text-muted-foreground text-sm">Loading...</div>
      </div>
    );
  }

  if (error || !config) {
    return (
      <div className="space-y-6">
        <h2 className="text-lg font-semibold text-foreground">Strategy</h2>
        <Card>
          <CardContent className="py-12">
            <div className="text-center text-muted-foreground text-sm">
              <FlaskConical className="mx-auto h-8 w-8 mb-2 opacity-30" />
              <p>{error ? `Error: ${error}` : "No configuration available."}</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-foreground">Strategy</h2>
        <Badge variant={config.portfolio.enabled ? "profit" : "warning"}>
          {config.portfolio.enabled ? "Portfolio Active" : "Portfolio Inactive"}
        </Badge>
      </div>

      {/* Active config */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4" /> Active Configuration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            {/* Portfolio settings */}
            <div>
              <h4 className="text-sm font-medium text-muted mb-3">Portfolio</h4>
              <dl className="space-y-2 text-sm">
                <ConfigRow label="Starting Capital" value={`$${config.portfolio.starting_capital.toLocaleString()}`} />
                <ConfigRow label="Max Trades / Day" value={String(config.portfolio.max_trades_per_day)} />
                <ConfigRow label="Max Trades / Run" value={String(config.portfolio.max_trades_per_run)} />
                <ConfigRow label="Monthly DD Limit" value={`${(config.portfolio.monthly_drawdown_limit * 100).toFixed(0)}%`} />
                <ConfigRow label="Lot Scaling" value={`1 lot per $${config.portfolio.lot_per_equity.toLocaleString()}`} />
                <ConfigRow label="Max Equity Risk" value={`${(config.portfolio.max_equity_risk_pct * 100).toFixed(0)}%`} />
                <ConfigRow label="Calls Only" value={config.portfolio.calls_only ? "Yes" : "No"} />
              </dl>
            </div>

            {/* Event settings */}
            <div>
              <h4 className="text-sm font-medium text-muted mb-3">Event Signals</h4>
              <dl className="space-y-2 text-sm">
                <ConfigRow label="Enabled" value={config.event.enabled ? "Yes" : "No"} />
                <ConfigRow label="Budget Mode" value={config.event.budget_mode} />
                <ConfigRow label="Max Event Trades" value={String(config.event.max_trades)} />
                <ConfigRow label="SPX Drop Threshold" value={`${(config.event.spx_drop_threshold * 100).toFixed(1)}%`} />
                <ConfigRow label="VIX Spike Threshold" value={`${(config.event.vix_spike_threshold * 100).toFixed(0)}%`} />
                <ConfigRow label="DTE Range" value={`${config.event.min_dte} – ${config.event.max_dte}`} />
                <ConfigRow label="Delta Range" value={`${config.event.min_delta} – ${config.event.max_delta}`} />
                <ConfigRow label="Rally Avoidance" value={config.event.rally_avoidance ? "On" : "Off"} />
              </dl>
            </div>
          </div>

          {/* Decision settings */}
          <div className="mt-6 pt-6 border-t border-border">
            <h4 className="text-sm font-medium text-muted mb-3">Decision Schedule</h4>
            <dl className="grid grid-cols-2 gap-2 text-sm lg:grid-cols-4">
              <ConfigRow label="Entry Times" value={config.decision.entry_times} />
              <ConfigRow label="DTE Targets" value={config.decision.dte_targets} />
              <ConfigRow label="Delta Targets" value={config.decision.delta_targets} />
              <ConfigRow label="Spread Width" value={`${config.decision.spread_width_points} pts`} />
            </dl>
          </div>
        </CardContent>
      </Card>

      {/* Backtest performance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4" /> Backtest Performance (Sharpe-Optimal)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
            <StatCard
              title="Sharpe Ratio"
              value={BACKTEST_METRICS.sharpe.toFixed(2)}
              icon={TrendingUp}
            />
            <StatCard
              title="Return"
              value={`${BACKTEST_METRICS.return_pct.toFixed(1)}%`}
              icon={TrendingUp}
              trend="up"
            />
            <StatCard
              title="Max Drawdown"
              value={`${BACKTEST_METRICS.max_dd_pct.toFixed(1)}%`}
              icon={TrendingDown}
              trend="down"
            />
            <StatCard
              title="Total Trades"
              value={String(BACKTEST_METRICS.total_trades)}
              icon={BarChart3}
            />
            <StatCard
              title="Win Rate"
              value={`${(BACKTEST_METRICS.win_rate * 100).toFixed(0)}%`}
              icon={Target}
            />
            <StatCard
              title="Profit Factor"
              value={BACKTEST_METRICS.profit_factor.toFixed(2)}
              icon={BarChart3}
            />
          </div>
        </CardContent>
      </Card>

      {/* Walk-forward validation */}
      <Card>
        <CardHeader>
          <CardTitle>Walk-Forward Validation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="pb-2 text-xs font-medium text-muted">Window</th>
                  <th className="pb-2 text-xs font-medium text-muted text-right">Train Sharpe</th>
                  <th className="pb-2 text-xs font-medium text-muted text-right">Test Sharpe</th>
                  <th className="pb-2 text-xs font-medium text-muted text-center">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {WALKFORWARD_WINDOWS.map((w) => (
                  <tr key={w.window} className="border-b border-border/50">
                    <td className="py-2 text-foreground-secondary">{w.window}</td>
                    <td className="py-2 text-right text-foreground-secondary">{w.train_sharpe.toFixed(2)}</td>
                    <td className="py-2 text-right text-foreground-secondary">{w.test_sharpe.toFixed(2)}</td>
                    <td className="py-2 text-center">
                      <Badge variant={w.verdict === "PASS" ? "profit" : "loss"}>{w.verdict}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            All 9 windows pass validation — positive test Sharpe across every out-of-sample period.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Simple key-value row for the config panel.
 */
function ConfigRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-medium text-foreground">{value}</dd>
    </div>
  );
}
