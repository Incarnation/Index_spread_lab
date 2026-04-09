import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/shared/StatCard";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { formatCurrency, formatDateTime, timeAgo } from "@/lib/utils";
import {
  fetchModelOps,
  fetchModelAccuracy,
  fetchModelCalibration,
  fetchModelPnlAttribution,
  fetchModelPredictions,
  fetchPortfolioConfig,
  type ModelOpsResponse,
  type ModelAccuracyResponse,
  type ModelCalibrationResponse,
  type ModelPnlAttributionResponse,
  type ModelPredictionRow,
  type ModelPredictionsResponse,
} from "@/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  ReferenceLine,
  Legend,
} from "recharts";
import { Brain, Activity, Database, AlertTriangle, TrendingUp, TrendingDown, Info } from "lucide-react";

/**
 * Model Monitor page -- comprehensive model health dashboard with:
 * - Model status and active version info
 * - Accuracy/precision/recall over time
 * - Calibration curve
 * - Confusion matrix
 * - PnL attribution
 * - Individual prediction browser
 */
export function ModelMonitorPage() {
  const { tick } = useAutoRefresh(60_000);
  const [ops, setOps] = useState<ModelOpsResponse | null>(null);
  const [accuracy, setAccuracy] = useState<ModelAccuracyResponse | null>(null);
  const [calibration, setCalibration] = useState<ModelCalibrationResponse | null>(null);
  const [attribution, setAttribution] = useState<ModelPnlAttributionResponse | null>(null);
  const [predictions, setPredictions] = useState<ModelPredictionsResponse | null>(null);
  const [predPage, setPredPage] = useState(0);
  const [predFilter, setPredFilter] = useState<"all" | "TRADE" | "SKIP">("all");
  const [portfolioEnabled, setPortfolioEnabled] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setError(null);
    Promise.all([
      fetchModelOps(undefined, ac.signal).then((o) => {
        if (!ac.signal.aborted) setOps(o);
      }),
      fetchModelAccuracy(undefined, "week", ac.signal).then((a) => {
        if (!ac.signal.aborted) setAccuracy(a);
      }),
      fetchModelCalibration(undefined, 10, ac.signal).then((c) => {
        if (!ac.signal.aborted) setCalibration(c);
      }),
      fetchModelPnlAttribution(undefined, ac.signal).then((p) => {
        if (!ac.signal.aborted) setAttribution(p);
      }),
      fetchPortfolioConfig(ac.signal)
        .then((cfg) => {
          if (!ac.signal.aborted) setPortfolioEnabled(cfg.portfolio.enabled);
        })
        .catch(() => {
          if (!ac.signal.aborted) {
          }
        }),
    ]).catch((e) => {
      if (!ac.signal.aborted) setError(e.message ?? "Failed to load model data");
    });
    return () => ac.abort();
  }, [tick]);

  useEffect(() => {
    const ac = new AbortController();
    const dec = predFilter === "all" ? undefined : predFilter;
    fetchModelPredictions(50, predPage * 50, undefined, dec, undefined, undefined, ac.signal)
      .then((p) => {
        if (!ac.signal.aborted) setPredictions(p);
      })
      .catch((e) => {
        if (!ac.signal.aborted) setError(e.message ?? "Failed to load predictions");
      });
    return () => ac.abort();
  }, [predPage, predFilter, tick]);

  const active = ops?.active_model_version;
  const run = ops?.latest_training_run;

  // Compute confusion matrix totals
  const confMatrix = accuracy?.windows.reduce(
    (acc, w) => ({
      tp: acc.tp + w.true_positive,
      fp: acc.fp + w.false_positive,
      tn: acc.tn + w.true_negative,
      fn: acc.fn + w.false_negative,
    }),
    { tp: 0, fp: 0, tn: 0, fn: 0 }
  );

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-foreground">Model Monitor</h2>

      {portfolioEnabled && (
        <div className="rounded-lg border border-accent/30 bg-accent/5 p-3 flex items-start gap-2">
          <Info className="h-4 w-4 text-accent mt-0.5 shrink-0" />
          <p className="text-sm text-accent">
            Portfolio management is active. ML models are running in shadow mode &mdash;
            predictions are logged but do not influence trade decisions.
          </p>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-loss/30 bg-loss-bg p-3 text-sm text-loss">{error}</div>
      )}

      {/* Warnings */}
      {ops && ops.warnings.length > 0 && (
        <div className="rounded-lg border border-warning/30 bg-warning-bg p-3">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className="h-4 w-4 text-warning" />
            <span className="text-xs font-medium text-warning">Model Warnings</span>
          </div>
          {ops.warnings.map((w, i) => (
            <p key={i} className="text-xs text-warning/80">{w}</p>
          ))}
        </div>
      )}

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard title="Model Name" value={ops?.model_name ?? "—"} icon={Brain} />
        <StatCard title="Total Predictions" value={ops?.counts.model_predictions?.toLocaleString() ?? "—"} icon={Database} />
        <StatCard title="Predictions (24h)" value={ops?.counts.model_predictions_24h?.toLocaleString() ?? "—"} icon={Activity} />
        <StatCard title="Last Prediction" value={timeAgo(ops?.latest_prediction_ts)} icon={Activity} />
      </div>

      {/* Active model info */}
      {active && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-4 w-4" /> Active Model
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm lg:grid-cols-4">
              <div>
                <span className="text-xs text-muted">Version</span>
                <div className="text-foreground">{active.version}</div>
              </div>
              <div>
                <span className="text-xs text-muted">Rollout</span>
                <div><Badge variant={active.rollout_status === "active" ? "profit" : "warning"}>{active.rollout_status}</Badge></div>
              </div>
              <div>
                <span className="text-xs text-muted">Created</span>
                <div className="text-foreground-secondary">{formatDateTime(active.created_at_utc)}</div>
              </div>
              <div>
                <span className="text-xs text-muted">Promoted</span>
                <div className="text-foreground-secondary">{formatDateTime(active.promoted_at_utc)}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Accuracy over time */}
      {accuracy && accuracy.windows.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Accuracy Over Time (Weekly)</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={accuracy.windows.map((w) => ({
                period: w.period?.slice(0, 10) ?? "",
                accuracy: w.accuracy != null ? +(w.accuracy * 100).toFixed(1) : null,
                precision: w.precision != null ? +(w.precision * 100).toFixed(1) : null,
                recall: w.recall != null ? +(w.recall * 100).toFixed(1) : null,
              }))}>
                <XAxis dataKey="period" tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={{ stroke: "#1e1e2e" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
                <Tooltip contentStyle={{ backgroundColor: "#111118", border: "1px solid #1e1e2e", borderRadius: "6px", fontSize: "12px", color: "#e4e4ef" }} />
                <Legend wrapperStyle={{ fontSize: "11px", color: "#a0a0b8" }} />
                <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#3b82f6" strokeWidth={1.5} dot={{ r: 2 }} />
                <Line type="monotone" dataKey="precision" name="Precision" stroke="#10b981" strokeWidth={1.5} dot={{ r: 2 }} />
                <Line type="monotone" dataKey="recall" name="Recall" stroke="#f59e0b" strokeWidth={1.5} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Calibration curve */}
        {calibration && calibration.bins.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Calibration Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <ScatterChart margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                  <XAxis
                    dataKey="predicted_avg"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 10, fill: "#6b6b80" }}
                    tickLine={false}
                    axisLine={{ stroke: "#1e1e2e" }}
                    name="Predicted"
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  />
                  <YAxis
                    dataKey="observed_rate"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 10, fill: "#6b6b80" }}
                    tickLine={false}
                    axisLine={false}
                    name="Observed"
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  />
                  <Tooltip contentStyle={{ backgroundColor: "#111118", border: "1px solid #1e1e2e", borderRadius: "6px", fontSize: "12px", color: "#e4e4ef" }} />
                  <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#3b3b50" strokeDasharray="3 3" />
                  <Scatter
                    data={calibration.bins.filter((b) => b.observed_rate != null)}
                    fill="#3b82f6"
                  />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="mt-1 text-xs text-muted-foreground text-center">
                Dashed line = perfect calibration. Points above = model underestimates; below = overestimates.
              </p>
            </CardContent>
          </Card>
        )}

        {/* Confusion matrix */}
        {confMatrix && (confMatrix.tp + confMatrix.fp + confMatrix.tn + confMatrix.fn) > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Confusion Matrix (All Time)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-0 text-center text-sm max-w-xs mx-auto">
                <div />
                <div className="py-2 text-xs text-muted font-medium">Actual Win</div>
                <div className="py-2 text-xs text-muted font-medium">Actual Loss</div>

                <div className="py-3 text-xs text-muted font-medium text-right pr-3">Pred TRADE</div>
                <div className="py-3 bg-profit-bg rounded-tl-md border border-border font-semibold text-profit">
                  {confMatrix.tp}
                  <div className="text-[10px] text-muted-foreground">TP</div>
                </div>
                <div className="py-3 bg-loss-bg rounded-tr-md border border-border font-semibold text-loss">
                  {confMatrix.fp}
                  <div className="text-[10px] text-muted-foreground">FP</div>
                </div>

                <div className="py-3 text-xs text-muted font-medium text-right pr-3">Pred SKIP</div>
                <div className="py-3 bg-warning-bg rounded-bl-md border border-border font-semibold text-warning">
                  {confMatrix.fn}
                  <div className="text-[10px] text-muted-foreground">FN</div>
                </div>
                <div className="py-3 bg-card rounded-br-md border border-border font-semibold text-foreground-secondary">
                  {confMatrix.tn}
                  <div className="text-[10px] text-muted-foreground">TN</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* PnL Attribution */}
      {attribution && (
        <Card>
          <CardHeader>
            <CardTitle>PnL Attribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-5 mb-4">
              <StatCard
                title="Baseline PnL"
                value={formatCurrency(attribution.baseline_pnl)}
                trend={attribution.baseline_pnl >= 0 ? "up" : "down"}
                subtitle={`${attribution.total_candidates} candidates`}
              />
              <StatCard
                title="Model PnL"
                value={formatCurrency(attribution.model_pnl)}
                trend={attribution.model_pnl >= 0 ? "up" : "down"}
                subtitle={`${attribution.trade_count} traded`}
                icon={TrendingUp}
              />
              <StatCard
                title="Losses Avoided"
                value={formatCurrency(attribution.saved_pnl)}
                trend="up"
                icon={TrendingUp}
              />
              <StatCard
                title="Wins Missed"
                value={formatCurrency(attribution.missed_pnl)}
                trend="down"
                icon={TrendingDown}
              />
              <StatCard
                title="Net Impact"
                value={formatCurrency(attribution.net_impact)}
                trend={attribution.net_impact >= 0 ? "up" : "down"}
                subtitle={`${attribution.skip_count} skipped`}
              />
            </div>

            <ResponsiveContainer width="100%" height={180}>
              <BarChart
                data={[
                  { name: "Baseline", value: attribution.baseline_pnl },
                  { name: "Model", value: attribution.model_pnl },
                  { name: "Saved", value: attribution.saved_pnl },
                  { name: "Missed", value: -attribution.missed_pnl },
                  { name: "Net Impact", value: attribution.net_impact },
                ]}
                layout="vertical"
              >
                <XAxis type="number" tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={{ stroke: "#1e1e2e" }} tickFormatter={(v: number) => `$${v.toFixed(0)}`} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: "#a0a0b8" }} tickLine={false} axisLine={false} width={80} />
                <Tooltip contentStyle={{ backgroundColor: "#111118", border: "1px solid #1e1e2e", borderRadius: "6px", fontSize: "12px", color: "#e4e4ef" }} formatter={(v: number | undefined) => [formatCurrency(v ?? 0)]} />
                <ReferenceLine x={0} stroke="#2a2a3e" />
                <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Prediction browser */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Prediction Browser</CardTitle>
            <div className="flex gap-1">
              {(["all", "TRADE", "SKIP"] as const).map((f) => (
                <Button key={f} variant={predFilter === f ? "default" : "ghost"} size="sm" onClick={() => { setPredFilter(f); setPredPage(0); }}>
                  {f === "all" ? "All" : f}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {predictions && (
            <>
              <DataTable
                columns={[
                  { key: "prediction_id", header: "ID", className: "w-14" },
                  { key: "created_at", header: "Time", render: (r: ModelPredictionRow) => formatDateTime(r.created_at) },
                  {
                    key: "decision_hint",
                    header: "Decision",
                    render: (r: ModelPredictionRow) => (
                      <Badge variant={r.decision_hint === "TRADE" ? "profit" : "loss"}>{r.decision_hint ?? "—"}</Badge>
                    ),
                  },
                  { key: "probability_win", header: "P(Win)", render: (r: ModelPredictionRow) => r.probability_win != null ? `${(r.probability_win * 100).toFixed(1)}%` : "—" },
                  { key: "expected_value", header: "EV", className: "text-right", render: (r: ModelPredictionRow) => formatCurrency(r.expected_value) },
                  { key: "score_raw", header: "Score", render: (r: ModelPredictionRow) => r.score_raw?.toFixed(2) ?? "—" },
                  {
                    key: "realized_pnl",
                    header: "Actual PnL",
                    className: "text-right",
                    render: (r: ModelPredictionRow) => {
                      const pnl = r.realized_pnl;
                      if (pnl == null) return <span className="text-muted-foreground">pending</span>;
                      return <span className={pnl >= 0 ? "text-profit font-medium" : "text-loss font-medium"}>{formatCurrency(pnl)}</span>;
                    },
                  },
                  {
                    key: "label_status",
                    header: "Status",
                    render: (r: ModelPredictionRow) => (
                      <Badge variant={r.label_status === "resolved" ? "muted" : "warning"}>{r.label_status ?? "—"}</Badge>
                    ),
                  },
                ]}
                data={predictions.items}
                keyFn={(r) => r.prediction_id}
              />
              <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                <span>{predictions.total} total predictions</span>
                <div className="flex gap-2">
                  <Button variant="ghost" size="sm" disabled={predPage === 0} onClick={() => setPredPage((p) => p - 1)}>
                    Previous
                  </Button>
                  <span className="py-1">Page {predPage + 1}</span>
                  <Button variant="ghost" size="sm" disabled={(predPage + 1) * 50 >= predictions.total} onClick={() => setPredPage((p) => p + 1)}>
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Latest training run */}
      {run && (
        <Card>
          <CardHeader>
            <CardTitle>Latest Training Run</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm lg:grid-cols-4">
              <div>
                <span className="text-xs text-muted">Status</span>
                <div>
                  <Badge variant={run.status === "COMPLETED" ? "profit" : run.status === "FAILED" ? "loss" : "warning"}>
                    {run.status}
                  </Badge>
                </div>
              </div>
              <div>
                <span className="text-xs text-muted">Started</span>
                <div className="text-foreground-secondary">{formatDateTime(run.started_at_utc)}</div>
              </div>
              <div>
                <span className="text-xs text-muted">Train / Test</span>
                <div className="text-foreground-secondary">{run.rows_train} / {run.rows_test}</div>
              </div>
              {run.skip_reason && (
                <div>
                  <span className="text-xs text-muted">Skip Reason</span>
                  <div className="text-warning text-sm">{run.skip_reason}</div>
                </div>
              )}
            </div>
            {run.gate && (
              <div className="mt-3">
                <span className="text-xs text-muted">Promotion Gate: </span>
                <Badge variant={run.gate.passed ? "profit" : "loss"}>{run.gate.passed ? "PASSED" : "FAILED"}</Badge>
                <div className="mt-2 grid grid-cols-2 gap-2 lg:grid-cols-4">
                  {Object.entries(run.gate.checks).map(([key, check]) => (
                    <div key={key} className="rounded-md bg-background p-2 text-xs">
                      <span className="text-muted">{key}</span>
                      <div className={check.pass ? "text-profit" : "text-loss"}>
                        {typeof check.value === "number" ? check.value.toFixed(3) : String(check.value)}
                        <span className="text-muted-foreground"> / {check.threshold}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
