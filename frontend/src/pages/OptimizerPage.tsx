import { useEffect, useState } from "react";
import { StatCard } from "@/components/shared/StatCard";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  fetchOptimizerRuns,
  fetchOptimizerResults,
  fetchParetoFrontier,
  fetchConfigComparison,
  fetchWalkforwardResults,
  type OptimizerRun,
  type OptimizerResult,
  type WalkforwardRow,
} from "@/api";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
  ReferenceLine,
} from "recharts";
import {
  Activity,
  BarChart3,
  GitCompare,
  Target,
  TrendingUp,
  Layers,
  AlertCircle,
} from "lucide-react";

/**
 * Optimizer dashboard with tabs for runs, Pareto frontier, results, comparison,
 * and walk-forward analysis.
 */
export function OptimizerPage() {
  const [tab, setTab] = useState("runs");
  const [runs, setRuns] = useState<OptimizerRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    fetchOptimizerRuns(50, 0, ac.signal)
      .then((d) => { setRuns(d.runs); setLoading(false); })
      .catch((e) => { if (!ac.signal.aborted) { setError(e.message); setLoading(false); } });
    return () => ac.abort();
  }, []);

  useEffect(() => {
    if (runs.length > 0 && !selectedRunId) {
      setSelectedRunId(runs[0].run_id);
    }
  }, [runs, selectedRunId]);

  if (loading) return <div className="p-6 text-muted-foreground">Loading optimizer data...</div>;
  if (error) return <div className="p-6 text-destructive">Error: {error}</div>;
  if (runs.length === 0) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-4">Optimizer Dashboard</h1>
        <div className="rounded-lg border border-border p-12 text-center text-muted-foreground">
          <Layers className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium mb-2">No optimizer runs found</p>
          <p className="text-sm">Run an optimizer and ingest results to see them here.</p>
          <pre className="mt-4 text-xs bg-muted p-3 rounded inline-block text-left">
            python scripts/backtest_strategy.py --optimize-staged{"\n"}
            python scripts/ingest_optimizer_results.py --run-name "my-run"
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Optimizer Dashboard</h1>
        <select
          className="rounded-md border border-input bg-background px-3 py-1.5 text-sm"
          value={selectedRunId ?? ""}
          onChange={(e) => setSelectedRunId(e.target.value)}
        >
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_name ?? r.run_id} ({r.optimizer_mode})
            </option>
          ))}
        </select>
      </div>

      {/* Summary stats for selected run */}
      {selectedRunId && <RunSummary runs={runs} runId={selectedRunId} />}

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="runs"><Layers className="h-3.5 w-3.5 mr-1.5" />Runs</TabsTrigger>
          <TabsTrigger value="pareto"><Target className="h-3.5 w-3.5 mr-1.5" />Pareto</TabsTrigger>
          <TabsTrigger value="results"><BarChart3 className="h-3.5 w-3.5 mr-1.5" />Results</TabsTrigger>
          <TabsTrigger value="compare"><GitCompare className="h-3.5 w-3.5 mr-1.5" />Compare</TabsTrigger>
          <TabsTrigger value="walkforward"><TrendingUp className="h-3.5 w-3.5 mr-1.5" />Walk-Forward</TabsTrigger>
        </TabsList>

        <TabsContent value="runs"><RunsTable runs={runs} onSelect={setSelectedRunId} /></TabsContent>
        <TabsContent value="pareto">{selectedRunId && <ParetoTab runId={selectedRunId} />}</TabsContent>
        <TabsContent value="results">{selectedRunId && <ResultsTab runId={selectedRunId} />}</TabsContent>
        <TabsContent value="compare">{selectedRunId && <CompareTab runId={selectedRunId} />}</TabsContent>
        <TabsContent value="walkforward">{selectedRunId && <WalkForwardTab runId={selectedRunId} />}</TabsContent>
      </Tabs>
    </div>
  );
}

/** Summary stat cards for the selected run. */
function RunSummary({ runs, runId }: { runs: OptimizerRun[]; runId: string }) {
  const run = runs.find((r) => r.run_id === runId);
  if (!run) return null;
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      <StatCard title="Configs" value={run.num_configs?.toLocaleString() ?? "—"} icon={Layers} />
      <StatCard title="Best Sharpe" value={run.best_sharpe?.toFixed(2) ?? "—"} icon={Target} />
      <StatCard title="Best Return" value={run.best_return_pct != null ? `${run.best_return_pct.toFixed(0)}%` : "—"} icon={TrendingUp} />
      <StatCard title="Pareto Configs" value={String(run.pareto_count)} icon={Activity} />
      <StatCard title="Status" value={run.status} icon={BarChart3} />
    </div>
  );
}

/** Table of all optimizer runs. */
function RunsTable({ runs, onSelect }: { runs: OptimizerRun[]; onSelect: (id: string) => void }) {
  const columns = [
    { key: "run_name", label: "Name" },
    { key: "optimizer_mode", label: "Mode" },
    { key: "num_configs", label: "Configs" },
    { key: "best_sharpe", label: "Best Sharpe", render: (v: number | null) => v?.toFixed(2) ?? "—" },
    { key: "best_return_pct", label: "Best Return", render: (v: number | null) => v != null ? `${v.toFixed(0)}%` : "—" },
    { key: "pareto_count", label: "Pareto" },
    { key: "status", label: "Status" },
    { key: "started_at", label: "Started", render: (v: string | null) => v ? new Date(v).toLocaleDateString() : "—" },
  ];

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-muted/50">
          <tr>{columns.map((c) => <th key={c.key} className="text-left px-4 py-2 font-medium text-muted-foreground">{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.run_id}
              className="border-t border-border hover:bg-muted/30 cursor-pointer"
              onClick={() => onSelect(run.run_id)}
            >
              {columns.map((c) => {
                const val = (run as Record<string, unknown>)[c.key];
                return (
                  <td key={c.key} className="px-4 py-2">
                    {c.render ? c.render(val as never) : String(val ?? "—")}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Inline error banner for sub-tab fetch failures. */
function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
      <AlertCircle className="h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

/** Pareto frontier scatter plot (Sharpe vs Max DD). */
function ParetoTab({ runId }: { runId: string }) {
  const [data, setData] = useState<OptimizerResult[]>([]);
  const [allResults, setAllResults] = useState<OptimizerResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    Promise.all([
      fetchParetoFrontier(runId, ac.signal),
      fetchOptimizerResults(runId, { limit: 500, sortBy: "sharpe" }, ac.signal),
    ]).then(([pareto, all]) => {
      setData(pareto.pareto);
      setAllResults(all.results);
      setLoading(false);
    }).catch((e) => { if (!ac.signal.aborted) { setError(e.message); setLoading(false); } });
    return () => ac.abort();
  }, [runId]);

  if (loading) return <div className="py-8 text-muted-foreground">Loading Pareto data...</div>;
  if (error) return <ErrorBanner message={`Failed to load Pareto data: ${error}`} />;

  const nonPareto = allResults.filter((r) => !r.is_pareto);
  const paretoPoints = data.map((r) => ({
    sharpe: r.sharpe,
    maxDD: Math.abs(r.max_dd_pct),
    returnPct: r.return_pct,
    winRate: r.win_rate,
    id: r.id,
  }));
  const otherPoints = nonPareto.slice(0, 300).map((r) => ({
    sharpe: r.sharpe,
    maxDD: Math.abs(r.max_dd_pct),
    returnPct: r.return_pct,
    id: r.id,
  }));

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Pareto Frontier: Sharpe vs Max Drawdown</h3>
      <p className="text-sm text-muted-foreground">{data.length} Pareto-optimal configs highlighted. Hover a dot for details.</p>
      <div className="h-[500px] border border-border rounded-lg p-4 bg-card">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis type="number" dataKey="maxDD" name="Max DD (%)" unit="%" label={{ value: "Max Drawdown (%)", position: "bottom" }} />
            <YAxis type="number" dataKey="sharpe" name="Sharpe" label={{ value: "Sharpe Ratio", angle: -90, position: "insideLeft" }} />
            <ZAxis type="number" dataKey="returnPct" range={[20, 200]} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<ParetoTooltip />} />
            <Scatter name="Other" data={otherPoints} fill="hsl(var(--muted-foreground))" opacity={0.15} />
            <Scatter name="Pareto" data={paretoPoints} fill="hsl(var(--chart-1))" opacity={0.85} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ParetoTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: Record<string, unknown> }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-popover border border-border rounded p-3 text-sm shadow-lg">
      <p>Sharpe: <strong>{(d.sharpe as number).toFixed(2)}</strong></p>
      <p>Max DD: <strong>{(d.maxDD as number).toFixed(1)}%</strong></p>
      <p>Return: <strong>{(d.returnPct as number).toFixed(0)}%</strong></p>
      {d.winRate !== undefined && <p>Win Rate: <strong>{((d.winRate as number) * 100).toFixed(1)}%</strong></p>}
    </div>
  );
}

/** Paginated results table. */
function ResultsTab({ runId }: { runId: string }) {
  const [results, setResults] = useState<OptimizerResult[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState("sharpe");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pageSize = 50;

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchOptimizerResults(runId, { sortBy, limit: pageSize, offset: page * pageSize }, ac.signal)
      .then((d) => { setResults(d.results); setTotal(d.total); setLoading(false); })
      .catch((e) => { if (!ac.signal.aborted) { setError(e.message); setLoading(false); } });
    return () => ac.abort();
  }, [runId, sortBy, page]);

  if (loading) return <div className="py-8 text-muted-foreground">Loading results...</div>;
  if (error) return <ErrorBanner message={`Failed to load results: ${error}`} />;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-sm text-muted-foreground">{total.toLocaleString()} total results</span>
        <select
          className="rounded-md border border-input bg-background px-2 py-1 text-sm"
          value={sortBy}
          onChange={(e) => { setSortBy(e.target.value); setPage(0); }}
        >
          <option value="sharpe">Sort by Sharpe</option>
          <option value="return_pct">Sort by Return</option>
          <option value="max_dd_pct">Sort by Max DD</option>
          <option value="win_rate">Sort by Win Rate</option>
          <option value="total_trades">Sort by Trades</option>
        </select>
      </div>
      <div className="rounded-lg border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              <th className="text-left px-3 py-2 font-medium text-muted-foreground">#</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Sharpe</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Return%</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Max DD%</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Win Rate</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Trades</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Final Equity</th>
              <th className="text-center px-3 py-2 font-medium text-muted-foreground">Pareto</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r, i) => (
              <tr key={r.id} className="border-t border-border hover:bg-muted/30">
                <td className="px-3 py-1.5 text-muted-foreground">{page * pageSize + i + 1}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.sharpe?.toFixed(2)}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.return_pct?.toFixed(0)}%</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.max_dd_pct?.toFixed(1)}%</td>
                <td className="px-3 py-1.5 text-right font-mono">{((r.win_rate ?? 0) * 100).toFixed(1)}%</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.total_trades}</td>
                <td className="px-3 py-1.5 text-right font-mono">${r.final_equity?.toLocaleString()}</td>
                <td className="px-3 py-1.5 text-center">{r.is_pareto ? "★" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between">
        <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
        <span className="text-sm text-muted-foreground">Page {page + 1} of {Math.max(1, Math.ceil(total / pageSize))}</span>
        <Button variant="outline" size="sm" disabled={(page + 1) * pageSize >= total} onClick={() => setPage(page + 1)}>Next</Button>
      </div>
    </div>
  );
}

/** Config comparison (top results side by side). */
function CompareTab({ runId }: { runId: string }) {
  const [top, setTop] = useState<OptimizerResult[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [comparison, setComparison] = useState<{ configs: OptimizerResult[]; differing_columns: string[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchOptimizerResults(runId, { sortBy: "sharpe", limit: 20 }, ac.signal)
      .then((d) => { setTop(d.results); setLoading(false); })
      .catch((e) => { if (!ac.signal.aborted) { setError(e.message); setLoading(false); } });
    return () => ac.abort();
  }, [runId]);

  const toggleSelect = (id: number) => {
    setSelected((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 4 ? [...prev, id] : prev);
  };

  const doCompare = () => {
    if (selected.length < 2) return;
    setCompareError(null);
    const ac = new AbortController();
    fetchConfigComparison(selected, ac.signal)
      .then(setComparison)
      .catch((e) => { setCompareError(e.message); });
  };

  if (loading) return <div className="py-8 text-muted-foreground">Loading...</div>;
  if (error) return <ErrorBanner message={`Failed to load configs: ${error}`} />;

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">Select 2-4 configs to compare side-by-side. Differing parameters are highlighted.</p>
      <div className="flex gap-2 items-center">
        <Button onClick={doCompare} disabled={selected.length < 2} size="sm">
          <GitCompare className="h-3.5 w-3.5 mr-1.5" />Compare ({selected.length})
        </Button>
      </div>
      {compareError && <ErrorBanner message={`Comparison failed: ${compareError}`} />}
      <div className="rounded-lg border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              <th className="text-center px-3 py-2 w-10"></th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Sharpe</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Return%</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Max DD%</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Win Rate</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Trades</th>
            </tr>
          </thead>
          <tbody>
            {top.map((r) => (
              <tr key={r.id} className={`border-t border-border hover:bg-muted/30 cursor-pointer ${selected.includes(r.id) ? "bg-accent/20" : ""}`} onClick={() => toggleSelect(r.id)}>
                <td className="px-3 py-1.5 text-center"><input type="checkbox" checked={selected.includes(r.id)} readOnly /></td>
                <td className="px-3 py-1.5 text-right font-mono">{r.sharpe?.toFixed(2)}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.return_pct?.toFixed(0)}%</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.max_dd_pct?.toFixed(1)}%</td>
                <td className="px-3 py-1.5 text-right font-mono">{((r.win_rate ?? 0) * 100).toFixed(1)}%</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.total_trades}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {comparison && (
        <div className="mt-6 space-y-3">
          <h3 className="text-lg font-semibold">Comparison ({comparison.configs.length} configs)</h3>
          <p className="text-sm text-muted-foreground">{comparison.differing_columns.length} parameters differ</p>
          <div className="rounded-lg border border-border overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left px-3 py-2 font-medium text-muted-foreground">Parameter</th>
                  {comparison.configs.map((c, i) => (
                    <th key={i} className="text-right px-3 py-2 font-medium text-muted-foreground">Config {i + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison.differing_columns.map((col) => (
                  <tr key={col} className="border-t border-border bg-yellow-50/50 dark:bg-yellow-900/10">
                    <td className="px-3 py-1 font-mono text-xs">{col}</td>
                    {comparison.configs.map((c, i) => (
                      <td key={i} className="px-3 py-1 text-right font-mono text-xs">{String((c as Record<string, unknown>)[col] ?? "—")}</td>
                    ))}
                  </tr>
                ))}
                {["sharpe", "return_pct", "max_dd_pct", "win_rate", "total_trades", "final_equity"].map((col) => (
                  <tr key={col} className="border-t border-border">
                    <td className="px-3 py-1 font-mono text-xs font-bold">{col}</td>
                    {comparison.configs.map((c, i) => (
                      <td key={i} className="px-3 py-1 text-right font-mono text-xs font-bold">{String((c as Record<string, unknown>)[col] ?? "—")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

/** Walk-forward results tab. */
function WalkForwardTab({ runId }: { runId: string }) {
  const [data, setData] = useState<WalkforwardRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchWalkforwardResults(runId, ac.signal)
      .then((d) => { setData(d.walkforward); setLoading(false); })
      .catch((e) => { if (!ac.signal.aborted) { setError(e.message); setLoading(false); } });
    return () => ac.abort();
  }, [runId]);

  if (loading) return <div className="py-8 text-muted-foreground">Loading walk-forward data...</div>;
  if (error) return <ErrorBanner message={`Failed to load walk-forward data: ${error}`} />;

  if (data.length === 0) {
    return (
      <div className="py-8 text-center text-muted-foreground">
        <p>No walk-forward results for this run.</p>
        <p className="text-xs mt-1">Run --walkforward and ingest with --walkforward-csv to see results here.</p>
      </div>
    );
  }

  const chartData = data
    .filter((r) => r.train_sharpe != null && r.test_sharpe != null)
    .map((r) => ({
      window: r.window_label ?? "",
      trainSharpe: r.train_sharpe,
      testSharpe: r.test_sharpe,
      decayRatio: r.decay_ratio,
    }));

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Walk-Forward Validation</h3>

      {chartData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="border border-border rounded-lg p-4 bg-card">
            <h4 className="text-sm font-medium mb-3">Train vs Test Sharpe</h4>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="window" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="trainSharpe" name="Train Sharpe" fill="hsl(var(--chart-1))" />
                  <Bar dataKey="testSharpe" name="Test Sharpe" fill="hsl(var(--chart-2))" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="border border-border rounded-lg p-4 bg-card">
            <h4 className="text-sm font-medium mb-3">Train vs Test Sharpe (Scatter)</h4>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis type="number" dataKey="trainSharpe" name="Train Sharpe" label={{ value: "Train", position: "bottom" }} />
                  <YAxis type="number" dataKey="testSharpe" name="Test Sharpe" label={{ value: "Test", angle: -90, position: "insideLeft" }} />
                  <ReferenceLine stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" segment={[{ x: -1, y: -1 }, { x: 5, y: 5 }]} />
                  <Tooltip />
                  <Scatter data={chartData} fill="hsl(var(--chart-1))" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      <div className="rounded-lg border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              <th className="text-left px-3 py-2 font-medium text-muted-foreground">Window</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Train Sharpe</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Test Sharpe</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Train Return</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Test Return</th>
              <th className="text-right px-3 py-2 font-medium text-muted-foreground">Decay Ratio</th>
            </tr>
          </thead>
          <tbody>
            {data.map((r) => (
              <tr key={r.id} className="border-t border-border">
                <td className="px-3 py-1.5">{r.window_label ?? "—"}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.train_sharpe?.toFixed(2) ?? "—"}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.test_sharpe?.toFixed(2) ?? "—"}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.train_return != null ? `${r.train_return.toFixed(0)}%` : "—"}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.test_return != null ? `${r.test_return.toFixed(0)}%` : "—"}</td>
                <td className="px-3 py-1.5 text-right font-mono">{r.decay_ratio?.toFixed(2) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default OptimizerPage;
