import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  fetchGexSnapshots,
  fetchGexCurve,
  fetchGexDtes,
  type GexSnapshot,
  type GexCurvePoint,
} from "@/api";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import { formatDateTime } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

const UNDERLYINGS = ["SPX", "SPY"] as const;
const SOURCES = ["CBOE", "TRADIER", "All"] as const;

/**
 * GEX / Market page -- gamma exposure visualization by strike.
 */
export function GexPage() {
  const { tick } = useAutoRefresh(60_000);
  const [underlying, setUnderlying] = useState<string>("SPX");
  const [source, setSource] = useState<string>("CBOE");
  const [snapshots, setSnapshots] = useState<GexSnapshot[]>([]);
  const [selectedSnap, setSelectedSnap] = useState<GexSnapshot | null>(null);
  const [dtes, setDtes] = useState<number[]>([]);
  const [selectedDte, setSelectedDte] = useState<number | undefined>();
  const [curve, setCurve] = useState<GexCurvePoint[]>([]);

  useEffect(() => {
    const ac = new AbortController();
    const sourceParam = source === "All" ? undefined : source;
    fetchGexSnapshots(10, underlying, sourceParam, ac.signal)
      .then((snaps) => {
        if (!ac.signal.aborted) {
          setSnapshots(snaps);
          if (snaps.length > 0 && (!selectedSnap || selectedSnap.underlying !== underlying || selectedSnap.source !== (sourceParam ?? selectedSnap.source))) {
            setSelectedSnap(snaps[0]);
          }
        }
      })
      .catch(() => {
        if (!ac.signal.aborted) {
        }
      });
    return () => ac.abort();
  }, [underlying, source, tick]);

  useEffect(() => {
    if (!selectedSnap) return;
    const ac = new AbortController();
    setSelectedDte(undefined);
    fetchGexDtes(selectedSnap.snapshot_id, ac.signal)
      .then((d) => {
        if (!ac.signal.aborted) setDtes(d);
      })
      .catch(() => {
        if (!ac.signal.aborted) {
        }
      });
    return () => ac.abort();
  }, [selectedSnap]);

  useEffect(() => {
    if (!selectedSnap) return;
    const ac = new AbortController();
    fetchGexCurve(selectedSnap.snapshot_id, selectedDte, undefined, ac.signal)
      .then((c) => {
        if (!ac.signal.aborted) setCurve(c);
      })
      .catch(() => {
        if (!ac.signal.aborted) {
        }
      });
    return () => ac.abort();
  }, [selectedSnap, selectedDte]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-foreground">GEX / Market</h2>
        <div className="flex items-center gap-3">
          <div className="flex gap-1">
            {SOURCES.map((s) => (
              <Button key={s} variant={source === s ? "default" : "ghost"} size="sm" onClick={() => setSource(s)}>
                {s}
              </Button>
            ))}
          </div>
          <div className="flex gap-1">
            {UNDERLYINGS.map((u) => (
              <Button key={u} variant={underlying === u ? "default" : "ghost"} size="sm" onClick={() => setUnderlying(u)}>
                {u}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Snapshot selector */}
      <div className="flex gap-2 flex-wrap items-center">
        {snapshots.slice(0, 5).map((s) => (
          <Button
            key={s.snapshot_id}
            variant={selectedSnap?.snapshot_id === s.snapshot_id ? "default" : "outline"}
            size="sm"
            onClick={() => setSelectedSnap(s)}
          >
            {formatDateTime(s.ts)}
          </Button>
        ))}
      </div>

      {/* Snapshot info */}
      {selectedSnap && (
        <div className="flex gap-3 text-xs text-muted-foreground">
          <span>Spot: <strong className="text-foreground">{selectedSnap.spot_price?.toFixed(2) ?? "—"}</strong></span>
          <span>Net GEX: <strong className="text-foreground">{selectedSnap.gex_net?.toExponential(2) ?? "—"}</strong></span>
          <span>Zero Gamma: <strong className="text-foreground">{selectedSnap.zero_gamma_level?.toFixed(0) ?? "—"}</strong></span>
          <Badge variant="muted">{selectedSnap.source}</Badge>
        </div>
      )}

      {/* DTE filter */}
      {dtes.length > 0 && (
        <div className="flex gap-1 flex-wrap">
          <Button variant={!selectedDte ? "default" : "ghost"} size="sm" onClick={() => setSelectedDte(undefined)}>All DTE</Button>
          {dtes.map((d) => (
            <Button key={d} variant={selectedDte === d ? "default" : "ghost"} size="sm" onClick={() => setSelectedDte(d)}>
              {d}d
            </Button>
          ))}
        </div>
      )}

      {/* GEX chart */}
      {curve.length > 0 && (
        <div className="rounded-lg border border-border bg-card p-4">
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={curve}>
              <XAxis dataKey="strike" tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={{ stroke: "#1e1e2e" }} />
              <YAxis tick={{ fontSize: 10, fill: "#6b6b80" }} tickLine={false} axisLine={false} tickFormatter={(v: number) => v.toExponential(1)} />
              <Tooltip contentStyle={{ backgroundColor: "#111118", border: "1px solid #1e1e2e", borderRadius: "6px", fontSize: "12px", color: "#e4e4ef" }} />
              <Legend wrapperStyle={{ fontSize: "11px", color: "#a0a0b8" }} />
              {selectedSnap?.zero_gamma_level != null && (
                <ReferenceLine x={selectedSnap.zero_gamma_level} stroke="#f59e0b" strokeDasharray="3 3" label={{ value: "Zero Gamma", fill: "#f59e0b", fontSize: 10 }} />
              )}
              {selectedSnap?.spot_price && (
                <ReferenceLine x={Math.round(selectedSnap.spot_price)} stroke="#3b82f6" strokeDasharray="3 3" label={{ value: "Spot", fill: "#3b82f6", fontSize: 10 }} />
              )}
              <Bar dataKey="gex_calls" name="Calls" fill="#10b981" opacity={0.7} />
              <Bar dataKey="gex_puts" name="Puts" fill="#ef4444" opacity={0.7} />
              <Line type="monotone" dataKey="gex_net" name="Net" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
