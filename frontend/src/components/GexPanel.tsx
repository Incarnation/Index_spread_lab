import React from "react";
import { Card, Group, Loader, MultiSelect, SegmentedControl, Select, Text } from "@mantine/core";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchGexCurve, type GexCurvePoint, type GexExpirationItem, type GexSnapshot } from "../api";
import { formatTs } from "../utils/format";

type GexPanelProps = {
  snapshots: GexSnapshot[];
  selectedSnapshot: GexSnapshot | null;
  onSelectedSnapshotChange: (snapshot: GexSnapshot | null) => void;
  dtes: number[];
  expirations: GexExpirationItem[];
  selectedDte: string;
  onSelectedDteChange: (value: string) => void;
  selectedCustomExpirations: string[];
  onSelectedCustomExpirationsChange: (values: string[]) => void;
  loading: boolean;
  curve: GexCurvePoint[];
};

type ChartRow = {
  strike: number;
  gex_net: number;
  gex_calls: number;
  gex_puts: number;
};

type ChartView = "composed" | "heatmap";

type ExpirationChoice = {
  expiration: string;
  dte_days: number | null;
};

type HeatmapRow = {
  expiration: string;
  dte_days: number | null;
  byStrike: Map<number, number>;
};

function toNumeric(value: number | null): number {
  return typeof value === "number" ? value : 0;
}

function findNearestStrike(target: number | null | undefined, strikes: number[]): number | null {
  if (typeof target !== "number" || strikes.length === 0) return null;
  let nearest = strikes[0];
  let nearestDistance = Math.abs(nearest - target);
  for (let index = 1; index < strikes.length; index += 1) {
    const strike = strikes[index];
    const distance = Math.abs(strike - target);
    if (distance < nearestDistance) {
      nearest = strike;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function formatCompactNumber(value: number): string {
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (absolute >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (absolute >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toFixed(0);
}

function heatmapColor(value: number | undefined, maxAbs: number): string {
  if (typeof value !== "number") return "#f1f3f5";
  if (maxAbs <= 0) return "#dee2e6";
  const ratio = Math.min(1, Math.abs(value) / maxAbs);
  const alpha = 0.15 + 0.75 * ratio;
  if (value >= 0) return `rgba(47, 158, 68, ${alpha})`;
  return `rgba(224, 49, 49, ${alpha})`;
}

export function GexPanel({
  snapshots,
  selectedSnapshot,
  onSelectedSnapshotChange,
  dtes,
  expirations,
  selectedDte,
  onSelectedDteChange,
  selectedCustomExpirations,
  onSelectedCustomExpirationsChange,
  loading,
  curve,
}: GexPanelProps) {
  const [chartView, setChartView] = React.useState<ChartView>("composed");
  const [heatmapRows, setHeatmapRows] = React.useState<HeatmapRow[]>([]);
  const [heatmapStrikes, setHeatmapStrikes] = React.useState<number[]>([]);
  const [heatmapMaxAbs, setHeatmapMaxAbs] = React.useState<number>(0);
  const [heatmapLoading, setHeatmapLoading] = React.useState<boolean>(false);
  const [heatmapError, setHeatmapError] = React.useState<string | null>(null);

  const chartRows = React.useMemo<ChartRow[]>(
    () =>
      curve
        .map((point) => ({
          strike: Number(point.strike),
          gex_net: toNumeric(point.gex_net),
          gex_calls: toNumeric(point.gex_calls),
          gex_puts: toNumeric(point.gex_puts),
        }))
        .sort((a, b) => a.strike - b.strike),
    [curve],
  );
  const curveStrikes = React.useMemo(() => chartRows.map((row) => row.strike), [chartRows]);

  const snapshotOptions = React.useMemo(
    () =>
      snapshots.map((snapshot) => ({
        value: String(snapshot.snapshot_id),
        label: `Batch #${snapshot.snapshot_id} · ${formatTs(snapshot.ts)}`,
      })),
    [snapshots],
  );

  const dteOptions = React.useMemo(
    () => [
      { value: "all", label: "All" },
      { value: "custom", label: "Custom (pick dates)" },
      ...dtes.map((dte) => ({ value: String(dte), label: `${dte}` })),
    ],
    [dtes],
  );

  const expirationOptions = React.useMemo(
    () =>
      expirations.map((expiration) => ({
        value: expiration.expiration,
        label:
          expiration.dte_days == null
            ? expiration.expiration
            : `${expiration.expiration} (DTE ${expiration.dte_days})`,
      })),
    [expirations],
  );

  const needsCustomSelections = selectedDte === "custom" && selectedCustomExpirations.length === 0;
  const selectedSnapshotId = selectedSnapshot?.snapshot_id ?? null;

  const heatmapExpirations = React.useMemo<ExpirationChoice[]>(() => {
    if (selectedDte === "custom") {
      const byDate = new Map(expirations.map((row) => [row.expiration, row]));
      return selectedCustomExpirations
        .map((expiration) => byDate.get(expiration))
        .filter((row): row is GexExpirationItem => row != null)
        .map((row) => ({ expiration: row.expiration, dte_days: row.dte_days }));
    }
    if (selectedDte === "all") {
      return expirations.map((row) => ({ expiration: row.expiration, dte_days: row.dte_days }));
    }
    const dteValue = Number(selectedDte);
    if (!Number.isFinite(dteValue)) return [];
    return expirations
      .filter((row) => row.dte_days === dteValue)
      .map((row) => ({ expiration: row.expiration, dte_days: row.dte_days }));
  }, [expirations, selectedCustomExpirations, selectedDte]);

  const nearestSpotStrike = React.useMemo(
    () => findNearestStrike(selectedSnapshot?.spot_price, curveStrikes),
    [curveStrikes, selectedSnapshot?.spot_price],
  );
  const nearestZeroGammaStrike = React.useMemo(
    () => findNearestStrike(selectedSnapshot?.zero_gamma_level, curveStrikes),
    [curveStrikes, selectedSnapshot?.zero_gamma_level],
  );

  const heatmapLabelStep = React.useMemo(() => {
    if (heatmapStrikes.length <= 12) return 1;
    return Math.ceil(heatmapStrikes.length / 12);
  }, [heatmapStrikes.length]);

  React.useEffect(() => {
    if (chartView !== "heatmap") return;
    if (selectedSnapshotId == null || needsCustomSelections || heatmapExpirations.length === 0) {
      setHeatmapRows([]);
      setHeatmapStrikes([]);
      setHeatmapMaxAbs(0);
      setHeatmapError(null);
      setHeatmapLoading(false);
      return;
    }

    let cancelled = false;
    setHeatmapLoading(true);
    setHeatmapError(null);

    Promise.all(
      heatmapExpirations.map(async (expiration) => {
        const points = await fetchGexCurve(selectedSnapshotId, undefined, [expiration.expiration]);
        const byStrike = new Map<number, number>();
        for (const point of points) {
          byStrike.set(Number(point.strike), toNumeric(point.gex_net));
        }
        return {
          expiration: expiration.expiration,
          dte_days: expiration.dte_days,
          byStrike,
        } satisfies HeatmapRow;
      }),
    )
      .then((rows) => {
        if (cancelled) return;
        const strikeSet = new Set<number>();
        let maxAbs = 0;
        for (const row of rows) {
          row.byStrike.forEach((value, strike) => {
            strikeSet.add(strike);
            const abs = Math.abs(value);
            if (abs > maxAbs) maxAbs = abs;
          });
        }
        setHeatmapRows(rows);
        setHeatmapStrikes(Array.from(strikeSet).sort((a, b) => a - b));
        setHeatmapMaxAbs(maxAbs);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setHeatmapRows([]);
        setHeatmapStrikes([]);
        setHeatmapMaxAbs(0);
        setHeatmapError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setHeatmapLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [chartView, heatmapExpirations, needsCustomSelections, selectedSnapshotId]);

  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm" wrap="wrap">
        <Text fw={600}>Gamma exposure (GEX)</Text>
        <Group gap="sm">
          <Select
            label="Capture batch"
            data={snapshotOptions}
            value={selectedSnapshot ? String(selectedSnapshot.snapshot_id) : null}
            onChange={(value) => {
              const id = value ? Number(value) : null;
              const snapshot = id ? snapshots.find((row) => row.snapshot_id === id) ?? null : null;
              onSelectedSnapshotChange(snapshot);
            }}
            w={260}
            placeholder="Select capture batch"
          />
          <Select label="DTE" data={dteOptions} value={selectedDte} onChange={(value) => onSelectedDteChange(value || "all")} w={220} />
          <div>
            <Text size="xs" c="dimmed" mb={4}>
              View
            </Text>
            <SegmentedControl
              value={chartView}
              onChange={(value) => setChartView(value as ChartView)}
              data={[
                { label: "Composed", value: "composed" },
                { label: "Heatmap", value: "heatmap" },
              ]}
            />
          </div>
          {selectedDte === "custom" && (
            <MultiSelect
              label="Expirations"
              data={expirationOptions}
              value={selectedCustomExpirations}
              onChange={onSelectedCustomExpirationsChange}
              placeholder="Pick upcoming dates"
              searchable
              clearable
              w={300}
            />
          )}
        </Group>
      </Group>

      {selectedSnapshot && (
        <Group gap="lg" mb="sm">
          <Text size="sm" c="dimmed">
            Spot: {selectedSnapshot.spot_price ?? "—"}
          </Text>
          <Text size="sm" c="dimmed">
            GEX Net: {selectedSnapshot.gex_net ?? "—"}
          </Text>
          <Text size="sm" c="dimmed">
            Zero Gamma: {selectedSnapshot.zero_gamma_level ?? "—"}
          </Text>
        </Group>
      )}

      {loading && chartView === "composed" && (
        <Group gap="xs">
          <Loader size="sm" />
          <Text c="dimmed" size="sm">
            Loading GEX…
          </Text>
        </Group>
      )}

      {chartView === "heatmap" && heatmapLoading && (
        <Group gap="xs">
          <Loader size="sm" />
          <Text c="dimmed" size="sm">
            Loading heatmap…
          </Text>
        </Group>
      )}

      {!loading && !heatmapLoading && needsCustomSelections && (
        <Text c="dimmed">Pick one or more expiration dates to render GEX data.</Text>
      )}

      {chartView === "composed" && !loading && curve.length === 0 && !needsCustomSelections && (
        <Text c="dimmed">No GEX data yet. Run snapshots and GEX first.</Text>
      )}

      {chartView === "composed" && !loading && curve.length > 0 && (
        <div style={{ width: "100%", height: 320 }}>
          <ResponsiveContainer>
            <ComposedChart data={chartRows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="strike" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} tickFormatter={(value: number) => formatCompactNumber(value)} />
              <Tooltip
                formatter={(value: number | string | undefined) => {
                  if (typeof value === "number") return formatCompactNumber(value);
                  const parsed = Number(value);
                  return Number.isFinite(parsed) ? formatCompactNumber(parsed) : String(value ?? "");
                }}
                labelFormatter={(value) => `Strike ${String(value ?? "")}`}
              />
              <Legend />
              <ReferenceLine y={0} stroke="#868e96" />
              {nearestSpotStrike != null && (
                <ReferenceLine x={nearestSpotStrike} stroke="#495057" strokeDasharray="4 4" label="Spot" />
              )}
              {nearestZeroGammaStrike != null && (
                <ReferenceLine x={nearestZeroGammaStrike} stroke="#f08c00" strokeDasharray="4 4" label="Zero Gamma" />
              )}
              <Bar dataKey="gex_calls" name="Calls GEX" stackId="gex" fill="#2f9e44" />
              <Bar dataKey="gex_puts" name="Puts GEX" stackId="gex" fill="#e03131" />
              <Line type="monotone" dataKey="gex_net" name="Net GEX" stroke="#228be6" strokeWidth={2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {chartView === "heatmap" && !heatmapLoading && heatmapError && (
        <Text c="red" size="sm">
          Failed to load heatmap data: {heatmapError}
        </Text>
      )}

      {chartView === "heatmap" && !heatmapLoading && !heatmapError && heatmapRows.length === 0 && !needsCustomSelections && (
        <Text c="dimmed">No heatmap data available for this selection.</Text>
      )}

      {chartView === "heatmap" && !heatmapLoading && !heatmapError && heatmapRows.length > 0 && heatmapStrikes.length > 0 && (
        <div style={{ width: "100%" }}>
          <Group gap={8} mb="xs" align="center">
            <Text size="xs" c="dimmed">
              Negative
            </Text>
            <div
              style={{
                width: 160,
                height: 8,
                borderRadius: 999,
                background:
                  "linear-gradient(90deg, rgba(224, 49, 49, 0.9) 0%, rgba(248, 249, 250, 1) 50%, rgba(47, 158, 68, 0.9) 100%)",
              }}
            />
            <Text size="xs" c="dimmed">
              Positive
            </Text>
          </Group>

          <div
            style={{
              overflowX: "auto",
              overflowY: "auto",
              maxHeight: 360,
              border: "1px solid #e9ecef",
              borderRadius: 8,
              padding: 8,
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: `160px repeat(${heatmapStrikes.length}, 12px)`,
                gap: 2,
                minWidth: 160 + heatmapStrikes.length * 14,
              }}
            >
              <div />
              {heatmapStrikes.map((strike, index) => (
                <div
                  key={`strike-${strike}`}
                  style={{ fontSize: 10, lineHeight: "12px", color: "#868e96", textAlign: "center", minHeight: 12 }}
                >
                  {index % heatmapLabelStep === 0 ? strike : ""}
                </div>
              ))}

              {heatmapRows.map((row) => (
                <React.Fragment key={row.expiration}>
                  <Text size="xs" style={{ alignSelf: "center" }}>
                    {row.dte_days == null ? row.expiration : `${row.expiration} (DTE ${row.dte_days})`}
                  </Text>
                  {heatmapStrikes.map((strike) => {
                    const value = row.byStrike.get(strike);
                    return (
                      <div
                        key={`${row.expiration}-${strike}`}
                        title={`${row.expiration} | Strike ${strike} | Net GEX ${
                          value == null ? "No data" : formatCompactNumber(value)
                        }`}
                        style={{
                          width: 12,
                          height: 12,
                          borderRadius: 2,
                          border: "1px solid #f1f3f5",
                          backgroundColor: heatmapColor(value, heatmapMaxAbs),
                        }}
                      />
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
