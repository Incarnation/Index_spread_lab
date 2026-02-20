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
import {
  GEX_UNDERLYING_OPTIONS,
  GEX_ZERO_DTE_ONLY_SENTINEL,
  getSnapshotTradingDateIso,
} from "../constants/gex";
import { formatTs } from "../utils/format";

type GexPanelProps = {
  snapshots: GexSnapshot[];
  selectedSnapshot: GexSnapshot | null;
  onSelectedSnapshotChange: (snapshot: GexSnapshot | null) => void;
  selectedUnderlying: string;
  onSelectedUnderlyingChange: (value: string | null) => void;
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

const GEX_VIEW_STORAGE_KEY = "dashboard.gex.chartView";
const GEX_STRIKE_COUNT_STORAGE_KEY = "dashboard.gex.strikeCount";
const GEX_STRIKE_COUNT_OPTIONS = ["50", "100", "150", "all"] as const;
type GexStrikeCountOption = (typeof GEX_STRIKE_COUNT_OPTIONS)[number];

/**
 * Load the persisted GEX chart view, defaulting to composed mode.
 */
function readStoredGexView(): ChartView {
  try {
    const raw = window.localStorage.getItem(GEX_VIEW_STORAGE_KEY);
    return raw === "heatmap" ? "heatmap" : "composed";
  } catch {
    return "composed";
  }
}

/**
 * Save the currently selected GEX chart view.
 */
function writeStoredGexView(value: ChartView): void {
  try {
    window.localStorage.setItem(GEX_VIEW_STORAGE_KEY, value);
  } catch {
    // Ignore storage write errors.
  }
}

/**
 * Load the persisted strike-window size for GEX charts.
 */
function readStoredGexStrikeCount(): GexStrikeCountOption {
  try {
    const raw = window.localStorage.getItem(GEX_STRIKE_COUNT_STORAGE_KEY);
    if (raw && GEX_STRIKE_COUNT_OPTIONS.includes(raw as GexStrikeCountOption)) {
      return raw as GexStrikeCountOption;
    }
    return "all";
  } catch {
    return "all";
  }
}

/**
 * Save the selected strike-window size for GEX charts.
 */
function writeStoredGexStrikeCount(value: GexStrikeCountOption): void {
  try {
    window.localStorage.setItem(GEX_STRIKE_COUNT_STORAGE_KEY, value);
  } catch {
    // Ignore storage write errors.
  }
}

/**
 * Build a snapshot dropdown label that always includes the underlying symbol.
 */
function formatSnapshotOptionLabel(snapshot: GexSnapshot): string {
  const underlying = snapshot.underlying?.trim() || "UNKNOWN";
  return `${underlying} · Batch #${snapshot.snapshot_id} · ${formatTs(snapshot.ts)}`;
}

/**
 * Normalize symbol values before applying equality checks.
 */
function normalizeUnderlying(value: string | null | undefined): string {
  return (value ?? "").trim().toUpperCase();
}

/**
 * Build the synthetic strict 0DTE option for custom expiration mode.
 */
function buildZeroDteOnlyOption(snapshot: GexSnapshot | null): { value: string; label: string } | null {
  const tradingDate = getSnapshotTradingDateIso(snapshot);
  if (!tradingDate) return null;
  return {
    value: GEX_ZERO_DTE_ONLY_SENTINEL,
    label: `${tradingDate} (Today, 0DTE only)`,
  };
}

/**
 * Keep custom expiration selection deterministic and make 0DTE mode exclusive.
 */
function normalizeCustomExpirationSelection(values: string[]): string[] {
  const cleaned = Array.from(
    new Set(values.map((value) => value.trim()).filter((value) => value.length > 0)),
  );
  if (cleaned.includes(GEX_ZERO_DTE_ONLY_SENTINEL)) {
    return [GEX_ZERO_DTE_ONLY_SENTINEL];
  }
  return cleaned;
}

/** Coerce nullable numeric fields from API rows into chart-safe numbers. */
function toNumeric(value: number | null): number {
  return typeof value === "number" ? value : 0;
}

/** Find the closest plotted strike to a target level (spot or zero-gamma). */
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

/**
 * Keep only the nearest N strikes to a reference level (spot by default).
 */
function selectNearestStrikes(allStrikes: number[], target: number | null | undefined, limit: number): number[] {
  if (limit <= 0 || allStrikes.length <= limit) {
    return [...allStrikes].sort((a, b) => a - b);
  }
  const fallbackTarget = allStrikes.length > 0 ? allStrikes[Math.floor(allStrikes.length / 2)] : 0;
  const reference = typeof target === "number" ? target : fallbackTarget;
  return [...allStrikes]
    .sort((a, b) => {
      const distanceDiff = Math.abs(a - reference) - Math.abs(b - reference);
      if (distanceDiff !== 0) return distanceDiff;
      return a - b;
    })
    .slice(0, limit)
    .sort((a, b) => a - b);
}

/** Compact large GEX values for axis ticks and tooltips. */
function formatCompactNumber(value: number): string {
  const absolute = Math.abs(value);
  if (absolute >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (absolute >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (absolute >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toFixed(0);
}

/** Convert heatmap magnitude/sign into a red-gray-green color scale. */
function heatmapColor(value: number | undefined, maxAbs: number): string {
  if (typeof value !== "number") return "#f1f3f5";
  if (maxAbs <= 0) return "#dee2e6";
  const ratio = Math.min(1, Math.abs(value) / maxAbs);
  const alpha = 0.15 + 0.75 * ratio;
  if (value >= 0) return `rgba(47, 158, 68, ${alpha})`;
  return `rgba(224, 49, 49, ${alpha})`;
}

/**
 * Render an enriched tooltip for strike-level call/put/net values.
 */
function renderTooltipContent(
  payload: Array<{ name?: string; value?: number }> | undefined,
  label: number | string | undefined,
): React.ReactNode {
  if (!payload || payload.length === 0) return null;
  const net = payload.find((row) => row.name === "Net GEX")?.value;
  const calls = payload.find((row) => row.name === "Calls GEX")?.value;
  const puts = payload.find((row) => row.name === "Puts GEX")?.value;

  return (
    <div style={{ background: "white", border: "1px solid #dee2e6", borderRadius: 8, padding: 10 }}>
      <Text fw={600} size="sm" mb={4}>
        Strike {label == null ? "—" : String(label)}
      </Text>
      <Text size="xs" c="#1c7ed6">
        Net: {typeof net === "number" ? `${formatCompactNumber(net)} (${net.toFixed(0)})` : "—"}
      </Text>
      <Text size="xs" c="#2b8a3e">
        Calls: {typeof calls === "number" ? `${formatCompactNumber(calls)} (${calls.toFixed(0)})` : "—"}
      </Text>
      <Text size="xs" c="#c92a2a">
        Puts: {typeof puts === "number" ? `${formatCompactNumber(puts)} (${puts.toFixed(0)})` : "—"}
      </Text>
    </div>
  );
}

/**
 * Render GEX analytics with two chart modes:
 * - composed chart for stacked calls/puts + net line
 * - strike/expiration heatmap for cross-expiry structure
 */
export function GexPanel({
  snapshots,
  selectedSnapshot,
  onSelectedSnapshotChange,
  selectedUnderlying,
  onSelectedUnderlyingChange,
  dtes,
  expirations,
  selectedDte,
  onSelectedDteChange,
  selectedCustomExpirations,
  onSelectedCustomExpirationsChange,
  loading,
  curve,
}: GexPanelProps) {
  const [chartView, setChartView] = React.useState<ChartView>(() => readStoredGexView());
  const [selectedStrikeCount, setSelectedStrikeCount] = React.useState<GexStrikeCountOption>(() =>
    readStoredGexStrikeCount()
  );
  const [heatmapRows, setHeatmapRows] = React.useState<HeatmapRow[]>([]);
  const [heatmapStrikes, setHeatmapStrikes] = React.useState<number[]>([]);
  const [heatmapMaxAbs, setHeatmapMaxAbs] = React.useState<number>(0);
  const [heatmapLoading, setHeatmapLoading] = React.useState<boolean>(false);
  const [heatmapError, setHeatmapError] = React.useState<string | null>(null);

  const normalizedUnderlying = normalizeUnderlying(selectedUnderlying);
  const filteredSnapshots = React.useMemo(
    () => snapshots.filter((snapshot) => normalizeUnderlying(snapshot.underlying) === normalizedUnderlying),
    [normalizedUnderlying, snapshots],
  );
  const strikeSelectionTarget = React.useMemo(() => {
    if (!selectedSnapshot) return null;
    if (normalizeUnderlying(selectedSnapshot.underlying) !== normalizedUnderlying) return null;
    return typeof selectedSnapshot.spot_price === "number" ? selectedSnapshot.spot_price : null;
  }, [normalizedUnderlying, selectedSnapshot]);
  const strikeLimit = selectedStrikeCount === "all" ? null : Number(selectedStrikeCount);
  const allChartRows = React.useMemo<ChartRow[]>(
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
  const chartRows = React.useMemo<ChartRow[]>(() => {
    if (strikeLimit == null || allChartRows.length <= strikeLimit) {
      return allChartRows;
    }
    const selectedStrikes = new Set(selectNearestStrikes(allChartRows.map((row) => row.strike), strikeSelectionTarget, strikeLimit));
    return allChartRows.filter((row) => selectedStrikes.has(row.strike));
  }, [allChartRows, strikeLimit, strikeSelectionTarget]);
  const curveStrikes = React.useMemo(() => chartRows.map((row) => row.strike), [chartRows]);

  const snapshotOptions = React.useMemo(
    () =>
      filteredSnapshots.map((snapshot) => ({
        value: String(snapshot.snapshot_id),
        label: formatSnapshotOptionLabel(snapshot),
      })),
    [filteredSnapshots],
  );
  const selectedSnapshotForDisplay = React.useMemo(() => {
    if (!selectedSnapshot) return null;
    if (normalizeUnderlying(selectedSnapshot.underlying) !== normalizedUnderlying) {
      return null;
    }
    return filteredSnapshots.find((row) => row.snapshot_id === selectedSnapshot.snapshot_id) ?? null;
  }, [filteredSnapshots, normalizedUnderlying, selectedSnapshot]);
  const underlyingOptions = React.useMemo(
    () => GEX_UNDERLYING_OPTIONS.map((symbol) => ({ value: symbol, label: symbol })),
    [],
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
    () => {
      const realExpirationOptions = expirations.map((expiration) => ({
        value: expiration.expiration,
        label:
          expiration.dte_days == null
            ? expiration.expiration
            : `${expiration.expiration} (DTE ${expiration.dte_days})`,
      }));
      const zeroDteOption = buildZeroDteOnlyOption(selectedSnapshotForDisplay);
      return zeroDteOption ? [zeroDteOption, ...realExpirationOptions] : realExpirationOptions;
    },
    [expirations, selectedSnapshotForDisplay],
  );

  const needsCustomSelections = selectedDte === "custom" && selectedCustomExpirations.length === 0;
  const selectedSnapshotId = selectedSnapshotForDisplay?.snapshot_id ?? null;

  const heatmapExpirations = React.useMemo<ExpirationChoice[]>(() => {
    if (selectedDte === "custom") {
      if (selectedCustomExpirations.includes(GEX_ZERO_DTE_ONLY_SENTINEL)) {
        const tradingDate = getSnapshotTradingDateIso(selectedSnapshotForDisplay);
        if (!tradingDate) return [];
        return expirations
          .filter((row) => row.expiration === tradingDate)
          .map((row) => ({ expiration: row.expiration, dte_days: row.dte_days }));
      }
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
  }, [expirations, selectedCustomExpirations, selectedDte, selectedSnapshotForDisplay]);

  const nearestSpotStrike = React.useMemo(
    () => findNearestStrike(selectedSnapshotForDisplay?.spot_price, curveStrikes),
    [curveStrikes, selectedSnapshotForDisplay?.spot_price],
  );
  const nearestZeroGammaStrike = React.useMemo(
    () => findNearestStrike(selectedSnapshotForDisplay?.zero_gamma_level, curveStrikes),
    [curveStrikes, selectedSnapshotForDisplay?.zero_gamma_level],
  );
  const visibleHeatmapStrikes = React.useMemo(() => {
    if (strikeLimit == null || heatmapStrikes.length <= strikeLimit) {
      return heatmapStrikes;
    }
    return selectNearestStrikes(heatmapStrikes, strikeSelectionTarget, strikeLimit);
  }, [heatmapStrikes, strikeLimit, strikeSelectionTarget]);

  const heatmapLabelStep = React.useMemo(() => {
    if (visibleHeatmapStrikes.length <= 12) return 1;
    return Math.ceil(visibleHeatmapStrikes.length / 12);
  }, [visibleHeatmapStrikes.length]);

  /**
   * Persist selected chart view between reloads for analyst convenience.
   */
  React.useEffect(() => {
    writeStoredGexView(chartView);
  }, [chartView]);

  /**
   * Persist selected strike-window size between reloads.
   */
  React.useEffect(() => {
    writeStoredGexStrikeCount(selectedStrikeCount);
  }, [selectedStrikeCount]);

  /**
   * Load per-expiration curves for heatmap mode and normalize them into a
   * shared strike grid so each expiration row can be rendered consistently.
   */
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
            label="Underlying"
            data={underlyingOptions}
            value={selectedUnderlying}
            onChange={onSelectedUnderlyingChange}
            w={120}
          />
          <Select
            label="Capture batch"
            data={snapshotOptions}
            value={selectedSnapshotForDisplay ? String(selectedSnapshotForDisplay.snapshot_id) : null}
            onChange={(value) => {
              const id = value ? Number(value) : null;
              const snapshot = id ? filteredSnapshots.find((row) => row.snapshot_id === id) ?? null : null;
              onSelectedSnapshotChange(snapshot);
            }}
            w={320}
            placeholder="Select capture batch"
          />
          <Select label="DTE" data={dteOptions} value={selectedDte} onChange={(value) => onSelectedDteChange(value || "all")} w={220} />
          <Select
            label="Strikes"
            data={[
              { value: "50", label: "50" },
              { value: "100", label: "100" },
              { value: "150", label: "150" },
              { value: "all", label: "All" },
            ]}
            value={selectedStrikeCount}
            onChange={(value) => {
              if (!value || !GEX_STRIKE_COUNT_OPTIONS.includes(value as GexStrikeCountOption)) {
                setSelectedStrikeCount("all");
                return;
              }
              setSelectedStrikeCount(value as GexStrikeCountOption);
            }}
            w={120}
          />
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
              onChange={(values) => onSelectedCustomExpirationsChange(normalizeCustomExpirationSelection(values))}
              placeholder="Pick expiration dates"
              searchable
              clearable
              w={300}
            />
          )}
        </Group>
      </Group>

      {selectedSnapshotForDisplay && (
        <Group gap="lg" mb="sm">
          <Text size="sm" c="dimmed">
            Spot: {selectedSnapshotForDisplay.spot_price ?? "—"}
          </Text>
          <Text size="sm" c="dimmed">
            GEX Net: {selectedSnapshotForDisplay.gex_net ?? "—"}
          </Text>
          <Text size="sm" c="dimmed">
            Zero Gamma: {selectedSnapshotForDisplay.zero_gamma_level ?? "—"}
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
                content={({ payload, label }) => renderTooltipContent(payload as Array<{ name?: string; value?: number }> | undefined, label)}
              />
              <Legend />
              <ReferenceLine y={0} stroke="#495057" strokeWidth={2} />
              {nearestSpotStrike != null && (
                <ReferenceLine x={nearestSpotStrike} stroke="#495057" strokeDasharray="4 4" label="Spot" />
              )}
              {nearestZeroGammaStrike != null && (
                <ReferenceLine x={nearestZeroGammaStrike} stroke="#f08c00" strokeDasharray="4 4" label="Zero Gamma" />
              )}
              <Bar dataKey="gex_calls" name="Calls GEX" stackId="gex" fill="#2b8a3e" />
              <Bar dataKey="gex_puts" name="Puts GEX" stackId="gex" fill="#c92a2a" />
              <Line type="monotone" dataKey="gex_net" name="Net GEX" stroke="#1c7ed6" strokeWidth={2} dot={false} />
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

      {chartView === "heatmap" && !heatmapLoading && !heatmapError && heatmapRows.length > 0 && visibleHeatmapStrikes.length > 0 && (
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
                gridTemplateColumns: `160px repeat(${visibleHeatmapStrikes.length}, 12px)`,
                gap: 2,
                minWidth: 160 + visibleHeatmapStrikes.length * 14,
              }}
            >
              <div />
              {visibleHeatmapStrikes.map((strike, index) => (
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
                  {visibleHeatmapStrikes.map((strike) => {
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
