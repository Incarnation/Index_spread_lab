import React from "react";
import { Card, Group, Loader, MultiSelect, Select, Text } from "@mantine/core";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { GexCurvePoint, GexExpirationItem, GexSnapshot } from "../api";
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

      {loading && (
        <Group gap="xs">
          <Loader size="sm" />
          <Text c="dimmed" size="sm">
            Loading GEX…
          </Text>
        </Group>
      )}

      {!loading && needsCustomSelections && <Text c="dimmed">Pick one or more expiration dates to render a custom GEX curve.</Text>}

      {!loading && curve.length === 0 && !needsCustomSelections && (
        <Text c="dimmed">No GEX data yet. Run snapshots and GEX first.</Text>
      )}

      {!loading && curve.length > 0 && (
        <div style={{ width: "100%", height: 320 }}>
          <ResponsiveContainer>
            <LineChart data={curve}>
              <XAxis dataKey="strike" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Line type="monotone" dataKey="gex_net" stroke="#228be6" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}
