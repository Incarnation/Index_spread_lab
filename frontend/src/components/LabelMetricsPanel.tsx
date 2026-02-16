import { Badge, Card, Group, Loader, ScrollArea, Table, Text } from "@mantine/core";
import type { LabelMetricsResponse } from "../api";

type LabelMetricsPanelProps = {
  metrics: LabelMetricsResponse | null;
  loading: boolean;
  error: string | null;
};

function formatPct(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatMoney(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}$${value.toFixed(2)}`;
}

export function LabelMetricsPanel({ metrics, loading, error }: LabelMetricsPanelProps) {
  const summary = metrics?.summary ?? null;

  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Label metrics (TP50 vs TP100 @ expiry)</Text>
        {loading ? (
          <Group gap="xs">
            <Loader size="sm" />
            <Text c="dimmed" size="sm">
              Loading…
            </Text>
          </Group>
        ) : (
          <Text c="dimmed" size="sm">
            Last {metrics?.lookback_days ?? 0} days
          </Text>
        )}
      </Group>

      <Group gap="xs" mb="md">
        <Badge variant="light" color="blue">
          Resolved {summary?.resolved ?? 0}
        </Badge>
        <Badge variant="light" color="green">
          TP50 {formatPct(summary?.tp50_rate ?? null)}
        </Badge>
        <Badge variant="light" color="violet">
          TP100 @ expiry {formatPct(summary?.tp100_at_expiry_rate ?? null)}
        </Badge>
        <Badge variant="light" color="gray">
          Avg PnL {formatMoney(summary?.avg_realized_pnl ?? null)}
        </Badge>
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Spread side</Table.Th>
              <Table.Th>Resolved</Table.Th>
              <Table.Th>TP50</Table.Th>
              <Table.Th>TP50 rate</Table.Th>
              <Table.Th>TP100 @ expiry</Table.Th>
              <Table.Th>TP100 @ expiry rate</Table.Th>
              <Table.Th>Avg realized PnL</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {(metrics?.by_side ?? []).map((row) => (
              <Table.Tr key={row.spread_side}>
                <Table.Td>
                  <Badge variant="light">{row.spread_side.toUpperCase()}</Badge>
                </Table.Td>
                <Table.Td>{row.resolved}</Table.Td>
                <Table.Td>{row.tp50}</Table.Td>
                <Table.Td>{formatPct(row.tp50_rate)}</Table.Td>
                <Table.Td>{row.tp100_at_expiry}</Table.Td>
                <Table.Td>{formatPct(row.tp100_at_expiry_rate)}</Table.Td>
                <Table.Td>{formatMoney(row.avg_realized_pnl)}</Table.Td>
              </Table.Tr>
            ))}
            {(metrics?.by_side ?? []).length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={7}>
                  <Text c="dimmed">No resolved labels yet. Run feature-builder + decision + labeler first.</Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Card>
  );
}
