import { Badge, Card, Group, Loader, ScrollArea, Table, Text } from "@mantine/core";
import type { StrategyMetricsResponse } from "../api";

type StrategyQualityPanelProps = {
  metrics: StrategyMetricsResponse | null;
  loading: boolean;
  error: string | null;
};

/** Format ratio metrics (0-1) as percentages for badge/table display. */
function formatPct(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

/** Format dollar metrics with sign and two decimals. */
function formatMoney(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}$${value.toFixed(2)}`;
}

/**
 * Show strategy quality and risk metrics for model/policy evaluation.
 *
 * Metrics include both return-oriented measures (expectancy) and risk guards
 * (drawdown, tail-loss proxy, margin usage) overall and by spread side.
 */
export function StrategyQualityPanel({ metrics, loading, error }: StrategyQualityPanelProps) {
  const summary = metrics?.summary ?? null;
  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Strategy quality and risk (v2)</Text>
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
          Win50 {formatPct(summary?.tp50_rate ?? null)}
        </Badge>
        <Badge variant="light" color="violet">
          Win100 @ expiry {formatPct(summary?.tp100_at_expiry_rate ?? null)}
        </Badge>
        <Badge variant="light" color="green">
          Expectancy {formatMoney(summary?.expectancy ?? null)}
        </Badge>
        <Badge variant="light" color="orange">
          Drawdown {formatMoney(summary?.max_drawdown ?? null)}
        </Badge>
        <Badge variant="light" color="red">
          Tail loss proxy {formatMoney(summary?.tail_loss_proxy ?? null)}
        </Badge>
        <Badge variant="light" color="gray">
          Avg margin usage {formatMoney(summary?.avg_margin_usage ?? null)}
        </Badge>
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Side</Table.Th>
              <Table.Th>Resolved</Table.Th>
              <Table.Th>Win50</Table.Th>
              <Table.Th>Win100 @ expiry</Table.Th>
              <Table.Th>Expectancy</Table.Th>
              <Table.Th>Drawdown</Table.Th>
              <Table.Th>Tail loss proxy</Table.Th>
              <Table.Th>Avg margin usage</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {(metrics?.by_side ?? []).map((row) => (
              <Table.Tr key={row.spread_side}>
                <Table.Td>
                  <Badge variant="light">{row.spread_side.toUpperCase()}</Badge>
                </Table.Td>
                <Table.Td>{row.resolved}</Table.Td>
                <Table.Td>{formatPct(row.tp50_rate)}</Table.Td>
                <Table.Td>{formatPct(row.tp100_at_expiry_rate)}</Table.Td>
                <Table.Td>{formatMoney(row.expectancy)}</Table.Td>
                <Table.Td>{formatMoney(row.max_drawdown)}</Table.Td>
                <Table.Td>{formatMoney(row.tail_loss_proxy)}</Table.Td>
                <Table.Td>{formatMoney(row.avg_margin_usage)}</Table.Td>
              </Table.Tr>
            ))}
            {(metrics?.by_side ?? []).length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={8}>
                  <Text c="dimmed">No resolved labels yet. Run feature-builder + labeler first.</Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Card>
  );
}

