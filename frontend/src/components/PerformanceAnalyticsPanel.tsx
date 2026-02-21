import React from "react";
import { Badge, Card, Group, Loader, ScrollArea, SegmentedControl, SimpleGrid, Table, Text } from "@mantine/core";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PerformanceAnalyticsBreakdownRow, PerformanceAnalyticsMode, PerformanceAnalyticsResponse } from "../api";

type PerformanceAnalyticsPanelProps = {
  combined: PerformanceAnalyticsResponse | null;
  realized: PerformanceAnalyticsResponse | null;
  loading: boolean;
  error: string | null;
  lookbackDays: number;
  onLookbackDaysChange: (days: number) => void;
};

/** Format ratio values (0..1) as percentages for badge/table display. */
function formatPct(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

/** Format dollar values with sign and fixed precision. */
function formatMoney(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}$${value.toFixed(2)}`;
}

/** Build chart tooltip lines for cumulative and daily PnL values. */
function tooltipFormatter(value: number | string | undefined, name: string | undefined): [string, string] {
  const numericValue = typeof value === "number" ? value : Number(value);
  if (name === "cumulative_pnl") return [formatMoney(numericValue), "Cumulative PnL"];
  if (name === "daily_pnl") return [formatMoney(numericValue), "Daily PnL"];
  return [String(value), name ?? ""];
}

/** Convert segmented-control lookback value into a supported day window. */
function parseLookbackValue(value: string): number {
  const parsed = Number(value);
  if (parsed === 1 || parsed === 7 || parsed === 30 || parsed === 90) return parsed;
  return 30;
}

/** Render one compact breakdown table for a specific analytics dimension. */
function BreakdownTable({
  title,
  rows,
}: {
  title: string;
  rows: PerformanceAnalyticsBreakdownRow[];
}) {
  return (
    <Card withBorder radius="md" p="sm">
      <Text fw={600} size="sm" mb="xs">
        {title}
      </Text>
      <ScrollArea type="auto" mah={220}>
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Bucket</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Trades</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Win rate</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Net PnL</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Expectancy</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((row) => (
              <Table.Tr key={`${title}-${row.bucket}`}>
                <Table.Td>{row.bucket}</Table.Td>
                <Table.Td style={{ textAlign: "right" }}>{row.trade_count}</Table.Td>
                <Table.Td style={{ textAlign: "right" }}>{formatPct(row.win_rate)}</Table.Td>
                <Table.Td style={{ textAlign: "right" }}>{formatMoney(row.net_pnl)}</Table.Td>
                <Table.Td style={{ textAlign: "right" }}>{formatMoney(row.expectancy)}</Table.Td>
              </Table.Tr>
            ))}
            {rows.length === 0 && (
              <Table.Tr>
                <Table.Td colSpan={5}>
                  <Text c="dimmed" size="sm">
                    No rows in this lookback window.
                  </Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Card>
  );
}

/**
 * Show aggregate PnL analytics with realized/combined mode toggle.
 *
 * This panel renders KPI badges, an equity-curve chart, and grouped breakdown
 * tables backed by the aggregate-first `/api/performance-analytics` payload.
 */
export function PerformanceAnalyticsPanel({
  combined,
  realized,
  loading,
  error,
  lookbackDays,
  onLookbackDaysChange,
}: PerformanceAnalyticsPanelProps) {
  const [mode, setMode] = React.useState<PerformanceAnalyticsMode>("combined");
  const active = mode === "combined" ? combined : realized;
  const summary = active?.summary ?? null;

  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>PnL analytics (aggregate-first)</Text>
        <Group gap="xs">
          <SegmentedControl
            value={String(lookbackDays)}
            onChange={(nextValue) => onLookbackDaysChange(parseLookbackValue(nextValue))}
            data={[
              { value: "1", label: "1D" },
              { value: "7", label: "7D" },
              { value: "30", label: "30D" },
              { value: "90", label: "90D" },
            ]}
          />
          <SegmentedControl
            value={mode}
            onChange={(nextMode) => setMode((nextMode as PerformanceAnalyticsMode) || "combined")}
            data={[
              { value: "combined", label: "Realized + Unrealized" },
              { value: "realized", label: "Realized only" },
            ]}
          />
          {loading ? (
            <Group gap="xs">
              <Loader size="sm" />
              <Text c="dimmed" size="sm">
                Loading…
              </Text>
            </Group>
          ) : (
            <Text c="dimmed" size="sm">
              Last {active?.lookback_days ?? lookbackDays} days
            </Text>
          )}
        </Group>
      </Group>

      <Group gap="xs" mb="md">
        <Badge variant="light" color="blue">
          Net {formatMoney(summary?.net_pnl ?? null)}
        </Badge>
        <Badge variant="light" color="teal">
          Realized {formatMoney(summary?.realized_net_pnl ?? null)}
        </Badge>
        <Badge variant="light" color="grape">
          Unrealized {formatMoney(summary?.unrealized_net_pnl ?? null)}
        </Badge>
        <Badge variant="light" color="green">
          Win rate {formatPct(summary?.win_rate ?? null)}
        </Badge>
        <Badge variant="light" color="cyan">
          Avg win {formatMoney(summary?.avg_win ?? null)}
        </Badge>
        <Badge variant="light" color="orange">
          Avg loss {formatMoney(summary?.avg_loss ?? null)}
        </Badge>
        <Badge variant="light" color="violet">
          Profit factor {typeof summary?.profit_factor === "number" ? summary.profit_factor.toFixed(2) : "—"}
        </Badge>
        <Badge variant="light" color="red">
          Max drawdown {formatMoney(summary?.max_drawdown ?? null)}
        </Badge>
      </Group>

      <Card withBorder radius="md" p="sm" mb="md">
        <Text fw={600} size="sm" mb="xs">
          Equity curve
        </Text>
        {active?.equity_curve?.length ? (
          <div style={{ width: "100%", height: 260 }}>
            <ResponsiveContainer>
              <LineChart data={active.equity_curve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" minTickGap={28} />
                <YAxis />
                <Tooltip formatter={tooltipFormatter} />
                <Line type="monotone" dataKey="cumulative_pnl" stroke="#1c7ed6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="daily_pnl" stroke="#15aabf" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Text c="dimmed" size="sm">
            No equity points available yet.
          </Text>
        )}
      </Card>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="sm">
        <BreakdownTable title="By side" rows={active?.breakdowns.side ?? []} />
        <BreakdownTable title="By DTE bucket" rows={active?.breakdowns.dte_bucket ?? []} />
        <BreakdownTable title="By delta bucket" rows={active?.breakdowns.delta_bucket ?? []} />
        <BreakdownTable title="By weekday" rows={active?.breakdowns.weekday ?? []} />
        <BreakdownTable title="By entry hour" rows={active?.breakdowns.hour ?? []} />
        <BreakdownTable title="By source" rows={active?.breakdowns.source ?? []} />
      </SimpleGrid>

      {!loading && !error && !active?.summary && (
        <Text c="dimmed" size="sm" mt="sm">
          Analytics snapshot is not ready yet. Run trade PnL and performance analytics jobs, then refresh.
        </Text>
      )}
    </Card>
  );
}
