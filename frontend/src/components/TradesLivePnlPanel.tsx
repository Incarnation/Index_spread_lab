import { Badge, Card, Group, Loader, ScrollArea, Table, Text } from "@mantine/core";
import type { TradeLeg, TradeRow } from "../api";
import { formatTs } from "../utils/format";

type TradesLivePnlPanelProps = {
  trades: TradeRow[];
  loading: boolean;
  error: string | null;
};

/** Format PnL values as signed USD text with graceful null handling. */
function formatMoney(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}$${value.toFixed(2)}`;
}

/** Build a compact human-readable label for one trade leg. */
function legLabel(leg: TradeLeg): string {
  const strike = typeof leg.strike === "number" ? leg.strike.toFixed(0) : "—";
  const right = leg.option_right ?? "";
  return `${leg.side} ${strike}${right} x${leg.qty}`;
}

/** Map numeric PnL to semantic text color used in the table. */
function pnlColor(value: number | null): string | undefined {
  if (typeof value !== "number" || Number.isNaN(value)) return undefined;
  if (value > 0) return "green";
  if (value < 0) return "red";
  return "dimmed";
}

/**
 * Render live and realized PnL state for each recorded trade.
 *
 * This panel is the operational view for open/closed positions, last marks,
 * TP/SL targets, and current profit state per trade.
 */
export function TradesLivePnlPanel({ trades, loading, error }: TradesLivePnlPanelProps) {
  const openCount = trades.filter((t) => t.status === "OPEN").length;
  const closedCount = trades.filter((t) => t.status === "CLOSED").length;

  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Group gap="xs">
          <Text fw={600}>Trades + Live PnL</Text>
          <Badge variant="light" color="green">
            OPEN {openCount}
          </Badge>
          <Badge variant="light" color="gray">
            CLOSED {closedCount}
          </Badge>
        </Group>
        {loading ? (
          <Group gap="xs">
            <Loader size="sm" />
            <Text c="dimmed" size="sm">
              Loading…
            </Text>
          </Group>
        ) : (
          <Text c="dimmed" size="sm">
            {trades.length} rows
          </Text>
        )}
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Trade</Table.Th>
              <Table.Th>Entry Time</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>DTE</Table.Th>
              <Table.Th>Expiration</Table.Th>
              <Table.Th>Legs</Table.Th>
              <Table.Th>Entry Credit</Table.Th>
              <Table.Th>Current Exit</Table.Th>
              <Table.Th>Current PnL</Table.Th>
              <Table.Th>Realized PnL</Table.Th>
              <Table.Th>TP / SL</Table.Th>
              <Table.Th>Last Mark</Table.Th>
              <Table.Th>Marks</Table.Th>
              <Table.Th>Exit Reason</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {trades.map((trade) => (
              <Table.Tr key={trade.trade_id}>
                <Table.Td>
                  <Text size="sm" fw={600}>
                    #{trade.trade_id}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {trade.strategy_type}
                  </Text>
                </Table.Td>
                <Table.Td>{formatTs(trade.entry_time)}</Table.Td>
                <Table.Td>
                  <Badge variant="light" color={trade.status === "OPEN" ? "green" : "gray"}>
                    {trade.status}
                  </Badge>
                </Table.Td>
                <Table.Td>{trade.target_dte ?? "—"}</Table.Td>
                <Table.Td>{trade.expiration ?? "—"}</Table.Td>
                <Table.Td>
                  <Text size="xs" maw={260}>
                    {trade.legs.length > 0 ? trade.legs.map(legLabel).join(" | ") : "—"}
                  </Text>
                </Table.Td>
                <Table.Td>{typeof trade.entry_credit === "number" ? trade.entry_credit.toFixed(2) : "—"}</Table.Td>
                <Table.Td>{typeof trade.current_exit_cost === "number" ? trade.current_exit_cost.toFixed(2) : "—"}</Table.Td>
                <Table.Td>
                  <Text c={pnlColor(trade.current_pnl)}>{formatMoney(trade.current_pnl)}</Text>
                </Table.Td>
                <Table.Td>
                  <Text c={pnlColor(trade.realized_pnl)}>{formatMoney(trade.realized_pnl)}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs">TP {formatMoney(trade.take_profit_target)}</Text>
                  <Text size="xs">SL {formatMoney(trade.stop_loss_target)}</Text>
                </Table.Td>
                <Table.Td>{trade.last_mark_ts ? formatTs(trade.last_mark_ts) : "—"}</Table.Td>
                <Table.Td>{trade.mark_count}</Table.Td>
                <Table.Td>{trade.exit_reason ?? "—"}</Table.Td>
              </Table.Tr>
            ))}
            {trades.length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={14}>
                  <Text c="dimmed">No trades yet. Run the decision job and then run trade PnL updates.</Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Card>
  );
}
