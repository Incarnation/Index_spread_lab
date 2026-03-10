import React from "react";
import { Badge, Card, Group, Loader, ScrollArea, Select, Table, Text, UnstyledButton } from "@mantine/core";
import type { TradeLeg, TradeRow } from "../api";
import { formatTs } from "../utils/format";

type TradesLivePnlPanelProps = {
  trades: TradeRow[];
  loading: boolean;
  error: string | null;
};

type TimeWindowFilter = "all" | "24h" | "7d" | "30d";
type TradeSortField = "entry_time" | "current_pnl" | "realized_pnl";
type SortDirection = "asc" | "desc";

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

/** Infer spread side from strategy naming conventions used in trade rows. */
function inferTradeSide(trade: TradeRow): "put" | "call" | "unknown" {
  const strategy = trade.strategy_type.toLowerCase();
  if (strategy.includes("put")) return "put";
  if (strategy.includes("call")) return "call";
  return "unknown";
}

/** Evaluate if a trade entry timestamp belongs to the selected time window. */
function withinTimeWindow(ts: string, window: TimeWindowFilter): boolean {
  if (window === "all") return true;
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return true;
  const ageMs = Date.now() - date.getTime();
  const maxAgeMs =
    window === "24h" ? 24 * 60 * 60 * 1000 : window === "7d" ? 7 * 24 * 60 * 60 * 1000 : 30 * 24 * 60 * 60 * 1000;
  return ageMs <= maxAgeMs;
}

/** Render sortable header labels with an inline direction arrow. */
function sortHeaderLabel(field: TradeSortField, activeField: TradeSortField, direction: SortDirection, title: string): string {
  if (field !== activeField) return title;
  return `${title} ${direction === "asc" ? "↑" : "↓"}`;
}

/**
 * Render live and realized PnL state for each recorded trade.
 *
 * This panel is the operational view for open/closed positions, last marks,
 * TP/SL targets, and current profit state per trade.
 */
export function TradesLivePnlPanel({ trades, loading, error }: TradesLivePnlPanelProps) {
  const [timeWindow, setTimeWindow] = React.useState<TimeWindowFilter>("all");
  const [statusFilter, setStatusFilter] = React.useState<string>("all");
  const [dteFilter, setDteFilter] = React.useState<string>("all");
  const [sideFilter, setSideFilter] = React.useState<string>("all");
  const [sortField, setSortField] = React.useState<TradeSortField>("entry_time");
  const [sortDirection, setSortDirection] = React.useState<SortDirection>("desc");

  /** Toggle one sortable column and direction state. */
  const handleSortChange = React.useCallback((nextField: TradeSortField) => {
    setSortField((previousField) => {
      if (previousField === nextField) {
        setSortDirection((previousDirection) => (previousDirection === "asc" ? "desc" : "asc"));
        return previousField;
      }
      setSortDirection("desc");
      return nextField;
    });
  }, []);

  const dteOptions = React.useMemo(
    () => [
      { value: "all", label: "All DTE" },
      ...Array.from(new Set(trades.map((trade) => trade.target_dte).filter((value): value is number => typeof value === "number")))
        .sort((a, b) => a - b)
        .map((value) => ({ value: String(value), label: `DTE ${value}` })),
    ],
    [trades],
  );

  const filteredTrades = React.useMemo(() => {
    return trades.filter((trade) => {
      if (!withinTimeWindow(trade.entry_time, timeWindow)) return false;
      if (statusFilter !== "all" && trade.status !== statusFilter) return false;
      if (dteFilter !== "all" && trade.target_dte !== Number(dteFilter)) return false;
      if (sideFilter !== "all" && inferTradeSide(trade) !== sideFilter) return false;
      return true;
    });
  }, [dteFilter, sideFilter, statusFilter, timeWindow, trades]);

  const sortedTrades = React.useMemo(() => {
    const rows = [...filteredTrades];
    rows.sort((left, right) => {
      if (sortField === "current_pnl" || sortField === "realized_pnl") {
        const leftValue = typeof left[sortField] === "number" ? (left[sortField] as number) : Number.NEGATIVE_INFINITY;
        const rightValue = typeof right[sortField] === "number" ? (right[sortField] as number) : Number.NEGATIVE_INFINITY;
        const delta = leftValue - rightValue;
        return sortDirection === "asc" ? delta : -delta;
      }
      const leftTs = new Date(left.entry_time).getTime();
      const rightTs = new Date(right.entry_time).getTime();
      const delta = leftTs - rightTs;
      return sortDirection === "asc" ? delta : -delta;
    });
    return rows;
  }, [filteredTrades, sortDirection, sortField]);

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
            {sortedTrades.length} / {trades.length} rows
          </Text>
        )}
      </Group>

      <Group mb="sm" gap="sm" wrap="wrap">
        <Select
          label="Time window"
          data={[
            { value: "all", label: "All time" },
            { value: "24h", label: "Last 24h" },
            { value: "7d", label: "Last 7 days" },
            { value: "30d", label: "Last 30 days" },
          ]}
          value={timeWindow}
          onChange={(value) => setTimeWindow((value as TimeWindowFilter) || "all")}
          w={170}
        />
        <Select
          label="Status"
          data={[
            { value: "all", label: "All statuses" },
            { value: "OPEN", label: "OPEN" },
            { value: "CLOSED", label: "CLOSED" },
            { value: "ROLLED", label: "ROLLED" },
          ]}
          value={statusFilter}
          onChange={(value) => setStatusFilter(value || "all")}
          w={170}
        />
        <Select label="DTE" data={dteOptions} value={dteFilter} onChange={(value) => setDteFilter(value || "all")} w={140} />
        <Select
          label="Side"
          data={[
            { value: "all", label: "All sides" },
            { value: "put", label: "Put spreads" },
            { value: "call", label: "Call spreads" },
          ]}
          value={sideFilter}
          onChange={(value) => setSideFilter(value || "all")}
          w={150}
        />
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders stickyHeader stickyHeaderOffset={0}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Trade</Table.Th>
              <Table.Th aria-sort={sortField === "entry_time" ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}>
                <UnstyledButton onClick={() => handleSortChange("entry_time")} aria-label="Sort by entry time">
                  {sortHeaderLabel("entry_time", sortField, sortDirection, "Entry Time")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>DTE</Table.Th>
              <Table.Th>Expiration</Table.Th>
              <Table.Th>Legs</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Entry Credit</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Current Exit</Table.Th>
              <Table.Th style={{ textAlign: "right" }} aria-sort={sortField === "current_pnl" ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}>
                <UnstyledButton onClick={() => handleSortChange("current_pnl")} aria-label="Sort by current PnL">
                  {sortHeaderLabel("current_pnl", sortField, sortDirection, "Current PnL")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th style={{ textAlign: "right" }} aria-sort={sortField === "realized_pnl" ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}>
                <UnstyledButton onClick={() => handleSortChange("realized_pnl")} aria-label="Sort by realized PnL">
                  {sortHeaderLabel("realized_pnl", sortField, sortDirection, "Realized PnL")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th>TP / SL</Table.Th>
              <Table.Th>Last Mark</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Marks</Table.Th>
              <Table.Th>Exit Reason</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sortedTrades.map((trade) => (
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
                <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                  {typeof trade.entry_credit === "number" ? trade.entry_credit.toFixed(2) : "—"}
                </Table.Td>
                <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                  {typeof trade.current_exit_cost === "number" ? trade.current_exit_cost.toFixed(2) : "—"}
                </Table.Td>
                <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                  <Text c={pnlColor(trade.current_pnl)}>{formatMoney(trade.current_pnl)}</Text>
                </Table.Td>
                <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                  <Text c={pnlColor(trade.realized_pnl)}>{formatMoney(trade.realized_pnl)}</Text>
                </Table.Td>
                <Table.Td>
                  <Text size="xs">TP {formatMoney(trade.take_profit_target)}</Text>
                  <Text size="xs">SL {formatMoney(trade.stop_loss_target)}</Text>
                </Table.Td>
                <Table.Td>{trade.last_mark_ts ? formatTs(trade.last_mark_ts) : "—"}</Table.Td>
                <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>{trade.mark_count}</Table.Td>
                <Table.Td>{trade.exit_reason ?? "—"}</Table.Td>
              </Table.Tr>
            ))}
            {sortedTrades.length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={14}>
                  <Text c="dimmed">
                    No trades for this filter. Next step: run decision now, then run trade PnL to update marks and realized outcomes.
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
