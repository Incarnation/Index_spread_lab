import React from "react";
import { Badge, Card, Code, Group, Loader, ScrollArea, Select, Table, Text, UnstyledButton } from "@mantine/core";
import type { ChainSnapshot } from "../api";
import { formatTs, truncate } from "../utils/format";

type ChainSnapshotsPanelProps = {
  items: ChainSnapshot[];
  loading: boolean;
  error: string | null;
};

type SnapshotSortField = "snapshot_id" | "ts" | "target_dte";
type SortDirection = "asc" | "desc";

/** Render sort header text with direction arrows for active columns. */
function sortHeaderLabel(field: SnapshotSortField, activeField: SnapshotSortField, direction: SortDirection, title: string): string {
  if (field !== activeField) return title;
  return `${title} ${direction === "asc" ? "↑" : "↓"}`;
}

/**
 * Show recently captured option-chain snapshot batches.
 *
 * This gives quick visibility into data freshness, target DTE coverage,
 * and checksum identifiers used for audit/debug workflows.
 */
export function ChainSnapshotsPanel({ items, loading, error }: ChainSnapshotsPanelProps) {
  const [underlyingFilter, setUnderlyingFilter] = React.useState<string>("all");
  const [sortField, setSortField] = React.useState<SnapshotSortField>("ts");
  const [sortDirection, setSortDirection] = React.useState<SortDirection>("desc");

  /** Toggle selected sort column and direction state for snapshot rows. */
  const handleSortChange = React.useCallback((nextField: SnapshotSortField) => {
    setSortField((previousField) => {
      if (previousField === nextField) {
        setSortDirection((previousDirection) => (previousDirection === "asc" ? "desc" : "asc"));
        return previousField;
      }
      setSortDirection("desc");
      return nextField;
    });
  }, []);

  const filteredItems = React.useMemo(
    () => (underlyingFilter === "all" ? items : items.filter((item) => item.underlying === underlyingFilter)),
    [items, underlyingFilter],
  );

  const sortedItems = React.useMemo(() => {
    const rows = [...filteredItems];
    rows.sort((left, right) => {
      if (sortField === "snapshot_id") {
        const delta = left.snapshot_id - right.snapshot_id;
        return sortDirection === "asc" ? delta : -delta;
      }
      if (sortField === "target_dte") {
        const delta = left.target_dte - right.target_dte;
        return sortDirection === "asc" ? delta : -delta;
      }
      const leftTs = new Date(left.ts).getTime();
      const rightTs = new Date(right.ts).getTime();
      const delta = leftTs - rightTs;
      return sortDirection === "asc" ? delta : -delta;
    });
    return rows;
  }, [filteredItems, sortDirection, sortField]);

  const underlyingOptions = React.useMemo(
    () => [
      { value: "all", label: "All underlyings" },
      ...Array.from(new Set(items.map((item) => item.underlying)))
        .sort()
        .map((value) => ({ value, label: value })),
    ],
    [items],
  );

  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Latest chain snapshots</Text>
        {loading ? (
          <Group gap="xs">
            <Loader size="sm" />
            <Text c="dimmed" size="sm">
              Loading…
            </Text>
          </Group>
        ) : (
          <Text c="dimmed" size="sm">
            {sortedItems.length} / {items.length} rows
          </Text>
        )}
      </Group>

      <Group mb="sm" gap="sm">
        <Select
          label="Underlying"
          data={underlyingOptions}
          value={underlyingFilter}
          onChange={(value) => setUnderlyingFilter(value || "all")}
          w={200}
        />
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders stickyHeader stickyHeaderOffset={0}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>
                <UnstyledButton onClick={() => handleSortChange("snapshot_id")}>
                  {sortHeaderLabel("snapshot_id", sortField, sortDirection, "ID")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th>
                <UnstyledButton onClick={() => handleSortChange("ts")}>
                  {sortHeaderLabel("ts", sortField, sortDirection, "Time")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th>Underlying</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>
                <UnstyledButton onClick={() => handleSortChange("target_dte")}>
                  {sortHeaderLabel("target_dte", sortField, sortDirection, "DTE")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th>Expiration</Table.Th>
              <Table.Th>Checksum</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sortedItems.map((item) => (
              <Table.Tr key={item.snapshot_id}>
                <Table.Td>{item.snapshot_id}</Table.Td>
                <Table.Td>{formatTs(item.ts)}</Table.Td>
                <Table.Td>
                  <Badge variant="light">{item.underlying}</Badge>
                </Table.Td>
                <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>{item.target_dte}</Table.Td>
                <Table.Td>{item.expiration}</Table.Td>
                <Table.Td>
                  <Code>{truncate(item.checksum, 12)}</Code>
                </Table.Td>
              </Table.Tr>
            ))}
            {sortedItems.length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={6}>
                  <Text c="dimmed">
                    No snapshots for this filter. Next step: run snapshot now (or run full pipeline) and refresh this panel.
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
