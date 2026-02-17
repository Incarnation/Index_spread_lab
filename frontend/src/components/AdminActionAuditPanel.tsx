import { Badge, Card, Group, ScrollArea, Table, Text } from "@mantine/core";
import type { AdminActionAuditEntry } from "../hooks/useAdminRuns";
import { formatTs } from "../utils/format";

type AdminActionAuditPanelProps = {
  entries: AdminActionAuditEntry[];
};

/**
 * Render a compact audit feed of manual admin actions and outcomes.
 */
export function AdminActionAuditPanel({ entries }: AdminActionAuditPanelProps) {
  return (
    <Card withBorder radius="md" mt="md" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Admin action audit</Text>
        <Text size="sm" c="dimmed">
          {entries.length} recent actions
        </Text>
      </Group>

      <ScrollArea type="auto">
        <Table withTableBorder withColumnBorders striped highlightOnHover stickyHeader stickyHeaderOffset={0}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Time</Table.Th>
              <Table.Th>Operator</Table.Th>
              <Table.Th>Action</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Duration</Table.Th>
              <Table.Th>Detail</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {entries.map((entry) => (
              <Table.Tr key={entry.id}>
                <Table.Td>{formatTs(entry.completedAt)}</Table.Td>
                <Table.Td>{entry.operator}</Table.Td>
                <Table.Td>{entry.action}</Table.Td>
                <Table.Td>
                  <Badge variant="light" color={entry.status === "success" ? "green" : "red"}>
                    {entry.status.toUpperCase()}
                  </Badge>
                </Table.Td>
                <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>{entry.durationMs} ms</Table.Td>
                <Table.Td>{entry.detail}</Table.Td>
              </Table.Tr>
            ))}
            {entries.length === 0 && (
              <Table.Tr>
                <Table.Td colSpan={6}>
                  <Text c="dimmed">
                    No actions recorded yet. Run an admin action to capture a timestamped audit row.
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

