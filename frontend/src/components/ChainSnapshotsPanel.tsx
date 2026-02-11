import { Badge, Card, Code, Group, Loader, ScrollArea, Table, Text } from "@mantine/core";
import type { ChainSnapshot } from "../api";
import { formatTs, truncate } from "../utils/format";

type ChainSnapshotsPanelProps = {
  items: ChainSnapshot[];
  loading: boolean;
  error: string | null;
};

export function ChainSnapshotsPanel({ items, loading, error }: ChainSnapshotsPanelProps) {
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
            {items.length} rows
          </Text>
        )}
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>ID</Table.Th>
              <Table.Th>Time</Table.Th>
              <Table.Th>Underlying</Table.Th>
              <Table.Th>DTE</Table.Th>
              <Table.Th>Expiration</Table.Th>
              <Table.Th>Checksum</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {items.map((item) => (
              <Table.Tr key={item.snapshot_id}>
                <Table.Td>{item.snapshot_id}</Table.Td>
                <Table.Td>{formatTs(item.ts)}</Table.Td>
                <Table.Td>
                  <Badge variant="light">{item.underlying}</Badge>
                </Table.Td>
                <Table.Td>{item.target_dte}</Table.Td>
                <Table.Td>{item.expiration}</Table.Td>
                <Table.Td>
                  <Code>{truncate(item.checksum, 12)}</Code>
                </Table.Td>
              </Table.Tr>
            ))}
            {items.length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={6}>
                  <Text c="dimmed">No snapshots yet. Try clicking “Run snapshot now”.</Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Card>
  );
}
