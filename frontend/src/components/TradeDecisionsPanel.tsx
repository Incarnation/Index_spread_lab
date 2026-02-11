import { Badge, Button, Card, Group, Loader, ScrollArea, Table, Text } from "@mantine/core";
import type { TradeDecision } from "../api";
import { getDecisionSummary } from "../utils/decision";
import { formatTs } from "../utils/format";

type TradeDecisionsPanelProps = {
  decisions: TradeDecision[];
  loading: boolean;
  error: string | null;
  deletingDecisionId: number | null;
  onViewDecision: (decision: TradeDecision) => void;
  onDeleteDecision: (decisionId: number) => void;
};

export function TradeDecisionsPanel({
  decisions,
  loading,
  error,
  deletingDecisionId,
  onViewDecision,
  onDeleteDecision,
}: TradeDecisionsPanelProps) {
  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Trade decisions</Text>
        {loading ? (
          <Group gap="xs">
            <Loader size="sm" />
            <Text c="dimmed" size="sm">
              Loading…
            </Text>
          </Group>
        ) : (
          <Text c="dimmed" size="sm">
            {decisions.length} rows
          </Text>
        )}
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>ID</Table.Th>
              <Table.Th>Time</Table.Th>
              <Table.Th>Decision</Table.Th>
              <Table.Th>DTE</Table.Th>
              <Table.Th>Delta</Table.Th>
              <Table.Th>Legs</Table.Th>
              <Table.Th>Credit</Table.Th>
              <Table.Th>Score</Table.Th>
              <Table.Th>GEX</Table.Th>
              <Table.Th>Zero Gamma</Table.Th>
              <Table.Th>VIX</Table.Th>
              <Table.Th>Reason</Table.Th>
              <Table.Th>Actions</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {decisions.map((decision) => {
              const summary = getDecisionSummary(decision);
              return (
                <Table.Tr key={decision.decision_id}>
                  <Table.Td>{decision.decision_id}</Table.Td>
                  <Table.Td>{formatTs(decision.ts)}</Table.Td>
                  <Table.Td>
                    <Badge variant="light" color={decision.decision === "TRADE" ? "green" : "gray"}>
                      {decision.decision}
                    </Badge>
                  </Table.Td>
                  <Table.Td>{decision.target_dte}</Table.Td>
                  <Table.Td>{decision.delta_target}</Table.Td>
                  <Table.Td>{summary ? `${summary.shortStrike} / ${summary.longStrike}` : "—"}</Table.Td>
                  <Table.Td>{summary?.credit?.toFixed(2) ?? "—"}</Table.Td>
                  <Table.Td>{typeof decision.score === "number" ? decision.score.toFixed(3) : "—"}</Table.Td>
                  <Table.Td>{summary?.gexNet?.toFixed(0) ?? "—"}</Table.Td>
                  <Table.Td>{summary?.zeroGamma?.toFixed(2) ?? "—"}</Table.Td>
                  <Table.Td>{summary?.vix?.toFixed(2) ?? "—"}</Table.Td>
                  <Table.Td>{decision.reason ?? "—"}</Table.Td>
                  <Table.Td>
                    <Group gap="xs">
                      <Button size="xs" variant="light" onClick={() => onViewDecision(decision)}>
                        View
                      </Button>
                      <Button
                        size="xs"
                        color="red"
                        variant="outline"
                        loading={deletingDecisionId === decision.decision_id}
                        onClick={() => onDeleteDecision(decision.decision_id)}
                      >
                        Delete
                      </Button>
                    </Group>
                  </Table.Td>
                </Table.Tr>
              );
            })}
            {decisions.length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={13}>
                  <Text c="dimmed">No decisions yet. Run the decision job to populate.</Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Card>
  );
}
