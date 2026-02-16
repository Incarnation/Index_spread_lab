import { Badge, Card, Code, Group, ScrollArea, Text } from "@mantine/core";
import type { RunDecisionResult, RunQuotesResult, RunSnapshotResult, RunTradePnlResult } from "../api";

type ResultPayload = RunSnapshotResult | RunQuotesResult | RunDecisionResult | RunTradePnlResult;

type RunResultCardProps = {
  title: string;
  result: ResultPayload;
  scrollHeight?: number;
};

/**
 * Display raw admin-run payloads with a compact status badge.
 *
 * This keeps manual run diagnostics visible without leaving the dashboard.
 */
export function RunResultCard({ title, result, scrollHeight }: RunResultCardProps) {
  const body = <Code block>{JSON.stringify(result, null, 2)}</Code>;

  return (
    <Card withBorder radius="md" mt="md" p="md">
      <Group justify="space-between" mb="xs">
        <Text fw={600}>{title}</Text>
        <Badge variant="light" color={result.skipped ? "yellow" : "green"}>
          {result.skipped ? "Skipped" : "Inserted"}
        </Badge>
      </Group>
      {typeof scrollHeight === "number" ? (
        <ScrollArea h={scrollHeight} type="auto">
          {body}
        </ScrollArea>
      ) : (
        body
      )}
    </Card>
  );
}
