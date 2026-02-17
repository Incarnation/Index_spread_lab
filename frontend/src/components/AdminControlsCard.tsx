import type { ChangeEvent } from "react";
import { Badge, Button, Card, Group, Stack, Text, TextInput } from "@mantine/core";
import type { PipelineRunState } from "../hooks/useAdminRuns";

type AdminControlsCardProps = {
  environmentLabel: string;
  isProduction: boolean;
  adminKey: string;
  onAdminKeyChange: (value: string) => void;
  onRefresh: () => void;
  onRunFullPipeline: () => void;
  onRunSnapshot: () => void;
  onRunQuotes: () => void;
  onRunDecision: () => void;
  onRunTradePnl: () => void;
  actionLoading: {
    snapshot: boolean;
    quotes: boolean;
    decision: boolean;
    tradePnl: boolean;
    pipeline: boolean;
  };
  isAnyActionRunning: boolean;
  pipelineRun: PipelineRunState | null;
};

/**
 * Render manual admin controls for jobs and optional API key input.
 *
 * Buttons map directly to backend admin endpoints so operators can trigger
 * jobs on demand while troubleshooting or validating pipeline state.
 */
export function AdminControlsCard({
  environmentLabel,
  isProduction,
  adminKey,
  onAdminKeyChange,
  onRefresh,
  onRunFullPipeline,
  onRunSnapshot,
  onRunQuotes,
  onRunDecision,
  onRunTradePnl,
  actionLoading,
  isAnyActionRunning,
  pipelineRun,
}: AdminControlsCardProps) {
  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" wrap="wrap" gap="xs" mb="sm">
        <Group gap="xs">
          <Text fw={600}>Manual controls</Text>
          <Badge variant="light" color={isProduction ? "orange" : "blue"}>
            {environmentLabel}
          </Badge>
          <Badge variant="outline" color={adminKey.trim() ? "green" : "gray"}>
            Admin key {adminKey.trim() ? "set" : "not set"}
          </Badge>
        </Group>
        {pipelineRun?.running && (
          <Text size="sm" c="dimmed">
            Full pipeline is running… single-step actions are locked.
          </Text>
        )}
      </Group>

      <Stack gap="md">
        <Group gap="sm" wrap="wrap">
          <Button onClick={onRefresh} variant="light" disabled={isAnyActionRunning}>
            Refresh
          </Button>
          <Button
            onClick={onRunFullPipeline}
            color="teal"
            loading={actionLoading.pipeline}
            disabled={isAnyActionRunning && !actionLoading.pipeline}
          >
            Run full pipeline
          </Button>
          <Button
            onClick={onRunSnapshot}
            loading={actionLoading.snapshot}
            disabled={isAnyActionRunning && !actionLoading.snapshot}
          >
            Run snapshot now
          </Button>
          <Button
            onClick={onRunQuotes}
            variant="outline"
            loading={actionLoading.quotes}
            disabled={isAnyActionRunning && !actionLoading.quotes}
          >
            Run quotes now
          </Button>
          <Button
            onClick={onRunDecision}
            variant="outline"
            loading={actionLoading.decision}
            disabled={isAnyActionRunning && !actionLoading.decision}
          >
            Run decision now
          </Button>
          <Button
            onClick={onRunTradePnl}
            variant="outline"
            loading={actionLoading.tradePnl}
            disabled={isAnyActionRunning && !actionLoading.tradePnl}
          >
            Run trade PnL now
          </Button>
        </Group>

        <TextInput
          label="Admin key (optional)"
          description="Used for protected admin endpoints in production."
          value={adminKey}
          onChange={(e: ChangeEvent<HTMLInputElement>) => onAdminKeyChange(e.currentTarget.value)}
          placeholder="X-API-Key"
          w={360}
        />

        {isProduction && (
          <Text size="sm" c="dimmed">
            Production guardrail: high-impact actions require confirmation before execution.
          </Text>
        )}
      </Stack>
    </Card>
  );
}
