import { Badge, Card, Group, Stepper, Text } from "@mantine/core";
import type { PipelineRunState } from "../hooks/useAdminRuns";
import { formatTs } from "../utils/format";

type PipelineRunStepperCardProps = {
  pipelineRun: PipelineRunState | null;
};

/**
 * Map pipeline step statuses to the stepper's active index.
 */
function computeActiveStepIndex(pipelineRun: PipelineRunState): number {
  const runningIndex = pipelineRun.steps.findIndex((step) => step.status === "running");
  if (runningIndex >= 0) return runningIndex;
  const firstPending = pipelineRun.steps.findIndex((step) => step.status === "pending");
  if (firstPending >= 0) return firstPending;
  return pipelineRun.steps.length;
}

/**
 * Render a per-step pipeline execution summary with progress state.
 */
export function PipelineRunStepperCard({ pipelineRun }: PipelineRunStepperCardProps) {
  if (pipelineRun == null) return null;

  const activeStep = computeActiveStepIndex(pipelineRun);

  return (
    <Card withBorder radius="md" mt="md" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Full pipeline run</Text>
        <Badge variant="light" color={pipelineRun.running ? "blue" : pipelineRun.summary?.includes("successfully") ? "green" : "red"}>
          {pipelineRun.running ? "Running" : "Completed"}
        </Badge>
      </Group>

      <Stepper active={activeStep} size="sm" orientation="horizontal" allowNextStepsSelect={false}>
        {pipelineRun.steps.map((step) => (
          <Stepper.Step
            key={step.id}
            label={step.label}
            description={
              step.status === "running"
                ? "Running…"
                : step.status === "pending"
                  ? "Pending"
                  : step.detail ?? (step.status === "success" ? "Completed" : "Failed")
            }
            color={step.status === "error" ? "red" : step.status === "success" ? "green" : "blue"}
          />
        ))}
      </Stepper>

      <Group mt="sm" gap="md">
        <Text size="sm" c="dimmed">
          Started: {formatTs(pipelineRun.startedAt)}
        </Text>
        <Text size="sm" c="dimmed">
          Completed: {pipelineRun.completedAt ? formatTs(pipelineRun.completedAt) : "—"}
        </Text>
      </Group>

      <Text mt="xs" size="sm">
        {pipelineRun.summary ?? "Running pipeline steps..."}
      </Text>
    </Card>
  );
}

