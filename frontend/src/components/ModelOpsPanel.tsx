import { Badge, Card, Group, Loader, Table, Text } from "@mantine/core";
import type { ModelOpsResponse } from "../api";

type ModelOpsPanelProps = {
  ops: ModelOpsResponse | null;
  loading: boolean;
  error: string | null;
};

/** Convert ISO timestamps to local display text while handling nulls safely. */
function fmtTs(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

/** Convert ratio values (0-1) to percentage strings for metric rows. */
function fmtPct(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

/** Convert PnL-like metrics to signed currency strings. */
function fmtMoney(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}$${value.toFixed(2)}`;
}

/** Format the latest training attempt with an explicit skip or failure reason when present. */
function fmtTrainingAttempt(run: ModelOpsResponse["latest_training_run"]): string {
  if (!run) return "—";
  const detail = run.skip_reason ?? run.notes;
  return `#${run.training_run_id} (${run.status}${detail ? `: ${detail}` : ""})`;
}

/**
 * Render training/promotion/prediction health indicators for model operations.
 *
 * The card is intended for quick operational checks: whether models are being
 * trained, whether gates pass, and whether prediction traffic is active.
 */
export function ModelOpsPanel({ ops, loading, error }: ModelOpsPanelProps) {
  const gatePassed = ops?.latest_training_run?.gate?.passed;
  const gateColor = gatePassed === true ? "green" : gatePassed === false ? "red" : "gray";
  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Model Ops</Text>
        {loading ? (
          <Group gap="xs">
            <Loader size="sm" />
            <Text c="dimmed" size="sm">
              Loading…
            </Text>
          </Group>
        ) : (
          <Text c="dimmed" size="sm">
            Model: {ops?.model_name ?? "—"}
          </Text>
        )}
      </Group>

      <Group gap="xs" mb="md">
        <Badge variant="light" color="blue">
          Models {ops?.counts.model_versions ?? 0}
        </Badge>
        <Badge variant="light" color="violet">
          Training runs {ops?.counts.training_runs ?? 0}
        </Badge>
        <Badge variant="light" color="indigo">
          Predictions 24h {ops?.counts.model_predictions_24h ?? 0}
        </Badge>
        <Badge variant="light" color={gateColor}>
          Gate {gatePassed === true ? "PASS" : gatePassed === false ? "FAIL" : "N/A"}
        </Badge>
        <Badge variant="light" color={(ops?.active_model_version?.is_active ?? false) ? "green" : "gray"}>
          Active model {(ops?.active_model_version?.is_active ?? false) ? "YES" : "NO"}
        </Badge>
      </Group>

      <Table striped highlightOnHover withTableBorder withColumnBorders stickyHeader stickyHeaderOffset={0}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Item</Table.Th>
            <Table.Th>Value</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          <Table.Tr>
            <Table.Td>Latest model version</Table.Td>
            <Table.Td>{ops?.latest_model_version?.version ?? "—"}</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Latest rollout status</Table.Td>
            <Table.Td>{ops?.latest_model_version?.rollout_status ?? "—"}</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Latest training attempt</Table.Td>
            <Table.Td>{fmtTrainingAttempt(ops?.latest_training_run ?? null)}</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Latest prediction timestamp</Table.Td>
            <Table.Td>{fmtTs(ops?.latest_prediction_ts)}</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Validation TP50 rate</Table.Td>
            <Table.Td>{fmtPct(ops?.latest_model_version?.metrics?.tp50_rate_test)}</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Validation expectancy</Table.Td>
            <Table.Td>{fmtMoney(ops?.latest_model_version?.metrics?.expectancy_test)}</Table.Td>
          </Table.Tr>
          <Table.Tr>
            <Table.Td>Validation drawdown</Table.Td>
            <Table.Td>{fmtMoney(ops?.latest_model_version?.metrics?.max_drawdown_test)}</Table.Td>
          </Table.Tr>
        </Table.Tbody>
      </Table>

      {(ops?.warnings?.length ?? 0) > 0 && !loading && !error && (
        <Text mt="sm" c="dimmed" size="sm">
          Warnings: {ops?.warnings.join(", ")}. If models are missing, continue running feature-builder + labeler until enough
          resolved rows are available for trainer.
        </Text>
      )}
    </Card>
  );
}

