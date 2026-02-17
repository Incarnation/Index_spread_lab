import { Alert, Badge, Card, Group, Loader, Paper, SimpleGrid, Stack, Text } from "@mantine/core";
import type { AdminPreflightResponse } from "../api";
import { formatTs } from "../utils/format";

type PipelineStatusBarProps = {
  preflight: AdminPreflightResponse | null;
  loading: boolean;
  authRequired: boolean;
};

type StageIndicator = {
  label: string;
  statusLabel: string;
  color: "green" | "yellow" | "red" | "gray";
  ts: string | null;
  ageMinutes: number | null;
};

/**
 * Compute minutes elapsed from a timestamp to now.
 */
function ageMinutesFromIso(value: string | null): number | null {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  const age = Math.floor((Date.now() - parsed.getTime()) / 60_000);
  return age >= 0 ? age : 0;
}

/**
 * Convert freshness age and warning state into a compact indicator chip.
 */
function buildStageIndicator(params: {
  label: string;
  ts: string | null;
  thresholdMinutes: number;
  hasMissingWarning: boolean;
}): StageIndicator {
  const { label, ts, thresholdMinutes, hasMissingWarning } = params;
  const ageMinutes = ageMinutesFromIso(ts);
  if (hasMissingWarning || ageMinutes == null) {
    return {
      label,
      statusLabel: "Missing",
      color: "red",
      ts,
      ageMinutes,
    };
  }
  if (ageMinutes <= thresholdMinutes) {
    return {
      label,
      statusLabel: "Fresh",
      color: "green",
      ts,
      ageMinutes,
    };
  }
  if (ageMinutes <= thresholdMinutes * 3) {
    return {
      label,
      statusLabel: "Stale",
      color: "yellow",
      ts,
      ageMinutes,
    };
  }
  return {
    label,
    statusLabel: "Outdated",
    color: "red",
    ts,
    ageMinutes,
  };
}

/**
 * Render top-level pipeline freshness badges and warning chips.
 */
export function PipelineStatusBar({ preflight, loading, authRequired }: PipelineStatusBarProps) {
  const indicators: StageIndicator[] = preflight
    ? [
        buildStageIndicator({
          label: "Quotes",
          ts: preflight.latest.quote_ts,
          thresholdMinutes: 10,
          hasMissingWarning: preflight.warnings.includes("no_underlying_quotes"),
        }),
        buildStageIndicator({
          label: "Snapshot",
          ts: preflight.latest.snapshot_ts,
          thresholdMinutes: 20,
          hasMissingWarning: preflight.warnings.includes("no_chain_snapshots"),
        }),
        buildStageIndicator({
          label: "GEX",
          ts: preflight.latest.gex_ts,
          thresholdMinutes: 20,
          hasMissingWarning: preflight.warnings.includes("no_gex_snapshots"),
        }),
        buildStageIndicator({
          label: "Decision",
          ts: preflight.latest.decision_ts,
          thresholdMinutes: 120,
          hasMissingWarning: preflight.warnings.includes("no_trade_decisions"),
        }),
        buildStageIndicator({
          label: "Trade PnL",
          ts: preflight.latest.trade_mark_ts,
          thresholdMinutes: 30,
          hasMissingWarning: preflight.warnings.includes("no_trades"),
        }),
      ]
    : [];

  return (
    <Card withBorder radius="md" mt="md" p="md">
      <Group justify="space-between" align="center" mb="sm">
        <Text fw={600}>Pipeline status</Text>
        {loading ? (
          <Group gap="xs">
            <Loader size="sm" />
            <Text size="sm" c="dimmed">
              Refreshing freshness checks…
            </Text>
          </Group>
        ) : (
          <Text size="sm" c="dimmed">
            Staleness thresholds: quotes 10m, snapshot/GEX 20m, trade PnL 30m, decision 120m
          </Text>
        )}
      </Group>

      {authRequired && (
        <Alert color="yellow" title="Admin key needed for full diagnostics" mb="sm">
          Enter an admin key to load pipeline preflight data and warning counts in production.
        </Alert>
      )}

      {!authRequired && !loading && preflight == null && (
        <Alert color="gray" title="Preflight unavailable" mb="sm">
          Unable to read admin preflight data. Public cards still update, but freshness diagnostics are unavailable.
        </Alert>
      )}

      {!authRequired && preflight != null && (
        <Stack gap="sm">
          <SimpleGrid cols={{ base: 1, sm: 2, lg: 5 }}>
            {indicators.map((indicator) => (
              <Paper key={indicator.label} withBorder radius="md" p="sm">
                <Group justify="space-between" mb={6}>
                  <Text fw={600} size="sm">
                    {indicator.label}
                  </Text>
                  <Badge variant="light" color={indicator.color}>
                    {indicator.statusLabel}
                  </Badge>
                </Group>
                <Text size="xs" c="dimmed">
                  Last run: {indicator.ts ? formatTs(indicator.ts) : "—"}
                </Text>
                <Text size="xs" c="dimmed">
                  Age: {typeof indicator.ageMinutes === "number" ? `${indicator.ageMinutes}m ago` : "unknown"}
                </Text>
              </Paper>
            ))}
          </SimpleGrid>

          <Group gap="xs">
            <Text size="sm" fw={600}>
              Active warnings:
            </Text>
            {preflight.warnings.length > 0 ? (
              preflight.warnings.map((warning) => (
                <Badge key={warning} variant="light" color="yellow">
                  {warning}
                </Badge>
              ))
            ) : (
              <Badge variant="light" color="green">
                none
              </Badge>
            )}
          </Group>
        </Stack>
      )}
    </Card>
  );
}

