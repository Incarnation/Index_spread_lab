import React from "react";
import { Link } from "react-router-dom";
import { Alert, Badge, Box, Button, Container, Group, Stack, Text, TextInput, Title } from "@mantine/core";
import { API_BASE } from "./api";
import {
  ChainSnapshotsPanel,
  GexPanel,
  LabelMetricsPanel,
  ModelOpsPanel,
  PipelineStatusBar,
  StrategyQualityPanel,
  TradeDecisionDetailsDrawer,
  TradeDecisionsPanel,
  TradesLivePnlPanel,
} from "./components";
import { useAuth } from "./contexts/AuthContext";
import { useGexData, useSnapshotsDecisions } from "./hooks";

type EnvironmentInfo = {
  label: string;
  isProduction: boolean;
  apiTargetLabel: string;
};

/**
 * Derive a user-facing environment label from API base URL and build mode.
 */
function deriveEnvironmentInfo(apiBase: string): EnvironmentInfo {
  const normalized = apiBase.trim().toLowerCase();
  if (!normalized) {
    return {
      label: import.meta.env.MODE === "production" ? "Production" : "Local",
      isProduction: import.meta.env.MODE === "production",
      apiTargetLabel: "same-origin",
    };
  }
  if (normalized.includes("localhost") || normalized.includes("127.0.0.1")) {
    return {
      label: "Local",
      isProduction: false,
      apiTargetLabel: apiBase,
    };
  }
  if (normalized.includes("railway.app")) {
    return {
      label: "Production",
      isProduction: true,
      apiTargetLabel: apiBase,
    };
  }
  if (import.meta.env.MODE === "production") {
    return {
      label: "Production",
      isProduction: true,
      apiTargetLabel: apiBase || "same-origin",
    };
  }
  return {
    label: "Preview",
    isProduction: false,
    apiTargetLabel: apiBase || "same-origin",
  };
}

/**
 * Read frontend admin key from Vite env and normalize optional wrapping quotes.
 */
function readFrontendAdminKey(): string {
  const raw = (import.meta.env.VITE_ADMIN_API_KEY as string | undefined)?.trim() ?? "";
  if (raw.length >= 2 && ((raw.startsWith('"') && raw.endsWith('"')) || (raw.startsWith("'") && raw.endsWith("'")))) {
    return raw.slice(1, -1).trim();
  }
  return raw;
}

/**
 * Mask secret values for read-only display in the UI.
 */
function maskSecret(value: string): string {
  if (!value) return "Not configured";
  if (value.length <= 6) return `${"*".repeat(Math.max(0, value.length - 2))}${value.slice(-2)}`;
  return `${value.slice(0, 2)}${"*".repeat(value.length - 4)}${value.slice(-2)}`;
}

/**
 * Render the main IndexSpreadLab dashboard page and wire cross-panel interactions.
 *
 * This container composes all major cards/panels and coordinates shared
 * concerns such as global error display and decision drawer state.
 */
export function DashboardApp() {
  const { user, logout } = useAuth();
  const [error, setError] = React.useState<string | null>(null);
  const adminKey = React.useMemo(() => readFrontendAdminKey(), []);
  const [drawerOpen, setDrawerOpen] = React.useState<boolean>(false);
  const [drawerDecision, setDrawerDecision] = React.useState<{
    decision_id: number;
  } | null>(null);
  const environmentInfo = React.useMemo(() => deriveEnvironmentInfo(API_BASE), []);

  /** Clear the top-level error banner after successful actions. */
  const clearError = React.useCallback(() => {
    setError(null);
  }, []);

  /** Capture an error message from hooks/actions and show it in the UI. */
  const handleError = React.useCallback((message: string) => {
    setError(message);
  }, []);

  const {
    items,
    decisions,
    trades,
    labelMetrics,
    modelOps,
    strategyMetrics,
    loading,
    decisionsLoading,
    tradesLoading,
    labelMetricsLoading,
    modelOpsLoading,
    strategyMetricsLoading,
    preflight,
    preflightLoading,
    preflightAuthRequired,
    refresh,
  } = useSnapshotsDecisions({ adminKey, onError: handleError });

  const {
    gexSnapshots,
    selectedGexSnapshot,
    setSelectedGexSnapshot,
    selectedUnderlying,
    handleSelectedUnderlyingChange,
    gexDtes,
    gexExpirations,
    selectedDte,
    selectedCustomExpirations,
    setSelectedCustomExpirations,
    handleSelectedDteChange,
    gexCurve,
    gexLoading,
  } = useGexData({ onError: handleError });

  return (
    <Box bg="gray.0" mih="100vh" py="xl">
      <Container size="lg">
        <TradeDecisionDetailsDrawer
          opened={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          decision={decisions.find((d) => d.decision_id === drawerDecision?.decision_id) ?? null}
        />

        <Group justify="space-between" align="flex-start">
          <Box>
            <Title order={2}>IndexSpreadLab</Title>
            <Text c="dimmed" mt={4}>
              React dashboard (MVP)
            </Text>
            <Stack mt="xs" gap={6}>
              <Badge variant="light">Snapshots</Badge>
              <Text size="sm" c="dimmed">
                API target: {environmentInfo.apiTargetLabel}
              </Text>
            </Stack>
          </Box>
          <Group gap="xs">
            {user?.is_admin && (
              <Button variant="subtle" size="xs" component={Link} to="/admin/auth-audit">
                Auth Audit
              </Button>
            )}
            {user && (
              <Badge variant="outline" size="sm">
                {user.username}
              </Badge>
            )}
            <Button variant="subtle" size="xs" onClick={logout}>
              Log out
            </Button>
          </Group>
        </Group>

        <PipelineStatusBar preflight={preflight} loading={preflightLoading} authRequired={preflightAuthRequired} />

        {error && (
          <Alert mt="md" color="red" title="Error">
            <Text>
              {error} <Text span c="dimmed">(Is the backend running on port 8000?)</Text>
            </Text>
          </Alert>
        )}

        <GexPanel
          snapshots={gexSnapshots}
          selectedSnapshot={selectedGexSnapshot}
          onSelectedSnapshotChange={setSelectedGexSnapshot}
          selectedUnderlying={selectedUnderlying}
          onSelectedUnderlyingChange={handleSelectedUnderlyingChange}
          dtes={gexDtes}
          expirations={gexExpirations}
          selectedDte={selectedDte}
          onSelectedDteChange={handleSelectedDteChange}
          selectedCustomExpirations={selectedCustomExpirations}
          onSelectedCustomExpirationsChange={setSelectedCustomExpirations}
          loading={gexLoading}
          curve={gexCurve}
        />

        <TradeDecisionsPanel
          decisions={decisions}
          loading={decisionsLoading}
          error={error}
          onViewDecision={(decision) => {
            setDrawerDecision({ decision_id: decision.decision_id });
            setDrawerOpen(true);
          }}
        />

        <LabelMetricsPanel metrics={labelMetrics} loading={labelMetricsLoading} error={error} />
        <ModelOpsPanel ops={modelOps} loading={modelOpsLoading} error={error} />
        <StrategyQualityPanel metrics={strategyMetrics} loading={strategyMetricsLoading} error={error} />

        <TradesLivePnlPanel trades={trades} loading={tradesLoading} error={error} />

        <ChainSnapshotsPanel items={items} loading={loading} error={error} />
      </Container>
    </Box>
  );
}
