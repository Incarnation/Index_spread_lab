import React from "react";
import { Alert, Badge, Box, Container, Text, Title } from "@mantine/core";
import {
  AdminControlsCard,
  ChainSnapshotsPanel,
  GexPanel,
  LabelMetricsPanel,
  ModelOpsPanel,
  RunResultCard,
  StrategyQualityPanel,
  TradeDecisionDetailsDrawer,
  TradeDecisionsPanel,
  TradesLivePnlPanel,
} from "./components";
import { useAdminRuns, useDecisionDeletion, useGexData, useSnapshotsDecisions } from "./hooks";

/**
 * Render the main IndexSpreadLab dashboard page and wire cross-panel interactions.
 *
 * This container composes all major cards/panels and coordinates shared
 * concerns such as global error display, drawer state, and manual admin runs.
 */
export function DashboardApp() {
  const [error, setError] = React.useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = React.useState<boolean>(false);
  const [drawerDecision, setDrawerDecision] = React.useState<{
    decision_id: number;
  } | null>(null);

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
    refresh,
  } = useSnapshotsDecisions({ onError: handleError });

  const {
    gexSnapshots,
    selectedGexSnapshot,
    setSelectedGexSnapshot,
    gexDtes,
    gexExpirations,
    selectedDte,
    selectedCustomExpirations,
    setSelectedCustomExpirations,
    handleSelectedDteChange,
    gexCurve,
    gexLoading,
  } = useGexData({ onError: handleError });

  const {
    adminKey,
    setAdminKey,
    runResult,
    runQuotesResult,
    runDecisionResult,
    runTradePnlResult,
    runSnapshot,
    runQuotes,
    runDecision,
    runTradePnl,
  } = useAdminRuns({
    onRefresh: refresh,
    onError: handleError,
    onClearError: clearError,
  });

  const { deletingDecisionId, successMessage, deleteDecision } = useDecisionDeletion({
    adminKey,
    activeDrawerDecisionId: drawerDecision?.decision_id ?? null,
    onDeleteActiveDrawerDecision: () => {
      setDrawerOpen(false);
      setDrawerDecision(null);
    },
    onRefresh: refresh,
    onError: handleError,
    onClearError: clearError,
  });

  /** Refresh all dashboard datasets while clearing stale error text. */
  const handleRefresh = React.useCallback(() => {
    clearError();
    refresh();
  }, [clearError, refresh]);

  return (
    <Box bg="gray.0" mih="100vh" py="xl">
      <Container size="lg">
        <TradeDecisionDetailsDrawer
          opened={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          decision={decisions.find((d) => d.decision_id === drawerDecision?.decision_id) ?? null}
        />

        <Box>
          <Title order={2}>IndexSpreadLab</Title>
          <Text c="dimmed" mt={4}>
            React dashboard (MVP)
          </Text>
          <Badge variant="light" mt="xs">
            Snapshots
          </Badge>
        </Box>

        <AdminControlsCard
          adminKey={adminKey}
          onAdminKeyChange={setAdminKey}
          onRefresh={handleRefresh}
          onRunSnapshot={() => {
            void runSnapshot();
          }}
          onRunQuotes={() => {
            void runQuotes();
          }}
          onRunDecision={() => {
            void runDecision();
          }}
          onRunTradePnl={() => {
            void runTradePnl();
          }}
        />

        {error && (
          <Alert mt="md" color="red" title="Error">
            <Text>
              {error} <Text span c="dimmed">(Is the backend running on port 8000?)</Text>
            </Text>
          </Alert>
        )}

        {successMessage && (
          <Alert mt="md" color="green" title="Success">
            <Text>{successMessage}</Text>
          </Alert>
        )}

        {runResult && <RunResultCard title="Snapshot run result" result={runResult} scrollHeight={220} />}
        {runQuotesResult && <RunResultCard title="Quote run result" result={runQuotesResult} />}
        {runDecisionResult && <RunResultCard title="Decision run result" result={runDecisionResult} />}
        {runTradePnlResult && <RunResultCard title="Trade PnL run result" result={runTradePnlResult} />}

        <GexPanel
          snapshots={gexSnapshots}
          selectedSnapshot={selectedGexSnapshot}
          onSelectedSnapshotChange={setSelectedGexSnapshot}
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
          deletingDecisionId={deletingDecisionId}
          onViewDecision={(decision) => {
            setDrawerDecision({ decision_id: decision.decision_id });
            setDrawerOpen(true);
          }}
          onDeleteDecision={(decisionId) => {
            void deleteDecision(decisionId);
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
