import React from "react";
import { Badge, Button, Card, Group, Loader, ScrollArea, Select, Table, Text, UnstyledButton } from "@mantine/core";
import type { TradeDecision } from "../api";
import { getDecisionSummary } from "../utils/decision";
import { formatTs } from "../utils/format";

type TradeDecisionsPanelProps = {
  decisions: TradeDecision[];
  loading: boolean;
  error: string | null;
  onViewDecision: (decision: TradeDecision) => void;
  deletingDecisionId?: number | null;
  onDeleteDecision?: (decisionId: number) => void;
};

type TimeWindowFilter = "all" | "24h" | "7d" | "30d";
type DecisionSortField = "ts" | "target_dte" | "score";
type SortDirection = "asc" | "desc";

/**
 * Parse a decision timestamp and evaluate if it falls inside a selected window.
 */
function withinTimeWindow(ts: string, window: TimeWindowFilter): boolean {
  if (window === "all") return true;
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return true;
  const nowMs = Date.now();
  const ageMs = nowMs - date.getTime();
  const maxAgeMs =
    window === "24h" ? 24 * 60 * 60 * 1000 : window === "7d" ? 7 * 24 * 60 * 60 * 1000 : 30 * 24 * 60 * 60 * 1000;
  return ageMs <= maxAgeMs;
}

/**
 * Render a sortable table header label with direction indicator.
 */
function sortHeaderLabel(field: DecisionSortField, activeField: DecisionSortField, direction: SortDirection, title: string): string {
  if (field !== activeField) return title;
  return `${title} ${direction === "asc" ? "↑" : "↓"}`;
}

/**
 * Render latest decision-engine outputs and row-level actions.
 *
 * Operators can inspect selected legs/context and open the detail drawer.
 * Delete actions are optional and are rendered only when provided by caller.
 */
export function TradeDecisionsPanel({
  decisions,
  loading,
  error,
  deletingDecisionId,
  onViewDecision,
  onDeleteDecision,
}: TradeDecisionsPanelProps) {
  const [timeWindow, setTimeWindow] = React.useState<TimeWindowFilter>("all");
  const [dteFilter, setDteFilter] = React.useState<string>("all");
  const [sideFilter, setSideFilter] = React.useState<string>("all");
  const [sortField, setSortField] = React.useState<DecisionSortField>("ts");
  const [sortDirection, setSortDirection] = React.useState<SortDirection>("desc");

  /**
   * Toggle one sortable column while preserving previous direction behavior.
   */
  const handleSortChange = React.useCallback((nextField: DecisionSortField) => {
    setSortField((previousField) => {
      if (previousField === nextField) {
        setSortDirection((previousDirection) => (previousDirection === "asc" ? "desc" : "asc"));
        return previousField;
      }
      setSortDirection("desc");
      return nextField;
    });
  }, []);

  const dteOptions = React.useMemo(
    () => [
      { value: "all", label: "All DTE" },
      ...Array.from(new Set(decisions.map((decision) => decision.target_dte)))
        .sort((a, b) => a - b)
        .map((value) => ({ value: String(value), label: `DTE ${value}` })),
    ],
    [decisions],
  );

  const filteredDecisions = React.useMemo(() => {
    return decisions.filter((decision) => {
      if (!withinTimeWindow(decision.ts, timeWindow)) return false;
      if (dteFilter !== "all" && decision.target_dte !== Number(dteFilter)) return false;
      if (sideFilter !== "all") {
        const summary = getDecisionSummary(decision);
        if (summary?.spreadSide !== sideFilter) return false;
      }
      return true;
    });
  }, [decisions, dteFilter, sideFilter, timeWindow]);

  const sortedDecisions = React.useMemo(() => {
    const rows = [...filteredDecisions];
    rows.sort((a, b) => {
      if (sortField === "target_dte") {
        const delta = a.target_dte - b.target_dte;
        return sortDirection === "asc" ? delta : -delta;
      }
      if (sortField === "score") {
        const left = typeof a.score === "number" ? a.score : Number.NEGATIVE_INFINITY;
        const right = typeof b.score === "number" ? b.score : Number.NEGATIVE_INFINITY;
        const delta = left - right;
        return sortDirection === "asc" ? delta : -delta;
      }
      const leftTs = new Date(a.ts).getTime();
      const rightTs = new Date(b.ts).getTime();
      const delta = leftTs - rightTs;
      return sortDirection === "asc" ? delta : -delta;
    });
    return rows;
  }, [filteredDecisions, sortDirection, sortField]);

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
            {sortedDecisions.length} / {decisions.length} rows
          </Text>
        )}
      </Group>

      <Group mb="sm" gap="sm" wrap="wrap">
        <Select
          label="Time window"
          data={[
            { value: "all", label: "All time" },
            { value: "24h", label: "Last 24h" },
            { value: "7d", label: "Last 7 days" },
            { value: "30d", label: "Last 30 days" },
          ]}
          value={timeWindow}
          onChange={(value) => setTimeWindow((value as TimeWindowFilter) || "all")}
          w={170}
        />
        <Select label="DTE" data={dteOptions} value={dteFilter} onChange={(value) => setDteFilter(value || "all")} w={140} />
        <Select
          label="Side"
          data={[
            { value: "all", label: "All sides" },
            { value: "put", label: "Put spreads" },
            { value: "call", label: "Call spreads" },
          ]}
          value={sideFilter}
          onChange={(value) => setSideFilter(value || "all")}
          w={150}
        />
      </Group>

      <ScrollArea type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders stickyHeader stickyHeaderOffset={0}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>ID</Table.Th>
              <Table.Th>
                <UnstyledButton onClick={() => handleSortChange("ts")}>
                  {sortHeaderLabel("ts", sortField, sortDirection, "Time")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th>Decision</Table.Th>
              <Table.Th>
                <UnstyledButton onClick={() => handleSortChange("target_dte")}>
                  {sortHeaderLabel("target_dte", sortField, sortDirection, "DTE")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th>Delta</Table.Th>
              <Table.Th>Side</Table.Th>
              <Table.Th>Legs</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Credit</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>
                <UnstyledButton onClick={() => handleSortChange("score")}>
                  {sortHeaderLabel("score", sortField, sortDirection, "Score")}
                </UnstyledButton>
              </Table.Th>
              <Table.Th style={{ textAlign: "right" }}>GEX</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>Zero Gamma</Table.Th>
              <Table.Th style={{ textAlign: "right" }}>VIX</Table.Th>
              <Table.Th>Reason</Table.Th>
              <Table.Th>Actions</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sortedDecisions.map((decision) => {
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
                  <Table.Td>{summary?.spreadSide ? summary.spreadSide.toUpperCase() : "—"}</Table.Td>
                  <Table.Td>{summary ? `${summary.shortStrike} / ${summary.longStrike}` : "—"}</Table.Td>
                  <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    {summary?.credit?.toFixed(2) ?? "—"}
                  </Table.Td>
                  <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    {typeof decision.score === "number" ? decision.score.toFixed(3) : "—"}
                  </Table.Td>
                  <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>{summary?.gexNet?.toFixed(0) ?? "—"}</Table.Td>
                  <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    {summary?.zeroGamma?.toFixed(2) ?? "—"}
                  </Table.Td>
                  <Table.Td style={{ textAlign: "right", whiteSpace: "nowrap" }}>{summary?.vix?.toFixed(2) ?? "—"}</Table.Td>
                  <Table.Td>{decision.reason ?? "—"}</Table.Td>
                  <Table.Td>
                    <Group gap="xs">
                      <Button size="xs" variant="light" onClick={() => onViewDecision(decision)}>
                        View
                      </Button>
                      {onDeleteDecision && (
                        <Button
                          size="xs"
                          color="red"
                          variant="outline"
                          loading={deletingDecisionId === decision.decision_id}
                          onClick={() => onDeleteDecision(decision.decision_id)}
                        >
                          Delete
                        </Button>
                      )}
                    </Group>
                  </Table.Td>
                </Table.Tr>
              );
            })}
            {sortedDecisions.length === 0 && !loading && !error && (
              <Table.Tr>
                <Table.Td colSpan={14}>
                  <Text c="dimmed">
                    No decisions for this filter. Next step: run full pipeline (recommended) or run decision now after snapshot and
                    feature-builder.
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
