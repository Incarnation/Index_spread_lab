import React from "react";
import {
  fetchChainSnapshots,
  fetchGexCurve,
  fetchGexDtes,
  fetchGexSnapshots,
  fetchTradeDecisions,
  runDecisionNow,
  runQuotesNow,
  runSnapshotNow,
  type ChainSnapshot,
  type GexCurvePoint,
  type GexSnapshot,
  type RunDecisionResult,
  type RunQuotesResult,
  type RunSnapshotResult,
  type TradeDecision,
} from "./api";
import {
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Code,
  Container,
  Drawer,
  Group,
  Loader,
  ScrollArea,
  Select,
  Table,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// Shorten strings for compact table display.
function truncate(s: string, n: number) {
  return s.length <= n ? s : `${s.slice(0, n)}…`;
}

// Parse JSON payloads that may be stored as strings or objects.
function parseJsonRecord(value: Record<string, unknown> | string | null): Record<string, unknown> | null {
  if (!value) return null;
  if (typeof value === "string") {
    try {
      return JSON.parse(value) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  return value;
}

export function App() {
  const [items, setItems] = React.useState<ChainSnapshot[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState<boolean>(true);
  const [adminKey, setAdminKey] = React.useState<string>("");
  const [runResult, setRunResult] = React.useState<RunSnapshotResult | null>(null);
  const [runQuotesResult, setRunQuotesResult] = React.useState<RunQuotesResult | null>(null);
  const [runDecisionResult, setRunDecisionResult] = React.useState<RunDecisionResult | null>(null);
  const [gexSnapshots, setGexSnapshots] = React.useState<GexSnapshot[]>([]);
  const [selectedGexSnapshot, setSelectedGexSnapshot] = React.useState<GexSnapshot | null>(null);
  const [gexDtes, setGexDtes] = React.useState<number[]>([]);
  const [selectedDte, setSelectedDte] = React.useState<string>("all");
  const [gexCurve, setGexCurve] = React.useState<GexCurvePoint[]>([]);
  const [gexLoading, setGexLoading] = React.useState<boolean>(false);
  const [decisions, setDecisions] = React.useState<TradeDecision[]>([]);
  const [decisionsLoading, setDecisionsLoading] = React.useState<boolean>(true);
  const [drawerOpen, setDrawerOpen] = React.useState<boolean>(false);
  const [drawerDecision, setDrawerDecision] = React.useState<TradeDecision | null>(null);

  const gexSnapshotOptions = React.useMemo(
    () =>
      gexSnapshots.map((s) => ({
        value: String(s.snapshot_id),
        label: `#${s.snapshot_id} · ${s.ts}`,
      })),
    [gexSnapshots],
  );
  const gexDteOptions = React.useMemo(
    () => [{ value: "all", label: "All" }, ...gexDtes.map((d) => ({ value: String(d), label: `${d}` }))],
    [gexDtes],
  );

  const refresh = React.useCallback(() => {
    setError(null);
    setLoading(true);
    setDecisionsLoading(true);
    Promise.all([fetchChainSnapshots(50), fetchTradeDecisions(50)])
      .then(([snapshotRows, decisionRows]) => {
        setItems(snapshotRows);
        setDecisions(decisionRows);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setLoading(false);
        setDecisionsLoading(false);
      });
  }, []);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  // Extract compact summary fields from chosen_legs_json.
  const decisionSummary = React.useCallback((row: TradeDecision) => {
    const data = parseJsonRecord(row.chosen_legs_json as Record<string, unknown> | string | null);
    if (!data) return null;
    const short = data["short"] as Record<string, unknown> | undefined;
    const long = data["long"] as Record<string, unknown> | undefined;
    const credit = data["credit"] as number | undefined;
    const context = data["context"] as Record<string, unknown> | undefined;
    if (!short || !long) return null;
    return {
      shortStrike: short["strike"] as number | undefined,
      longStrike: long["strike"] as number | undefined,
      credit: typeof credit === "number" ? credit : undefined,
      gexNet: typeof context?.["gex_net"] === "number" ? (context["gex_net"] as number) : undefined,
      zeroGamma:
        typeof context?.["zero_gamma_level"] === "number" ? (context["zero_gamma_level"] as number) : undefined,
      vix: typeof context?.["vix"] === "number" ? (context["vix"] as number) : undefined,
    };
  }, []);

  const drawerPayload = React.useMemo(
    () => (drawerDecision ? parseJsonRecord(drawerDecision.chosen_legs_json as Record<string, unknown> | string | null) : null),
    [drawerDecision],
  );
  const drawerStrategyParams = React.useMemo(
    () =>
      drawerDecision
        ? parseJsonRecord(drawerDecision.strategy_params_json as Record<string, unknown> | string | null)
        : null,
    [drawerDecision],
  );

  React.useEffect(() => {
    let cancelled = false;
    setGexLoading(true);
    fetchGexSnapshots(20)
      .then((rows) => {
        if (cancelled) return;
        setGexSnapshots(rows);
        if (rows.length > 0) {
          setSelectedGexSnapshot(rows[0]);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setGexLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    if (!selectedGexSnapshot) return;
    let cancelled = false;
    fetchGexDtes(selectedGexSnapshot.snapshot_id)
      .then((rows) => {
        if (cancelled) return;
        setGexDtes(rows);
        if (selectedDte !== "all" && !rows.includes(Number(selectedDte))) {
          setSelectedDte("all");
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedGexSnapshot, selectedDte]);

  React.useEffect(() => {
    if (!selectedGexSnapshot) return;
    let cancelled = false;
    setGexLoading(true);
    const dteVal = selectedDte === "all" ? undefined : Number(selectedDte);
    fetchGexCurve(selectedGexSnapshot.snapshot_id, dteVal)
      .then((points) => {
        if (!cancelled) setGexCurve(points);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setGexLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedGexSnapshot, selectedDte]);

  const runSnapshot = React.useCallback(async () => {
    setError(null);
    setRunResult(null);
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    try {
      const result = await runSnapshotNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunResult(result);
      refresh();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, refresh]);

  const runQuotes = React.useCallback(async () => {
    setError(null);
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    try {
      const result = await runQuotesNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunQuotesResult(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey]);

  const runDecision = React.useCallback(async () => {
    setError(null);
    setRunDecisionResult(null);
    try {
      const result = await runDecisionNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunDecisionResult(result);
      refresh();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, refresh]);

  return (
    <Box bg="gray.0" mih="100vh" py="xl">
      <Container size="lg">
        <Drawer
          opened={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          position="right"
          size="lg"
          title={drawerDecision ? `Decision #${drawerDecision.decision_id}` : "Decision details"}
        >
          {!drawerDecision && <Text c="dimmed">No decision selected.</Text>}
          {drawerDecision && (
            <Box>
              <Text size="sm" c="dimmed">
                Time: {drawerDecision.ts}
              </Text>
              <Text size="sm" c="dimmed">
                Decision: {drawerDecision.decision} · DTE {drawerDecision.target_dte} · Δ {drawerDecision.delta_target}
              </Text>
              <Text size="sm" c="dimmed">
                Source: {drawerDecision.decision_source} · Ruleset {drawerDecision.ruleset_version}
              </Text>

              <Text fw={600} mt="md" mb="xs">
                Chosen legs JSON
              </Text>
              <ScrollArea h={240} type="auto">
                <Code block>{drawerPayload ? JSON.stringify(drawerPayload, null, 2) : "—"}</Code>
              </ScrollArea>

              <Text fw={600} mt="md" mb="xs">
                Strategy params JSON
              </Text>
              <ScrollArea h={160} type="auto">
                <Code block>{drawerStrategyParams ? JSON.stringify(drawerStrategyParams, null, 2) : "—"}</Code>
              </ScrollArea>
            </Box>
          )}
        </Drawer>

        <Group justify="space-between" align="flex-end">
          <div>
            <Title order={2}>SPX Tools</Title>
            <Text c="dimmed" mt={4}>
              React dashboard (MVP)
            </Text>
          </div>
          <Badge variant="light">Snapshots</Badge>
        </Group>

        <Card withBorder radius="md" mt="lg" p="md">
          <Group justify="space-between" align="flex-end" wrap="wrap" gap="md">
            <Group>
              <Button onClick={refresh} variant="light">
                Refresh
              </Button>
              <Button onClick={runSnapshot}>Run snapshot now</Button>
              <Button onClick={runQuotes} variant="outline">
                Run quotes now
              </Button>
              <Button onClick={runDecision} variant="outline">
                Run decision now
              </Button>
            </Group>
            <TextInput
              label="Admin key (optional)"
              value={adminKey}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAdminKey(e.currentTarget.value)}
              placeholder="X-API-Key"
              w={320}
            />
          </Group>
        </Card>

        {error && (
          <Alert mt="md" color="red" title="Error">
            <Text>
              {error} <Text span c="dimmed">(Is the backend running on port 8000?)</Text>
            </Text>
          </Alert>
        )}

        {runResult && (
          <Card withBorder radius="md" mt="md" p="md">
            <Group justify="space-between" mb="xs">
              <Text fw={600}>Snapshot run result</Text>
              <Badge variant="light" color={runResult.skipped ? "yellow" : "green"}>
                {runResult.skipped ? "Skipped" : "Inserted"}
              </Badge>
            </Group>
            <ScrollArea h={220} type="auto">
              <Code block>{JSON.stringify(runResult, null, 2)}</Code>
            </ScrollArea>
          </Card>
        )}

        {runQuotesResult && (
          <Card withBorder radius="md" mt="md" p="md">
            <Group justify="space-between" mb="xs">
              <Text fw={600}>Quote run result</Text>
              <Badge variant="light" color={runQuotesResult.skipped ? "yellow" : "green"}>
                {runQuotesResult.skipped ? "Skipped" : "Inserted"}
              </Badge>
            </Group>
            <Code block>{JSON.stringify(runQuotesResult, null, 2)}</Code>
          </Card>
        )}

        {runDecisionResult && (
          <Card withBorder radius="md" mt="md" p="md">
            <Group justify="space-between" mb="xs">
              <Text fw={600}>Decision run result</Text>
              <Badge variant="light" color={runDecisionResult.skipped ? "yellow" : "green"}>
                {runDecisionResult.skipped ? "Skipped" : "Inserted"}
              </Badge>
            </Group>
            <Code block>{JSON.stringify(runDecisionResult, null, 2)}</Code>
          </Card>
        )}

        <Card withBorder radius="md" mt="lg" p="md">
          <Group justify="space-between" align="center" mb="sm" wrap="wrap">
            <Text fw={600}>Gamma exposure (GEX)</Text>
            <Group gap="sm">
              <Select
                label="Snapshot"
                data={gexSnapshotOptions}
                value={selectedGexSnapshot ? String(selectedGexSnapshot.snapshot_id) : null}
                onChange={(value) => {
                  const id = value ? Number(value) : null;
                  const snap = id ? gexSnapshots.find((s) => s.snapshot_id === id) || null : null;
                  setSelectedGexSnapshot(snap);
                }}
                w={260}
                placeholder="Select snapshot"
              />
              <Select
                label="DTE"
                data={gexDteOptions}
                value={selectedDte}
                onChange={(value) => setSelectedDte(value || "all")}
                w={120}
              />
            </Group>
          </Group>

          {selectedGexSnapshot && (
            <Group gap="lg" mb="sm">
              <Text size="sm" c="dimmed">
                Spot: {selectedGexSnapshot.spot_price ?? "—"}
              </Text>
              <Text size="sm" c="dimmed">
                GEX Net: {selectedGexSnapshot.gex_net ?? "—"}
              </Text>
              <Text size="sm" c="dimmed">
                Zero Gamma: {selectedGexSnapshot.zero_gamma_level ?? "—"}
              </Text>
            </Group>
          )}

          {gexLoading && (
            <Group gap="xs">
              <Loader size="sm" />
              <Text c="dimmed" size="sm">
                Loading GEX…
              </Text>
            </Group>
          )}

          {!gexLoading && gexCurve.length === 0 && (
            <Text c="dimmed">No GEX data yet. Run snapshots and GEX first.</Text>
          )}

          {!gexLoading && gexCurve.length > 0 && (
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <LineChart data={gexCurve}>
                  <XAxis dataKey="strike" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="gex_net" stroke="#228be6" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card withBorder radius="md" mt="lg" p="md">
          <Group justify="space-between" align="center" mb="sm">
            <Text fw={600}>Trade decisions</Text>
            {decisionsLoading ? (
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
                  <Table.Th>Time (UTC)</Table.Th>
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
                  <Table.Th>Details</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {decisions.map((d) => {
                  const summary = decisionSummary(d);
                  return (
                    <Table.Tr key={d.decision_id}>
                      <Table.Td>{d.decision_id}</Table.Td>
                      <Table.Td>{d.ts}</Table.Td>
                      <Table.Td>
                        <Badge variant="light" color={d.decision === "TRADE" ? "green" : "gray"}>
                          {d.decision}
                        </Badge>
                      </Table.Td>
                      <Table.Td>{d.target_dte}</Table.Td>
                      <Table.Td>{d.delta_target}</Table.Td>
                      <Table.Td>
                        {summary ? `${summary.shortStrike} / ${summary.longStrike}` : "—"}
                      </Table.Td>
                      <Table.Td>{summary?.credit?.toFixed(2) ?? "—"}</Table.Td>
                      <Table.Td>{typeof d.score === "number" ? d.score.toFixed(3) : "—"}</Table.Td>
                      <Table.Td>{summary?.gexNet?.toFixed(0) ?? "—"}</Table.Td>
                      <Table.Td>{summary?.zeroGamma?.toFixed(2) ?? "—"}</Table.Td>
                      <Table.Td>{summary?.vix?.toFixed(2) ?? "—"}</Table.Td>
                      <Table.Td>{d.reason ?? "—"}</Table.Td>
                      <Table.Td>
                        <Button
                          size="xs"
                          variant="light"
                          onClick={() => {
                            setDrawerDecision(d);
                            setDrawerOpen(true);
                          }}
                        >
                          View
                        </Button>
                      </Table.Td>
                    </Table.Tr>
                  );
                })}
                {decisions.length === 0 && !decisionsLoading && !error && (
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
                  <Table.Th>Time (UTC)</Table.Th>
                  <Table.Th>Underlying</Table.Th>
                  <Table.Th>DTE</Table.Th>
                  <Table.Th>Expiration</Table.Th>
                  <Table.Th>Checksum</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {items.map((x) => (
                  <Table.Tr key={x.snapshot_id}>
                    <Table.Td>{x.snapshot_id}</Table.Td>
                    <Table.Td>{x.ts}</Table.Td>
                    <Table.Td>
                      <Badge variant="light">{x.underlying}</Badge>
                    </Table.Td>
                    <Table.Td>{x.target_dte}</Table.Td>
                    <Table.Td>{x.expiration}</Table.Td>
                    <Table.Td>
                      <Code>{truncate(x.checksum, 12)}</Code>
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
      </Container>
    </Box>
  );
}

