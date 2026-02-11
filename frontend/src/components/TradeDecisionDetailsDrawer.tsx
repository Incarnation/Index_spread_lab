import { Box, Code, Drawer, ScrollArea, Text } from "@mantine/core";
import type { TradeDecision } from "../api";
import { formatTs, parseJsonRecord } from "../utils/format";

type TradeDecisionDetailsDrawerProps = {
  opened: boolean;
  onClose: () => void;
  decision: TradeDecision | null;
};

export function TradeDecisionDetailsDrawer({ opened, onClose, decision }: TradeDecisionDetailsDrawerProps) {
  const payload = decision ? parseJsonRecord(decision.chosen_legs_json as Record<string, unknown> | string | null) : null;
  const strategyParams = decision ? parseJsonRecord(decision.strategy_params_json as Record<string, unknown> | string | null) : null;

  return (
    <Drawer opened={opened} onClose={onClose} position="right" size="lg" title={decision ? `Decision #${decision.decision_id}` : "Decision details"}>
      {!decision && <Text c="dimmed">No decision selected.</Text>}
      {decision && (
        <Box>
          <Text size="sm" c="dimmed">
            Time: {formatTs(decision.ts)}
          </Text>
          <Text size="sm" c="dimmed">
            Decision: {decision.decision} · DTE {decision.target_dte} · Δ {decision.delta_target}
          </Text>
          <Text size="sm" c="dimmed">
            Source: {decision.decision_source} · Ruleset {decision.ruleset_version}
          </Text>

          <Text fw={600} mt="md" mb="xs">
            Chosen legs JSON
          </Text>
          <ScrollArea h={240} type="auto">
            <Code block>{payload ? JSON.stringify(payload, null, 2) : "—"}</Code>
          </ScrollArea>

          <Text fw={600} mt="md" mb="xs">
            Strategy params JSON
          </Text>
          <ScrollArea h={160} type="auto">
            <Code block>{strategyParams ? JSON.stringify(strategyParams, null, 2) : "—"}</Code>
          </ScrollArea>
        </Box>
      )}
    </Drawer>
  );
}
