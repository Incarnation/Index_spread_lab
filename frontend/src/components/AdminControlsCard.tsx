import type { ChangeEvent } from "react";
import { Button, Card, Group, TextInput } from "@mantine/core";

type AdminControlsCardProps = {
  adminKey: string;
  onAdminKeyChange: (value: string) => void;
  onRefresh: () => void;
  onRunSnapshot: () => void;
  onRunQuotes: () => void;
  onRunDecision: () => void;
  onRunTradePnl: () => void;
};

/**
 * Render manual admin controls for jobs and optional API key input.
 *
 * Buttons map directly to backend admin endpoints so operators can trigger
 * jobs on demand while troubleshooting or validating pipeline state.
 */
export function AdminControlsCard({
  adminKey,
  onAdminKeyChange,
  onRefresh,
  onRunSnapshot,
  onRunQuotes,
  onRunDecision,
  onRunTradePnl,
}: AdminControlsCardProps) {
  return (
    <Card withBorder radius="md" mt="lg" p="md">
      <Group justify="space-between" align="flex-end" wrap="wrap" gap="md">
        <Group>
          <Button onClick={onRefresh} variant="light">
            Refresh
          </Button>
          <Button onClick={onRunSnapshot}>Run snapshot now</Button>
          <Button onClick={onRunQuotes} variant="outline">
            Run quotes now
          </Button>
          <Button onClick={onRunDecision} variant="outline">
            Run decision now
          </Button>
          <Button onClick={onRunTradePnl} variant="outline">
            Run trade PnL now
          </Button>
        </Group>
        <TextInput
          label="Admin key (optional)"
          value={adminKey}
          onChange={(e: ChangeEvent<HTMLInputElement>) => onAdminKeyChange(e.currentTarget.value)}
          placeholder="X-API-Key"
          w={320}
        />
      </Group>
    </Card>
  );
}
