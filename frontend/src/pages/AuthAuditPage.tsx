/**
 * Auth Audit page: lists login/logout/session_expiry events from auth_audit_log.
 * Admin-only; non-admins are redirected to the dashboard.
 */

import React, { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Alert, Box, Button, Container, Group, Loader, Table, Text, Title } from "@mantine/core";
import { useAuth } from "../contexts/AuthContext";
import { fetchAuthAudit, type AuthAuditEvent, type AuthAuditResponse } from "../api";

const PAGE_SIZE = 100;

/** Format ISO timestamp in user's local timezone for display. */
function formatAuditTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  });
}

/**
 * Derive a short browser/device label from User-Agent (Chrome, Safari, Firefox, etc.).
 * Most browsers send a UA starting with "Mozilla/5.0" for legacy reasons; we parse the real engine.
 */
function formatUserAgentShort(ua: string | null): string {
  if (!ua || !ua.trim()) return "—";
  const s = ua.trim();
  // Order matters: Chrome includes "Safari", Edge includes "Chrome"
  if (s.includes("Edg/")) {
    const m = s.match(/Edg\/([\d.]+)/);
    return m ? `Edge ${m[1]}` : "Edge";
  }
  if (s.includes("Chrome/") && !s.includes("Chromium")) {
    const m = s.match(/Chrome\/([\d.]+)/);
    return m ? `Chrome ${m[1]}` : "Chrome";
  }
  if (s.includes("Firefox/")) {
    const m = s.match(/Firefox\/([\d.]+)/);
    return m ? `Firefox ${m[1]}` : "Firefox";
  }
  if (s.includes("Safari/") && !s.includes("Chrome")) {
    const m = s.match(/Version\/([\d.]+).*Safari/);
    return m ? `Safari ${m[1]}` : "Safari";
  }
  return s.length > 50 ? `${s.slice(0, 47)}…` : s;
}

export function AuthAuditPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [data, setData] = useState<AuthAuditResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchAuthAudit(PAGE_SIZE, page * PAGE_SIZE, null, null);
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load audit log");
      if (e instanceof Error && (e.message.includes("403") || e.message.includes("Forbidden"))) {
        navigate("/", { replace: true });
      }
    } finally {
      setLoading(false);
    }
  }, [page, navigate]);

  useEffect(() => {
    if (!user) return;
    if (!user.is_admin) {
      navigate("/", { replace: true });
      return;
    }
    load();
  }, [user, load, navigate]);

  if (!user) return null;
  if (!user.is_admin) return null;

  return (
    <Box bg="gray.0" mih="100vh" py="xl">
      <Container size="xl">
        <Box mb="lg">
          <Button variant="subtle" size="xs" component={Link} to="/">
            ← Dashboard
          </Button>
        </Box>
        <Title order={2} mb="xs">
          Auth Audit Log
        </Title>
        <Text c="dimmed" size="sm" mb="md">
          Login, logout, and session expiry events. Admin only.
        </Text>

        {error && (
          <Alert color="red" title="Error" mb="md">
            {error}
          </Alert>
        )}

        {loading && (
          <Loader size="sm" />
        )}

        {!loading && data && (
          <>
            <Text size="sm" c="dimmed" mb="sm">
              Total: {data.total} · Showing {data.events.length} (offset {data.offset})
            </Text>
            <Table striped highlightOnHover withTableBorder>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Time</Table.Th>
                  <Table.Th>Event</Table.Th>
                  <Table.Th>User</Table.Th>
                  <Table.Th>IP</Table.Th>
                  <Table.Th>Country</Table.Th>
                  <Table.Th>Browser</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {data.events.map((ev: AuthAuditEvent) => (
                  <Table.Tr key={ev.id}>
                    <Table.Td>{formatAuditTime(ev.occurred_at)}</Table.Td>
                    <Table.Td>{ev.event_type}</Table.Td>
                    <Table.Td>{ev.username ?? (ev.user_id != null ? `id:${ev.user_id}` : "—")}</Table.Td>
                    <Table.Td>{ev.ip_address ?? "—"}</Table.Td>
                    <Table.Td>{ev.country ?? "—"}</Table.Td>
                    <Table.Td>
                      <Text size="xs" title={ev.user_agent ?? undefined}>
                        {formatUserAgentShort(ev.user_agent)}
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
            <Group mt="md" gap="xs">
              <Button
                variant="light"
                size="xs"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                Previous
              </Button>
              <Button
                variant="light"
                size="xs"
                disabled={data.offset + data.events.length >= data.total}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </Group>
          </>
        )}
      </Container>
    </Box>
  );
}
