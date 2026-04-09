/**
 * Auth Audit page: lists login/logout/session_expiry events from auth_audit_log.
 * Admin-only; non-admins are redirected to the dashboard.
 */

import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { fetchAuthAudit, type AuthAuditEvent, type AuthAuditResponse } from "@/api";
import { DataTable } from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";

const PAGE_SIZE = 100;

function formatAuditTime(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" });
}

function formatUserAgentShort(ua: string | null): string {
  if (!ua?.trim()) return "—";
  const s = ua.trim();
  if (s.includes("Edg/")) return s.match(/Edg\/([\d.]+)/)?.[0] ?? "Edge";
  if (s.includes("Chrome/") && !s.includes("Chromium")) return s.match(/Chrome\/([\d.]+)/)?.[0] ?? "Chrome";
  if (s.includes("Firefox/")) return s.match(/Firefox\/([\d.]+)/)?.[0] ?? "Firefox";
  if (s.includes("Safari/") && !s.includes("Chrome")) return s.match(/Version\/([\d.]+)/)?.[0] ?? "Safari";
  return s.length > 50 ? `${s.slice(0, 47)}…` : s;
}

export function AuthAuditPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [data, setData] = useState<AuthAuditResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [detail, setDetail] = useState<AuthAuditEvent | null>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchAuthAudit(PAGE_SIZE, page * PAGE_SIZE, null, null, signal);
      if (!signal?.aborted) setData(res);
    } catch (e) {
      if (signal?.aborted) return;
      setError(e instanceof Error ? e.message : "Failed to load audit log");
      if (e instanceof Error && (e.message.includes("403") || e.message.includes("Forbidden"))) {
        navigate("/", { replace: true });
      }
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [page, navigate]);

  useEffect(() => {
    if (!user) return;
    if (!user.is_admin) { navigate("/", { replace: true }); return; }
    const ac = new AbortController();
    load(ac.signal);
    return () => ac.abort();
  }, [user, load, navigate]);

  if (!user?.is_admin) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Auth Audit Log</h2>
          <p className="text-sm text-muted-foreground">Login, logout, and session expiry events. Admin only.</p>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/">← Dashboard</Link>
        </Button>
      </div>

      {error && (
        <div className="rounded-md bg-loss-bg border border-loss/30 p-3 text-sm text-loss">{error}</div>
      )}

      {loading && <p className="text-sm text-muted">Loading...</p>}

      {!loading && data && (
        <>
          <p className="text-xs text-muted-foreground">
            Total: {data.total} · Showing {data.events.length} (offset {data.offset})
          </p>

          <DataTable
            columns={[
              { key: "occurred_at", header: "Time", render: (ev: AuthAuditEvent) => formatAuditTime(ev.occurred_at) },
              {
                key: "event_type",
                header: "Event",
                render: (ev: AuthAuditEvent) => (
                  <Badge variant={ev.event_type === "login" ? "profit" : ev.event_type === "logout" ? "muted" : "warning"}>
                    {ev.event_type}
                  </Badge>
                ),
              },
              { key: "username", header: "User", render: (ev: AuthAuditEvent) => ev.username ?? (ev.user_id != null ? `id:${ev.user_id}` : "—") },
              { key: "ip_address", header: "IP", render: (ev: AuthAuditEvent) => ev.ip_address ?? "—" },
              { key: "country", header: "Country", render: (ev: AuthAuditEvent) => ev.country ?? "—" },
              { key: "user_agent", header: "Browser", render: (ev: AuthAuditEvent) => (
                <span className="text-xs" title={ev.user_agent ?? undefined}>{formatUserAgentShort(ev.user_agent)}</span>
              )},
            ]}
            data={data.events}
            keyFn={(ev) => ev.id}
            onRowClick={setDetail}
          />

          <div className="flex gap-2">
            <Button variant="ghost" size="sm" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>
              Previous
            </Button>
            <Button variant="ghost" size="sm" disabled={data.offset + data.events.length >= data.total} onClick={() => setPage((p) => p + 1)}>
              Next
            </Button>
          </div>
        </>
      )}

      {/* Detail drawer */}
      {detail && (
        <div className="fixed inset-y-0 right-0 z-50 w-[480px] border-l border-border bg-card shadow-2xl overflow-y-auto">
          <div className="flex items-center justify-between p-4 border-b border-border">
            <h3 className="text-sm font-medium text-foreground">Event Details</h3>
            <Button variant="ghost" size="icon" onClick={() => setDetail(null)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="p-4 space-y-3 text-sm">
            <p className="text-muted-foreground">
              {formatAuditTime(detail.occurred_at)} · {detail.event_type} · {detail.username ?? "—"}
            </p>
            {detail.geo_json && Object.keys(detail.geo_json).length > 0 && (
              <div>
                <span className="text-xs text-muted font-medium">Geo / IP Lookup</span>
                <pre className="mt-1 rounded-md bg-background p-3 text-xs text-foreground-secondary overflow-x-auto max-h-64">
                  {JSON.stringify(detail.geo_json, null, 2)}
                </pre>
              </div>
            )}
            {detail.details && Object.keys(detail.details).length > 0 && (
              <div>
                <span className="text-xs text-muted font-medium">Other Details</span>
                <pre className="mt-1 rounded-md bg-background p-3 text-xs text-foreground-secondary overflow-x-auto max-h-48">
                  {JSON.stringify(detail.details, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
