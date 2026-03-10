import React from "react";
import {
  runFeatureBuilderNow,
  runGexNow,
  runLabelerNow,
  runPromotionGatesNow,
  runDecisionNow,
  runQuotesNow,
  runShadowInferenceNow,
  runSnapshotNow,
  runTrainerNow,
  runTradePnlNow,
  type GenericAdminRunResult,
  type RunDecisionResult,
  type RunQuotesResult,
  type RunSnapshotResult,
  type RunTradePnlResult,
} from "../api";

type UseAdminRunsArgs = {
  adminKey: string;
  onRefresh: () => void;
  onError: (message: string) => void;
  onClearError: () => void;
};

type ActionLoadingState = {
  snapshot: boolean;
  quotes: boolean;
  decision: boolean;
  tradePnl: boolean;
  pipeline: boolean;
};

type ActionLoadingKey = keyof ActionLoadingState;

type RunAdminActionOptions<T extends GenericAdminRunResult> = {
  actionKey: ActionLoadingKey;
  actionLabel: string;
  request: (apiKey?: string) => Promise<T>;
  refreshAfter?: boolean;
  onSuccess?: (result: T) => void;
  quiet?: boolean;
};

type PipelineStepStatus = "pending" | "running" | "success" | "error";

export type PipelineStepState = {
  id: string;
  label: string;
  status: PipelineStepStatus;
  detail: string | null;
  durationMs: number | null;
};

export type PipelineRunState = {
  running: boolean;
  startedAt: string;
  completedAt: string | null;
  steps: PipelineStepState[];
  summary: string | null;
};

export type AdminActionAuditEntry = {
  id: number;
  operator: string;
  action: string;
  status: "success" | "error";
  startedAt: string;
  completedAt: string;
  durationMs: number;
  detail: string;
};

export type AdminActionToast = {
  id: number;
  title: string;
  message: string;
  color: "green" | "yellow" | "red" | "blue";
};

type UseAdminRunsResult = {
  runResult: RunSnapshotResult | null;
  runQuotesResult: RunQuotesResult | null;
  runDecisionResult: RunDecisionResult | null;
  runTradePnlResult: RunTradePnlResult | null;
  actionLoading: ActionLoadingState;
  isAnyActionRunning: boolean;
  pipelineRun: PipelineRunState | null;
  actionAudit: AdminActionAuditEntry[];
  toasts: AdminActionToast[];
  dismissToast: (toastId: number) => void;
  runSnapshot: () => Promise<void>;
  runQuotes: () => Promise<void>;
  runDecision: () => Promise<void>;
  runTradePnl: () => Promise<void>;
  runFullPipeline: () => Promise<void>;
};

/** Build concise step detail text from a heterogeneous admin-run payload. */
function getRunDetail(result: GenericAdminRunResult): string {
  // Prefer explicit run counters when available (e.g. multi-trade decisions).
  const resultRecord = result as Record<string, unknown>;
  const tradesCreatedCount = resultRecord["trades_created_count"];
  if (typeof tradesCreatedCount === "number") {
    const selectionMeta = resultRecord["selection_meta"];
    const clippedBy =
      selectionMeta && typeof selectionMeta === "object" && selectionMeta !== null
        ? (selectionMeta as Record<string, unknown>)["clipped_by"]
        : null;
    if (typeof clippedBy === "string" && clippedBy.trim().length > 0) {
      return `created_trades=${tradesCreatedCount} clipped_by=${clippedBy}`;
    }
    return `created_trades=${tradesCreatedCount}`;
  }
  if (typeof result.reason === "string" && result.reason.trim()) return result.reason;
  if (typeof result.status === "string" && result.status.trim()) return result.status;
  if (typeof result.error === "string" && result.error.trim()) return result.error;
  if (typeof result.ok === "boolean") return result.ok ? "ok" : "failed";
  return "completed";
}

/**
 * Manage admin-triggered job actions and their result payloads.
 *
 * The hook keeps endpoint responses separate per action so the dashboard can
 * show the most recent run output for snapshot/quotes/decision/trade-pnl jobs.
 * It also tracks pipeline step progress, action audit history, and toast-style
 * feedback so operators can quickly verify outcomes.
 */
export function useAdminRuns({ adminKey, onRefresh, onError, onClearError }: UseAdminRunsArgs): UseAdminRunsResult {
  const [runResult, setRunResult] = React.useState<RunSnapshotResult | null>(null);
  const [runQuotesResult, setRunQuotesResult] = React.useState<RunQuotesResult | null>(null);
  const [runDecisionResult, setRunDecisionResult] = React.useState<RunDecisionResult | null>(null);
  const [runTradePnlResult, setRunTradePnlResult] = React.useState<RunTradePnlResult | null>(null);
  const [actionLoading, setActionLoading] = React.useState<ActionLoadingState>({
    snapshot: false,
    quotes: false,
    decision: false,
    tradePnl: false,
    pipeline: false,
  });
  const [pipelineRun, setPipelineRun] = React.useState<PipelineRunState | null>(null);
  const [actionAudit, setActionAudit] = React.useState<AdminActionAuditEntry[]>([]);
  const [toasts, setToasts] = React.useState<AdminActionToast[]>([]);
  const nextAuditIdRef = React.useRef<number>(1);
  const nextToastIdRef = React.useRef<number>(1);
  const toastTimersRef = React.useRef<Set<ReturnType<typeof setTimeout>>>(new Set());

  // Clear all pending toast timers on unmount
  React.useEffect(() => {
    const timers = toastTimersRef.current;
    return () => { timers.forEach(clearTimeout); timers.clear(); };
  }, []);

  const normalizedAdminKey = React.useMemo(() => {
    const value = adminKey.trim();
    return value.length > 0 ? value : undefined;
  }, [adminKey]);

  /**
   * Remove one toast message after it is dismissed by timeout or user action.
   */
  const dismissToast = React.useCallback((toastId: number) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== toastId));
  }, []);

  /**
   * Add a temporary toast notification to the action feedback stack.
   */
  const pushToast = React.useCallback(
    (title: string, message: string, color: AdminActionToast["color"]) => {
      const id = nextToastIdRef.current;
      nextToastIdRef.current += 1;
      setToasts((prev) => [...prev, { id, title, message, color }].slice(-5));
      const timer = window.setTimeout(() => {
        toastTimersRef.current.delete(timer);
        dismissToast(id);
      }, 7000);
      toastTimersRef.current.add(timer);
    },
    [dismissToast],
  );

  /**
   * Record one admin action into the audit feed with timing and outcome data.
   */
  const pushAuditEntry = React.useCallback((entry: Omit<AdminActionAuditEntry, "id">) => {
    const id = nextAuditIdRef.current;
    nextAuditIdRef.current += 1;
    setActionAudit((prev) => [{ id, ...entry }, ...prev].slice(0, 40));
  }, []);

  /**
   * Execute one admin API action with consistent loading, error, and audit flow.
   */
  const runAdminAction = React.useCallback(
    async <T extends GenericAdminRunResult>({
      actionKey,
      actionLabel,
      request,
      refreshAfter = false,
      onSuccess,
      quiet = false,
    }: RunAdminActionOptions<T>): Promise<T | null> => {
      const startedAt = new Date().toISOString();
      const startedAtMs = Date.now();
      setActionLoading((prev) => ({ ...prev, [actionKey]: true }));
      try {
        const result = await request(normalizedAdminKey);
        onSuccess?.(result);
        if (refreshAfter) onRefresh();
        const durationMs = Date.now() - startedAtMs;
        const detail = getRunDetail(result);
        pushAuditEntry({
          operator: "dashboard_user",
          action: actionLabel,
          status: "success",
          startedAt,
          completedAt: new Date().toISOString(),
          durationMs,
          detail,
        });
        if (!quiet) {
          const skipped = result.skipped === true;
          pushToast(actionLabel, skipped ? `Skipped: ${detail}` : `Completed: ${detail}`, skipped ? "yellow" : "green");
        }
        return result;
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        const durationMs = Date.now() - startedAtMs;
        onError(message);
        pushAuditEntry({
          operator: "dashboard_user",
          action: actionLabel,
          status: "error",
          startedAt,
          completedAt: new Date().toISOString(),
          durationMs,
          detail: message,
        });
        if (!quiet) pushToast(actionLabel, message, "red");
        return null;
      } finally {
        setActionLoading((prev) => ({ ...prev, [actionKey]: false }));
      }
    },
    [normalizedAdminKey, onError, onRefresh, pushAuditEntry, pushToast],
  );

  /**
   * Run the snapshot job now, then refresh read models that depend on it.
   */
  const runSnapshot = React.useCallback(async () => {
    onClearError();
    setRunResult(null);
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    setRunTradePnlResult(null);
    await runAdminAction({
      actionKey: "snapshot",
      actionLabel: "Run snapshot now",
      request: runSnapshotNow,
      refreshAfter: true,
      onSuccess: (result) => {
        setRunResult(result as RunSnapshotResult);
      },
    });
  }, [onClearError, runAdminAction]);

  /**
   * Run quote ingestion now. This updates marks/underlying context data.
   */
  const runQuotes = React.useCallback(async () => {
    onClearError();
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    setRunTradePnlResult(null);
    await runAdminAction({
      actionKey: "quotes",
      actionLabel: "Run quotes now",
      request: runQuotesNow,
      refreshAfter: true,
      onSuccess: (result) => {
        setRunQuotesResult(result as RunQuotesResult);
      },
    });
  }, [onClearError, runAdminAction]);

  /**
   * Run the decision engine now, then refresh panels that show decisions/trades.
   */
  const runDecision = React.useCallback(async () => {
    onClearError();
    setRunDecisionResult(null);
    setRunTradePnlResult(null);
    await runAdminAction({
      actionKey: "decision",
      actionLabel: "Run decision now",
      request: runDecisionNow,
      refreshAfter: true,
      onSuccess: (result) => {
        setRunDecisionResult(result as RunDecisionResult);
      },
    });
  }, [onClearError, runAdminAction]);

  /**
   * Run mark-to-market PnL updates now, then refresh the trades panel.
   */
  const runTradePnl = React.useCallback(async () => {
    onClearError();
    setRunTradePnlResult(null);
    await runAdminAction({
      actionKey: "tradePnl",
      actionLabel: "Run trade PnL now",
      request: runTradePnlNow,
      refreshAfter: true,
      onSuccess: (result) => {
        setRunTradePnlResult(result as RunTradePnlResult);
      },
    });
  }, [onClearError, runAdminAction]);

  /**
   * Run the full manual pipeline sequence with per-step progress state.
   */
  const runFullPipeline = React.useCallback(async () => {
    onClearError();
    const startedAt = new Date().toISOString();
    const startedAtMs = Date.now();
    setActionLoading((prev) => ({ ...prev, pipeline: true }));

    const initialSteps: PipelineStepState[] = [
      { id: "quotes", label: "Quotes", status: "pending", detail: null, durationMs: null },
      { id: "snapshot", label: "Snapshot", status: "pending", detail: null, durationMs: null },
      { id: "gex", label: "GEX", status: "pending", detail: null, durationMs: null },
      { id: "feature", label: "Feature builder", status: "pending", detail: null, durationMs: null },
      { id: "labeler", label: "Labeler", status: "pending", detail: null, durationMs: null },
      { id: "decision", label: "Decision", status: "pending", detail: null, durationMs: null },
      { id: "tradePnl", label: "Trade PnL", status: "pending", detail: null, durationMs: null },
    ];
    setPipelineRun({
      running: true,
      startedAt,
      completedAt: null,
      steps: initialSteps,
      summary: null,
    });

    const requests: Array<{ id: string; label: string; request: (apiKey?: string) => Promise<GenericAdminRunResult> }> = [
      { id: "quotes", label: "Quotes", request: runQuotesNow },
      { id: "snapshot", label: "Snapshot", request: runSnapshotNow },
      { id: "gex", label: "GEX", request: runGexNow },
      { id: "feature", label: "Feature builder", request: runFeatureBuilderNow },
      { id: "labeler", label: "Labeler", request: runLabelerNow },
      { id: "decision", label: "Decision", request: runDecisionNow },
      { id: "tradePnl", label: "Trade PnL", request: runTradePnlNow },
    ];

    let failed = false;
    const summaryParts: string[] = [];

    for (const step of requests) {
      const stepStartMs = Date.now();
      setPipelineRun((prev) =>
        prev == null
          ? prev
          : {
              ...prev,
              steps: prev.steps.map((row) =>
                row.id === step.id ? { ...row, status: "running", detail: null, durationMs: null } : row,
              ),
            },
      );

      try {
        const result = await step.request(normalizedAdminKey);
        const durationMs = Date.now() - stepStartMs;
        const detail = getRunDetail(result);
        summaryParts.push(`${step.label}: ${detail}`);

        setPipelineRun((prev) =>
          prev == null
            ? prev
            : {
                ...prev,
                steps: prev.steps.map((row) =>
                  row.id === step.id ? { ...row, status: "success", detail, durationMs } : row,
                ),
              },
        );

        if (step.id === "snapshot") setRunResult(result as RunSnapshotResult);
        if (step.id === "quotes") setRunQuotesResult(result as RunQuotesResult);
        if (step.id === "decision") setRunDecisionResult(result as RunDecisionResult);
        if (step.id === "tradePnl") setRunTradePnlResult(result as RunTradePnlResult);
      } catch (error: unknown) {
        failed = true;
        const durationMs = Date.now() - stepStartMs;
        const message = error instanceof Error ? error.message : String(error);
        onError(message);
        setPipelineRun((prev) =>
          prev == null
            ? prev
            : {
                ...prev,
                steps: prev.steps.map((row) =>
                  row.id === step.id ? { ...row, status: "error", detail: message, durationMs } : row,
                ),
              },
        );
        break;
      }
    }

    const completedAt = new Date().toISOString();
    const durationMs = Date.now() - startedAtMs;
    const summary = failed ? "Pipeline stopped due to step failure." : "Pipeline completed successfully.";

    if (!failed) onRefresh();
    setPipelineRun((prev) =>
      prev == null
        ? prev
        : {
            ...prev,
            running: false,
            completedAt,
            summary,
          },
    );

    pushAuditEntry({
      operator: "dashboard_user",
      action: "Run full pipeline",
      status: failed ? "error" : "success",
      startedAt,
      completedAt,
      durationMs,
      detail: failed ? "Stopped on failed step." : summaryParts.join(" | "),
    });
    pushToast("Run full pipeline", summary, failed ? "red" : "green");
    setActionLoading((prev) => ({ ...prev, pipeline: false }));
  }, [normalizedAdminKey, onClearError, onError, onRefresh, pushAuditEntry, pushToast]);

  return {
    runResult,
    runQuotesResult,
    runDecisionResult,
    runTradePnlResult,
    actionLoading,
    isAnyActionRunning: Object.values(actionLoading).some(Boolean),
    pipelineRun,
    actionAudit,
    toasts,
    dismissToast,
    runSnapshot,
    runQuotes,
    runDecision,
    runTradePnl,
    runFullPipeline,
  };
}
