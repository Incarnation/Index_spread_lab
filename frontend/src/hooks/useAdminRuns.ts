import React from "react";
import {
  runDecisionNow,
  runQuotesNow,
  runSnapshotNow,
  runTradePnlNow,
  type RunDecisionResult,
  type RunQuotesResult,
  type RunSnapshotResult,
  type RunTradePnlResult,
} from "../api";

type UseAdminRunsArgs = {
  onRefresh: () => void;
  onError: (message: string) => void;
  onClearError: () => void;
};

type UseAdminRunsResult = {
  adminKey: string;
  setAdminKey: (value: string) => void;
  runResult: RunSnapshotResult | null;
  runQuotesResult: RunQuotesResult | null;
  runDecisionResult: RunDecisionResult | null;
  runTradePnlResult: RunTradePnlResult | null;
  runSnapshot: () => Promise<void>;
  runQuotes: () => Promise<void>;
  runDecision: () => Promise<void>;
  runTradePnl: () => Promise<void>;
};

/**
 * Manage admin-triggered job actions and their result payloads.
 *
 * The hook keeps endpoint responses separate per action so the dashboard can
 * show the most recent run output for snapshot/quotes/decision/trade-pnl jobs.
 */
export function useAdminRuns({ onRefresh, onError, onClearError }: UseAdminRunsArgs): UseAdminRunsResult {
  const [adminKey, setAdminKey] = React.useState<string>("");
  const [runResult, setRunResult] = React.useState<RunSnapshotResult | null>(null);
  const [runQuotesResult, setRunQuotesResult] = React.useState<RunQuotesResult | null>(null);
  const [runDecisionResult, setRunDecisionResult] = React.useState<RunDecisionResult | null>(null);
  const [runTradePnlResult, setRunTradePnlResult] = React.useState<RunTradePnlResult | null>(null);

  /**
   * Run the snapshot job now, then refresh read models that depend on it.
   */
  const runSnapshot = React.useCallback(async () => {
    onClearError();
    setRunResult(null);
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    setRunTradePnlResult(null);
    try {
      const result = await runSnapshotNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunResult(result);
      onRefresh();
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, onClearError, onError, onRefresh]);

  /**
   * Run quote ingestion now. This updates marks/underlying context data.
   */
  const runQuotes = React.useCallback(async () => {
    onClearError();
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    setRunTradePnlResult(null);
    try {
      const result = await runQuotesNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunQuotesResult(result);
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, onClearError, onError]);

  /**
   * Run the decision engine now, then refresh panels that show decisions/trades.
   */
  const runDecision = React.useCallback(async () => {
    onClearError();
    setRunDecisionResult(null);
    setRunTradePnlResult(null);
    try {
      const result = await runDecisionNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunDecisionResult(result);
      onRefresh();
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, onClearError, onError, onRefresh]);

  /**
   * Run mark-to-market PnL updates now, then refresh the trades panel.
   */
  const runTradePnl = React.useCallback(async () => {
    onClearError();
    setRunTradePnlResult(null);
    try {
      const result = await runTradePnlNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunTradePnlResult(result);
      onRefresh();
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, onClearError, onError, onRefresh]);

  return {
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
  };
}
