import React from "react";
import {
  runDecisionNow,
  runQuotesNow,
  runSnapshotNow,
  type RunDecisionResult,
  type RunQuotesResult,
  type RunSnapshotResult,
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
  runSnapshot: () => Promise<void>;
  runQuotes: () => Promise<void>;
  runDecision: () => Promise<void>;
};

export function useAdminRuns({ onRefresh, onError, onClearError }: UseAdminRunsArgs): UseAdminRunsResult {
  const [adminKey, setAdminKey] = React.useState<string>("");
  const [runResult, setRunResult] = React.useState<RunSnapshotResult | null>(null);
  const [runQuotesResult, setRunQuotesResult] = React.useState<RunQuotesResult | null>(null);
  const [runDecisionResult, setRunDecisionResult] = React.useState<RunDecisionResult | null>(null);

  const runSnapshot = React.useCallback(async () => {
    onClearError();
    setRunResult(null);
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    try {
      const result = await runSnapshotNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunResult(result);
      onRefresh();
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, onClearError, onError, onRefresh]);

  const runQuotes = React.useCallback(async () => {
    onClearError();
    setRunQuotesResult(null);
    setRunDecisionResult(null);
    try {
      const result = await runQuotesNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunQuotesResult(result);
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : String(e));
    }
  }, [adminKey, onClearError, onError]);

  const runDecision = React.useCallback(async () => {
    onClearError();
    setRunDecisionResult(null);
    try {
      const result = await runDecisionNow(adminKey.trim() ? adminKey.trim() : undefined);
      setRunDecisionResult(result);
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
    runSnapshot,
    runQuotes,
    runDecision,
  };
}
