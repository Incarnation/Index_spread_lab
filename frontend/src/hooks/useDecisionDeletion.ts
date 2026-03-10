import React from "react";
import { deleteTradeDecision } from "../api";

type UseDecisionDeletionArgs = {
  adminKey: string;
  activeDrawerDecisionId: number | null;
  onDeleteActiveDrawerDecision: () => void;
  onRefresh: () => void;
  onError: (message: string) => void;
  onClearError: () => void;
};

type UseDecisionDeletionResult = {
  deletingDecisionId: number | null;
  successMessage: string | null;
  deleteDecision: (decisionId: number) => Promise<void>;
};

/**
 * Handle deletion of persisted trade-decision rows and related UI feedback.
 *
 * The hook owns confirmation flow, optimistic drawer cleanup, success toast
 * text, and refresh/error callbacks after API completion.
 */
export function useDecisionDeletion({
  adminKey,
  activeDrawerDecisionId,
  onDeleteActiveDrawerDecision,
  onRefresh,
  onError,
  onClearError,
}: UseDecisionDeletionArgs): UseDecisionDeletionResult {
  const [deletingDecisionId, setDeletingDecisionId] = React.useState<number | null>(null);
  const [successMessage, setSuccessMessage] = React.useState<string | null>(null);
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  React.useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  /**
   * Delete a decision after confirmation and refresh table state.
   */
  const deleteDecision = React.useCallback(
    async (decisionId: number) => {
      const ok = window.confirm(`Delete trade decision #${decisionId}?`);
      if (!ok) return;

      onClearError();
      setDeletingDecisionId(decisionId);
      try {
        await deleteTradeDecision(decisionId, adminKey.trim() ? adminKey.trim() : undefined);
        if (activeDrawerDecisionId === decisionId) {
          onDeleteActiveDrawerDecision();
        }
        setSuccessMessage(`Decision #${decisionId} deleted.`);
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = window.setTimeout(() => {
          setSuccessMessage((prev) => (prev === `Decision #${decisionId} deleted.` ? null : prev));
          timerRef.current = null;
        }, 3000);
        onRefresh();
      } catch (e: unknown) {
        onError(e instanceof Error ? e.message : String(e));
      } finally {
        setDeletingDecisionId(null);
      }
    },
    [activeDrawerDecisionId, adminKey, onClearError, onDeleteActiveDrawerDecision, onError, onRefresh],
  );

  return {
    deletingDecisionId,
    successMessage,
    deleteDecision,
  };
}
