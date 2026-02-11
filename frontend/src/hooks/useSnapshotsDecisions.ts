import React from "react";
import { fetchChainSnapshots, fetchTradeDecisions, type ChainSnapshot, type TradeDecision } from "../api";

type UseSnapshotsDecisionsArgs = {
  onError: (message: string) => void;
};

type UseSnapshotsDecisionsResult = {
  items: ChainSnapshot[];
  decisions: TradeDecision[];
  loading: boolean;
  decisionsLoading: boolean;
  refresh: () => void;
};

export function useSnapshotsDecisions({ onError }: UseSnapshotsDecisionsArgs): UseSnapshotsDecisionsResult {
  const [items, setItems] = React.useState<ChainSnapshot[]>([]);
  const [decisions, setDecisions] = React.useState<TradeDecision[]>([]);
  const [loading, setLoading] = React.useState<boolean>(true);
  const [decisionsLoading, setDecisionsLoading] = React.useState<boolean>(true);

  const refresh = React.useCallback(() => {
    setLoading(true);
    setDecisionsLoading(true);
    Promise.all([fetchChainSnapshots(50), fetchTradeDecisions(50)])
      .then(([snapshotRows, decisionRows]) => {
        setItems(snapshotRows);
        setDecisions(decisionRows);
      })
      .catch((e: unknown) => onError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setLoading(false);
        setDecisionsLoading(false);
      });
  }, [onError]);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    items,
    decisions,
    loading,
    decisionsLoading,
    refresh,
  };
}
