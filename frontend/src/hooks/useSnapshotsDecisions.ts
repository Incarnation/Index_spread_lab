import React from "react";
import { fetchChainSnapshots, fetchTradeDecisions, fetchTrades, type ChainSnapshot, type TradeDecision, type TradeRow } from "../api";

type UseSnapshotsDecisionsArgs = {
  onError: (message: string) => void;
};

type UseSnapshotsDecisionsResult = {
  items: ChainSnapshot[];
  decisions: TradeDecision[];
  trades: TradeRow[];
  loading: boolean;
  decisionsLoading: boolean;
  tradesLoading: boolean;
  refresh: () => void;
};

export function useSnapshotsDecisions({ onError }: UseSnapshotsDecisionsArgs): UseSnapshotsDecisionsResult {
  const [items, setItems] = React.useState<ChainSnapshot[]>([]);
  const [decisions, setDecisions] = React.useState<TradeDecision[]>([]);
  const [trades, setTrades] = React.useState<TradeRow[]>([]);
  const [loading, setLoading] = React.useState<boolean>(true);
  const [decisionsLoading, setDecisionsLoading] = React.useState<boolean>(true);
  const [tradesLoading, setTradesLoading] = React.useState<boolean>(true);

  const refresh = React.useCallback(() => {
    setLoading(true);
    setDecisionsLoading(true);
    setTradesLoading(true);
    Promise.all([fetchChainSnapshots(50), fetchTradeDecisions(50), fetchTrades(100)])
      .then(([snapshotRows, decisionRows, tradeRows]) => {
        setItems(snapshotRows);
        setDecisions(decisionRows);
        setTrades(tradeRows);
      })
      .catch((e: unknown) => onError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setLoading(false);
        setDecisionsLoading(false);
        setTradesLoading(false);
      });
  }, [onError]);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    items,
    decisions,
    trades,
    loading,
    decisionsLoading,
    tradesLoading,
    refresh,
  };
}
