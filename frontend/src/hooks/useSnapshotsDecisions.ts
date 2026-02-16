import React from "react";
import {
  fetchChainSnapshots,
  fetchLabelMetrics,
  fetchTradeDecisions,
  fetchTrades,
  type ChainSnapshot,
  type LabelMetricsResponse,
  type TradeDecision,
  type TradeRow,
} from "../api";

type UseSnapshotsDecisionsArgs = {
  onError: (message: string) => void;
};

type UseSnapshotsDecisionsResult = {
  items: ChainSnapshot[];
  decisions: TradeDecision[];
  trades: TradeRow[];
  labelMetrics: LabelMetricsResponse | null;
  loading: boolean;
  decisionsLoading: boolean;
  tradesLoading: boolean;
  labelMetricsLoading: boolean;
  refresh: () => void;
};

export function useSnapshotsDecisions({ onError }: UseSnapshotsDecisionsArgs): UseSnapshotsDecisionsResult {
  const [items, setItems] = React.useState<ChainSnapshot[]>([]);
  const [decisions, setDecisions] = React.useState<TradeDecision[]>([]);
  const [trades, setTrades] = React.useState<TradeRow[]>([]);
  const [labelMetrics, setLabelMetrics] = React.useState<LabelMetricsResponse | null>(null);
  const [loading, setLoading] = React.useState<boolean>(true);
  const [decisionsLoading, setDecisionsLoading] = React.useState<boolean>(true);
  const [tradesLoading, setTradesLoading] = React.useState<boolean>(true);
  const [labelMetricsLoading, setLabelMetricsLoading] = React.useState<boolean>(true);

  const refresh = React.useCallback(() => {
    setLoading(true);
    setDecisionsLoading(true);
    setTradesLoading(true);
    setLabelMetricsLoading(true);
    Promise.all([fetchChainSnapshots(50), fetchTradeDecisions(50), fetchTrades(100), fetchLabelMetrics(90)])
      .then(([snapshotRows, decisionRows, tradeRows, metrics]) => {
        setItems(snapshotRows);
        setDecisions(decisionRows);
        setTrades(tradeRows);
        setLabelMetrics(metrics);
      })
      .catch((e: unknown) => onError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setLoading(false);
        setDecisionsLoading(false);
        setTradesLoading(false);
        setLabelMetricsLoading(false);
      });
  }, [onError]);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    items,
    decisions,
    trades,
    labelMetrics,
    loading,
    decisionsLoading,
    tradesLoading,
    labelMetricsLoading,
    refresh,
  };
}
