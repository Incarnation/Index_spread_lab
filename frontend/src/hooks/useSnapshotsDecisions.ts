import React from "react";
import {
  fetchChainSnapshots,
  fetchLabelMetrics,
  fetchModelOps,
  fetchStrategyMetrics,
  fetchTradeDecisions,
  fetchTrades,
  type ChainSnapshot,
  type LabelMetricsResponse,
  type ModelOpsResponse,
  type StrategyMetricsResponse,
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
  modelOps: ModelOpsResponse | null;
  strategyMetrics: StrategyMetricsResponse | null;
  loading: boolean;
  decisionsLoading: boolean;
  tradesLoading: boolean;
  labelMetricsLoading: boolean;
  modelOpsLoading: boolean;
  strategyMetricsLoading: boolean;
  refresh: () => void;
};

/**
 * Manage core dashboard datasets and loading states.
 *
 * This hook centralizes fetch orchestration for snapshots, decisions, trades,
 * label metrics, strategy metrics, and model-ops metrics so the page renders
 * from a single source of truth.
 */
export function useSnapshotsDecisions({ onError }: UseSnapshotsDecisionsArgs): UseSnapshotsDecisionsResult {
  const [items, setItems] = React.useState<ChainSnapshot[]>([]);
  const [decisions, setDecisions] = React.useState<TradeDecision[]>([]);
  const [trades, setTrades] = React.useState<TradeRow[]>([]);
  const [labelMetrics, setLabelMetrics] = React.useState<LabelMetricsResponse | null>(null);
  const [modelOps, setModelOps] = React.useState<ModelOpsResponse | null>(null);
  const [strategyMetrics, setStrategyMetrics] = React.useState<StrategyMetricsResponse | null>(null);
  const [loading, setLoading] = React.useState<boolean>(true);
  const [decisionsLoading, setDecisionsLoading] = React.useState<boolean>(true);
  const [tradesLoading, setTradesLoading] = React.useState<boolean>(true);
  const [labelMetricsLoading, setLabelMetricsLoading] = React.useState<boolean>(true);
  const [modelOpsLoading, setModelOpsLoading] = React.useState<boolean>(true);
  const [strategyMetricsLoading, setStrategyMetricsLoading] = React.useState<boolean>(true);

  /**
   * Refresh all dashboard data sources in one coordinated request cycle.
   *
   * We intentionally reset each loading flag first so each panel can render
   * predictable loading states while the parallel API calls resolve.
   */
  const refresh = React.useCallback(() => {
    setLoading(true);
    setDecisionsLoading(true);
    setTradesLoading(true);
    setLabelMetricsLoading(true);
    setModelOpsLoading(true);
    setStrategyMetricsLoading(true);
    Promise.all([fetchChainSnapshots(50), fetchTradeDecisions(50), fetchTrades(100), fetchLabelMetrics(90), fetchModelOps(), fetchStrategyMetrics(90)])
      .then(([snapshotRows, decisionRows, tradeRows, metrics, ops, strategy]) => {
        setItems(snapshotRows);
        setDecisions(decisionRows);
        setTrades(tradeRows);
        setLabelMetrics(metrics);
        setModelOps(ops);
        setStrategyMetrics(strategy);
      })
      .catch((e: unknown) => onError(e instanceof Error ? e.message : String(e)))
      .finally(() => {
        setLoading(false);
        setDecisionsLoading(false);
        setTradesLoading(false);
        setLabelMetricsLoading(false);
        setModelOpsLoading(false);
        setStrategyMetricsLoading(false);
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
    modelOps,
    strategyMetrics,
    loading,
    decisionsLoading,
    tradesLoading,
    labelMetricsLoading,
    modelOpsLoading,
    strategyMetricsLoading,
    refresh,
  };
}
