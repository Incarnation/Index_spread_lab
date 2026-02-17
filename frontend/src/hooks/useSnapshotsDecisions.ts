import React from "react";
import {
  fetchAdminPreflight,
  fetchChainSnapshots,
  fetchLabelMetrics,
  fetchModelOps,
  fetchStrategyMetrics,
  fetchTradeDecisions,
  fetchTrades,
  type AdminPreflightResponse,
  type ChainSnapshot,
  type LabelMetricsResponse,
  type ModelOpsResponse,
  type StrategyMetricsResponse,
  type TradeDecision,
  type TradeRow,
} from "../api";

type UseSnapshotsDecisionsArgs = {
  adminKey: string;
  onError: (message: string) => void;
};

type UseSnapshotsDecisionsResult = {
  items: ChainSnapshot[];
  decisions: TradeDecision[];
  trades: TradeRow[];
  labelMetrics: LabelMetricsResponse | null;
  modelOps: ModelOpsResponse | null;
  strategyMetrics: StrategyMetricsResponse | null;
  preflight: AdminPreflightResponse | null;
  loading: boolean;
  decisionsLoading: boolean;
  tradesLoading: boolean;
  labelMetricsLoading: boolean;
  modelOpsLoading: boolean;
  strategyMetricsLoading: boolean;
  preflightLoading: boolean;
  preflightAuthRequired: boolean;
  refresh: () => void;
};

/**
 * Manage core dashboard datasets and loading states.
 *
 * This hook centralizes fetch orchestration for snapshots, decisions, trades,
 * label metrics, strategy metrics, model-ops metrics, and admin preflight
 * freshness diagnostics so the page renders from a single source of truth.
 */
export function useSnapshotsDecisions({ adminKey, onError }: UseSnapshotsDecisionsArgs): UseSnapshotsDecisionsResult {
  const [items, setItems] = React.useState<ChainSnapshot[]>([]);
  const [decisions, setDecisions] = React.useState<TradeDecision[]>([]);
  const [trades, setTrades] = React.useState<TradeRow[]>([]);
  const [labelMetrics, setLabelMetrics] = React.useState<LabelMetricsResponse | null>(null);
  const [modelOps, setModelOps] = React.useState<ModelOpsResponse | null>(null);
  const [strategyMetrics, setStrategyMetrics] = React.useState<StrategyMetricsResponse | null>(null);
  const [preflight, setPreflight] = React.useState<AdminPreflightResponse | null>(null);
  const [loading, setLoading] = React.useState<boolean>(true);
  const [decisionsLoading, setDecisionsLoading] = React.useState<boolean>(true);
  const [tradesLoading, setTradesLoading] = React.useState<boolean>(true);
  const [labelMetricsLoading, setLabelMetricsLoading] = React.useState<boolean>(true);
  const [modelOpsLoading, setModelOpsLoading] = React.useState<boolean>(true);
  const [strategyMetricsLoading, setStrategyMetricsLoading] = React.useState<boolean>(true);
  const [preflightLoading, setPreflightLoading] = React.useState<boolean>(true);
  const [preflightAuthRequired, setPreflightAuthRequired] = React.useState<boolean>(false);

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
    setPreflightLoading(true);

    const trimmedAdminKey = adminKey.trim();
    const preflightPromise = fetchAdminPreflight(trimmedAdminKey ? trimmedAdminKey : undefined)
      .then((payload) => {
        setPreflight(payload);
        setPreflightAuthRequired(false);
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        setPreflight(null);
        if (message.includes("HTTP 401")) {
          setPreflightAuthRequired(true);
          return;
        }
        setPreflightAuthRequired(false);
        onError(message);
      })
      .finally(() => {
        setPreflightLoading(false);
      });

    Promise.all([
      fetchChainSnapshots(50),
      fetchTradeDecisions(50),
      fetchTrades(100),
      fetchLabelMetrics(90),
      fetchModelOps(),
      fetchStrategyMetrics(90),
      preflightPromise,
    ])
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
  }, [adminKey, onError]);

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
    preflight,
    loading,
    decisionsLoading,
    tradesLoading,
    labelMetricsLoading,
    modelOpsLoading,
    strategyMetricsLoading,
    preflightLoading,
    preflightAuthRequired,
    refresh,
  };
}
