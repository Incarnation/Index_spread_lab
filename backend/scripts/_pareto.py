"""Shared Pareto-frontier utilities for optimizer results.

Single source of truth for the (Sharpe vs max-drawdown) Pareto computation
that both ``backtest_strategy.extract_pareto_frontier`` and
``ingest_optimizer_results._compute_pareto`` previously implemented in
parallel (see M6 in ``OFFLINE_PIPELINE_AUDIT.md``).

Convention
----------
* ``sharpe`` -- larger is better.
* ``max_dd_pct`` -- the drawdown *magnitude* expressed as a positive
  percentage (e.g. ``12.5`` means a 12.5% peak-to-trough drawdown).
  Smaller magnitudes are better. This matches the values produced by
  ``backtest_strategy._compute_metrics`` and stored in
  ``backtest_results.csv``.

A row ``i`` is dominated when there exists another row ``j`` with
``sharpe[j] >= sharpe[i]`` AND ``max_dd_pct[j] <= max_dd_pct[i]`` and at
least one of those inequalities is strict. Pareto-optimal rows are the
non-dominated set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SHARPE_COL = "sharpe"
MAX_DD_COL = "max_dd_pct"


def compute_pareto_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean ``Series`` flagging Pareto-optimal rows.

    Parameters
    ----------
    df : Optimizer results DataFrame. Must contain the columns ``sharpe``
        and ``max_dd_pct``.

    Returns
    -------
    pd.Series
        Boolean series (aligned to ``df.index``) where ``True`` marks a
        Pareto-optimal row. The mask preserves ``df`` order; consumers
        that want a ranked output should sort externally.

    Raises
    ------
    KeyError
        If either ``sharpe`` or ``max_dd_pct`` is missing.
    """
    missing = [c for c in (SHARPE_COL, MAX_DD_COL) if c not in df.columns]
    if missing:
        raise KeyError(
            f"compute_pareto_mask: missing required column(s): {missing}"
        )

    n = len(df)
    if n == 0:
        return pd.Series([], index=df.index, dtype=bool)

    # Pull values into NumPy once -- the O(n^2) loop below is hot when
    # n grows past ~5k rows, and avoiding pandas attribute lookups in
    # the inner loop is a measurable win.
    is_pareto = np.ones(n, dtype=bool)
    sharpe = df[SHARPE_COL].to_numpy()
    dd = df[MAX_DD_COL].to_numpy()

    for i in range(n):
        if not is_pareto[i]:
            continue
        # Walk the (still-) candidate set; the early ``break`` makes
        # the practical complexity closer to O(n * k) where k is the
        # average frontier depth.
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if sharpe[j] >= sharpe[i] and dd[j] <= dd[i]:
                # j dominates i iff at least one inequality is strict.
                if sharpe[j] > sharpe[i] or dd[j] < dd[i]:
                    is_pareto[i] = False
                    break

    return pd.Series(is_pareto, index=df.index)


def extract_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Return only Pareto-optimal rows, sorted by Sharpe descending.

    Thin convenience wrapper around :func:`compute_pareto_mask` for
    callers (e.g. ``backtest_strategy.py``) that want a ready-to-print
    frontier table rather than a boolean mask.

    Parameters
    ----------
    df : Optimizer results DataFrame with ``sharpe`` and ``max_dd_pct``.

    Returns
    -------
    pd.DataFrame
        Filtered frontier rows, sorted by ``sharpe`` descending and
        with the index reset.
    """
    mask = compute_pareto_mask(df)
    return (
        df[mask]
        .sort_values(SHARPE_COL, ascending=False)
        .reset_index(drop=True)
    )
