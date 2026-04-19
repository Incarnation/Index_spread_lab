"""Shared regime classification utilities for backtest tooling.

Provides a lightweight day-level regime tagger that assigns each trading
day to one bucket per dimension (VIX level, SPX move, term structure).

**Current consumers** (verified via grep, last refreshed 2026-04-16):

- ``backend/scripts/backtest_strategy.py`` -- imports
  :func:`compute_regime_metrics` for per-backtest regime metric tables.
- ``backend/scripts/generate_training_data.py`` -- only reads this file
  for cache-key fingerprinting (it never imports the symbols), so a
  bug-fix here invalidates the candidate cache for that day. See L6 in
  ``OFFLINE_PIPELINE_AUDIT.md``.
- ``backend/tests/test_regime_utils.py`` -- unit tests added in Wave 4.

``regime_analysis.py`` was the historical second consumer and remains a
**candidate** for adoption (the L6 follow-up in
``OFFLINE_PIPELINE_AUDIT.md``), but it currently rolls its own
classification logic. Aligning the two is intentionally out of scope
until ``regime_analysis.py`` is part of an end-to-end pipeline.

All percentage-change inputs use the **decimal-fraction** convention
(``-0.01`` = 1% drop) so the helpers here are interchangeable across:

- ``backtest_strategy.precompute_daily_signals`` (decimal)
- ``regime_analysis.enrich_with_daily_features`` (decimal after the
  M2 unification -- see ``OFFLINE_PIPELINE_AUDIT.md``)
- ``services.event_signals`` runtime context (decimal)

Pass ``prev_spx_return`` and ``term_structure`` directly from any of the
above without unit conversion. See ``OFFLINE_PIPELINE_AUDIT.md`` for the
broader unit-consistency story.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# -- VIX level buckets (at entry) ------------------------------------------

VIX_THRESHOLDS = [
    (15.0, "vix_low"),
    (20.0, "vix_medium"),
    (25.0, "vix_high"),
    (float("inf"), "vix_extreme"),
]


def classify_vix(vix: float | None) -> str:
    """Classify a VIX reading into low / medium / high / extreme."""
    if vix is None or np.isnan(vix):
        return "vix_unknown"
    for threshold, label in VIX_THRESHOLDS:
        if vix < threshold:
            return label
    return "vix_extreme"


# -- SPX prior-day move buckets (decimal fractions) -------------------------

SPX_THRESHOLDS = [
    (-0.02, "spx_big_drop"),
    (-0.005, "spx_small_drop"),
    (0.005, "spx_flat"),
    (float("inf"), "spx_rally"),
]


def classify_spx_move(prev_spx_return: float | None) -> str:
    """Classify prior-day SPX return into big drop / small drop / flat / rally.

    Uses decimal fractions: -0.02 = 2% drop.
    """
    if prev_spx_return is None or np.isnan(prev_spx_return):
        return "spx_unknown"
    for threshold, label in SPX_THRESHOLDS:
        if prev_spx_return < threshold:
            return label
    return "spx_rally"


# -- Term structure buckets (VIX9D / VIX) -----------------------------------

def classify_term_structure(term_structure: float | None) -> str:
    """Classify term structure as normal (< 1.0) or inverted (>= 1.0)."""
    if term_structure is None or np.isnan(term_structure):
        return "ts_unknown"
    return "ts_inverted" if term_structure >= 1.0 else "ts_normal"


# -- Combined regime tagger -------------------------------------------------

REGIME_DIMENSIONS = ["vix_regime", "spx_regime", "ts_regime"]


def classify_day_regime(daily_signals_row: pd.Series) -> dict[str, str]:
    """Tag a single day with its regime across all dimensions.

    Parameters
    ----------
    daily_signals_row : One row from the ``precompute_daily_signals`` DataFrame
        (indexed by day). Expected columns: ``vix``, ``prev_spx_return``,
        ``term_structure``.

    Returns
    -------
    Dict with keys ``vix_regime``, ``spx_regime``, ``ts_regime``.
    """
    vix = daily_signals_row.get("vix")
    prev_ret = daily_signals_row.get("prev_spx_return")
    ts = daily_signals_row.get("term_structure")
    return {
        "vix_regime": classify_vix(vix if pd.notna(vix) else None),
        "spx_regime": classify_spx_move(prev_ret if pd.notna(prev_ret) else None),
        "ts_regime": classify_term_structure(ts if pd.notna(ts) else None),
    }


def compute_regime_metrics(
    curve_df: pd.DataFrame,
    daily_signals: pd.DataFrame,
) -> dict[str, Any]:
    """Compute per-regime performance breakdowns from a backtest equity curve.

    Parameters
    ----------
    curve_df : DataFrame of DayRecord dicts (columns: day, equity, daily_pnl,
        n_trades, lots, status, event_signals).
    daily_signals : Precomputed daily signal features (indexed by day).

    Returns
    -------
    Flat dict with keys like ``vix_high_trades``, ``vix_high_pnl``,
    ``vix_high_win_rate``, ``spx_big_drop_sharpe``, etc.
    """
    if curve_df.empty:
        return {}

    regime_tags: list[dict[str, str]] = []
    for day in curve_df["day"]:
        if day in daily_signals.index:
            regime_tags.append(classify_day_regime(daily_signals.loc[day]))
        else:
            regime_tags.append({d: "unknown" for d in REGIME_DIMENSIONS})

    tagged = curve_df.copy()
    for dim in REGIME_DIMENSIONS:
        tagged[dim] = [t[dim] for t in regime_tags]

    metrics: dict[str, Any] = {}
    traded_mask = tagged["n_trades"] > 0

    for dim in REGIME_DIMENSIONS:
        prefix = dim.replace("_regime", "")
        for regime_val, grp in tagged.groupby(dim):
            key = f"{prefix}_{regime_val.split('_', 1)[-1]}" if "_" in regime_val else f"{prefix}_{regime_val}"

            grp_traded = grp[grp["n_trades"] > 0]
            n_trades_days = len(grp_traded)
            total_pnl = float(grp["daily_pnl"].sum())
            win_days = int((grp_traded["daily_pnl"] > 0).sum()) if n_trades_days > 0 else 0
            win_rate = win_days / max(n_trades_days, 1)

            pnl_series = grp["daily_pnl"]
            sharpe = 0.0
            if len(pnl_series) > 1 and pnl_series.std() > 0:
                sharpe = float(pnl_series.mean() / pnl_series.std() * np.sqrt(252))

            metrics[f"{key}_trades"] = n_trades_days
            metrics[f"{key}_pnl"] = round(total_pnl, 2)
            metrics[f"{key}_win_rate"] = round(win_rate, 4)
            metrics[f"{key}_sharpe"] = round(sharpe, 2)

    return metrics
