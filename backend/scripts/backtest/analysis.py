"""Capital-budgeted backtest engine with scheduled + event-driven layers.

Simulates a fixed-capital portfolio with configurable:
  - Position sizing (gradual lot scaling or fixed)
  - Monthly drawdown stop-loss
  - Scheduled trade selection (calls-only or both, by credit_to_width)
  - Event-driven signals (SPX drop, VIX spike, rally avoidance, etc.)
  - Exhaustive parameter grid search with CSV export

Usage::

    # Single run with defaults ($20k, 2/day, 15% monthly stop)
    python scripts/backtest_strategy.py

    # Custom single run
    python scripts/backtest_strategy.py --capital 20000 --max-trades 2 --monthly-stop 0.15

    # Exhaustive optimizer (sweeps ~1000+ configs, exports CSV)
    python scripts/backtest_strategy.py --optimize

    # Quick comparison of preset configs
    python scripts/backtest_strategy.py --compare

Continuous Improvement Workflow
-------------------------------
**Weekly** -- Compare live paper-trade PnL against backtest expectations.
Check that win-rate and average trade PnL are within 1 standard deviation
of the backtest distribution.  Investigate if 3+ consecutive losing days
occur that the backtest did not predict.

**Monthly** -- After accumulating a full calendar month of live data:

1. Export latest production data (chains, underlying quotes, calendar)::

       python scripts/export_production_data.py --tables all

2. Regenerate training candidates.  The pipeline auto-merges Databento
   files (historical) with production exports (recent), so no separate
   download step is needed for dates covered by the production DB::

       python scripts/generate_training_data.py --workers 4

3. Re-run the optimizer with the expanded dataset::

       python scripts/backtest_strategy.py --optimize

4. Run walk-forward validation with auto-generated windows::

       python scripts/backtest_strategy.py --walkforward --wf-auto

5. If the current config's out-of-sample Sharpe drops >30% vs the
   original walkforward result for 2+ consecutive months, re-select
   the best config from the updated Pareto frontier and update
   ``.env`` PORTFOLIO_*/EVENT_* values accordingly.

**On new data sources** -- When adding FRD data for earlier years (2022-2024),
regenerate the full training set and re-run ``--optimize --walkforward``
to validate the strategy on the longer history.  Larger datasets
reduce overfitting risk and increase confidence in parameter choices.

**Production DB as sole data source** -- Once the production system has
accumulated enough data, Databento/FRD downloads are no longer needed.
The ``export_production_data.py`` script exports ``chain_snapshots``,
``option_chain_rows``, and ``underlying_quotes`` as per-day Parquet files
that the training pipeline consumes transparently via auto-fallback
loaders.  Run ``--wf-auto`` to auto-size walk-forward windows to the
available data range.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

def _locate_scripts_dir() -> Path:
    """Find ``backend/scripts/`` from this file's location.

    Works whether the file is the original monolith (``backend/scripts/
    backtest_strategy.py``) or a submodule of the post-split package
    (``backend/scripts/backtest/<sub>.py``).  We walk up the parent
    chain looking for a directory named ``scripts`` whose sibling
    ``spx_backend`` package exists -- that's the canonical scripts
    directory regardless of nesting.
    """
    for parent in Path(__file__).resolve().parents:
        if parent.name == "scripts" and (parent.parent / "spx_backend").is_dir():
            return parent
    # Fall back to direct parent so behaviour is unchanged when running
    # from an unfamiliar layout (e.g. a test fixture); spx_backend
    # imports below will fail loudly rather than silently mis-resolve.
    return Path(__file__).resolve().parent


_SCRIPTS_DIR = _locate_scripts_dir()
_BACKEND_DIR = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
# Add backend root so ``spx_backend.config`` is importable for the
# M10 alignment-warn helper.  Falls back gracefully when the package
# is unavailable (e.g. minimal CI sandboxes).
sys.path.insert(0, str(_BACKEND_DIR))

from _constants import CONTRACT_MULT, CONTRACTS, MARGIN_PER_LOT
from _pareto import extract_pareto_frontier as _pareto_extract
from regime_utils import compute_regime_metrics

# Shared event-signal evaluator (Wave 5 of OFFLINE_PIPELINE_AUDIT.md).
# Both the live ``EventSignalDetector`` and the backtest's wrapper now
# delegate to ``evaluate_event_signals`` so signal-firing rules stay
# byte-identical between paths.  A failed import (e.g. a Python sandbox
# without the spx_backend package on sys.path) is treated as a hard
# error -- a divergent backtest is worse than a missing one.
from spx_backend.services.event_signals import (
    EventThresholds,
    evaluate_event_signals,
)

# Shared candidate-leg-identity dedup helper (audit M3).  Both the live
# ``DecisionJob`` and this backtest engine now use the same key shape
# so a candidate row that maps to the exact same (side, expiration,
# short_symbol, long_symbol) as another row in the same day-window is
# never double-counted regardless of DataFrame index.
from spx_backend.services.candidate_dedupe import candidate_dedupe_key

DATA_DIR = _SCRIPTS_DIR.parents[1] / "data"
DEFAULT_CSV = DATA_DIR / "training_candidates.csv"
RESULTS_CSV = DATA_DIR / "backtest_results.csv"

# Backtest-local mirror of the live ``_SPX_2D_MAX_CALENDAR_GAP_DAYS``.
# Kept as a module constant for any non-detector code paths that want
# to introspect or document the gap threshold; the actual gating now
# lives in ``evaluate_event_signals`` (single source of truth).
_SPX_2D_MAX_CALENDAR_GAP_DAYS = 4

from .engine import (
    BacktestResult,
    EventConfig,
    FullConfig,
    PortfolioConfig,
    RegimeThrottle,
    TradingConfig,
    _opt_str,
    _opt_val,
    _safe_bool,
    compute_effective_pnl,
    precompute_daily_signals,
    precompute_pnl_columns,
    run_backtest,
)
from .optimizer import (
    OPTIMIZER_SL_VALUES,
    OPTIMIZER_TP_VALUES,
    _build_optimizer_grid,
    _run_grid,
    run_event_only_optimizer,
    run_optimizer,
    run_selective_optimizer,
    run_staged_optimizer,
)


def _build_comparison_configs() -> list[tuple[str, FullConfig]]:
    """Hand-picked configs for quick side-by-side comparison."""
    return [
        ("$20k | 2/day | 15% stop | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | 20% stop | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.20, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | no stop  | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=None, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 1/day | 15% stop | gradual | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=1,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | 15% stop | fixed 1 | calls", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=999_999, calls_only=True),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2/day | 15% stop | gradual | both sides", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=False),
            event=EventConfig(enabled=False),
        )),
        ("$20k | 2+1ev | 15% stop | shared  | drop puts", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=True, budget_mode="shared", spx_drop_threshold=-0.01,
                              side_preference="puts", min_dte=5, max_dte=7, min_delta=0.15, max_delta=0.25),
        )),
        ("$20k | 2+1ev | 15% stop | separate | drop puts", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=True, budget_mode="separate", max_event_trades=1,
                              spx_drop_threshold=-0.01, side_preference="puts",
                              min_dte=5, max_dte=7, min_delta=0.15, max_delta=0.25),
        )),
        ("$20k | 2/day | 15% stop | gradual | calls + rally avoid", FullConfig(
            portfolio=PortfolioConfig(starting_capital=20_000, max_trades_per_day=2,
                                      monthly_drawdown_limit=0.15, lot_per_equity=10_000, calls_only=True),
            event=EventConfig(enabled=True, rally_avoidance=True, rally_threshold=0.01),
        )),
    ]
def print_summary(result: BacktestResult) -> None:
    """Print a one-block summary of a backtest result."""
    r = result
    cap = r.config.portfolio.starting_capital
    print(f"\n{'=' * 80}")
    print(f"  {r.label}")
    print(f"{'=' * 80}")
    print(f"  Start: ${cap:,.0f}  ->  Final: ${r.final_equity:,.0f}  "
          f"({r.total_return_pct:+.1f}%)")
    print(f"  Annualised: {r.annualised_return_pct:+.0f}%  |  "
          f"Max DD: {r.max_drawdown_pct:.1f}%  |  Trough: ${r.trough:,.0f}")
    print(f"  Sharpe: {r.sharpe:.2f}  |  "
          f"Trades: {r.total_trades}  |  "
          f"Traded {r.days_traded}d  Stopped {r.days_stopped}d  "
          f"Win {r.win_days}/{r.days_traded} "
          f"({r.win_days / max(r.days_traded, 1) * 100:.0f}%)")
def print_monthly(result: BacktestResult) -> None:
    """Print month-by-month breakdown."""
    print(f"\n{'Month':>8}  {'PnL':>10}  {'Return':>8}  {'Equity':>10}  "
          f"{'Lots':>5}  {'Traded':>7}  {'Stopped':>8}  {'Events':>7}  {'Worst Day':>10}")
    print("-" * 90)
    for _, row in result.monthly.iterrows():
        print(f"{row['month']:>8}  ${row['pnl']:>9,.0f}  "
              f"{row['return_pct']:>+6.1f}%  ${row['end_equity']:>9,.0f}  "
              f"{row['avg_lots']:>5.1f}  {int(row['days_traded']):>5}d  "
              f"{int(row['days_stopped']):>6}d  {int(row['event_days']):>5}d  "
              f"${row['worst_day']:>9,.0f}")
def print_comparison_table(results: list[BacktestResult]) -> None:
    """Print a compact comparison of multiple configs."""
    print(f"\n{'=' * 115}")
    print("  CONFIGURATION COMPARISON")
    print(f"{'=' * 115}")
    print(f"{'Configuration':<50} {'Final':>10} {'Return':>8} "
          f"{'MaxDD':>6} {'Trough':>10} {'Sharpe':>7} "
          f"{'Trades':>7} {'Stopped':>8}")
    print("-" * 115)
    for r in results:
        print(f"{r.label:<50} ${r.final_equity:>9,.0f} "
              f"{r.total_return_pct:>+6.0f}% "
              f"{r.max_drawdown_pct:>5.0f}% "
              f"${r.trough:>9,.0f} "
              f"{r.sharpe:>7.2f} "
              f"{r.total_trades:>7} "
              f"{r.days_stopped:>6}d")
def print_optimizer_top(results_df: pd.DataFrame, metric: str, n: int = 10) -> None:
    """Print the top-N configs from the optimizer by a given metric."""
    asc = metric == "max_dd_pct"
    top = results_df.sort_values(metric, ascending=asc).head(n)

    print(f"\n{'=' * 115}")
    print(f"  TOP {n} BY {metric.upper()}")
    print(f"{'=' * 115}")
    cols = ["p_max_trades_per_day", "p_monthly_drawdown_limit", "p_lot_per_equity",
            "p_calls_only", "t_tp_pct", "t_sl_mult", "t_max_vix",
            "t_avoid_opex", "t_width_filter", "t_entry_count",
            "e_enabled", "e_budget_mode", "e_spx_drop_threshold",
            "e_rally_avoidance", "final_equity", "return_pct", "max_dd_pct",
            "sharpe", "total_trades", "days_stopped"]
    display_cols = [c for c in cols if c in top.columns]
    print(top[display_cols].to_string(index=False))
PARETO_CSV = DATA_DIR / "pareto_frontier.csv"
WALKFORWARD_CSV = DATA_DIR / "walkforward_results.csv"
ANALYSIS_PARAMS = [
    "p_max_trades_per_day",
    "p_monthly_drawdown_limit",
    "p_lot_per_equity",
    "p_calls_only",
    "p_min_dte",
    "p_max_delta",
    "t_tp_pct",
    "t_sl_mult",
    "t_max_vix",
    "t_max_term_structure",
    "t_avoid_opex",
    "t_prefer_event_days",
    "t_width_filter",
    "t_entry_count",
    "e_enabled",
    "e_signal_mode",
    "e_budget_mode",
    "e_spx_drop_threshold",
    "e_spx_drop_2d_threshold",
    "e_vix_spike_threshold",
    "e_vix_elevated_threshold",
    "e_rally_avoidance",
    "e_side_preference",
]
def _deduplicate_results(rdf: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Collapse configs that produce identical outcome metrics.

    Parameters
    ----------
    rdf : Full optimizer results DataFrame.

    Returns
    -------
    Tuple of (deduplicated DataFrame keeping first of each group,
    original count, unique count).
    """
    metric_cols = ["final_equity", "return_pct", "max_dd_pct", "sharpe",
                   "total_trades", "days_traded", "days_stopped", "win_days"]
    rounded = rdf[metric_cols].round(4)
    dedup = rdf.loc[~rounded.duplicated(keep="first")]
    return dedup, len(rdf), len(dedup)
def _parameter_importance(rdf: pd.DataFrame) -> None:
    """Print grouped-mean tables showing how each parameter affects outcomes.

    For each parameter in ANALYSIS_PARAMS, groups the results by that
    parameter's values and prints the mean Sharpe, return, and max DD,
    plus the spread (max-min) across values.
    """
    metrics = ["sharpe", "return_pct", "max_dd_pct"]
    print(f"\n{'=' * 100}")
    print("  PARAMETER IMPORTANCE (mean metric by parameter value)")
    print(f"{'=' * 100}")

    importance: list[tuple[str, float]] = []

    for param in ANALYSIS_PARAMS:
        if param not in rdf.columns:
            continue
        grouped = rdf.groupby(param)[metrics].mean()
        if len(grouped) < 2:
            continue

        sharpe_spread = grouped["sharpe"].max() - grouped["sharpe"].min()
        importance.append((param, sharpe_spread))

        print(f"\n  {param} (Sharpe spread: {sharpe_spread:.2f})")
        print(f"  {'Value':<25} {'Sharpe':>8} {'Return%':>10} {'MaxDD%':>8}")
        print("  " + "-" * 55)
        for val, row in grouped.iterrows():
            val_str = str(val) if val is not None else "None"
            print(f"  {val_str:<25} {row['sharpe']:>8.2f} "
                  f"{row['return_pct']:>+9.0f}% {row['max_dd_pct']:>7.1f}%")

    importance.sort(key=lambda x: -x[1])
    print(f"\n{'=' * 100}")
    print("  PARAMETER RANKING (by Sharpe spread across values)")
    print(f"{'=' * 100}")
    for rank, (param, spread) in enumerate(importance, 1):
        print(f"  {rank}. {param:<35} spread = {spread:.2f}")
def extract_pareto_frontier(rdf: pd.DataFrame) -> pd.DataFrame:
    """Find Pareto-optimal configs: no other config has both higher Sharpe AND lower max DD.

    Thin wrapper around :func:`_pareto.extract_pareto_frontier` (the
    shared implementation that ``ingest_optimizer_results.py`` also
    uses, see M6 in ``OFFLINE_PIPELINE_AUDIT.md``).

    Parameters
    ----------
    rdf : Optimizer results DataFrame with ``sharpe`` and ``max_dd_pct`` columns.

    Returns
    -------
    DataFrame with only Pareto-optimal rows, sorted by Sharpe descending.
    """
    return _pareto_extract(rdf)
def _print_pareto(pareto: pd.DataFrame) -> None:
    """Print the Pareto frontier in a readable table."""
    print(f"\n{'=' * 130}")
    print(f"  PARETO FRONTIER (Sharpe vs Max DD) -- {len(pareto)} optimal configs")
    print(f"{'=' * 130}")
    cols = ["p_max_trades_per_day", "p_monthly_drawdown_limit", "p_lot_per_equity",
            "p_calls_only", "t_tp_pct", "t_sl_mult", "t_max_vix",
            "t_avoid_opex", "t_width_filter",
            "e_enabled", "e_budget_mode", "e_spx_drop_threshold",
            "e_rally_avoidance", "final_equity", "return_pct", "max_dd_pct",
            "sharpe", "win_rate"]
    display_cols = [c for c in cols if c in pareto.columns]
    print(pareto[display_cols].to_string(index=False))
def _robustness_check(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    rdf: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """Re-run top-N configs and report monthly consistency.

    Flags configs that depend on 1-2 monster months vs. those that
    are consistently profitable across the period.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    rdf : Optimizer results DataFrame.
    top_n : Number of top configs to re-evaluate.
    """
    print(f"\n{'=' * 140}")
    print(f"  ROBUSTNESS CHECK (monthly consistency for top-{top_n} by Sharpe)")
    print(f"{'=' * 140}")

    top = rdf.sort_values("sharpe", ascending=False).head(top_n)

    precomp_cache: dict[tuple, dict] = {}

    print(f"  {'#':>3} {'Trades/d':>8} {'Stop':>6} {'Lots':>8} {'Calls':>6} "
          f"{'Events':>7} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} "
          f"{'Profit Mo':>10} {'Worst Mo':>10} {'Best Mo':>9} {'Consistency':>12}")
    print("  " + "-" * 135)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        cfg = _row_to_config(row)
        key = (cfg.trading.width_filter, cfg.trading.entry_count)
        if key not in precomp_cache:
            precomp_cache[key] = _precompute_day_selections(
                df, width_filter=key[0], entry_count=key[1],
            )
        result = run_backtest(df, daily_signals, cfg, day_precomp=precomp_cache[key])
        m = result.monthly
        n_months = len(m)
        profitable_months = int((m["pnl"] > 0).sum())
        worst_month_pct = float(m["return_pct"].min())
        best_month_pct = float(m["return_pct"].max())
        consistency = profitable_months / max(n_months, 1)

        stop_str = f"{row.get('p_monthly_drawdown_limit', '')}"
        lots_str = "grad" if row.get("p_lot_per_equity", 0) < 100_000 else "fixed"
        evt_str = "yes" if row.get("e_enabled") else "no"

        print(f"  {rank:>3} {int(row.get('p_max_trades_per_day', 0)):>8} "
              f"{stop_str:>6} {lots_str:>8} {str(row.get('p_calls_only', '')):>6} "
              f"{evt_str:>7} {row['sharpe']:>7.2f} {row['return_pct']:>+7.0f}% "
              f"{row['max_dd_pct']:>5.0f}% "
              f"{profitable_months:>5}/{n_months:<4} "
              f"{worst_month_pct:>+9.1f}% {best_month_pct:>+8.1f}% "
              f"{consistency:>11.0%}")
def _row_to_config(row: pd.Series) -> FullConfig:
    """Reconstruct a FullConfig from an optimizer results CSV row.

    Parameters
    ----------
    row : One row from the optimizer results DataFrame.

    Returns
    -------
    FullConfig with portfolio, trading, and event settings from the row.
    """
    dd_limit = row.get("p_monthly_drawdown_limit")
    if pd.isna(dd_limit):
        dd_limit = None

    min_dte = row.get("p_min_dte")
    if pd.notna(min_dte):
        min_dte = int(min_dte)
    else:
        min_dte = None

    max_delta = row.get("p_max_delta")
    if pd.notna(max_delta):
        max_delta = float(max_delta)
    else:
        max_delta = None

    def _opt_float(key, default=None):
        """Extract a float from *row[key]*, returning *default* on NaN."""
        v = row.get(key)
        return float(v) if pd.notna(v) else default

    def _opt_int(key, default=None):
        """Extract an int from *row[key]*, returning *default* on NaN."""
        v = row.get(key)
        return int(v) if pd.notna(v) else default

    def _opt_bool(key, default=False):
        """Extract a bool from *row[key]*, returning *default* on NaN."""
        v = row.get(key, default)
        if pd.isna(v):
            return default
        return bool(v)

    pc = PortfolioConfig(
        starting_capital=float(row.get("p_starting_capital", 20_000)),
        max_trades_per_day=int(row.get("p_max_trades_per_day", 2)),
        monthly_drawdown_limit=dd_limit,
        lot_per_equity=float(row.get("p_lot_per_equity", 10_000)),
        calls_only=_opt_bool("p_calls_only", True),
        min_dte=min_dte,
        max_delta=max_delta,
    )

    tc = TradingConfig(
        tp_pct=float(row.get("t_tp_pct", 0.50)),
        sl_mult=_opt_float("t_sl_mult"),
        max_vix=_opt_float("t_max_vix"),
        max_term_structure=_opt_float("t_max_term_structure"),
        avoid_opex=_opt_bool("t_avoid_opex"),
        prefer_event_days=_opt_bool("t_prefer_event_days"),
        width_filter=_opt_float("t_width_filter"),
        entry_count=_opt_int("t_entry_count"),
    )

    ec = EventConfig(
        enabled=_opt_bool("e_enabled"),
        signal_mode=_opt_str(row.get("e_signal_mode"), "any"),
        budget_mode=_opt_str(row.get("e_budget_mode"), "shared"),
        max_event_trades=int(row.get("e_max_event_trades", 1)),
        spx_drop_threshold=float(row.get("e_spx_drop_threshold", -0.01)),
        spx_drop_2d_threshold=float(row.get("e_spx_drop_2d_threshold", -0.02)),
        spx_drop_min=_opt_float("e_spx_drop_min"),
        spx_drop_max=_opt_float("e_spx_drop_max"),
        vix_spike_threshold=float(row.get("e_vix_spike_threshold", 0.15)),
        vix_elevated_threshold=float(row.get("e_vix_elevated_threshold", 25.0)),
        term_inversion_threshold=float(row.get("e_term_inversion_threshold", 1.0)),
        side_preference=_opt_str(row.get("e_side_preference"), "puts"),
        min_dte=int(row.get("e_min_dte", 5)),
        max_dte=int(row.get("e_max_dte", 7)),
        min_delta=float(row.get("e_min_delta", 0.15)),
        max_delta=float(row.get("e_max_delta", 0.25)),
        rally_avoidance=_opt_bool("e_rally_avoidance"),
        rally_threshold=float(row.get("e_rally_threshold", 0.01)),
        event_only=_opt_bool("e_event_only"),
    )

    rt = RegimeThrottle(
        enabled=_opt_bool("r_enabled"),
        high_vix_threshold=float(row.get("r_high_vix_threshold", 30.0)),
        high_vix_multiplier=float(row.get("r_high_vix_multiplier", 0.5)),
        extreme_vix_threshold=float(row.get("r_extreme_vix_threshold", 40.0)),
        big_drop_threshold=float(row.get("r_big_drop_threshold", -0.02)),
        big_drop_multiplier=float(row.get("r_big_drop_multiplier", 0.5)),
        consecutive_loss_days=int(row.get("r_consecutive_loss_days", 3)),
        consecutive_loss_multiplier=float(row.get("r_consecutive_loss_multiplier", 0.5)),
    )
    return FullConfig(portfolio=pc, trading=tc, event=ec, regime=rt)
def run_analysis(results_csv: Path) -> pd.DataFrame:
    """Load optimizer CSV and print full analysis (A1-A4).

    Parameters
    ----------
    results_csv : Path to the optimizer results CSV.

    Returns
    -------
    The Pareto-frontier DataFrame (also exported to CSV).
    """
    rdf = pd.read_csv(results_csv)
    print(f"\nLoaded {len(rdf):,} optimizer results from {results_csv}")

    # A4: Dedup
    dedup, orig, unique = _deduplicate_results(rdf)
    print(f"  {unique} unique result profiles out of {orig} configs "
          f"({orig - unique} duplicates)")

    # A1: Parameter importance
    _parameter_importance(dedup)

    # Filter out configs with too few trades for statistical significance
    min_trades = 30
    before_len = len(dedup)
    dedup = dedup[dedup["total_trades"] >= min_trades]
    if len(dedup) < before_len:
        logger.info(
            "Filtered %d configs with < %d total trades",
            before_len - len(dedup), min_trades,
        )

    # A2: Pareto frontier (drop rows with missing Sharpe/DD to avoid NaN in dominance check)
    dedup = dedup.dropna(subset=["sharpe", "max_dd_pct"])
    pareto = extract_pareto_frontier(dedup)
    _print_pareto(pareto)
    PARETO_CSV.parent.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(PARETO_CSV, index=False)
    print(f"\n  Pareto frontier exported to {PARETO_CSV}")

    return pareto
WALKFORWARD_WINDOWS = [
    {
        "name": "W1",
        "train_start": "2025-03-01", "train_end": "2025-08-31",
        "test_start": "2025-09-01", "test_end": "2025-11-30",
    },
    {
        "name": "W2",
        "train_start": "2025-06-01", "train_end": "2025-11-30",
        "test_start": "2025-12-01", "test_end": "2026-02-28",
    },
    {
        "name": "W3",
        "train_start": "2025-09-01", "train_end": "2026-02-28",
        "test_start": "2026-03-01", "test_end": "2026-04-30",
    },
]
def generate_auto_windows(
    min_day: str,
    max_day: str,
    *,
    train_months: int = 3,
    test_months: int = 1,
    step_months: int = 1,
) -> list[dict[str, str]]:
    """Auto-generate rolling walk-forward windows from a data date range.

    Produces non-overlapping test windows, each preceded by a contiguous
    training window.  Windows that would extend beyond the data range are
    truncated.  Windows with less than one month of training or test data
    are dropped.

    Parameters
    ----------
    min_day : str
        Earliest date in the dataset (ISO format ``YYYY-MM-DD``).
    max_day : str
        Latest date in the dataset (ISO format ``YYYY-MM-DD``).
    train_months : int
        Length of each training window in months.
    test_months : int
        Length of each test window in months.
    step_months : int
        Stride between successive windows in months.

    Returns
    -------
    list[dict[str, str]]
        List of window dicts with keys ``name``, ``train_start``,
        ``train_end``, ``test_start``, ``test_end``.
    """
    from dateutil.relativedelta import relativedelta

    start = pd.Timestamp(min_day).date()
    end = pd.Timestamp(max_day).date()

    windows: list[dict[str, str]] = []
    cursor = start
    idx = 1

    while True:
        train_start = cursor
        train_end_raw = cursor + relativedelta(months=train_months) - timedelta(days=1)
        test_start = train_end_raw + timedelta(days=1)
        test_end_raw = test_start + relativedelta(months=test_months) - timedelta(days=1)

        # Truncate to data range
        train_end = min(train_end_raw, end)
        test_end = min(test_end_raw, end)

        # Need at least 30 days of train and 15 days of test
        if (train_end - train_start).days < 30:
            break
        if test_start > end or (test_end - test_start).days < 15:
            break

        windows.append({
            "name": f"Auto-W{idx}",
            "train_start": str(train_start),
            "train_end": str(train_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
        })
        idx += 1
        cursor += relativedelta(months=step_months)

        if cursor >= end:
            break

    return windows
def walkforward_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training candidates into train/test by date range.

    Parameters
    ----------
    df : Full training candidates DataFrame with ``day`` column.
    train_start, train_end : Inclusive date boundaries for training slice.
    test_start, test_end : Inclusive date boundaries for test slice.

    Returns
    -------
    Tuple of (train_df, test_df).
    """
    train = df[(df["day"] >= train_start) & (df["day"] <= train_end)]
    test = df[(df["day"] >= test_start) & (df["day"] <= test_end)]
    return train, test
def _run_window_optimizer(
    df_slice: pd.DataFrame,
    top_n: int = 10,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Run the full optimizer grid on a data slice and return top-N rows.

    Parameters
    ----------
    df_slice : Subset of training candidates for this window.
    top_n : Number of top configs to return.
    config_path : Optional YAML grid config. Uses YAML grid when provided,
        falls back to the hardcoded ``_build_optimizer_grid()`` otherwise.

    Returns
    -------
    DataFrame with top-N rows by Sharpe, including all config columns.
    """
    daily_signals = precompute_daily_signals(df_slice)
    if config_path is not None:
        from configs.optimizer.schema import build_configs_from_yaml
        configs = build_configs_from_yaml(config_path)
        if not configs:
            logger.warning("YAML grid %s produced zero configs — returning empty DataFrame", config_path)
            return pd.DataFrame()
        tp_vals = sorted({c.trading.tp_pct for c in configs})
        sl_vals = sorted({c.trading.sl_mult for c in configs}, key=lambda x: x or 0)
        precompute_pnl_columns(df_slice, tp_vals, sl_vals)
    else:
        configs = _build_optimizer_grid()

    precomp_cache: dict[tuple, dict] = {}
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        key = (cfg.trading.width_filter, cfg.trading.entry_count)
        if key not in precomp_cache:
            precomp_cache[key] = _precompute_day_selections(
                df_slice, width_filter=key[0], entry_count=key[1],
            )
        result = run_backtest(df_slice, daily_signals, cfg, day_precomp=precomp_cache[key])
        row = cfg.flat_dict()
        row.update({
            "final_equity": result.final_equity,
            "return_pct": result.total_return_pct,
            "ann_return_pct": result.annualised_return_pct,
            "max_dd_pct": result.max_drawdown_pct,
            "trough": result.trough,
            "sharpe": result.sharpe,
            "total_trades": result.total_trades,
            "days_traded": result.days_traded,
            "days_stopped": result.days_stopped,
            "win_days": result.win_days,
            "win_rate": result.win_days / max(result.days_traded, 1),
        })
        row.update(result.regime_metrics)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df.sort_values("sharpe", ascending=False).head(top_n).reset_index(drop=True)
def _config_signature(row: pd.Series) -> str:
    """Build a short human-readable label for a config row."""
    trades = int(row.get("p_max_trades_per_day", 0))
    stop = row.get("p_monthly_drawdown_limit")
    stop_s = f"{stop:.0%}" if pd.notna(stop) else "none"
    lots = "grad" if row.get("p_lot_per_equity", 0) < 100_000 else "fixed"
    calls = "C" if row.get("p_calls_only") else "B"
    evt = "E" if row.get("e_enabled") else ""
    _bm_raw = row.get("e_budget_mode", "") if row.get("e_enabled") else ""
    bm = _bm_raw[0].upper() if _bm_raw else ""
    return f"{trades}/d|{stop_s}|{lots}|{calls}{evt}{bm}"
def _config_key(row: pd.Series) -> tuple:
    """Build a hashable key from the config columns of a results row.

    NaN values are normalized to None so that two otherwise-identical
    configs always produce equal keys (``NaN != NaN`` in Python).
    """
    return tuple(
        _opt_val(row.get(c)) for c in [
            "p_starting_capital", "p_max_trades_per_day", "p_monthly_drawdown_limit",
            "p_lot_per_equity", "p_max_equity_risk_pct", "p_max_margin_pct",
            "p_calls_only", "p_min_dte", "p_max_delta",
            "t_tp_pct", "t_sl_mult", "t_max_vix", "t_max_term_structure",
            "t_avoid_opex", "t_prefer_event_days", "t_width_filter", "t_entry_count",
            "e_enabled", "e_signal_mode", "e_budget_mode", "e_max_event_trades",
            "e_spx_drop_threshold", "e_spx_drop_2d_threshold",
            "e_spx_drop_min", "e_spx_drop_max",
            "e_vix_spike_threshold", "e_vix_elevated_threshold",
            "e_term_inversion_threshold",
            "e_side_preference", "e_min_dte", "e_max_dte",
            "e_min_delta", "e_max_delta", "e_rally_avoidance", "e_rally_threshold",
            "e_event_only",
            "r_enabled", "r_high_vix_threshold", "r_high_vix_multiplier",
            "r_extreme_vix_threshold", "r_big_drop_threshold", "r_big_drop_multiplier",
            "r_consecutive_loss_days", "r_consecutive_loss_multiplier",
        ]
    )
def run_walkforward(
    df: pd.DataFrame,
    output_csv: Path = WALKFORWARD_CSV,
    top_n: int = 10,
    *,
    windows: list[dict[str, str]] | None = None,
    config_path: str | None = None,
) -> pd.DataFrame:
    """Run rolling walk-forward validation across all windows.

    For each window, runs the full optimizer on the train slice, then
    evaluates the top-N train configs on the test slice (out-of-sample).

    Parameters
    ----------
    df : Full training candidates DataFrame.
    output_csv : Path to export detailed results.
    top_n : Number of top configs per window.
    windows : Optional custom window list.  When ``None`` the hard-coded
        ``WALKFORWARD_WINDOWS`` are used.  Pass the output of
        ``generate_auto_windows()`` for data-driven window placement.
    config_path : Optional YAML grid config for the optimizer. When
        provided, walk-forward uses the same YAML grid as ``--optimize
        --config``. When ``None``, uses the hardcoded exhaustive grid.

    Returns
    -------
    DataFrame with per-window, per-config train vs test comparison.
    """
    effective_windows = windows if windows is not None else WALKFORWARD_WINDOWS
    all_rows: list[dict[str, Any]] = []
    config_appearances: dict[tuple, list[str]] = {}

    data_min = df["day"].min()
    data_max = df["day"].max()
    any_overlap = any(
        w["train_start"] <= data_max and w["test_end"] >= data_min
        for w in effective_windows
    )
    if not any_overlap:
        print(f"\n  WARNING: None of the {len(effective_windows)} walk-forward windows "
              f"overlap with data range [{data_min} .. {data_max}]. "
              f"Consider using --wf-auto to auto-generate windows.", flush=True)

    for window in effective_windows:
        wname = window["name"]
        print(f"\n{'=' * 100}")
        print(f"  WALK-FORWARD {wname}: "
              f"Train {window['train_start']} to {window['train_end']}  |  "
              f"Test {window['test_start']} to {window['test_end']}")
        print(f"{'=' * 100}")

        train_df, test_df = walkforward_split(
            df, window["train_start"], window["train_end"],
            window["test_start"], window["test_end"],
        )
        train_days = train_df["day"].nunique()
        test_days = test_df["day"].nunique()
        print(f"  Train: {len(train_df):,} candidates, {train_days} days")
        print(f"  Test:  {len(test_df):,} candidates, {test_days} days")

        if train_days < 10 or test_days < 5:
            print("  SKIP: insufficient data in this window")
            continue

        config_source = f"yaml:{Path(config_path).name}" if config_path else "exhaustive"
        print(f"  Optimizing on train set (source={config_source}) ...")
        t0 = time.time()
        top_train = _run_window_optimizer(train_df, top_n=top_n, config_path=config_path)
        train_elapsed = time.time() - t0
        print(f"  Train optimization done in {train_elapsed:.0f}s")

        test_signals = precompute_daily_signals(test_df)
        test_precomp_cache: dict[tuple, dict] = {}

        print(f"\n  {'#':>3} {'Config':<30} {'Tr Sharpe':>10} {'Te Sharpe':>10} "
              f"{'Tr Ret':>8} {'Te Ret':>8} {'Tr DD':>6} {'Te DD':>6} {'Verdict':>10}")
        print("  " + "-" * 100)

        for rank, (_, trow) in enumerate(top_train.iterrows(), 1):
            cfg = _row_to_config(trow)
            tpc_key = (cfg.trading.width_filter, cfg.trading.entry_count)
            if tpc_key not in test_precomp_cache:
                test_precomp_cache[tpc_key] = _precompute_day_selections(
                    test_df, width_filter=tpc_key[0], entry_count=tpc_key[1],
                )
            test_result = run_backtest(test_df, test_signals, cfg, day_precomp=test_precomp_cache[tpc_key])

            train_sharpe = float(trow["sharpe"])
            test_sharpe = test_result.sharpe
            train_ret = float(trow["return_pct"])
            test_ret = test_result.total_return_pct
            train_dd = float(trow["max_dd_pct"])
            test_dd = test_result.max_drawdown_pct

            # Verdict: how well does train performance predict test
            if test_sharpe > 0 and test_ret > 0:
                if train_sharpe <= 0:
                    verdict = "WEAK"
                else:
                    verdict = "PASS" if test_sharpe >= train_sharpe * 0.3 else "WEAK"
            elif test_ret > 0:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"

            sig = _config_signature(trow)
            key = _config_key(trow)
            if key not in config_appearances:
                config_appearances[key] = []
            config_appearances[key].append(wname)

            print(f"  {rank:>3} {sig:<30} {train_sharpe:>10.2f} {test_sharpe:>10.2f} "
                  f"{train_ret:>+7.0f}% {test_ret:>+7.0f}% "
                  f"{train_dd:>5.0f}% {test_dd:>5.0f}% {verdict:>10}")

            row_out = trow.to_dict()
            row_out.update({
                "window": wname,
                "rank_in_window": rank,
                "train_sharpe": train_sharpe,
                "test_sharpe": test_sharpe,
                "train_return_pct": train_ret,
                "test_return_pct": test_ret,
                "train_max_dd_pct": train_dd,
                "test_max_dd_pct": test_dd,
                "verdict": verdict,
                "config_source": config_source,
            })
            all_rows.append(row_out)

    # Cross-window summary
    print(f"\n{'=' * 100}")
    print("  CROSS-WINDOW SUMMARY: configs appearing in top-10 across multiple windows")
    print(f"{'=' * 100}")

    multi_window = {k: v for k, v in config_appearances.items() if len(v) > 1}
    if multi_window:
        wf_df = pd.DataFrame(all_rows)
        print(f"\n  {len(multi_window)} configs appear in 2+ windows:\n")

        for key, windows in sorted(multi_window.items(), key=lambda x: -len(x[1])):
            matches = wf_df[wf_df.apply(lambda r: _config_key(r) == key, axis=1)]
            if matches.empty:
                continue
            sig = _config_signature(matches.iloc[0])
            avg_test_sharpe = matches["test_sharpe"].mean()
            avg_test_ret = matches["test_return_pct"].mean()
            wlist = ",".join(windows)
            print(f"  {sig:<35} windows=[{wlist}]  "
                  f"avg_test_sharpe={avg_test_sharpe:.2f}  "
                  f"avg_test_return={avg_test_ret:+.0f}%")
    else:
        print("\n  No config appeared in the top-10 of more than one window.")
        print("  This suggests high regime-dependence; consider simpler / more robust configs.")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(output_csv, index=False)
    if results_df.empty:
        print(f"\n  Walk-forward: no valid windows produced results. Empty CSV at {output_csv}")
    else:
        print(f"\n  Walk-forward results exported to {output_csv}")

    return results_df
def run_holdout_evaluation(
    results_df: pd.DataFrame,
    df_holdout: pd.DataFrame,
    output_dir: Path,
    top_n: int = 5,
) -> pd.DataFrame:
    """Evaluate top optimizer configs on a held-out data slice.

    Picks the top-N configs by Sharpe from ``results_df``, runs each on
    ``df_holdout``, and writes a comparison CSV showing development vs
    holdout performance with a degradation percentage.

    Ranking uses ``test_sharpe`` (out-of-sample Sharpe from walk-forward
    windows) when that column exists in ``results_df``.  This gives a more
    realistic ranking than in-sample ``sharpe``.  When ``test_sharpe`` is
    absent (e.g. plain ``--optimize`` runs), ``sharpe`` is used instead.

    Parameters
    ----------
    results_df : Optimizer results (from --optimize or --walkforward).
    df_holdout : Held-out candidates DataFrame (never seen during optimization).
    output_dir : Directory to write ``holdout_results.csv``.
    top_n : Number of top configs to evaluate.

    Returns
    -------
    DataFrame with development + holdout metrics for each config.
    """
    if results_df.empty or df_holdout.empty:
        print("  Holdout: skipped (empty results or holdout data)")
        return pd.DataFrame()

    holdout_signals = precompute_daily_signals(df_holdout)
    holdout_days = df_holdout["day"].nunique()
    print(f"\n{'=' * 100}")
    print(f"  HOLDOUT EVALUATION: {holdout_days} trading days, "
          f"{len(df_holdout):,} candidates")
    print(f"{'=' * 100}")

    # Use test_sharpe (OOS) for ranking when available (walkforward results),
    # fall back to sharpe (train/full-sample) for optimizer results
    rank_col = "test_sharpe" if "test_sharpe" in results_df.columns else "sharpe"

    # Deduplicate: when walk-forward produces the same config across multiple
    # windows, keep only the row with the best ranking metric per unique config.
    # Uses _config_key (NaN-normalized tuple) for consistent identity.
    results_df = results_df.copy()
    results_df["_dedup_key"] = results_df.apply(_config_key, axis=1)
    deduped = (
        results_df
        .sort_values(rank_col, ascending=False)
        .drop_duplicates(subset=["_dedup_key"], keep="first")
        .drop(columns=["_dedup_key"])
    )
    top = deduped.nlargest(top_n, rank_col)
    precomp_cache: dict[tuple, dict] = {}
    rows: list[dict[str, Any]] = []

    print(f"\n  {'#':>3} {'Config':<30} {'Dev Sharpe':>11} {'HO Sharpe':>10} "
          f"{'Dev Ret':>8} {'HO Ret':>8} {'Degrade':>8}")
    print("  " + "-" * 90)

    for rank, (_, trow) in enumerate(top.iterrows(), 1):
        cfg = _row_to_config(trow)
        tpc_key = (cfg.trading.width_filter, cfg.trading.entry_count)
        if tpc_key not in precomp_cache:
            precomp_cache[tpc_key] = _precompute_day_selections(
                df_holdout, width_filter=tpc_key[0], entry_count=tpc_key[1],
            )
        ho_result = run_backtest(
            df_holdout, holdout_signals, cfg,
            day_precomp=precomp_cache[tpc_key],
        )

        dev_sharpe = float(trow["sharpe"])
        ho_sharpe = ho_result.sharpe
        dev_ret = float(trow["return_pct"])
        ho_ret = ho_result.total_return_pct

        degradation = (1 - ho_sharpe / dev_sharpe) * 100 if dev_sharpe > 0 else 0

        sig = _config_signature(trow)
        print(f"  {rank:>3} {sig:<30} {dev_sharpe:>11.2f} {ho_sharpe:>10.2f} "
              f"{dev_ret:>+7.0f}% {ho_ret:>+7.0f}% {degradation:>+7.0f}%")

        row_out = trow.to_dict()
        row_out.update({
            "dev_sharpe": dev_sharpe,
            "dev_return_pct": dev_ret,
            "dev_max_dd_pct": float(trow.get("max_dd_pct", 0)),
            "holdout_sharpe": ho_sharpe,
            "holdout_return_pct": ho_ret,
            "holdout_max_dd_pct": ho_result.max_drawdown_pct,
            "holdout_trades": ho_result.total_trades,
            "holdout_days": holdout_days,
            "degradation_pct": round(degradation, 1),
        })
        row_out.update({f"ho_{k}": v for k, v in ho_result.regime_metrics.items()})
        rows.append(row_out)

    holdout_df = pd.DataFrame(rows)
    ho_csv = output_dir / "holdout_results.csv"
    holdout_df.to_csv(ho_csv, index=False)
    print(f"\n  Holdout results exported to {ho_csv}")
    return holdout_df
