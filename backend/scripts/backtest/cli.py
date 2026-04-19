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
    compute_effective_pnl,
    precompute_daily_signals,
    precompute_pnl_columns,
    run_backtest,
)
from .optimizer import (
    _run_grid,
    run_event_only_optimizer,
    run_optimizer,
    run_selective_optimizer,
    run_staged_optimizer,
)
from .analysis import (
    ANALYSIS_PARAMS,
    PARETO_CSV,
    WALKFORWARD_CSV,
    WALKFORWARD_WINDOWS,
    _build_comparison_configs,
    _robustness_check,
    _row_to_config,
    extract_pareto_frontier,
    generate_auto_windows,
    print_comparison_table,
    print_monthly,
    print_optimizer_top,
    print_summary,
    run_analysis,
    run_holdout_evaluation,
    run_walkforward,
    walkforward_split,
)


def _alignment_warn_against_live_settings() -> None:
    """Print a banner if backtest defaults diverge from live settings.

    Backtest dataclasses (``TradingConfig``, ``PortfolioConfig``,
    ``EventConfig``) intentionally fix their own defaults so the script
    is reproducible without env vars; live behaviour is governed by
    ``spx_backend.config.settings`` (env-driven).  Silent divergence
    between the two is an attractive nuisance — the optimizer can sweep
    a parameter that production never enables and pick a config that
    looks great offline but is unreachable live.

    This guard reads the live settings (best-effort; settings module
    may not import in all CI sandboxes) and logs a single banner
    listing every parameter whose backtest dataclass default differs
    from the live setting default.  It does NOT raise — operators can
    still intentionally diverge for ablation studies; the message just
    surfaces what's happening.

    See M10 in OFFLINE_PIPELINE_AUDIT.md for the originating finding.
    """
    try:
        from spx_backend.config import settings as _live
    except Exception as exc:  # pragma: no cover -- import guard, env-dep
        logger.debug("Live-settings alignment skipped (settings unavailable: %s)", exc)
        return

    # (label, backtest-default-value, live-equivalent-value)
    pairs: list[tuple[str, Any, Any]] = [
        ("avoid_opex",         TradingConfig().avoid_opex,
                               getattr(_live, "decision_avoid_opex", None)),
        ("max_margin_pct",     PortfolioConfig().max_margin_pct,
                               getattr(_live, "portfolio_max_margin_pct", None)),
        ("max_equity_risk_pct", PortfolioConfig().max_equity_risk_pct,
                                getattr(_live, "portfolio_max_equity_risk_pct", None)),
        ("starting_capital",   PortfolioConfig().starting_capital,
                               getattr(_live, "portfolio_starting_capital", None)),
        ("monthly_drawdown_limit", PortfolioConfig().monthly_drawdown_limit,
                                   getattr(_live, "portfolio_monthly_drawdown_limit", None)),
    ]

    diverged = [(name, bt, live) for name, bt, live in pairs
                if live is not None and bt != live]
    if not diverged:
        return

    banner = ["", "─" * 72,
              "M10 alignment notice: backtest defaults diverge from live settings:"]
    for name, bt, live in diverged:
        banner.append(f"  {name:30s} backtest={bt!r:>10}   live={live!r}")
    banner.append("(optimizer YAMLs may still override these; see M10 in "
                  "OFFLINE_PIPELINE_AUDIT.md)")
    banner.append("─" * 72)
    for line in banner:
        logger.info(line)
def main() -> None:
    """Entry point for the backtest CLI."""
    parser = argparse.ArgumentParser(
        description="Capital-budgeted backtest with scheduled + event-driven layers",
    )
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help="Path to training_candidates.csv")
    parser.add_argument("--capital", type=float, default=20_000)
    parser.add_argument("--max-trades", type=int, default=2)
    parser.add_argument("--monthly-stop", type=float, default=0.15,
                        help="Monthly drawdown stop (0 to disable)")
    parser.add_argument("--lot-per-equity", type=float, default=10_000)
    parser.add_argument("--both-sides", action="store_true", default=False)
    parser.add_argument("--min-dte", type=int, default=None)
    parser.add_argument("--max-delta", type=float, default=None)

    # Event params
    parser.add_argument("--event", action="store_true", default=False,
                        help="Enable event-driven layer")
    parser.add_argument("--event-budget", type=str, default="shared",
                        choices=["shared", "separate"])
    parser.add_argument("--event-max-trades", type=int, default=1)
    parser.add_argument("--event-drop-threshold", type=float, default=-0.01)
    parser.add_argument("--event-side", type=str, default="puts",
                        choices=["puts", "calls", "best"])
    parser.add_argument("--event-min-dte", type=int, default=5)
    parser.add_argument("--event-max-dte", type=int, default=7)
    parser.add_argument("--event-min-delta", type=float, default=0.15)
    parser.add_argument("--event-max-delta", type=float, default=0.25)
    parser.add_argument("--rally-avoid", action="store_true", default=False)
    parser.add_argument("--rally-threshold", type=float, default=0.01)

    # Modes
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Run preset comparison configs")
    parser.add_argument("--optimize", action="store_true", default=False,
                        help="Run exhaustive grid-search optimizer (original mode)")
    parser.add_argument("--optimize-staged", action="store_true", default=False,
                        help="Run 3-stage optimizer: trading -> portfolio -> event "
                             "(requires multi-TP trajectory columns)")
    parser.add_argument("--optimize-event-only", action="store_true", default=False,
                        help="Run event-only optimizer: sweep SPX-drop strategies "
                             "that only trade on drop days (appends to results CSV)")
    parser.add_argument("--optimize-selective", action="store_true", default=False,
                        help="Run selective high-win-rate optimizer: conservative "
                             "params targeting 90%+ day win rate (appends to results CSV)")
    parser.add_argument("--analyze", action="store_true", default=False,
                        help="Deep-dive optimizer results (param importance, Pareto, robustness)")
    parser.add_argument("--walkforward", action="store_true", default=False,
                        help="Rolling walk-forward validation (train/test splits)")
    parser.add_argument("--wf-auto", action="store_true", default=False,
                        help="Auto-generate walk-forward windows from data date range")
    parser.add_argument("--wf-train-months", type=int, default=3,
                        help="Walk-forward train window in months (default: 3)")
    parser.add_argument("--wf-test-months", type=int, default=1,
                        help="Walk-forward test window in months (default: 1)")
    parser.add_argument("--wf-step-months", type=int, default=1,
                        help="Walk-forward step size in months (default: 1)")
    parser.add_argument("--top-n-trading", type=int, default=10,
                        help="Staged optimizer: top-N trading configs to carry into Stage 2 (default: 10)")
    parser.add_argument("--top-n-combined", type=int, default=5,
                        help="Staged optimizer: top-N combined configs to carry into Stage 3 (default: 5)")
    parser.add_argument("--output-csv", type=str, default=str(RESULTS_CSV),
                        help="CSV output path for optimizer results")
    parser.add_argument(
        "--walkforward-output-csv", type=str, default=None,
        help="Explicit CSV output path for walk-forward results. "
             "Defaults to <output-csv parent>/walkforward_results.csv "
             "for backward compat; pass an explicit per-run path "
             "(e.g. data/<run_name>_walkforward.csv) to avoid clobbering "
             "concurrent or sequential pipeline runs (see H4 in "
             "OFFLINE_PIPELINE_AUDIT.md).",
    )
    parser.add_argument("--backtest-workers", type=int,
                        default=max(1, (cpu_count() or 2) - 1),
                        help="Number of parallel workers for optimizer grid "
                             "(1 = sequential, default = cpu_count - 1)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML grid definition file "
                             "(overrides the hardcoded grid for --optimize)")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for the experiment run (enables experiment tracking)")
    parser.add_argument("--no-track", action="store_true", default=False,
                        help="Disable experiment tracking even when --experiment-name is set")
    parser.add_argument("--git-tag", action="store_true", default=False,
                        help="Create a git tag for the experiment run")

    parser.add_argument("--holdout-months", type=int, default=0,
                        help="Reserve the most recent N months as a true out-of-sample holdout "
                             "(0 = disabled, default). Data in the holdout is excluded from "
                             "--optimize and --walkforward, then the top configs are "
                             "automatically evaluated on it.")
    parser.add_argument("--holdout-end", type=str, default=None,
                        help="Explicit end date for the holdout window (YYYY-MM-DD). "
                             "Defaults to the last day in the data.")
    parser.add_argument("--holdout-top-n", type=int, default=5,
                        help="Number of top configs to evaluate on the holdout set (default: 5)")

    args = parser.parse_args()
    args.backtest_workers = max(1, args.backtest_workers)

    # M10 alignment guard: warn (don't fail) if backtest defaults
    # diverge from the live decision job's behaviour.  The audit
    # originally flagged a divergence in `avoid_opex` defaults; today
    # both sides default to False, but the optimizer YAML grids
    # frequently set it to True.  Surfacing this as a startup line
    # makes it obvious to operators which calendar gating actually
    # ran.  See M10 in OFFLINE_PIPELINE_AUDIT.md.
    _alignment_warn_against_live_settings()

    # --analyze only needs the results CSV, not the full training data
    if args.analyze:
        results_path = Path(args.output_csv)
        if not results_path.exists():
            logger.error("Results CSV not found: %s. Run --optimize first to generate it.", results_path)
            sys.exit(1)
        run_analysis(results_path)

        csv_path = Path(args.csv)
        if csv_path.exists():
            print("\nLoading training data for robustness check ...")
            df = pd.read_csv(csv_path)
            daily_signals = precompute_daily_signals(df)
            rdf = pd.read_csv(results_path)
            _robustness_check(df, daily_signals, rdf, top_n=20)
        else:
            print("\n  (skipping robustness check -- training CSV not found)")
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Validate required and optional columns
    required_cols = {"day", "entry_credit", "realized_pnl", "spread_side",
                     "dte_target", "delta_target", "credit_to_width", "spot", "vix",
                     "vix9d", "term_structure"}
    optional_cols = {"width_points", "is_opex_day", "is_fomc_day", "is_nfp_day",
                     "is_cpi_day", "hold_realized_pnl", "recovered_after_sl",
                     "hold_hit_tp50", "exit_reason", "final_pnl_at_expiry"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        logger.error("CSV missing required columns: %s", sorted(missing_required))
        sys.exit(2)
    missing_optional = optional_cols - set(df.columns)
    if missing_optional:
        logger.warning("CSV missing optional columns (will be ignored): %s", sorted(missing_optional))

    print(f"  {len(df):,} candidates across {df['day'].nunique()} trading days")
    print(f"  Period: {df['day'].min()} to {df['day'].max()}")

    # -- Holdout split --
    df_holdout = pd.DataFrame()
    if args.holdout_months > 0:
        from dateutil.relativedelta import relativedelta
        from datetime import datetime as _dt

        holdout_end_str = args.holdout_end or df["day"].max()
        holdout_end = _dt.strptime(holdout_end_str, "%Y%m%d") if len(holdout_end_str) == 8 \
            else _dt.strptime(holdout_end_str, "%Y-%m-%d")
        holdout_start = holdout_end - relativedelta(months=args.holdout_months)
        ho_start_str = holdout_start.strftime("%Y-%m-%d")
        ho_end_str = holdout_end.strftime("%Y-%m-%d")

        df_holdout = df[(df["day"] >= ho_start_str) & (df["day"] <= ho_end_str)].copy()
        df = df[df["day"] < ho_start_str].copy()

        ho_days = df_holdout["day"].nunique()
        dev_days = df["day"].nunique()
        print(f"\n  HOLDOUT SPLIT: reserving {ho_days} trading days "
              f"({ho_start_str} to {ho_end_str})")
        print(f"  Development set: {dev_days} trading days, "
              f"{len(df):,} candidates")
        print(f"  Holdout set:     {ho_days} trading days, "
              f"{len(df_holdout):,} candidates\n")

    print("Precomputing daily signals ...")
    daily_signals = precompute_daily_signals(df)

    if args.walkforward:
        wf_windows = None
        if args.wf_auto:
            min_day = df["day"].min()
            max_day = df["day"].max()
            wf_windows = generate_auto_windows(
                min_day, max_day,
                train_months=args.wf_train_months,
                test_months=args.wf_test_months,
                step_months=args.wf_step_months,
            )
            print(f"Auto-generated {len(wf_windows)} walk-forward windows from {min_day} to {max_day}")
        if args.walkforward_output_csv:
            wf_output = Path(args.walkforward_output_csv)
        else:
            wf_output = Path(args.output_csv).parent / "walkforward_results.csv"
        wf_results = run_walkforward(df, output_csv=wf_output, windows=wf_windows,
                                     config_path=args.config)

        if not df_holdout.empty and not wf_results.empty:
            run_holdout_evaluation(
                wf_results, df_holdout,
                output_dir=Path(args.output_csv).parent,
                top_n=args.holdout_top_n,
            )

    elif args.optimize_staged or args.optimize_event_only or args.optimize_selective or args.optimize:
        # Determine optimizer mode
        if args.optimize_staged:
            mode = "staged"
        elif args.optimize_event_only:
            mode = "event-only"
        elif args.optimize_selective:
            mode = "selective"
        else:
            mode = "yaml-config" if args.config else "exhaustive"

        # Set up experiment tracking
        tracker = None
        if args.experiment_name and not args.no_track:
            from experiment_tracker import CsvExperimentTracker
            tracker = CsvExperimentTracker()
            tracker.start_run(args.experiment_name, {
                "mode": mode,
                "csv": args.csv,
                "output_csv": args.output_csv,
                "workers": args.backtest_workers,
                "config_file": args.config,
            })

        # Run the selected optimizer (with tracker cleanup on failure)
        try:
            if args.optimize_staged:
                results_df = run_staged_optimizer(
                    df, daily_signals, Path(args.output_csv),
                    top_n_trading=args.top_n_trading,
                    top_n_combined=args.top_n_combined,
                    num_workers=args.backtest_workers,
                )
            elif args.optimize_event_only:
                results_df = run_event_only_optimizer(
                    df, daily_signals, Path(args.output_csv),
                    num_workers=args.backtest_workers,
                )
            elif args.optimize_selective:
                results_df = run_selective_optimizer(
                    df, daily_signals, Path(args.output_csv),
                    num_workers=args.backtest_workers,
                )
            elif args.config:
                from configs.optimizer.schema import build_configs_from_yaml
                configs = build_configs_from_yaml(args.config)
                if not configs:
                    print(f"\n  ERROR: YAML grid {args.config} produced zero configs — nothing to optimize.",
                          flush=True)
                    results_df = pd.DataFrame()
                else:
                    tp_vals = sorted({c.trading.tp_pct for c in configs})
                    sl_vals = sorted({c.trading.sl_mult for c in configs}, key=lambda x: x or 0)
                    precompute_pnl_columns(df, tp_vals, sl_vals)
                    results_df = _run_grid(configs, df, daily_signals, "YAML-Grid",
                                           num_workers=args.backtest_workers)
                    results_df.sort_values("sharpe", ascending=False, ignore_index=True, inplace=True)
                    results_df.to_csv(Path(args.output_csv), index=False)
                    print(f"\n  Results exported to {args.output_csv}", flush=True)
            else:
                results_df = run_optimizer(df, daily_signals, Path(args.output_csv),
                                           num_workers=args.backtest_workers)

            if not results_df.empty:
                for metric in ["sharpe", "return_pct", "max_dd_pct"]:
                    print_optimizer_top(results_df, metric)

            if not df_holdout.empty and not results_df.empty:
                run_holdout_evaluation(
                    results_df, df_holdout,
                    output_dir=Path(args.output_csv).parent,
                    top_n=args.holdout_top_n,
                )

            # Finalize experiment tracking
            if tracker is not None:
                tracker.log_metric("num_configs", len(results_df))
                if not results_df.empty:
                    tracker.log_metric("best_sharpe", float(results_df["sharpe"].max()))
                    tracker.log_metric("best_return_pct", float(results_df["return_pct"].max()))
                    tracker.log_metric("best_win_rate", float(results_df["win_rate"].max()))
                output_path = Path(args.output_csv)
                if output_path.exists():
                    tracker.log_artifact(output_path, "results.csv")
                if args.config:
                    tracker.log_artifact(Path(args.config), "config.yaml")
                top_n = results_df.nlargest(10, "sharpe") if not results_df.empty else results_df
                tracker.log_summary({
                    "mode": mode,
                    "total_configs": len(results_df),
                    "top_10_by_sharpe": top_n.to_dict("records"),
                })
                if args.git_tag:
                    tracker.create_git_tag()
                run_dir = tracker.current_run_dir
                tracker.end_run()
                print(f"\n  Experiment tracked: {run_dir}", flush=True)
        except BaseException:
            if tracker is not None:
                tracker.end_run(status="failed")
            raise

    elif args.compare:
        presets = _build_comparison_configs()
        results = []
        for label, cfg in presets:
            r = run_backtest(df, daily_signals, cfg, label)
            results.append(r)
        print_comparison_table(results)
        print("\n\nDetailed monthly breakdown for recommended config:")
        print_summary(results[0])
        print_monthly(results[0])

    else:
        pc = PortfolioConfig(
            starting_capital=args.capital,
            max_trades_per_day=args.max_trades,
            monthly_drawdown_limit=args.monthly_stop if args.monthly_stop > 0 else None,
            lot_per_equity=args.lot_per_equity,
            calls_only=not args.both_sides,
            min_dte=args.min_dte,
            max_delta=args.max_delta,
        )
        evc = EventConfig(
            enabled=args.event,
            budget_mode=args.event_budget,
            max_event_trades=args.event_max_trades,
            spx_drop_threshold=args.event_drop_threshold,
            side_preference=args.event_side,
            min_dte=args.event_min_dte,
            max_dte=args.event_max_dte,
            min_delta=args.event_min_delta,
            max_delta=args.event_max_delta,
            rally_avoidance=args.rally_avoid,
            rally_threshold=args.rally_threshold,
        )
        cfg = FullConfig(portfolio=pc, event=evc)

        stop_label = f"{pc.monthly_drawdown_limit:.0%} stop" if pc.monthly_drawdown_limit else "no stop"
        lots_label = "gradual" if pc.lot_per_equity < 100_000 else "fixed"
        event_label = " + events" if evc.enabled else ""
        label = (f"${pc.starting_capital/1000:.0f}k | "
                 f"{pc.max_trades_per_day}/day | "
                 f"{stop_label} | {lots_label}{event_label}")

        result = run_backtest(df, daily_signals, cfg, label)
        print_summary(result)
        print_monthly(result)
