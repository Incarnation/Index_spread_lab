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
    _opt_val,
    _precompute_day_selections,
    _safe_bool,
    precompute_daily_signals,
    precompute_pnl_columns,
    run_backtest,
)


def _build_optimizer_grid() -> list[FullConfig]:
    """Generate all valid parameter combinations for the exhaustive search.

    Returns
    -------
    List of FullConfig objects covering the full parameter space.
    Invalid combos (e.g. event params when event is disabled) are pruned.
    """
    configs: list[FullConfig] = []

    portfolio_grid = list(itertools.product(
        [20_000],                          # starting_capital
        [1, 2, 3],                         # max_trades_per_day
        [0.10, 0.15, 0.20, None],         # monthly_drawdown_limit
        [10_000, 20_000, 999_999],         # lot_per_equity
        [True, False],                     # calls_only
        [None],                            # min_dte (None = no filter)
        [None],                            # max_delta (None = no filter)
    ))

    # No-event configs
    for cap, trades, mstop, lpe, co, mdte, mdelta in portfolio_grid:
        pc = PortfolioConfig(
            starting_capital=cap, max_trades_per_day=trades,
            monthly_drawdown_limit=mstop, lot_per_equity=lpe,
            calls_only=co, min_dte=mdte, max_delta=mdelta,
        )
        configs.append(FullConfig(portfolio=pc, event=EventConfig(enabled=False)))

    # Event configs (only meaningful subsets)
    event_grid = list(itertools.product(
        ["shared", "separate"],            # budget_mode
        [1, 2],                            # max_event_trades
        [-0.005, -0.01, -0.015],           # spx_drop_threshold
        ["puts", "best"],                  # side_preference
        [3, 5],                            # event min_dte
        [5, 7, 10],                        # event max_dte
        [0.10, 0.15],                      # event min_delta
        [0.20, 0.25],                      # event max_delta
        [True, False],                     # rally_avoidance
    ))

    # Pair event configs with a focused set of portfolio configs
    portfolio_for_events = list(itertools.product(
        [20_000],                          # starting_capital
        [2, 3],                            # max_trades_per_day
        [0.15],                            # monthly_drawdown_limit
        [10_000],                          # lot_per_equity
        [True, False],                     # calls_only
    ))

    for (cap, trades, mstop, lpe, co) in portfolio_for_events:
        for (bmode, evt_max, drop_thr, side, emin_dte, emax_dte,
             emin_d, emax_d, rally) in event_grid:

            if emin_dte > emax_dte:
                continue
            if emin_d > emax_d:
                continue
            if bmode == "shared" and evt_max > 1:
                continue  # shared mode ignores max_event_trades

            pc = PortfolioConfig(
                starting_capital=cap, max_trades_per_day=trades,
                monthly_drawdown_limit=mstop, lot_per_equity=lpe,
                calls_only=co,
            )
            evc = EventConfig(
                enabled=True, budget_mode=bmode,
                max_event_trades=evt_max,
                spx_drop_threshold=drop_thr,
                side_preference=side,
                min_dte=emin_dte, max_dte=emax_dte,
                min_delta=emin_d, max_delta=emax_d,
                rally_avoidance=rally,
            )
            configs.append(FullConfig(portfolio=pc, event=evc))

    return configs
OPTIMIZER_TP_VALUES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
OPTIMIZER_SL_VALUES: list[float | None] = [None, 1.5, 2.0, 3.0]
def _build_staged_grid_stage1() -> list[FullConfig]:
    """Stage 1: Sweep trading params with fixed portfolio baseline.

    Fixes portfolio to current best (2/day, 15% DD, gradual sizing, calls-only)
    and sweeps all TradingConfig dimensions.
    """
    configs: list[FullConfig] = []

    trading_grid = itertools.product(
        OPTIMIZER_TP_VALUES,                # tp_pct
        OPTIMIZER_SL_VALUES,                # sl_mult
        [None, 30.0, 35.0],                # max_vix
        [None, 1.05, 1.10],                # max_term_structure
        [True, False],                      # avoid_opex
        [None, 10.0],                       # width_filter (None=all, 10=original)
        [None, 1, 2],                       # entry_count
        [None, 3, 5],                       # min_dte (via PortfolioConfig)
        [None, 0.20],                       # max_delta (via PortfolioConfig)
    )

    for (tp, sl, mvix, mts, aopex, wf, ec_n, mdte, mdelta) in trading_grid:
        tc = TradingConfig(
            tp_pct=tp, sl_mult=sl, max_vix=mvix,
            max_term_structure=mts, avoid_opex=aopex,
            width_filter=wf, entry_count=ec_n,
        )
        pc_with_dte = PortfolioConfig(
            starting_capital=20_000, max_trades_per_day=2,
            monthly_drawdown_limit=0.15, lot_per_equity=10_000,
            calls_only=True, min_dte=mdte, max_delta=mdelta,
        )
        configs.append(FullConfig(portfolio=pc_with_dte, trading=tc,
                                  event=EventConfig(enabled=False)))

    return configs
_StagedTradingWinner = tuple[TradingConfig, int | None, float | None]
def _build_staged_grid_stage2(
    top_trading_configs: list[_StagedTradingWinner],
) -> list[FullConfig]:
    """Stage 2: Sweep portfolio params using top trading configs from Stage 1.

    Each winner carries the best (min_dte, max_delta) from stage 1 so
    portfolio sweeping preserves DTE/delta filtering.
    """
    configs: list[FullConfig] = []

    portfolio_grid = itertools.product(
        [20_000],                          # starting_capital
        [1, 2, 3],                         # max_trades_per_day
        [0.10, 0.15, 0.20, None],         # monthly_drawdown_limit
        [10_000, 20_000, 999_999],         # lot_per_equity
        [True, False],                     # calls_only
    )

    for (cap, trades, mstop, lpe, co) in portfolio_grid:
        for tc, mdte, mdelta in top_trading_configs:
            pc = PortfolioConfig(
                starting_capital=cap, max_trades_per_day=trades,
                monthly_drawdown_limit=mstop, lot_per_equity=lpe,
                calls_only=co, min_dte=mdte, max_delta=mdelta,
            )
            configs.append(FullConfig(portfolio=pc, trading=tc,
                                      event=EventConfig(enabled=False)))

    return configs
def _build_staged_grid_stage3(
    top_combined: list[tuple[PortfolioConfig, TradingConfig]],
) -> list[FullConfig]:
    """Stage 3: Sweep event params using top portfolio+trading combos.

    Includes broader threshold sweeps for vix_spike_threshold,
    vix_elevated_threshold, spx_drop_2d_threshold, and signal_mode.
    """
    configs: list[FullConfig] = []

    event_grid = itertools.product(
        ["any", "spx_and_vix"],            # signal_mode
        ["shared", "separate"],            # budget_mode
        [1, 2],                            # max_event_trades
        [-0.005, -0.01, -0.015, -0.02, -0.03, -0.05],  # spx_drop_threshold
        [-0.01, -0.02, -0.03, -0.05],     # spx_drop_2d_threshold
        [0.10, 0.20],                      # vix_spike_threshold
        [20.0, 30.0],                      # vix_elevated_threshold
        ["puts", "best"],                  # side_preference
        [3, 5],                            # event min_dte
        [5, 7, 10],                        # event max_dte
        [0.10, 0.15],                      # event min_delta
        [0.20, 0.25],                      # event max_delta
        [True, False],                     # rally_avoidance
    )

    event_combos = list(event_grid)

    for (pc, tc) in top_combined:
        for (sig_mode, bmode, evt_max, drop_thr, drop_2d_thr,
             vix_spike_thr, vix_elev_thr,
             side, emin_dte, emax_dte,
             emin_d, emax_d, rally) in event_combos:

            if emin_dte > emax_dte:
                continue
            if emin_d > emax_d:
                continue
            if bmode == "shared" and evt_max > 1:
                continue

            evc = EventConfig(
                enabled=True,
                signal_mode=sig_mode,
                budget_mode=bmode,
                max_event_trades=evt_max,
                spx_drop_threshold=drop_thr,
                spx_drop_2d_threshold=drop_2d_thr,
                vix_spike_threshold=vix_spike_thr,
                vix_elevated_threshold=vix_elev_thr,
                side_preference=side,
                min_dte=emin_dte, max_dte=emax_dte,
                min_delta=emin_d, max_delta=emax_d,
                rally_avoidance=rally,
            )
            configs.append(FullConfig(portfolio=pc, trading=tc, event=evc))

    return configs
EVENT_ONLY_TP_VALUES = [0.50, 0.60, 0.70, 0.80]
EVENT_ONLY_SL_VALUES: list[float | None] = [None, 2.0, 3.0]
def _build_event_only_grid() -> list[FullConfig]:
    """Build a grid for event-only strategies that ONLY trade on SPX drop days.

    Sets ``EventConfig.event_only=True`` which suppresses all scheduled trades
    in ``run_backtest``, combined with wide SPX drop thresholds [-0.5% to -5%]
    and event-driven put credit spread parameters.  The result is a pure
    "sell puts after a market drop" strategy.
    """
    configs: list[FullConfig] = []

    grid = itertools.product(
        EVENT_ONLY_TP_VALUES,                          # tp_pct
        EVENT_ONLY_SL_VALUES,                          # sl_mult
        [-0.005, -0.01, -0.015, -0.02, -0.03, -0.05], # spx_drop_threshold (1d)
        [-0.01, -0.02, -0.03, -0.05],                 # spx_drop_2d_threshold
        ["puts", "best"],                              # side_preference
        [3, 5],                                        # event min_dte
        [5, 7, 10],                                    # event max_dte
        [0.10, 0.15],                                  # event min_delta
        [0.20, 0.25],                                  # event max_delta
        [1, 2, 3],                                     # max_event_trades
        [True, False],                                 # rally_avoidance
        [30.0, None],                                  # max_vix
    )

    for (tp, sl, drop_1d, drop_2d, side,
         emin_dte, emax_dte, emin_d, emax_d,
         evt_max, rally, mvix) in grid:

        if emin_dte > emax_dte:
            continue
        if emin_d > emax_d:
            continue

        tc = TradingConfig(
            tp_pct=tp, sl_mult=sl, max_vix=mvix,
            avoid_opex=True, width_filter=10.0,
        )
        pc = PortfolioConfig(
            starting_capital=20_000,
            max_trades_per_day=3,
            monthly_drawdown_limit=0.15,
            lot_per_equity=20_000,
            calls_only=False,
        )
        evc = EventConfig(
            enabled=True,
            event_only=True,
            signal_mode="any",
            budget_mode="shared",
            max_event_trades=evt_max,
            spx_drop_threshold=drop_1d,
            spx_drop_2d_threshold=drop_2d,
            side_preference=side,
            min_dte=emin_dte, max_dte=emax_dte,
            min_delta=emin_d, max_delta=emax_d,
            rally_avoidance=rally,
        )
        configs.append(FullConfig(portfolio=pc, trading=tc, event=evc))

    return configs
def run_event_only_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Run a focused optimizer sweep for event-only (SPX drop) strategies.

    Builds a grid of configs that only trade on SPX drop days, sweeping
    drop thresholds from -0.5% to -5%, DTE/delta ranges, TP/SL, and
    event parameters.  Results are appended to any existing results CSV
    so they can be analyzed alongside the staged optimizer output.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write/append results CSV.
    num_workers : Number of parallel backtest workers (1 = sequential).

    Returns
    -------
    DataFrame with event-only optimizer results.
    """
    precompute_pnl_columns(df, EVENT_ONLY_TP_VALUES, EVENT_ONLY_SL_VALUES)

    configs = _build_event_only_grid()
    print(f"  Event-only grid: {len(configs):,} configs", flush=True)
    results = _run_grid(configs, df, daily_signals, "Event-Only",
                        num_workers=num_workers)

    if results.empty:
        logger.warning("Event-only optimizer produced no results")
        return results

    if output_csv.exists():
        try:
            existing = pd.read_csv(output_csv)
            combined = pd.concat([existing, results], ignore_index=True)
            combined.to_csv(output_csv, index=False)
            print(f"\n  Appended {len(results):,} event-only results to {output_csv}"
                  f" (total: {len(combined):,})", flush=True)
        except Exception as exc:
            logger.warning("Failed to append to existing CSV, writing standalone: %s", exc)
            fallback = output_csv.with_stem(output_csv.stem + "_event_only")
            results.to_csv(fallback, index=False)
            print(f"\n  Exported {len(results):,} event-only results to {fallback}",
                  flush=True)
    else:
        results.to_csv(output_csv, index=False)
        print(f"\n  Exported {len(results):,} event-only results to {output_csv}",
              flush=True)

    return results.sort_values("sharpe", ascending=False).reset_index(drop=True)
SELECTIVE_TP_VALUES = [0.50, 0.60, 0.70]
SELECTIVE_SL_VALUES: list[float | None] = [None, 2.0, 3.0]
def _build_selective_grid() -> list[FullConfig]:
    """Build a grid targeting 90%+ win rate with selective trading.

    Key design choices for high win rate:
    - Tighter delta (0.05-0.15): further OTM spreads = higher probability of profit
    - Longer DTE (5-10): more time for theta decay to work
    - VIX filter (max 25-30): avoid chaotic high-vol environments
    - Conservative TP (0.50-0.70): take profit quickly
    - Width filter (10 pts): consistent spread geometry
    - Entry count filter (1-2): only best entries per day
    - Regime throttle enabled: reduce exposure in hostile environments
    """
    configs: list[FullConfig] = []

    grid = itertools.product(
        SELECTIVE_TP_VALUES,               # tp_pct
        SELECTIVE_SL_VALUES,               # sl_mult
        [20_000, 50_000, 100_000],         # starting_capital
        [1, 2],                            # max_trades_per_day
        [0.10, 0.15, 0.20],               # monthly_drawdown_limit
        [True, False],                     # calls_only
        [3, 5],                            # min_dte
        [0.15, 0.20],                      # max_delta
        [25.0, 30.0],                      # max_vix
        [True, False],                     # avoid_opex
        [1, 2],                            # entry_count
        [True, False],                     # regime_throttle enabled
    )

    for (tp, sl, cap, trades, mstop, co,
         mdte, mdelta, mvix, avopex, ecnt,
         rt_on) in grid:

        tc = TradingConfig(
            tp_pct=tp, sl_mult=sl, max_vix=mvix,
            avoid_opex=avopex, width_filter=10.0,
            entry_count=ecnt,
        )
        pc = PortfolioConfig(
            starting_capital=cap,
            max_trades_per_day=trades,
            monthly_drawdown_limit=mstop,
            lot_per_equity=cap,
            calls_only=co,
            min_dte=mdte,
            max_delta=mdelta,
        )
        rt = RegimeThrottle(enabled=rt_on)
        configs.append(FullConfig(
            portfolio=pc, trading=tc,
            event=EventConfig(enabled=False),
            regime=rt,
        ))

    return configs
def run_selective_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Run a focused optimizer sweep targeting high win-rate selective strategies.

    Uses conservative parameters (tight delta, long DTE, VIX filter) to find
    configs that achieve 90%+ day win-rate with at least 50 trades.  Results
    are appended to any existing results CSV.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write/append results CSV.
    num_workers : Number of parallel backtest workers (1 = sequential).

    Returns
    -------
    DataFrame with selective optimizer results.
    """
    precompute_pnl_columns(df, SELECTIVE_TP_VALUES, SELECTIVE_SL_VALUES)

    configs = _build_selective_grid()
    print(f"  Selective grid: {len(configs):,} configs", flush=True)
    results = _run_grid(configs, df, daily_signals, "Selective-HiWR",
                        num_workers=num_workers)

    if results.empty:
        logger.warning("Selective optimizer produced no results")
        return results

    if output_csv.exists():
        try:
            existing = pd.read_csv(output_csv)
            combined = pd.concat([existing, results], ignore_index=True)
            combined.to_csv(output_csv, index=False)
            print(f"\n  Appended {len(results):,} selective results to {output_csv}"
                  f" (total: {len(combined):,})", flush=True)
        except Exception as exc:
            logger.warning("Failed to append to existing CSV, writing standalone: %s", exc)
            fallback = output_csv.with_stem(output_csv.stem + "_selective")
            results.to_csv(fallback, index=False)
            print(f"\n  Exported {len(results):,} selective results to {fallback}",
                  flush=True)
    else:
        results.to_csv(output_csv, index=False)
        print(f"\n  Exported {len(results):,} selective results to {output_csv}",
              flush=True)

    hi_wr = results[
        (results["win_rate"] >= 0.90) & (results["total_trades"] >= 50)
    ]
    print(f"\n  Configs with 90%+ WR and 50+ trades: {len(hi_wr):,}", flush=True)
    if not hi_wr.empty:
        best = hi_wr.nlargest(5, "sharpe")
        print("  Top 5 high-WR configs:", flush=True)
        for _, r in best.iterrows():
            print(f"    sharpe={r['sharpe']:.2f}  WR={r['win_rate']:.1%}  "
                  f"ret={r['return_pct']:.0f}%  DD={r['max_dd_pct']:.1f}%  "
                  f"trades={int(r['total_trades'])}  "
                  f"tp={r['t_tp_pct']}  sl={r['t_sl_mult']}  "
                  f"vix<={r['t_max_vix']}", flush=True)

    return results.sort_values("sharpe", ascending=False).reset_index(drop=True)
_BACKTEST_WORKER_REF: dict[str, Any] = {}
def _init_backtest_worker(
    df_bytes: bytes,
    signals_bytes: bytes,
    precomp_bytes: bytes,
) -> None:
    """Initializer for each worker process in the backtest pool.

    Unpickles the shared read-only data once per worker, storing them in
    the module-level ``_BACKTEST_WORKER_REF`` dict so ``_backtest_worker``
    can access them without re-serializing on every task.
    """
    _BACKTEST_WORKER_REF["df"] = pickle.loads(df_bytes)
    _BACKTEST_WORKER_REF["daily_signals"] = pickle.loads(signals_bytes)
    _BACKTEST_WORKER_REF["precomp_cache"] = pickle.loads(precomp_bytes)
def _backtest_worker(cfg: FullConfig) -> dict[str, Any]:
    """Run a single backtest config inside a worker process.

    Reads shared data from ``_BACKTEST_WORKER_REF`` (set by the pool
    initializer) and returns the flat config + result metrics dict.
    """
    df = _BACKTEST_WORKER_REF["df"]
    daily_signals = _BACKTEST_WORKER_REF["daily_signals"]
    precomp_cache = _BACKTEST_WORKER_REF["precomp_cache"]

    tc = cfg.trading
    cache_key = (tc.width_filter, tc.entry_count)
    day_precomp = precomp_cache[cache_key]

    result = run_backtest(df, daily_signals, cfg, day_precomp=day_precomp)
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
    return row
def _build_precomp_cache(
    df: pd.DataFrame,
    configs: list[FullConfig],
) -> dict[tuple, dict]:
    """Pre-compute day selections for all unique (width_filter, entry_count) keys.

    Building these upfront (instead of lazily) is required before forking
    worker processes so every worker has the full cache available.
    """
    cache: dict[tuple, dict] = {}
    for cfg in configs:
        tc = cfg.trading
        key = (tc.width_filter, tc.entry_count)
        if key not in cache:
            cache[key] = _precompute_day_selections(
                df, width_filter=tc.width_filter, entry_count=tc.entry_count,
            )
    return cache
def _run_grid(
    configs: list[FullConfig],
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    stage_name: str,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Execute a list of configs and return results DataFrame.

    When *num_workers* > 1, distributes backtest runs across a
    ``multiprocessing.Pool``.  Falls back to a sequential loop when
    *num_workers* == 1 (useful for debugging).
    """
    print(f"\n[{stage_name}] {len(configs):,} configurations to evaluate"
          f" (workers={num_workers})", flush=True)
    t0 = time.time()

    precomp_cache = _build_precomp_cache(df, configs)

    if num_workers > 1:
        df_bytes = pickle.dumps(df)
        signals_bytes = pickle.dumps(daily_signals)
        precomp_bytes = pickle.dumps(precomp_cache)

        rows: list[dict[str, Any]] = []
        with Pool(
            num_workers,
            initializer=_init_backtest_worker,
            initargs=(df_bytes, signals_bytes, precomp_bytes),
        ) as pool:
            for i, row in enumerate(
                pool.imap_unordered(_backtest_worker, configs, chunksize=64),
            ):
                rows.append(row)
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    remaining = (len(configs) - i - 1) / rate
                    print(f"  {i+1:,}/{len(configs):,} done  "
                          f"({rate:.0f}/sec, ~{remaining:.0f}s remaining)",
                          flush=True)
    else:
        rows = []
        for i, cfg in enumerate(configs):
            tc = cfg.trading
            cache_key = (tc.width_filter, tc.entry_count)

            result = run_backtest(
                df, daily_signals, cfg, day_precomp=precomp_cache[cache_key],
            )
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

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(configs) - i - 1) / rate
                print(f"  {i+1:,}/{len(configs):,} done  "
                      f"({rate:.0f}/sec, ~{remaining:.0f}s remaining)",
                      flush=True)

    elapsed = time.time() - t0
    print(f"  [{stage_name}] Completed {len(configs):,} configs in {elapsed:.1f}s "
          f"({len(configs)/max(elapsed, 0.01):.0f}/sec)", flush=True)

    out = pd.DataFrame(rows)
    # L10 fix: ``Pool.imap_unordered`` yields results in completion
    # order, which depends on worker scheduling and therefore varies
    # across runs.  Sort by the canonical config key so the output
    # CSV is byte-identical for identical inputs (sequential vs
    # parallel, run N vs run N+1).  Use mergesort for stability.
    if not out.empty:
        try:
            out = out.assign(_sort_key=out.apply(_config_key, axis=1)) \
                     .sort_values("_sort_key", kind="mergesort") \
                     .drop(columns="_sort_key") \
                     .reset_index(drop=True)
        except Exception as exc:  # pragma: no cover -- defensive only
            logger.warning("Skipping deterministic sort (config_key failed: %s)", exc)
    return out
def run_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Run exhaustive grid search (original mode) and export results.

    Parameters
    ----------
    df : Training candidates DataFrame.
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write the full results CSV.
    num_workers : Number of parallel backtest workers (1 = sequential).

    Returns
    -------
    DataFrame with one row per config, sorted by Sharpe descending.
    """
    configs = _build_optimizer_grid()
    results_df = _run_grid(configs, df, daily_signals, "Optimizer",
                           num_workers=num_workers)
    results_df.to_csv(output_csv, index=False)
    print(f"\n  Full results exported to {output_csv}", flush=True)
    return results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
def run_staged_optimizer(
    df: pd.DataFrame,
    daily_signals: pd.DataFrame,
    output_csv: Path = RESULTS_CSV,
    top_n_trading: int = 10,
    top_n_combined: int = 5,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Run 3-stage optimizer: trading params -> portfolio params -> event params.

    Stage 1 sweeps all TradingConfig dimensions with fixed portfolio baseline.
    Stage 2 takes the top trading configs and sweeps portfolio params.
    Stage 3 takes the top combined configs and sweeps event params.

    Parameters
    ----------
    df : Training candidates DataFrame (must have multi-TP trajectory columns).
    daily_signals : Precomputed daily signal features.
    output_csv : Path to write combined results CSV.
    top_n_trading : Number of top trading configs to carry into Stage 2.
    top_n_combined : Number of top combined configs to carry into Stage 3.
    num_workers : Number of parallel backtest workers (1 = sequential).

    Returns
    -------
    Combined DataFrame with all stage results, sorted by Sharpe descending.
    """
    # Pre-compute PnL columns for all (tp, sl) combos
    precompute_pnl_columns(df, OPTIMIZER_TP_VALUES, OPTIMIZER_SL_VALUES)

    # --- Stage 1: Trading param sweep ---
    s1_configs = _build_staged_grid_stage1()
    s1_results = _run_grid(s1_configs, df, daily_signals, "Stage-1 Trading",
                           num_workers=num_workers)
    s1_sorted = s1_results.sort_values("sharpe", ascending=False)

    # Extract top trading configs (carrying min_dte / max_delta from stage 1)
    top_trading: list[_StagedTradingWinner] = []
    seen: set[str] = set()
    for _, row in s1_sorted.iterrows():
        key = (f"{row.get('t_tp_pct')}_{row.get('t_sl_mult')}_{row.get('t_max_vix')}_"
               f"{row.get('t_max_term_structure')}_{row.get('t_avoid_opex')}_"
               f"{row.get('t_width_filter')}_{row.get('t_entry_count')}_"
               f"{row.get('p_min_dte')}_{row.get('p_max_delta')}")
        if key in seen:
            continue
        seen.add(key)

        top_trading.append((
            TradingConfig(
                tp_pct=float(row.get("t_tp_pct", 0.50)),
                sl_mult=_opt_val(row.get("t_sl_mult")),
                max_vix=_opt_val(row.get("t_max_vix")),
                max_term_structure=_opt_val(row.get("t_max_term_structure")),
                avoid_opex=_safe_bool(row.get("t_avoid_opex")),
                width_filter=_opt_val(row.get("t_width_filter")),
                entry_count=int(row.get("t_entry_count")) if pd.notna(row.get("t_entry_count")) else None,
            ),
            int(row.get("p_min_dte")) if pd.notna(row.get("p_min_dte")) else None,
            float(row.get("p_max_delta")) if pd.notna(row.get("p_max_delta")) else None,
        ))
        if len(top_trading) >= top_n_trading:
            break

    print(f"\n  Top {len(top_trading)} trading configs selected for Stage 2", flush=True)

    # --- Stage 2: Portfolio param sweep ---
    s2_configs = _build_staged_grid_stage2(top_trading)
    s2_results = _run_grid(s2_configs, df, daily_signals, "Stage-2 Portfolio",
                           num_workers=num_workers)
    s2_sorted = s2_results.sort_values("sharpe", ascending=False)

    top_combined: list[tuple[PortfolioConfig, TradingConfig]] = []
    seen_combined: set[str] = set()
    _s2_key_cols = [c for c in s2_sorted.columns if c.startswith(("p_", "t_"))]
    for _, row in s2_sorted.iterrows():
        key = "|".join(str(row.get(c)) for c in _s2_key_cols)
        if key in seen_combined:
            continue
        seen_combined.add(key)

        dd_limit = _opt_val(row.get("p_monthly_drawdown_limit"))
        pc = PortfolioConfig(
            starting_capital=float(row.get("p_starting_capital", 20_000)),
            max_trades_per_day=int(row.get("p_max_trades_per_day", 2)),
            monthly_drawdown_limit=dd_limit,
            lot_per_equity=float(row.get("p_lot_per_equity", 10_000)),
            calls_only=_safe_bool(row.get("p_calls_only"), True),
            min_dte=int(row.get("p_min_dte")) if pd.notna(row.get("p_min_dte")) else None,
            max_delta=float(row.get("p_max_delta")) if pd.notna(row.get("p_max_delta")) else None,
        )
        tc = TradingConfig(
            tp_pct=float(row.get("t_tp_pct", 0.50)),
            sl_mult=_opt_val(row.get("t_sl_mult")),
            max_vix=_opt_val(row.get("t_max_vix")),
            max_term_structure=_opt_val(row.get("t_max_term_structure")),
            avoid_opex=_safe_bool(row.get("t_avoid_opex")),
            width_filter=_opt_val(row.get("t_width_filter")),
            entry_count=int(row.get("t_entry_count")) if pd.notna(row.get("t_entry_count")) else None,
        )
        top_combined.append((pc, tc))
        if len(top_combined) >= top_n_combined:
            break

    print(f"\n  Top {len(top_combined)} combined configs selected for Stage 3", flush=True)

    # --- Stage 3: Event param sweep ---
    s3_configs = _build_staged_grid_stage3(top_combined)
    s3_results = _run_grid(s3_configs, df, daily_signals, "Stage-3 Event",
                           num_workers=num_workers)

    # Combine all stages
    all_results = pd.concat([s1_results, s2_results, s3_results], ignore_index=True)
    all_results.to_csv(output_csv, index=False)
    print(f"\n  Combined results ({len(all_results):,} rows) exported to {output_csv}",
          flush=True)

    return all_results.sort_values("sharpe", ascending=False).reset_index(drop=True)
