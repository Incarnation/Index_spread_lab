#!/usr/bin/env python3
"""Offline training pipeline for SPX credit-spread bucket model.

Reads Databento .dbn.zst historical data, computes Black-Scholes implied
volatility and delta, constructs vertical credit spreads, labels outcomes
via forward-looking minute bars, builds production-compatible features, and
trains a hierarchical Bayesian-smoothed bucket model.

Usage:
    python generate_training_data.py                    # full pipeline
    python generate_training_data.py --max-days 5       # quick test run
    python generate_training_data.py --deploy            # train + insert model
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

def _locate_backend_dir() -> Path:
    """Return the canonical ``backend/`` directory for sys.path insertion.

    Walks up the parent chain from this file's location looking for a
    directory whose child ``spx_backend`` package exists -- that's the
    backend root regardless of whether this module lives at
    ``backend/scripts/generate_training_data.py`` (the monolith) or at
    ``backend/scripts/training/<sub>.py`` (a post-split submodule).

    Falls back to ``parents[1]`` so behaviour is unchanged for unfamiliar
    layouts; the spx_backend imports below will fail loudly rather than
    silently mis-resolve.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "spx_backend").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


_BACKEND = _locate_backend_dir()
sys.path.insert(0, str(_BACKEND))
sys.path.insert(0, str(_BACKEND / "scripts"))

from _constants import CONTRACT_MULT, CONTRACTS
from spx_backend.jobs.modeling import (
    extract_candidate_features,
    predict_with_bucket_model,
    summarize_strategy_quality,
    train_bucket_model,
)
from spx_backend.utils.pricing import mid_price

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = _BACKEND.parent / "data"


def _resolve_databento_dir() -> Path:
    """Return the Databento data directory (env-driven; L5 fix).

    Reads ``settings.databento_dir`` so the 120 GB tree can live on a
    dedicated SSD or NFS mount without code changes.  Relative paths
    are anchored to the repo root; absolute paths are used as-is.
    Falls back to ``data/databento`` when settings is unavailable.
    """
    try:
        from spx_backend.config import settings as _live  # type: ignore
        configured = Path(_live.databento_dir)
    except Exception:  # pragma: no cover -- import optional offline
        configured = Path("data/databento")
    if not configured.is_absolute():
        configured = _BACKEND.parent / configured
    return configured


DATABENTO_DIR = _resolve_databento_dir()
SPXW_CBBO = DATABENTO_DIR / "spxw" / "cbbo-1m"
SPXW_DEFS = DATABENTO_DIR / "spxw" / "definition"
SPXW_STATS = DATABENTO_DIR / "spxw" / "statistics"
SPX_CBBO = DATABENTO_DIR / "spx" / "cbbo-1m"
SPX_DEFS = DATABENTO_DIR / "spx" / "definition"
SPX_STATS = DATABENTO_DIR / "spx" / "statistics"
SPY_EQUITY_PATH = DATABENTO_DIR / "underlying" / "spy_equity_1m.parquet"
OFFLINE_GEX_CSV = DATA_DIR / "offline_gex_cache.csv"
CONTEXT_SNAPSHOTS_CSV = DATA_DIR / "context_snapshots_export.csv"
OUTPUT_CSV = DATA_DIR / "training_candidates.csv"
ECONOMIC_CALENDAR_CSV = DATA_DIR / "economic_calendar.csv"
CANDIDATES_CACHE_DIR = DATA_DIR / "candidates_cache"

# Production DB exports (from export_production_data.py)
PRODUCTION_CHAINS_DIR = DATA_DIR / "production_exports" / "chains"
PRODUCTION_UNDERLYING_DIR = DATA_DIR / "production_exports" / "underlying"

# FirstRateData parquets (converted from raw .txt downloads)
FRD_DIR = DATA_DIR / "firstratedata"
FRD_SPX = FRD_DIR / "spx_1min.parquet"
FRD_VIX = FRD_DIR / "vix_1min.parquet"
FRD_VIX9D = FRD_DIR / "vix9d_1min.parquet"
FRD_VVIX = FRD_DIR / "vvix_1min.parquet"
FRD_SKEW = FRD_DIR / "skew_daily.parquet"

# ---------------------------------------------------------------------------
# Pipeline constants (match production config defaults)
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
SPY_SPX_RATIO = 10.024
RISK_FREE_RATE = 0.043
DECISION_MINUTES_ET = [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0)]
DTE_TARGETS = [0, 1, 3, 5, 7, 10]
DELTA_TARGETS = [0.10, 0.15, 0.20, 0.25]
SPREAD_SIDES = ["put", "call"]
WIDTH_TARGETS = [5.0, 10.0, 15.0, 20.0]
WIDTH_POINTS = 10.0  # legacy default for single-width callers
# Restored to 0.50 in the Tier-2 TP-drift revert.  Commit 767f19a had
# raised this to 0.60 under the assumption that live
# ``trade_pnl_take_profit_pct`` was already 0.60 -- it was not, and
# has been 0.50 since the setting was introduced (commit 61b2eb8).
# The labels on disk (data/training_candidates.csv) and the deployed
# XGB model were both produced at 0.50, so the labeler constant was
# the only outlier.  Do not raise this to 0.60 (or any other level)
# without the coordinated Track F live-settings change; the C4 guard
# in ``_assert_sl_alignment_with_live_settings`` will block any
# mismatched re-labeling run.
TAKE_PROFIT_PCT = 0.50
STOP_LOSS_PCT = 2.00
LABEL_MARK_INTERVAL_MINUTES = 5
MIN_MID_PRICE = 0.05
MAX_IV = 5.0
MIN_IV = 0.01
IV_BISECT_ITERS = 60

# Active training grid config — overridden by --config at runtime
_ACTIVE_GRID: dict[str, Any] | None = None


def _get_grid_param(name: str) -> Any:
    """Return a grid parameter from the active config, falling back to the module constant."""
    if _ACTIVE_GRID is not None:
        return _ACTIVE_GRID[name]
    return globals()[name]


def _assert_sl_alignment_with_live_settings() -> None:
    """Fail loudly if the live SL/TP config drifts from the training
    labeler's hardcoded contract.  This is the C4 hybrid mitigation
    (see ``OFFLINE_PIPELINE_AUDIT.md``).

    The labeler in this script computes
    ``sl_thr = max_profit * STOP_LOSS_PCT`` (always SL-enabled) and
    ``tp_thr = max_profit * TAKE_PROFIT_PCT``.  If production
    overrides ``trade_pnl_stop_loss_basis``, disables SL via
    ``trade_pnl_stop_loss_enabled = False``, changes
    ``trade_pnl_stop_loss_pct`` / ``trade_pnl_take_profit_pct`` away
    from the active grid's values, training labels will silently
    misrepresent live trade outcomes and any downstream model /
    optimizer run will be calibrated to the wrong policy.

    The helper deliberately raises ``SystemExit`` rather than logging a
    warning so the pipeline cannot complete a labeling run with a stale
    hardcoded contract.

    Tier 1 strict-guard fix (E2E pipeline review): a previous version
    silently downgraded a settings-import failure (e.g.  PYTHONPATH
    misconfigured, partial spx_backend install) to a warning, which
    let the pipeline skip the entire alignment check.  Now a failed
    import is itself a hard error -- a missing alignment check is
    indistinguishable from a misaligned one for our purposes.

    Tier 1 TP-alignment fix (E2E pipeline review): previously only SL
    was checked.  Take-profit drift produces the same class of label
    bias (labels exit at a TP that production never hits, or vice
    versa) so we now mirror the SL check for TAKE_PROFIT_PCT.

    Tier 2 TP-drift revert (investigation 2026-04-16): commit 767f19a
    bumped the labeler's ``TAKE_PROFIT_PCT`` 0.50 -> 0.60 on the
    assumption that live ``trade_pnl_take_profit_pct`` was already
    0.60.  An audit found live has in fact been 0.50 since the
    setting was introduced (commit 61b2eb8), and the labels on disk
    plus the deployed XGB model were both produced at 0.50.  So the
    labeler constant was the only outlier and the correct fix was a
    1-line revert, not a multi-hour regen + retrain.  This guard
    remains the enforcement surface: if a future Track F decision
    moves live to 0.60 (or any other TP), raise ``TAKE_PROFIT_PCT``
    here in lockstep so this assertion continues to prevent silent
    label bias.
    """
    try:
        from spx_backend.config import settings as _live_settings
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(
            "[C4] Cannot verify SL/TP alignment because spx_backend.config "
            f"failed to import ({exc!r}). A failed import is treated as a "
            "hard error so misconfigurations cannot silently skip the "
            "labeler-vs-live alignment guard. Fix PYTHONPATH (or run from "
            "the backend/ directory) before retrying. "
            "See OFFLINE_PIPELINE_AUDIT.md C4."
        ) from exc

    expected_basis = "max_profit"
    if _live_settings.trade_pnl_stop_loss_basis != expected_basis:
        raise SystemExit(
            f"[C4] Training labeler hardcodes basis={expected_basis!r} but live "
            f"trade_pnl_stop_loss_basis={_live_settings.trade_pnl_stop_loss_basis!r}. "
            "Update _evaluate_outcome (or route through "
            "_label_helpers.evaluate_candidate_outcome) before re-labeling. "
            "See OFFLINE_PIPELINE_AUDIT.md C4."
        )
    if not _live_settings.trade_pnl_stop_loss_enabled:
        raise SystemExit(
            "[C4] Training labeler always simulates a stop-loss but live "
            "trade_pnl_stop_loss_enabled=False. Disable the labeler's SL "
            "branch (or hold-only labels) before continuing. "
            "See OFFLINE_PIPELINE_AUDIT.md C4."
        )

    grid_sl = float(_get_grid_param("STOP_LOSS_PCT"))
    live_sl = float(_live_settings.trade_pnl_stop_loss_pct)
    if abs(live_sl - grid_sl) > 1e-9:
        raise SystemExit(
            f"[C4] Active training STOP_LOSS_PCT={grid_sl} != live "
            f"trade_pnl_stop_loss_pct={live_sl}. If you intentionally "
            "want a what-if SL multiplier, update the YAML and the live "
            "config together (or temporarily comment out this assertion "
            "with a tracking issue). See OFFLINE_PIPELINE_AUDIT.md C4."
        )

    grid_tp = float(_get_grid_param("TAKE_PROFIT_PCT"))
    live_tp = float(_live_settings.trade_pnl_take_profit_pct)
    if abs(live_tp - grid_tp) > 1e-9:
        raise SystemExit(
            f"[C4] Active training TAKE_PROFIT_PCT={grid_tp} != live "
            f"trade_pnl_take_profit_pct={live_tp}. Take-profit drift "
            "between labels and live causes the same exit-bias class as "
            "stop-loss drift -- update the YAML and the live config "
            "together. See OFFLINE_PIPELINE_AUDIT.md C4."
        )

from .bs_gex_spot import (
    compute_offline_gex,
    derive_spx_from_parity,
    get_cbbo_snapshot_at,
)
from .io_loaders import (
    _available_day_files,
    _load_merged_gex,
    build_instrument_map,
    lag_daily_to_next_session,
    load_cbbo,
    load_daily_parquet,
    load_definitions,
    load_economic_calendar,
    load_frd_quotes,
    load_spy_equity,
    load_statistics,
    lookup_intraday_value,
    merge_underlying_quotes,
)
from .candidates import (
    _atomic_write_csv,
    _cache_day_candidates,
    _compute_code_version,
    _generate_candidates_for_day,
    _init_candidate_worker,
    _input_data_fingerprint,
    _load_cache_manifest,
    _load_cached_day,
    _save_cache_manifest,
    build_candidates_for_snapshot,
)
from .labeling import (
    LABELS_CACHE_DIR,
    TP_LEVELS,
    TRAJECTORY_COLUMNS,
    _compute_days_hash,
    _compute_label_code_hash,
    _determine_relabel_days,
    _load_labeled_cache_day,
    _load_labels_manifest,
    _save_labeled_cache_day,
    _save_labels_manifest,
    build_training_rows,
    deploy_model,
    label_candidates,
    label_candidates_fast,
    relabel_from_csv,
    walk_forward_validate,
)


def run_pipeline(
    *,
    max_days: int | None = None,
    deploy: bool = False,
    verbose: bool = True,
    workers: int = 8,
    force_regen: bool = False,
    cache_dir: Path = CANDIDATES_CACHE_DIR,
    label_cache_dir: Path = LABELS_CACHE_DIR,
    training_config: "TrainingGridConfig | None" = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Execute the full offline training pipeline end-to-end.

    Steps:
      1. Discover available trading days from Databento files.
      2. Load reference data (SPY equity bars, VIX / VIX9D CSVs).
      3. For each day × decision time × DTE × delta: construct candidates.
         (uses incremental per-day Parquet cache to skip unchanged days)
      4. Label all candidates with forward-looking cbbo-1m marks.
      5. Build production-compatible feature vectors.
      6. Walk-forward validation and final model training.
      7. (Optional) insert model into production DB as shadow.

    Parameters
    ----------
    force_regen : If True, ignore the candidate cache and regenerate all days.
    cache_dir : Directory for per-day Parquet candidate cache files.
    training_config : Optional YAML-loaded grid config. When provided,
        overrides module-level constants for entry times, DTEs, deltas,
        widths, and spread sides.
    """
    global _ACTIVE_GRID
    if training_config is not None:
        _ACTIVE_GRID = {
            "DECISION_MINUTES_ET": training_config.as_tuples(),
            "DTE_TARGETS": training_config.dte_targets,
            "DELTA_TARGETS": training_config.delta_targets,
            "SPREAD_SIDES": training_config.spread_sides,
            "WIDTH_TARGETS": training_config.width_targets,
            "TAKE_PROFIT_PCT": training_config.take_profit_pct,
            "STOP_LOSS_PCT": training_config.stop_loss_pct,
            "LABEL_MARK_INTERVAL_MINUTES": training_config.label_mark_interval_minutes,
        }
        print(f"[PIPELINE] Training config: {training_config.name} "
              f"(hash={training_config.content_hash()})", flush=True)
    else:
        _ACTIVE_GRID = None

    # Hybrid C4 mitigation: fail fast if live SL config diverges from the
    # labeler's hardcoded policy.  See OFFLINE_PIPELINE_AUDIT.md.
    _assert_sl_alignment_with_live_settings()
    t0 = time.time()

    # -- 1. Discover trading days --
    spxw_days = set(_available_day_files(SPXW_CBBO))
    spx_days = set(_available_day_files(SPX_CBBO))
    trading_days = sorted(spxw_days | spx_days)
    if not trading_days:
        logger.error("No trading days found in %s or %s", SPXW_CBBO, SPX_CBBO)
        sys.exit(2)
    # Apply date range filters (normalize YYYY-MM-DD to YYYYMMDD for comparison)
    if start_date:
        sd = start_date.replace("-", "")
        before = len(trading_days)
        trading_days = [d for d in trading_days if d >= sd]
        print(f"[PIPELINE] --start-date {start_date}: {before} -> {len(trading_days)} days", flush=True)
    if end_date:
        ed = end_date.replace("-", "")
        before = len(trading_days)
        trading_days = [d for d in trading_days if d <= ed]
        print(f"[PIPELINE] --end-date {end_date}: {before} -> {len(trading_days)} days", flush=True)
    if max_days:
        trading_days = trading_days[:max_days]
    print(
        f"[PIPELINE] {len(trading_days)} trading days: "
        f"{trading_days[0]} .. {trading_days[-1]}",
        flush=True,
    )

    # -- 2. Load reference data --
    print("[PIPELINE] Loading reference data ...", flush=True)
    spy_df = load_spy_equity()

    # FRD 1-min parquets provide full-history intraday coverage.
    empty_prod = pd.DataFrame(columns=["ts", "symbol", "last"])
    frd_spx = load_frd_quotes(FRD_SPX, "SPX")
    frd_vix = load_frd_quotes(FRD_VIX, "VIX")
    frd_vix9d = load_frd_quotes(FRD_VIX9D, "VIX9D")
    frd_vvix = load_frd_quotes(FRD_VVIX, "VVIX")
    uq_df = merge_underlying_quotes(empty_prod, frd_spx, frd_vix, frd_vix9d, frd_vvix)
    frd_total = len(frd_spx) + len(frd_vix) + len(frd_vix9d) + len(frd_vvix)

    # Daily SKEW from parquet (one value per trading day).
    # FRD is the primary source; production DB export extends forward.
    # H6 fix: both sources stamp values at session close, so we lag the
    # merged dict by one trading day before handing it to candidate
    # generation.  Looking up `skew_daily[day_date]` at an intraday
    # decision time was previously same-day lookahead.
    skew_eod = load_daily_parquet(FRD_SKEW)
    prod_skew_path = PRODUCTION_UNDERLYING_DIR / "SKEW_1min.parquet"
    if prod_skew_path.exists():
        prod_skew = load_daily_parquet(prod_skew_path)
        for d, v in prod_skew.items():
            skew_eod.setdefault(d, v)
    skew_daily = lag_daily_to_next_session(skew_eod)

    # Economic calendar (OPEX / FOMC / triple witching)
    cal_map = load_economic_calendar(ECONOMIC_CALENDAR_CSV)

    # Merge production context_snapshots (authoritative) with offline GEX cache (historical)
    cs_df = _load_merged_gex()

    gex_count = cs_df["gex_net"].notna().sum() if not cs_df.empty else 0
    print(f"  SPY equity rows : {len(spy_df)}")
    print(f"  Underlying quotes: {len(uq_df):,} rows (FRD={frd_total:,})")
    print(f"  SKEW daily dates : {len(skew_daily)}")
    print(f"  Calendar events  : {len(cal_map)}")
    print(f"  GEX snapshots    : {len(cs_df)} rows ({gex_count} with GEX)", flush=True)

    # -- 3. Generate candidates (parallel across days, with incremental cache) --
    code_version = _compute_code_version()
    manifest = _load_cache_manifest(cache_dir)
    grid_hash = training_config.content_hash() if training_config else "default"

    cache_valid = (
        not force_regen
        and manifest.get("code_version") == code_version
        and manifest.get("grid_hash") == grid_hash
    )
    if force_regen:
        print("[PIPELINE] --force-regen: ignoring candidate cache", flush=True)
    elif not cache_valid:
        if not manifest:
            reason = "no cache found"
        elif manifest.get("code_version") != code_version:
            reason = "code changed"
        else:
            reason = "grid config changed"
        print(f"[PIPELINE] Cache invalidated ({reason}), regenerating all days", flush=True)

    cached_days_info = manifest.get("days", {}) if cache_valid else {}

    # H3 fix: per-day input fingerprint check.  A day is "cached" only if
    # its stored {input_file: (mtime, size)} fingerprint matches the
    # current state on disk.  Mismatch -> regenerate just that day.
    invalidated_by_inputs: list[str] = []
    days_from_cache: list[str] = []
    days_to_generate: list[str] = []
    for d in trading_days:
        info = cached_days_info.get(d)
        if info is None:
            days_to_generate.append(d)
            continue
        stored_fp = info.get("inputs")
        current_fp = _input_data_fingerprint(d)
        # Older manifests have no "inputs" key.  Treat that as "unknown"
        # rather than mismatch so we don't force a 100% regeneration on
        # the first run after this change ships -- the cache will heal
        # itself organically as days get re-cached.
        if stored_fp is None or stored_fp == current_fp:
            days_from_cache.append(d)
        else:
            invalidated_by_inputs.append(d)
            days_to_generate.append(d)

    if invalidated_by_inputs:
        print(
            f"[PIPELINE] H3: {len(invalidated_by_inputs)} day(s) "
            f"invalidated by input-file fingerprint change "
            f"(first 5: {invalidated_by_inputs[:5]})",
            flush=True,
        )

    print(f"[PIPELINE] Cache: {len(days_from_cache)} days cached, "
          f"{len(days_to_generate)} days to generate", flush=True)

    all_candidates: list[dict] = []

    # Load cached days
    for day_str in days_from_cache:
        day_cands = _load_cached_day(cache_dir, day_str)
        all_candidates.extend(day_cands)
        if verbose:
            print(f"  [{day_str}] {len(day_cands)} candidates (cached)", flush=True)

    # Generate new days
    if days_to_generate:
        candidate_workers = min(workers, len(days_to_generate))
        print(
            f"[PIPELINE] Generating {len(days_to_generate)} days "
            f"({candidate_workers} workers) ...",
            flush=True,
        )

        import pickle
        spy_bytes = pickle.dumps(spy_df)
        uq_bytes = pickle.dumps(uq_df)

        new_manifest_days = dict(cached_days_info)

        t_gen_start = time.time()
        n_gen_total = len(days_to_generate)

        if candidate_workers > 1:
            with Pool(
                candidate_workers,
                initializer=_init_candidate_worker,
                initargs=(spy_bytes, uq_bytes, skew_daily, cal_map, _ACTIVE_GRID),
            ) as pool:
                for di, day_cands in enumerate(
                    pool.imap(_generate_candidates_for_day, days_to_generate),
                ):
                    day_str = days_to_generate[di]
                    all_candidates.extend(day_cands)
                    n_cached = _cache_day_candidates(cache_dir, day_str, day_cands)
                    new_manifest_days[day_str] = {
                        "rows": n_cached,
                        "file": f"{day_str}.parquet",
                        "inputs": _input_data_fingerprint(day_str),
                    }
                    if verbose:
                        print(
                            f"  [{day_str}] {len(day_cands)} candidates (new)",
                            flush=True,
                        )
                    done = di + 1
                    if done % 25 == 0 or done == n_gen_total:
                        elapsed = time.time() - t_gen_start
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (n_gen_total - done) / rate if rate > 0 else 0
                        print(
                            f"[PIPELINE] Candidates: {done}/{n_gen_total} days "
                            f"({len(all_candidates):,} candidates) "
                            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                            flush=True,
                        )
        else:
            _init_candidate_worker(spy_bytes, uq_bytes, skew_daily, cal_map, _ACTIVE_GRID)
            for di, day_str in enumerate(days_to_generate):
                day_cands = _generate_candidates_for_day(day_str)
                all_candidates.extend(day_cands)
                n_cached = _cache_day_candidates(cache_dir, day_str, day_cands)
                new_manifest_days[day_str] = {
                    "rows": n_cached,
                    "file": f"{day_str}.parquet",
                    "inputs": _input_data_fingerprint(day_str),
                }
                if verbose:
                    print(
                        f"  [{day_str}] {len(day_cands)} candidates (new)",
                        flush=True,
                    )
                done = di + 1
                if done % 25 == 0 or done == n_gen_total:
                    elapsed = time.time() - t_gen_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_gen_total - done) / rate if rate > 0 else 0
                    print(
                        f"[PIPELINE] Candidates: {done}/{n_gen_total} days "
                        f"({len(all_candidates):,} candidates) "
                        f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                        flush=True,
                    )

        del spy_bytes, uq_bytes

        # Update manifest
        _save_cache_manifest(cache_dir, {
            "code_version": code_version,
            "grid_hash": grid_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "days": new_manifest_days,
        })
    else:
        print("[PIPELINE] All days loaded from cache", flush=True)

    gc.collect()

    print(f"[PIPELINE] Total candidates: {len(all_candidates)}", flush=True)
    if not all_candidates:
        print("[PIPELINE] No candidates generated. Check data paths.")
        return

    # -- 4. Label candidates (with incremental label cache) --
    labels_cache = label_cache_dir
    labels_manifest = _load_labels_manifest(labels_cache)
    label_code_hash = _compute_label_code_hash()

    days_to_relabel = _determine_relabel_days(
        all_candidates, labels_manifest, trading_days, label_code_hash, force_regen,
        grid_hash=grid_hash,
    )
    cached_days_info = labels_manifest.get("days", {}) if not force_regen else {}

    # Split candidates by entry day
    cands_by_day: dict[str, list[dict]] = defaultdict(list)
    for c in all_candidates:
        cands_by_day[c["day"]].append(c)

    reusable_days = set(cands_by_day.keys()) - days_to_relabel
    reusable_from_cache = {d for d in reusable_days if d in cached_days_info}
    n_reuse = sum(cached_days_info.get(d, {}).get("rows", 0) for d in reusable_from_cache)
    n_relabel = sum(len(cands_by_day[d]) for d in days_to_relabel)

    print(f"[PIPELINE] Label cache: {len(reusable_from_cache)} days reusable "
          f"(~{n_reuse:,} candidates), {len(days_to_relabel)} days to relabel "
          f"({n_relabel:,} candidates)", flush=True)

    # Gather candidates that need labeling
    cands_to_label = []
    for day in sorted(days_to_relabel):
        cands_to_label.extend(cands_by_day[day])

    # Label the subset
    if cands_to_label:
        print(f"[PIPELINE] Labeling {len(cands_to_label):,} candidates "
              f"(forward-looking, workers={workers}) ...", flush=True)
        label_candidates_fast(cands_to_label, trading_days, workers=workers)

    # Save newly labeled candidates to cache (per entry day)
    new_labels_days_info = dict(cached_days_info)
    for day in sorted(days_to_relabel):
        day_cands = cands_by_day[day]
        max_expiry = max((c.get("expiry", "0000-00-00") for c in day_cands), default="0000-00-00")
        _save_labeled_cache_day(labels_cache, day, day_cands)
        new_labels_days_info[day] = {
            "rows": len(day_cands),
            "max_expiry": max_expiry,
            "file": f"{day}.parquet",
        }

    # Load cached labeled candidates for reusable days, verifying row counts
    cached_labeled: list[dict] = []
    missing_cache_days: list[str] = []
    for day in sorted(reusable_from_cache):
        expected_rows = len(cands_by_day[day])
        cached_rows = cached_days_info.get(day, {}).get("rows", 0)
        if cached_rows != expected_rows:
            logger.warning(
                "Label cache row mismatch for %s: cached=%d, expected=%d — relabeling",
                day, cached_rows, expected_rows,
            )
            missing_cache_days.append(day)
            continue
        day_labeled = _load_labeled_cache_day(labels_cache, day)
        if day_labeled:
            cached_labeled.extend(day_labeled)
        else:
            missing_cache_days.append(day)

    if missing_cache_days:
        logger.warning(
            "Label cache files missing for %d 'reusable' days — relabeling them: %s",
            len(missing_cache_days), missing_cache_days[:5],
        )
        fallback_cands = []
        for day in missing_cache_days:
            fallback_cands.extend(cands_by_day[day])
        if fallback_cands:
            label_candidates_fast(fallback_cands, trading_days, workers=workers)
            cands_to_label.extend(fallback_cands)
            for day in missing_cache_days:
                day_cands = cands_by_day[day]
                max_expiry = max((c.get("expiry", "0000-00-00") for c in day_cands), default="0000-00-00")
                _save_labeled_cache_day(labels_cache, day, day_cands)
                new_labels_days_info[day] = {
                    "rows": len(day_cands),
                    "max_expiry": max_expiry,
                    "file": f"{day}.parquet",
                }

    # Persist manifest AFTER recovery so repaired days are included
    _save_labels_manifest(labels_cache, {
        "code_hash": label_code_hash,
        "grid_hash": grid_hash,
        "trading_days_list": trading_days,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "days": new_labels_days_info,
    })

    # Merge: freshly labeled + cached
    labeled = cands_to_label + cached_labeled
    resolved = [c for c in labeled if c.get("resolved")]
    print(f"[PIPELINE] Labeled: {len(labeled)}, Resolved: {len(resolved)}", flush=True)

    if not resolved:
        print("[PIPELINE] No resolved candidates. Cannot train.")
        return

    # -- 5. Build features --
    print("[PIPELINE] Building feature vectors ...", flush=True)
    training_rows = build_training_rows(resolved, cs_df=cs_df)
    print(f"[PIPELINE] Training rows: {len(training_rows)}", flush=True)

    # -- 6. Save CSV --
    print(f"[PIPELINE] Saving candidates to {OUTPUT_CSV} ...", flush=True)
    flat_records = []
    for c in resolved:
        rec = {k: v for k, v in c.items() if not isinstance(v, (dict, list))}
        flat_records.append(rec)
    before_filter = len(flat_records)
    flat_records = [r for r in flat_records if r.get("entry_credit", 0) >= 0.01]
    if before_filter != len(flat_records):
        logger.info(
            "Filtered %d near-zero credit candidates",
            before_filter - len(flat_records),
        )
    out_df = pd.DataFrame(flat_records)
    # L8 fix: sort by (day, entry_dt, spread_id) so re-runs with
    # different cache-hit ratios produce byte-identical CSVs.  Without
    # this, freshly-labeled rows (cands_to_label) precede cached rows
    # (cached_labeled) and the order shifts whenever the cache shape
    # changes.  Missing tiebreaker columns are tolerated.
    sort_keys = [c for c in ("day", "entry_dt", "spread_id") if c in out_df.columns]
    if sort_keys:
        out_df = out_df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    # L9 fix (extended): write to <path>.tmp then atomically replace,
    # so a crash mid-write doesn't leave a truncated CSV that downstream
    # consumers (run_pipeline.py, backtest_strategy.py) might treat as
    # a complete dataset.
    _atomic_write_csv(out_df, OUTPUT_CSV)

    # -- 7. Walk-forward validation --
    print("[PIPELINE] Walk-forward validation ...", flush=True)
    results = walk_forward_validate(training_rows)

    if "error" in results:
        logger.warning("%s -- training on all data instead", results["error"])
        final_model = train_bucket_model(rows=training_rows)
    else:
        ts = results["test_summary"]
        print(f"\n{'=' * 60}")
        print("WALK-FORWARD RESULTS")
        print(f"{'=' * 60}")
        print(f"  Train : {results['train_count']} rows ({results['train_days']})")
        print(f"  Test  : {results['test_count']} rows ({results['test_days']})")

        def _fmt(label: str, key: str, fmt: str = ".1%", prefix: str = "") -> None:
            """Print a single metric line if the key exists in the test summary."""
            val = ts.get(key)
            if val is not None:
                print(f"  {label}: {prefix}{val:{fmt}}")
            else:
                print(f"  {label}: N/A")

        _fmt("TP50 rate      ", "tp50_rate")
        _fmt("TP100 rate     ", "tp100_at_expiry_rate")
        _fmt("Expectancy     ", "expectancy", ".2f", "$")
        _fmt("Max drawdown   ", "max_drawdown", ".2f", "$")
        _fmt("Tail loss proxy", "tail_loss_proxy", ".2f", "$")
        _fmt("Avg margin     ", "avg_margin_usage", ".2f", "$")
        print(f"{'=' * 60}\n")

        # -- 8. Train final model on ALL data --
        print("[PIPELINE] Training final model on all data ...")
        final_model = train_bucket_model(rows=training_rows)

    g = final_model.get("global", {})
    hier = final_model.get("bucket_hierarchy", {})
    print(f"  Buckets (full)          : {len(final_model.get('buckets', {}))}")
    print(f"  Buckets (relaxed_market): {len(hier.get('relaxed_market', {}))}")
    print(f"  Buckets (core)          : {len(hier.get('core', {}))}")
    print(f"  Prior strength          : {final_model.get('prior_strength', 'N/A')}")
    print(f"  Global TP50             : {g.get('prob_tp50', 'N/A')}")
    print(f"  Global E[PnL]           : {g.get('expected_pnl', 'N/A')}")

    if deploy:
        deploy_model(
            final_model,
            description=f"offline-pipeline {len(training_rows)} rows",
        )

    elapsed = time.time() - t0
    print(f"\n[PIPELINE] Done in {elapsed:.1f}s")
def precompute_offline_gex_to_csv(
    *, max_days: int | None = None, output: Path = OFFLINE_GEX_CSV,
) -> None:
    """Precompute offline GEX for all trading days and save to CSV.

    For each available trading day and decision time, loads CBBO, definitions,
    and statistics data, computes GEX via ``compute_offline_gex()``, and writes
    the results to a CSV with columns matching ``context_snapshots_export.csv``
    so the pipeline can merge both sources.

    Parameters
    ----------
    max_days:
        Optional limit on the number of days to process (for testing).
    output:
        Destination path for the cache CSV.
    """
    t0 = time.time()
    spxw_days = set(_available_day_files(SPXW_CBBO))
    spx_days = set(_available_day_files(SPX_CBBO))
    trading_days = sorted(spxw_days | spx_days)
    if max_days:
        trading_days = trading_days[:max_days]

    spy_df = load_spy_equity()
    frd_spx = load_frd_quotes(FRD_SPX, "SPX")
    empty_prod = pd.DataFrame(columns=["ts", "symbol", "last"])
    uq_df = merge_underlying_quotes(empty_prod, frd_spx)
    print(f"[GEX-PRECOMPUTE] {len(trading_days)} trading days, "
          f"SPY equity: {len(spy_df)} rows, FRD SPX: {len(frd_spx)} rows",
          flush=True)

    rows: list[dict] = []

    for day_str in trading_days:
        day_date = date(int(day_str[:4]), int(day_str[4:6]), int(day_str[6:]))

        def_df = load_definitions(day_str)
        if def_df is None:
            print(f"  [{day_str}] no definitions -- skip", flush=True)
            continue
        inst_map = build_instrument_map(def_df)
        del def_df
        if not inst_map:
            continue

        cbbo_df = load_cbbo(day_str)
        if cbbo_df is None or cbbo_df.empty:
            print(f"  [{day_str}] no cbbo -- skip", flush=True)
            continue

        oi_df = load_statistics(day_str)
        if oi_df is None or oi_df.empty:
            print(f"  [{day_str}] no statistics -- skip", flush=True)
            del cbbo_df, inst_map
            continue

        day_gex_count = 0
        for hour, minute in DECISION_MINUTES_ET:
            dec_et = datetime(
                day_date.year, day_date.month, day_date.day,
                hour, minute, tzinfo=ET,
            )
            dec_utc = dec_et.astimezone(timezone.utc)
            snapshot = get_cbbo_snapshot_at(cbbo_df, dec_utc)
            if snapshot.empty:
                continue

            # SPX spot: prefer FRD SPX direct price, fall back to SPY + parity
            spx_spot_frd = lookup_intraday_value(uq_df, "SPX", dec_utc)
            if spx_spot_frd is not None:
                spx_spot = spx_spot_frd
                spy_price = spx_spot / SPY_SPX_RATIO
            else:
                spy_mask = (
                    (spy_df["ts"] >= dec_utc - timedelta(minutes=5))
                    & (spy_df["ts"] <= dec_utc)
                )
                spy_rows = spy_df[spy_mask]
                if spy_rows.empty:
                    continue
                spy_price = float(spy_rows.iloc[-1]["close"])
                spx_spot = derive_spx_from_parity(snapshot, inst_map, day_date, spy_price)

            gex_net, zero_gamma = compute_offline_gex(
                snapshot, inst_map, oi_df, spx_spot, day_date,
            )

            rows.append({
                "ts": dec_utc.isoformat(),
                "gex_net": gex_net,
                "zero_gamma_level": zero_gamma,
                "spx_price": spx_spot,
                "spy_price": spy_price,
                "source": "offline_databento",
            })
            day_gex_count += 1

        print(f"  [{day_str}] {day_gex_count} GEX points computed", flush=True)
        del cbbo_df, inst_map, oi_df
        gc.collect()

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    non_null = df["gex_net"].notna().sum() if not df.empty else 0
    elapsed = time.time() - t0
    print(f"\n[GEX-PRECOMPUTE] Done in {elapsed:.1f}s")
    print(f"  Rows: {len(df)}, GEX non-null: {non_null}")
    print(f"  Saved to: {output}", flush=True)
def main() -> None:
    """Parse arguments and run the offline training pipeline."""
    parser = argparse.ArgumentParser(
        description="Offline SPX credit-spread training pipeline",
    )
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Limit processing to first N trading days (for quick tests)",
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help="Insert trained model into production model_versions as shadow",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-day progress output",
    )
    parser.add_argument(
        "--precompute-gex", action="store_true",
        help="Precompute offline GEX to CSV cache instead of running the pipeline",
    )
    parser.add_argument(
        "--relabel", action="store_true",
        help="Re-label existing training_candidates.csv with trajectory data "
             "(reloads CBBO marks for SL trades only, ~3-5 hours)",
    )
    parser.add_argument(
        "--model", type=str, default="bucket", choices=["bucket", "xgboost"],
        help="Model type: 'bucket' (default) runs full pipeline, "
             "'xgboost' reads existing training_candidates.csv and trains XGBoost",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers for the labeling step "
             "(default: min(8, cpu_count))",
    )
    parser.add_argument(
        "--force-regen", action="store_true",
        help="Ignore both candidate and label caches — regenerate and relabel all days from scratch",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=str(CANDIDATES_CACHE_DIR),
        help=f"Directory for per-day Parquet candidate cache (default: {CANDIDATES_CACHE_DIR})",
    )
    parser.add_argument(
        "--label-cache-dir", type=str, default=str(LABELS_CACHE_DIR),
        help=f"Directory for per-day Parquet label cache (default: {LABELS_CACHE_DIR})",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a training grid YAML config file. Overrides hardcoded "
             "entry times, DTEs, deltas, widths, and sides. "
             "(e.g. backend/configs/training/narrow.yaml)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Only process trading days on or after this date (YYYYMMDD or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Only process trading days on or before this date (YYYYMMDD or YYYY-MM-DD)",
    )
    args = parser.parse_args()

    if args.precompute_gex:
        precompute_offline_gex_to_csv(max_days=args.max_days)
        return

    if args.relabel:
        if args.config:
            sys.path.insert(0, str(_BACKEND))
            from configs.training.schema import load_training_config
            _cfg = load_training_config(args.config)
            global _ACTIVE_GRID
            _ACTIVE_GRID = {
                "DECISION_MINUTES_ET": _cfg.as_tuples(),
                "DTE_TARGETS": _cfg.dte_targets,
                "DELTA_TARGETS": _cfg.delta_targets,
                "SPREAD_SIDES": _cfg.spread_sides,
                "WIDTH_TARGETS": _cfg.width_targets,
                "TAKE_PROFIT_PCT": _cfg.take_profit_pct,
                "STOP_LOSS_PCT": _cfg.stop_loss_pct,
                "LABEL_MARK_INTERVAL_MINUTES": _cfg.label_mark_interval_minutes,
            }
            print(f"[RELABEL] Using config '{_cfg.name}' "
                  f"(TP={_cfg.take_profit_pct}, SL={_cfg.stop_loss_pct}) — "
                  f"applies to SL re-evaluation and mark downsampling only; "
                  f"non-SL rows keep existing labels from CSV", flush=True)
        relabel_from_csv()
        return

    if args.model == "xgboost":
        from xgb_model import main as xgb_main
        xgb_main()
        return

    training_config = None
    if args.config:
        sys.path.insert(0, str(_BACKEND))
        from configs.training.schema import load_training_config
        training_config = load_training_config(args.config)

    run_pipeline(
        max_days=args.max_days,
        deploy=args.deploy,
        verbose=not args.quiet,
        workers=args.workers,
        force_regen=args.force_regen,
        cache_dir=Path(args.cache_dir),
        label_cache_dir=Path(args.label_cache_dir),
        training_config=training_config,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
