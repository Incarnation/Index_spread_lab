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
    """Fail loudly if the live SL config drifts from the training labeler's
    hardcoded contract.  This is the C4 hybrid mitigation
    (see ``OFFLINE_PIPELINE_AUDIT.md``).

    The labeler in this script computes ``sl_thr = max_profit * STOP_LOSS_PCT``
    and always treats SL as enabled.  If production overrides
    ``trade_pnl_stop_loss_basis`` (e.g. to ``"max_loss"``), disables SL
    via ``trade_pnl_stop_loss_enabled = False``, or changes
    ``trade_pnl_stop_loss_pct`` away from the active grid's
    ``STOP_LOSS_PCT``, training labels will silently misrepresent live
    trade outcomes and any downstream model / optimizer run will be
    calibrated to the wrong policy.

    The helper deliberately raises ``SystemExit`` rather than logging a
    warning so the pipeline cannot complete a labeling run with a stale
    hardcoded contract.
    """
    try:
        from spx_backend.config import settings as _live_settings
    except Exception as exc:  # pragma: no cover - defensive
        # If settings can't be loaded (e.g. pytest with mocked env) we'd
        # rather skip the check than block tests.  Production runs always
        # import this module successfully because the rest of this file
        # already imports from spx_backend above.
        logger.warning("Skipping SL-alignment assertion: %s", exc)
        return

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

    grid_pct = float(_get_grid_param("STOP_LOSS_PCT"))
    live_pct = float(_live_settings.trade_pnl_stop_loss_pct)
    if abs(live_pct - grid_pct) > 1e-9:
        raise SystemExit(
            f"[C4] Active training STOP_LOSS_PCT={grid_pct} != live "
            f"trade_pnl_stop_loss_pct={live_pct}. If you intentionally "
            "want a what-if SL multiplier, update the YAML and the live "
            "config together (or temporarily comment out this assertion "
            "with a tracking issue). See OFFLINE_PIPELINE_AUDIT.md C4."
        )

from .io_loaders import (
    load_cbbo,
    load_definitions,
    build_instrument_map,
)
from .candidates import (
    _atomic_write_csv,
)


def _downsample_marks(
    marks: list[dict], interval_minutes: int = LABEL_MARK_INTERVAL_MINUTES,
) -> list[dict]:
    """Keep only marks whose minute is divisible by *interval_minutes*.

    Mirrors the production snapshot cadence (e.g. every 5 minutes) so that
    the offline labeler doesn't catch fleeting intraday touches that the
    production system would never observe.

    Parameters
    ----------
    marks            : list of mark dicts, each with a ``ts`` datetime key.
    interval_minutes : keep marks where ``ts.minute % interval == 0``.
                       A value of 1 means keep every mark (no filtering).

    Returns
    -------
    list[dict]  Filtered marks (same order, always a subset of input).
    """
    if interval_minutes <= 1:
        return marks
    return [m for m in marks if m["ts"].minute % interval_minutes == 0]
TP_LEVELS = [50, 60, 70, 80, 90, 100]
def _evaluate_outcome(entry_credit: float, marks: list[dict]) -> dict:
    """Evaluate credit-spread outcome from forward marks with full trajectory.

    Iterates ALL marks without early-returning at stop-loss, capturing
    trajectory data that enables analysis of any SL level (1x-5x) and any
    TP exit rule (50%-100% or hold-to-expiry) from a single pipeline run.

    For each TP level in TP_LEVELS (50, 60, 70, 80, 90, 100) the function
    records:
      - ``first_tpXX_pnl``        PnL at first crossing of XX% of max_profit
      - ``min_pnl_before_tpXX``   worst PnL before that crossing (for SL ordering)

    Additional trajectory columns:
      - ``max_favorable_pnl``     peak PnL reached at any point
      - ``max_adverse_pnl``       worst PnL seen across all marks
      - ``max_adverse_multiple``   max_adverse_pnl / max_profit
      - ``final_pnl_at_expiry``   PnL at the very last mark
      - ``hit_stop_loss``         True if PnL breached -SL threshold at any point
      - ``recovered_after_sl``    True if SL breached, then primary TP fired later

    The primary TP level is derived from ``TAKE_PROFIT_PCT`` in ``_ACTIVE_GRID``
    (e.g. 0.50 -> TP50, 0.60 -> TP60).  If the configured level is not in
    ``TP_LEVELS``, a warning is logged and 50 is used as the fallback.

    Backward-compatible fields (``hit_tp50``, ``realized_pnl``,
    ``exit_reason``) reflect the *configured* primary TP level (not always 50).

    Hold-through columns (no-SL + close-at-first-configured-TP strategy):
      - ``hold_hit_tp50``, ``hold_realized_pnl``, ``hold_exit_reason``,
        ``hold_hit_tp100_at_expiry``

    Parameters
    ----------
    entry_credit : float
        Credit received at entry (positive).
    marks : list[dict]
        Forward CBBO marks with short/long bid/ask.

    Returns
    -------
    dict
        Outcome fields with all multi-TP trajectory data.
    """
    _tp_nulls: dict = {}
    for lvl in TP_LEVELS:
        _tp_nulls[f"first_tp{lvl}_pnl"] = None
        _tp_nulls[f"min_pnl_before_tp{lvl}"] = None

    _trajectory_nulls = {
        "max_adverse_pnl": None, "max_adverse_multiple": None,
        "max_favorable_pnl": None,
        "final_pnl_at_expiry": None, "final_is_tp100": False,
        "hit_stop_loss": False, "recovered_after_sl": False,
        "hold_hit_tp50": False, "hold_realized_pnl": None,
        "hold_exit_reason": "NO_MARKS",
        "hold_hit_tp100_at_expiry": False,
        **_tp_nulls,
    }

    if not marks:
        return {
            "resolved": False, "hit_tp50": False,
            "hit_tp100_at_expiry": False,
            "realized_pnl": None, "exit_reason": "NO_MARKS",
            **_trajectory_nulls,
        }

    max_profit = entry_credit * CONTRACT_MULT * CONTRACTS
    sl_thr = max_profit * _get_grid_param("STOP_LOSS_PCT")

    # Derive the primary TP level from YAML config (e.g. 0.50 -> 50)
    primary_tp_lvl = round(_get_grid_param("TAKE_PROFIT_PCT") * 100)
    if primary_tp_lvl not in TP_LEVELS:
        logger.warning(
            "Configured TAKE_PROFIT_PCT %.2f maps to TP level %d which is not "
            "in TP_LEVELS %s — falling back to 50",
            _get_grid_param("TAKE_PROFIT_PCT"), primary_tp_lvl, TP_LEVELS,
        )
        primary_tp_lvl = 50

    # Per-TP-level thresholds and tracking
    tp_thresholds = {lvl: max_profit * (lvl / 100.0) for lvl in TP_LEVELS}
    first_tp_pnl: dict[int, float | None] = {lvl: None for lvl in TP_LEVELS}
    min_pnl_before_tp: dict[int, float | None] = {lvl: None for lvl in TP_LEVELS}

    max_adverse_pnl: float = 0.0
    max_favorable_pnl: float = 0.0
    sl_breach_pnl: float | None = None
    sl_breached_before_primary_tp: bool = False
    last_pnl: float | None = None

    for m in marks:
        s_mid = mid_price(m["short_bid"], m["short_ask"])
        l_mid = mid_price(m["long_bid"], m["long_ask"])
        if s_mid is None or l_mid is None:
            continue
        exit_cost = s_mid - l_mid
        pnl = (entry_credit - exit_cost) * CONTRACT_MULT * CONTRACTS
        last_pnl = pnl

        max_adverse_pnl = min(max_adverse_pnl, pnl)
        max_favorable_pnl = max(max_favorable_pnl, pnl)

        for lvl in TP_LEVELS:
            if first_tp_pnl[lvl] is None:
                if min_pnl_before_tp[lvl] is None:
                    min_pnl_before_tp[lvl] = pnl
                else:
                    min_pnl_before_tp[lvl] = min(min_pnl_before_tp[lvl], pnl)
                if pnl >= tp_thresholds[lvl]:
                    first_tp_pnl[lvl] = pnl

        if sl_breach_pnl is None and pnl <= -sl_thr:
            sl_breach_pnl = pnl
            if first_tp_pnl[primary_tp_lvl] is None:
                sl_breached_before_primary_tp = True

    if last_pnl is None:
        _trajectory_nulls["hold_exit_reason"] = "NO_VALID_MARKS"
        return {
            "resolved": False, "hit_tp50": False,
            "hit_tp100_at_expiry": False,
            "realized_pnl": None, "exit_reason": "NO_VALID_MARKS",
            **_trajectory_nulls,
        }

    max_adverse_multiple = (
        max_adverse_pnl / max_profit if max_profit > 0 else None
    )
    final_is_tp100 = bool(last_pnl >= max_profit)
    hit_stop_loss = sl_breach_pnl is not None
    recovered_after_sl = (
        sl_breached_before_primary_tp
        and first_tp_pnl[primary_tp_lvl] is not None
    )

    # --- Original outcome (with-SL, uses configured primary TP level) ---
    if sl_breached_before_primary_tp:
        orig_exit = "STOP_LOSS"
        orig_pnl = sl_breach_pnl
        orig_tp50 = False
        orig_tp100 = False
    elif first_tp_pnl[primary_tp_lvl] is not None:
        orig_exit = f"TAKE_PROFIT_{primary_tp_lvl}"
        orig_pnl = first_tp_pnl[primary_tp_lvl]
        orig_tp50 = True
        orig_tp100 = final_is_tp100
    else:
        orig_exit = "EXPIRY_OR_LAST_MARK"
        orig_pnl = last_pnl
        orig_tp50 = False
        orig_tp100 = final_is_tp100

    # --- Hold-through outcome (ignore SL, close at first configured TP) ---
    if first_tp_pnl[primary_tp_lvl] is not None:
        hold_exit = f"TAKE_PROFIT_{primary_tp_lvl}"
        hold_pnl = first_tp_pnl[primary_tp_lvl]
        hold_tp50 = True
    else:
        hold_exit = "EXPIRY_OR_LAST_MARK"
        hold_pnl = last_pnl
        hold_tp50 = False

    result = {
        "resolved": True,
        "hit_tp50": orig_tp50,
        "hit_tp100_at_expiry": orig_tp100,
        "realized_pnl": orig_pnl,
        "exit_reason": orig_exit,
        "max_adverse_pnl": max_adverse_pnl,
        "max_adverse_multiple": max_adverse_multiple,
        "max_favorable_pnl": max_favorable_pnl,
        "final_pnl_at_expiry": last_pnl,
        "final_is_tp100": final_is_tp100,
        "hit_stop_loss": hit_stop_loss,
        "recovered_after_sl": recovered_after_sl,
        "hold_hit_tp50": hold_tp50,
        "hold_realized_pnl": hold_pnl,
        "hold_exit_reason": hold_exit,
        "hold_hit_tp100_at_expiry": final_is_tp100,
    }

    # Multi-TP trajectory columns
    for lvl in TP_LEVELS:
        result[f"first_tp{lvl}_pnl"] = first_tp_pnl[lvl]
        result[f"min_pnl_before_tp{lvl}"] = min_pnl_before_tp[lvl]

    # Backward-compat aliases (same values as multi-TP keyed versions)
    result["min_pnl_before_tp50"] = min_pnl_before_tp[50]
    result["first_tp50_pnl"] = first_tp_pnl[50]

    return result
def label_candidates(
    candidates: list[dict], trading_days: list[str],
) -> list[dict]:
    """Label every candidate by loading forward cbbo-1m minute bars.

    For multi-DTE trades the forward days are loaded on-demand.  A small
    LRU cache keeps recently-loaded DataFrames and per-day symbol-to-
    instrument_id maps in memory to avoid re-reading files.

    Parameters
    ----------
    candidates   : list of candidate dicts (with entry info).
    trading_days : sorted list of YYYYMMDD strings.

    Returns
    -------
    list[dict]  Candidates augmented with outcome fields.
    """
    if not candidates:
        return []

    day_cache: dict[str, tuple[pd.DataFrame, dict[str, int]]] = {}
    max_cache = 15

    def _ensure_cached(day_str: str) -> tuple[pd.DataFrame, dict[str, int]] | None:
        """Load cbbo + build symbol->iid map; cache result."""
        if day_str in day_cache:
            return day_cache[day_str]
        cbbo = load_cbbo(day_str)
        if cbbo is None:
            return None
        def_df = load_definitions(day_str)
        if def_df is None:
            return None
        imap = build_instrument_map(def_df)
        sym_to_iid = {v["raw_symbol"]: k for k, v in imap.items()}
        day_cache[day_str] = (cbbo, sym_to_iid)
        while len(day_cache) > max_cache:
            oldest = min(day_cache)
            del day_cache[oldest]
        return (cbbo, sym_to_iid)

    labeled: list[dict] = []

    for ci, c in enumerate(candidates):
        entry_date = date.fromisoformat(c["day"])
        expiry_date = date.fromisoformat(c["expiry"])
        entry_dt = datetime.fromisoformat(c["entry_dt"])

        needed = [
            d for d in trading_days
            if entry_date <= date(int(d[:4]), int(d[4:6]), int(d[6:])) <= expiry_date
        ]
        if not needed:
            c.update(_evaluate_outcome(c["entry_credit"], []))
            labeled.append(c)
            continue

        short_sym = c["short_symbol"]
        long_sym = c["long_symbol"]
        marks: list[dict] = []

        for day_str in needed:
            cached = _ensure_cached(day_str)
            if cached is None:
                continue
            cbbo, sym_to_iid = cached

            s_iid = sym_to_iid.get(short_sym)
            l_iid = sym_to_iid.get(long_sym)

            # Fall back to entry-day instrument_ids if symbols not found
            if s_iid is None:
                s_iid = c["short_instrument_id"]
            if l_iid is None:
                l_iid = c["long_instrument_id"]

            mask = cbbo["instrument_id"].isin({s_iid, l_iid})
            relevant = cbbo.loc[mask]
            if relevant.empty:
                continue

            for ts_val, grp in relevant.groupby("ts"):
                ts_pd = pd.Timestamp(ts_val)
                if ts_pd.tzinfo is None:
                    ts_pd = ts_pd.tz_localize("UTC")
                ts_dt = ts_pd.to_pydatetime()
                if ts_dt <= entry_dt:
                    continue
                sr = grp[grp["instrument_id"] == s_iid]
                lr = grp[grp["instrument_id"] == l_iid]
                if sr.empty or lr.empty:
                    continue
                marks.append({
                    "ts": ts_dt,
                    "short_bid": float(sr.iloc[0]["bid_px_00"]),
                    "short_ask": float(sr.iloc[0]["ask_px_00"]),
                    "long_bid": float(lr.iloc[0]["bid_px_00"]),
                    "long_ask": float(lr.iloc[0]["ask_px_00"]),
                })

        marks.sort(key=lambda m: m["ts"])
        marks = _downsample_marks(marks, _get_grid_param("LABEL_MARK_INTERVAL_MINUTES"))

        c.update(_evaluate_outcome(c["entry_credit"], marks))
        labeled.append(c)

        if (ci + 1) % 50 == 0:
            print(f"  Labeled {ci + 1}/{len(candidates)} candidates", flush=True)

    return labeled
label_candidates_sequential = label_candidates
def _extract_marks_for_day(
    args: tuple[str, list[tuple]],
) -> list[tuple[int, list[dict]]]:
    """Worker function: load one day's CBBO once and extract marks for every candidate needing it.

    Top-level function so it can be pickled by multiprocessing.Pool.

    Parameters
    ----------
    args : tuple
        ``(day_str, candidate_requests)`` where *candidate_requests* is a list
        of ``(cand_idx, short_sym, long_sym, s_iid_fallback, l_iid_fallback,
        entry_dt_iso)`` tuples.

    Returns
    -------
    list[tuple[int, list[dict]]]
        ``(candidate_index, marks)`` pairs for every candidate that had at
        least one matching mark on this day.
    """
    day_str, candidate_requests = args

    cbbo = load_cbbo(day_str)
    if cbbo is None:
        return []
    def_df = load_definitions(day_str)
    if def_df is None:
        return []

    imap = build_instrument_map(def_df)
    sym_to_iid: dict[str, int] = {v["raw_symbol"]: k for k, v in imap.items()}

    # Vectorized pre-index: instrument_id -> ndarray of [ts, bid, ask] rows.
    # Replaces the expensive per-candidate DataFrame.isin() + groupby.
    idx: dict[int, np.ndarray] = {}
    for iid, grp in cbbo.groupby("instrument_id"):
        idx[int(iid)] = grp[["ts", "bid_px_00", "ask_px_00"]].values

    results: list[tuple[int, list[dict]]] = []

    for ci, short_sym, long_sym, s_iid_fb, l_iid_fb, entry_dt_iso in candidate_requests:
        entry_dt = datetime.fromisoformat(entry_dt_iso)

        s_iid = sym_to_iid.get(short_sym, s_iid_fb)
        l_iid = sym_to_iid.get(long_sym, l_iid_fb)

        s_rows = idx.get(s_iid)
        l_rows = idx.get(l_iid)
        if s_rows is None or l_rows is None:
            continue

        # Timestamp-keyed dicts for O(1) intersection
        s_dict = {ts: (bid, ask) for ts, bid, ask in s_rows}
        l_dict = {ts: (bid, ask) for ts, bid, ask in l_rows}

        marks: list[dict] = []
        for ts_val in sorted(set(s_dict) & set(l_dict)):
            ts_pd = pd.Timestamp(ts_val)
            if ts_pd.tzinfo is None:
                ts_pd = ts_pd.tz_localize("UTC")
            ts_dt = ts_pd.to_pydatetime()
            if ts_dt <= entry_dt:
                continue
            s_bid, s_ask = s_dict[ts_val]
            l_bid, l_ask = l_dict[ts_val]
            marks.append({
                "ts": ts_dt,
                "short_bid": float(s_bid),
                "short_ask": float(s_ask),
                "long_bid": float(l_bid),
                "long_ask": float(l_ask),
            })

        if marks:
            results.append((ci, marks))

    return results
def label_candidates_fast(
    candidates: list[dict],
    trading_days: list[str],
    *,
    workers: int = 8,
) -> list[dict]:
    """Label candidates using inverted day-iteration with optional multiprocessing.

    Instead of scanning the full CBBO DataFrame per candidate, this flips the
    loop: for each forward day, load CBBO once, pre-index by instrument_id,
    and extract marks for ALL candidates needing that day in a single pass.
    Multiprocessing distributes forward days across *workers* processes.

    Parameters
    ----------
    candidates   : list of candidate dicts (must include ``day``, ``expiry``,
                   ``entry_dt``, ``short_symbol``, ``long_symbol``,
                   ``short_instrument_id``, ``long_instrument_id``,
                   ``entry_credit``).
    trading_days : sorted list of ``YYYYMMDD`` strings covering the data range.
    workers      : number of parallel workers (1 = sequential, no Pool overhead).

    Returns
    -------
    list[dict]
        Candidates augmented with outcome fields from ``_evaluate_outcome``.
    """
    if not candidates:
        return []

    # Phase 1: build extraction plan — forward_day -> [(cand_idx, ...)]
    plan: dict[str, list[tuple]] = defaultdict(list)
    for ci, c in enumerate(candidates):
        entry_date = date.fromisoformat(c["day"])
        expiry_date = date.fromisoformat(c["expiry"])
        for d in trading_days:
            d_date = date(int(d[:4]), int(d[4:6]), int(d[6:]))
            if entry_date <= d_date <= expiry_date:
                plan[d].append((
                    ci,
                    c["short_symbol"],
                    c["long_symbol"],
                    c["short_instrument_id"],
                    c["long_instrument_id"],
                    c["entry_dt"],
                ))

    n_forward_days = len(plan)
    print(
        f"  Extraction plan: {n_forward_days} forward days "
        f"for {len(candidates)} candidates",
        flush=True,
    )

    # Phase 2 + 3: extract marks then evaluate in memory-bounded batches.
    # Each batch accumulates marks for its slice of candidates across all
    # forward days, evaluates outcomes, and frees memory before the next batch.
    BATCH_SIZE = 50_000
    n_cands = len(candidates)
    n_batches = (n_cands + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Processing in {n_batches} batch(es) of up to {BATCH_SIZE:,}", flush=True)

    t_label_start = time.time()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n_cands)
        batch_cis = set(range(batch_start, batch_end))
        batch_size_actual = batch_end - batch_start

        batch_marks: list[list[dict]] = [[] for _ in range(batch_size_actual)]

        batch_day_args: list[tuple[str, list[tuple]]] = []
        for day_str, requests in plan.items():
            filtered = [r for r in requests if r[0] in batch_cis]
            if filtered:
                batch_day_args.append((day_str, filtered))
        batch_day_args.sort()

        n_batch_days = len(batch_day_args)

        if workers > 1 and n_batch_days > 1:
            actual_workers = min(workers, n_batch_days)
            with Pool(actual_workers) as pool:
                for di, day_results in enumerate(
                    pool.imap_unordered(_extract_marks_for_day, batch_day_args),
                ):
                    for ci, marks in day_results:
                        batch_marks[ci - batch_start].extend(marks)
                    done = di + 1
                    if done % 25 == 0 or done == n_batch_days:
                        elapsed = time.time() - t_label_start
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (n_batch_days - done) / rate if rate > 0 else 0
                        print(
                            f"  Batch {batch_idx + 1}/{n_batches}: "
                            f"extracted marks {done}/{n_batch_days} forward days "
                            f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                            flush=True,
                        )
        else:
            for di, day_item in enumerate(batch_day_args):
                day_results = _extract_marks_for_day(day_item)
                for ci, marks in day_results:
                    batch_marks[ci - batch_start].extend(marks)
                done = di + 1
                if done % 25 == 0 or done == n_batch_days:
                    elapsed = time.time() - t_label_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_batch_days - done) / rate if rate > 0 else 0
                    print(
                        f"  Batch {batch_idx + 1}/{n_batches}: "
                        f"extracted marks {done}/{n_batch_days} forward days "
                        f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                        flush=True,
                    )

        for i in range(batch_size_actual):
            ci = batch_start + i
            c = candidates[ci]
            marks = sorted(batch_marks[i], key=lambda m: m["ts"])
            marks = _downsample_marks(marks, _get_grid_param("LABEL_MARK_INTERVAL_MINUTES"))
            c.update(_evaluate_outcome(c["entry_credit"], marks))
            batch_marks[i] = []

        print(
            f"  Batch {batch_idx + 1}/{n_batches}: "
            f"evaluated {batch_size_actual:,} candidates "
            f"({batch_end:,}/{n_cands:,} total)",
            flush=True,
        )
        del batch_marks

    print(f"  Labeling complete: {len(candidates)} candidates", flush=True)
    return candidates
LABELS_CACHE_DIR = DATA_DIR / "labels_cache"
def _load_labels_manifest(cache_dir: Path) -> dict:
    """Load the label cache manifest, or empty dict if missing/corrupt."""
    manifest_path = cache_dir / "labels_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Corrupt labels manifest %s: %s — treating as empty", manifest_path, exc)
    return {}
def _save_labels_manifest(cache_dir: Path, manifest: dict) -> None:
    """Atomically write the label cache manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "labels_manifest.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(cache_dir / "labels_manifest.json")
def _compute_label_code_hash() -> str:
    """Hash of label-related code sections for cache invalidation.

    Uses the full script hash (same as candidate cache) since labeling
    logic and evaluation logic are interleaved in the same file.
    """
    return _compute_code_version()
def _compute_days_hash(trading_days: list[str]) -> str:
    """Deterministic hash of the trading days list.

    Any change (new days added, days removed) invalidates entries
    whose expiry could be affected.
    """
    return hashlib.sha256(",".join(trading_days).encode()).hexdigest()[:16]
def _determine_relabel_days(
    all_candidates: list[dict],
    manifest: dict,
    trading_days: list[str],
    code_hash: str,
    force_regen: bool,
    grid_hash: str = "default",
) -> set[str]:
    """Determine which entry-days need re-labeling vs can be loaded from cache.

    Parameters
    ----------
    all_candidates : Full list of generated (unlabeled) candidates.
    manifest : Current labels_manifest.json contents.
    trading_days : Sorted list of all available trading day strings (compact YYYYMMDD).
    code_hash : Current label code hash.
    force_regen : Skip all caching.
    grid_hash : Hash of the training grid config. A change means different
        candidates were generated so all labels must be regenerated.

    Returns
    -------
    Set of entry-day strings (ISO YYYY-MM-DD) that need fresh labeling.
    """
    if force_regen or not manifest:
        return {c["day"] for c in all_candidates}

    if manifest.get("code_hash") != code_hash:
        return {c["day"] for c in all_candidates}

    if manifest.get("grid_hash", "default") != grid_hash:
        return {c["day"] for c in all_candidates}

    cached_days_info = manifest.get("days", {})
    prev_trading_days = set(manifest.get("trading_days_list", []))
    current_trading_days = set(trading_days)
    new_data_days = current_trading_days - prev_trading_days

    entry_days = sorted({c["day"] for c in all_candidates})
    days_to_relabel: set[str] = set()

    if not new_data_days:
        for day in entry_days:
            if day not in cached_days_info:
                days_to_relabel.add(day)
        return days_to_relabel

    # Convert compact YYYYMMDD to ISO for comparison with max_expiry
    min_new_compact = min(new_data_days)
    min_new_iso = f"{min_new_compact[:4]}-{min_new_compact[4:6]}-{min_new_compact[6:]}"

    for day in entry_days:
        info = cached_days_info.get(day)
        if info is None:
            days_to_relabel.add(day)
        else:
            cached_expiry = info.get("max_expiry", "9999-99-99")
            # Normalize: if cached_expiry is compact, convert to ISO
            if len(cached_expiry) == 8 and cached_expiry.isdigit():
                cached_expiry = f"{cached_expiry[:4]}-{cached_expiry[4:6]}-{cached_expiry[6:]}"
            if cached_expiry >= min_new_iso:
                days_to_relabel.add(day)

    return days_to_relabel
def _load_labeled_cache_day(cache_dir: Path, day_str: str) -> list[dict]:
    """Load labeled candidates for one entry day from the label cache.

    Returns an empty list if the file is missing or corrupt so callers
    can fall back to re-labeling.
    """
    fp = cache_dir / f"{day_str}.parquet"
    if not fp.exists():
        return []
    try:
        return pd.read_parquet(fp).to_dict("records")
    except Exception:
        logger.warning("Corrupt label cache %s — will re-label", fp)
        return []
def _save_labeled_cache_day(cache_dir: Path, day_str: str, candidates: list[dict]) -> None:
    """Write labeled candidates for one entry day to the label cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"{day_str}.parquet"
    if candidates:
        df = pd.DataFrame(candidates)
        safe_cols = [c for c in df.columns if not df[c].apply(lambda v: isinstance(v, (dict, list))).any()]
        df[safe_cols].to_parquet(fp, index=False)
    elif fp.exists():
        fp.unlink()
def build_training_rows(
    labeled_candidates: list[dict],
    cs_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Convert labeled candidates into training rows with production features.

    Uses ``extract_candidate_features`` from ``modeling.py`` to ensure the
    feature vectors are identical to what the production scoring path sees.

    Parameters
    ----------
    labeled_candidates:
        Candidates with resolved PnL labels from ``label_candidates``.
    cs_df:
        Optional context-snapshots DataFrame (from ``load_context_snapshots``)
        used to populate GEX-derived ``context_flags``.  When ``None`` or
        empty, context_flags defaults to ``[]`` (neutral regime).

    Returns
    -------
    list[dict]  Each dict has ``features``, ``realized_pnl``, ``hit_tp50``,
                ``hit_tp100_at_expiry``, ``margin_usage``, and metadata.
    """
    if cs_df is None:
        cs_df = pd.DataFrame(columns=["ts", "gex_net", "zero_gamma_level"])

    rows: list[dict[str, Any]] = []
    for c in labeled_candidates:
        if not c.get("resolved"):
            continue
        pnl = c.get("realized_pnl")
        if pnl is None or (isinstance(pnl, float) and math.isnan(pnl)):
            continue

        entry_dt_str = c.get("entry_dt", "")
        spread_side = c["spread_side"]
        spot = c.get("spot")

        context_flags: list[str] = []
        gex_source = "none"

        # Try production context_snapshots first
        if entry_dt_str and not cs_df.empty:
            try:
                dt_utc = pd.Timestamp(entry_dt_str).to_pydatetime()
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=ZoneInfo("UTC"))
                context_flags, gex_net, _ = lookup_gex_context(
                    cs_df, dt_utc, spread_side, spot,
                )
                if gex_net is not None:
                    gex_source = "production"
            except Exception:
                pass

        # Fall back to offline (Databento-computed) GEX
        if gex_source == "none":
            offline_gex = c.get("offline_gex_net")
            offline_zg = c.get("offline_zero_gamma")
            if offline_gex is not None:
                gex_source = "offline"
                if spread_side == "put":
                    context_flags.append("gex_support" if offline_gex >= 0 else "gex_headwind")
                else:
                    context_flags.append("gex_support" if offline_gex <= 0 else "gex_headwind")
                if spot is not None and offline_zg is not None:
                    if spread_side == "put":
                        context_flags.append(
                            "spot_above_zero_gamma" if spot >= offline_zg else "spot_below_zero_gamma"
                        )
                    else:
                        context_flags.append(
                            "spot_below_zero_gamma" if spot <= offline_zg else "spot_above_zero_gamma"
                        )

        candidate_json = {
            "spread_side": spread_side,
            "target_dte": c["dte_target"],
            "delta_target": c["delta_target"],
            "entry_credit": c["entry_credit"],
            "width_points": c["width_points"],
            "credit_to_width": c["credit_to_width"],
            "contracts": c["contracts"],
            "vix": c.get("vix"),
            "term_structure": c.get("term_structure"),
            "spy_price": c.get("spy_price"),
            "spx_price": c.get("spot"),
            "vvix": c.get("vvix"),
            "skew": c.get("skew"),
            "is_opex_day": c.get("is_opex_day", False),
            "is_fomc_day": c.get("is_fomc_day", False),
            "is_triple_witching": c.get("is_triple_witching", False),
            "is_cpi_day": c.get("is_cpi_day", False),
            "is_nfp_day": c.get("is_nfp_day", False),
            "context": {
                "vix": c.get("vix"),
                "term_structure": c.get("term_structure"),
                "spy_price": c.get("spy_price"),
                "spx_price": c.get("spot"),
            },
            "context_flags": context_flags,
            "cboe_context": {},
        }

        features = extract_candidate_features(
            candidate_json=candidate_json,
            max_loss_points=c.get("max_loss"),
            contract_multiplier=CONTRACT_MULT,
        )

        rows.append({
            "features": features,
            "realized_pnl": c["realized_pnl"],
            "hit_tp50": c["hit_tp50"],
            "hit_tp100_at_expiry": c.get("hit_tp100_at_expiry", False),
            "margin_usage": features.get("margin_usage", 0.0),
            "day": c["day"],
            "entry_dt": c["entry_dt"],
            "expiry": c["expiry"],
            "spread_side": c["spread_side"],
            "dte_target": c["dte_target"],
            "delta_target": c["delta_target"],
            "entry_credit": c["entry_credit"],
        })
    return rows
def walk_forward_validate(
    rows: list[dict], train_ratio: float = 0.67,
) -> dict:
    """Walk-forward validation: train on earliest *train_ratio* of days, test rest.

    Splits on **unique day boundaries** so no single trading day straddles
    the train/test partition.  Uses production ``train_bucket_model`` and
    ``predict_with_bucket_model`` to produce out-of-sample quality metrics
    via ``summarize_strategy_quality``.
    """
    sorted_rows = sorted(rows, key=lambda r: r["day"])
    unique_days = sorted(set(r["day"] for r in sorted_rows))
    split_day_idx = int(len(unique_days) * train_ratio)
    split_day = unique_days[min(split_day_idx, len(unique_days) - 1)]
    train_rows = [r for r in sorted_rows if r["day"] < split_day]
    test_rows = [r for r in sorted_rows if r["day"] >= split_day]

    if not train_rows or not test_rows:
        return {"error": "Insufficient data for walk-forward split"}

    model = train_bucket_model(rows=train_rows)

    test_pnls: list[float] = []
    test_margins: list[float] = []
    tp50_count = 0
    tp100_count = 0

    for row in test_rows:
        pnl = row.get("realized_pnl")
        if pnl is None or (isinstance(pnl, float) and math.isnan(pnl)):
            continue
        predict_with_bucket_model(model_payload=model, features=row["features"])
        test_pnls.append(float(pnl))
        test_margins.append(float(row.get("margin_usage", 0.0)))
        if row["hit_tp50"]:
            tp50_count += 1
        if row.get("hit_tp100_at_expiry"):
            tp100_count += 1

    return {
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "train_days": f"{train_rows[0]['day']} .. {train_rows[-1]['day']}",
        "test_days": f"{test_rows[0]['day']} .. {test_rows[-1]['day']}",
        "model": model,
        "test_summary": summarize_strategy_quality(
            realized_pnls=test_pnls,
            margin_usages=test_margins,
            hit_tp50_count=tp50_count,
            hit_tp100_count=tp100_count,
        ),
    }
def deploy_model(
    model_payload: dict, description: str = "offline-pipeline",
) -> None:
    """Insert the trained model into production ``model_versions`` as shadow.

    Reads DATABASE_URL from the environment or the backend ``.env`` file,
    converts asyncpg URLs to sync psycopg2 URLs, and performs a single
    INSERT.
    """
    import os

    from _env import load_project_env
    load_project_env()

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        logger.error("DATABASE_URL not set; cannot deploy model.")
        return

    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    sync_url = sync_url.replace("asyncpg://", "postgresql://")

    from sqlalchemy import create_engine, text as sa_text
    engine = create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(sa_text(
            "INSERT INTO model_versions (name, status, payload, description) "
            "VALUES (:name, 'shadow', :payload, :desc)"
        ), {
            "name": "offline_bucket_empirical_v1",
            "payload": json.dumps(model_payload, default=str),
            "desc": description,
        })
    print("[DEPLOY] Model inserted as shadow into model_versions")
TRAJECTORY_COLUMNS = [
    "max_adverse_pnl", "max_adverse_multiple", "max_favorable_pnl",
    "final_pnl_at_expiry", "final_is_tp100",
    "hit_stop_loss", "recovered_after_sl",
    "hold_hit_tp50", "hold_realized_pnl", "hold_exit_reason",
    "hold_hit_tp100_at_expiry",
    # Multi-TP trajectory columns (one pair per TP level; includes tp50 aliases)
    *[col for lvl in TP_LEVELS for col in (f"first_tp{lvl}_pnl", f"min_pnl_before_tp{lvl}")],
]
def relabel_from_csv(
    csv_path: Path = OUTPUT_CSV,
    output_path: Path | None = None,
) -> None:
    """Re-label the existing training CSV with full trajectory data.

    Only SL trades need CBBO data reloaded (their marks were discarded on
    the original early-return).  TP50 and EXPIRY trades get new columns
    inferred from existing data -- no CBBO reload needed.

    Parameters
    ----------
    csv_path : Path
        Input CSV (default ``training_candidates.csv``).
    output_path : Path | None
        Where to write the enhanced CSV.  Defaults to overwriting the input.
    """
    if output_path is None:
        output_path = csv_path

    # Hybrid C4 mitigation: relabeling reuses the same _evaluate_outcome
    # contract as the main pipeline, so the live SL config must agree.
    _assert_sl_alignment_with_live_settings()

    t0 = time.time()
    df = pd.read_csv(str(csv_path))
    print(f"[RELABEL] Loaded {len(df)} rows from {csv_path}", flush=True)

    sl_mask = df["exit_reason"] == "STOP_LOSS"
    sl_count = sl_mask.sum()
    non_sl_count = len(df) - sl_count
    print(f"[RELABEL] SL trades to re-evaluate: {sl_count}", flush=True)
    print(f"[RELABEL] Non-SL trades (inferred): {non_sl_count}", flush=True)

    # --- Infer trajectory columns for non-SL trades ---
    # TP50 trades: SL never breached before TP50, so hold outcome = same.
    # EXPIRY trades: neither SL nor TP50, so hold outcome = same.
    for col in TRAJECTORY_COLUMNS:
        if col not in df.columns:
            df[col] = None

    non_sl = ~sl_mask
    df.loc[non_sl, "hit_stop_loss"] = False
    df.loc[non_sl, "recovered_after_sl"] = False
    df.loc[non_sl, "hold_hit_tp50"] = df.loc[non_sl, "hit_tp50"]
    df.loc[non_sl, "hold_realized_pnl"] = df.loc[non_sl, "realized_pnl"]
    df.loc[non_sl, "hold_exit_reason"] = df.loc[non_sl, "exit_reason"]
    df.loc[non_sl, "hold_hit_tp100_at_expiry"] = df.loc[non_sl, "hit_tp100_at_expiry"]
    # For non-SL trades, multi-TP columns cannot be inferred from existing
    # data alone (only first_tp50_pnl was stored).  Leave new TP levels as
    # None -- a full --relabel or regeneration will populate them.

    if sl_count == 0:
        print("[RELABEL] No SL trades to re-evaluate. Saving.", flush=True)
        df.to_csv(str(output_path), index=False)
        elapsed = time.time() - t0
        print(f"[RELABEL] Done in {elapsed:.1f}s → {output_path}", flush=True)
        return

    # --- Re-label SL trades by reloading CBBO marks ---
    spxw_days = set(_available_day_files(SPXW_CBBO))
    spx_days = set(_available_day_files(SPX_CBBO))
    trading_days = sorted(spxw_days | spx_days)
    print(f"[RELABEL] {len(trading_days)} trading days available", flush=True)

    sl_rows = df[sl_mask].to_dict("records")

    # label_candidates expects list[dict] with entry info keys
    required_keys = [
        "day", "expiry", "entry_dt", "entry_credit",
        "short_symbol", "long_symbol",
        "short_instrument_id", "long_instrument_id",
    ]
    candidates = []
    for row in sl_rows:
        c = {k: row[k] for k in required_keys}
        candidates.append(c)

    print(f"[RELABEL] Re-labeling {len(candidates)} SL candidates ...", flush=True)
    relabeled = label_candidates(candidates, trading_days)
    print(f"[RELABEL] Re-labeled {len(relabeled)} candidates", flush=True)

    # Write the new trajectory columns back into the DataFrame
    sl_indices = df.index[sl_mask].tolist()
    for idx, c in zip(sl_indices, relabeled):
        for col in TRAJECTORY_COLUMNS:
            df.at[idx, col] = c.get(col)
        # Also update backward-compatible fields (should match, but
        # ensures consistency with the new _evaluate_outcome logic)
        df.at[idx, "hit_tp50"] = c.get("hit_tp50", False)
        df.at[idx, "realized_pnl"] = c.get("realized_pnl")
        df.at[idx, "exit_reason"] = c.get("exit_reason", "STOP_LOSS")
        df.at[idx, "hit_tp100_at_expiry"] = c.get("hit_tp100_at_expiry", False)

    df.to_csv(str(output_path), index=False)
    elapsed = time.time() - t0
    print(f"\n[RELABEL] Done in {elapsed:.1f}s → {output_path}", flush=True)

    # Summary stats
    sl_df = df[sl_mask]
    recovered = sl_df["recovered_after_sl"].sum()
    hold_tp50 = sl_df["hold_hit_tp50"].sum()
    print(f"[RELABEL] SL trades that would recover (hit TP50 after SL): "
          f"{recovered}/{sl_count} ({recovered/sl_count*100:.1f}%)", flush=True)
    print(f"[RELABEL] SL trades where TP50 fires at any point: "
          f"{hold_tp50}/{sl_count} ({hold_tp50/sl_count*100:.1f}%)", flush=True)

    hold_pnl_mean = sl_df["hold_realized_pnl"].mean()
    orig_pnl_mean = sl_df["realized_pnl"].mean()
    print(f"[RELABEL] SL trade avg PnL — close at SL: ${orig_pnl_mean:.0f}, "
          f"hold through: ${hold_pnl_mean:.0f}", flush=True)
