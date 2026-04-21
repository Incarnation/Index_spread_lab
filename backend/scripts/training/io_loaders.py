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


def _available_day_files(directory: Path) -> list[str]:
    """Return sorted YYYYMMDD day strings from .dbn.zst files in *directory*.

    Also scans ``PRODUCTION_CHAINS_DIR`` for ``.parquet`` files and merges
    them into the result, so both Databento and production-exported days
    appear in the unified trading-day list.
    """
    results: set[str] = set()
    for f in directory.glob("*.dbn.zst"):
        day_str = f.name.split(".")[0]
        if day_str.isdigit() and len(day_str) == 8:
            results.add(day_str)
    if PRODUCTION_CHAINS_DIR.exists():
        for f in PRODUCTION_CHAINS_DIR.glob("*.parquet"):
            day_str = f.stem
            if day_str.isdigit() and len(day_str) == 8:
                results.add(day_str)
    return sorted(results)
def _symbol_to_iid(sym: str) -> int:
    """Derive a stable integer instrument_id from an OCC option symbol.

    Uses the first 8 hex digits of an MD5 hash to produce a positive 32-bit
    integer.  Deterministic across runs so the same symbol always maps to the
    same ID within a pipeline execution.
    """
    return int(hashlib.md5(sym.encode()).hexdigest()[:8], 16)
_production_chain_cache: dict[str, pd.DataFrame | None] = {}
_PRODUCTION_CHAIN_CACHE_MAX = 15
def load_production_chain(day_str: str) -> pd.DataFrame | None:
    """Read a per-day production-exported Parquet file if it exists.

    Results are cached in ``_production_chain_cache`` (max 15 entries) so
    that ``load_definitions``, ``load_cbbo``, and ``load_statistics`` can
    all fall back for the same day without reading the parquet file three
    times.

    Returns the raw DataFrame with columns ``ts``, ``option_symbol``,
    ``expiration``, ``strike``, ``option_right``, ``bid``, ``ask``,
    ``open_interest``, ``delta``, ``gamma``.  Returns ``None`` when no
    export file exists for the requested day.
    """
    if day_str in _production_chain_cache:
        return _production_chain_cache[day_str]

    path = PRODUCTION_CHAINS_DIR / f"{day_str}.parquet"
    if not path.exists():
        _production_chain_cache[day_str] = None
        return None
    try:
        df = pd.read_parquet(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
    except Exception as exc:
        logger.warning("Cannot load production chain %s: %s", day_str, exc)
        df = None

    _production_chain_cache[day_str] = df
    while len(_production_chain_cache) > _PRODUCTION_CHAIN_CACHE_MAX:
        oldest = min(_production_chain_cache)
        del _production_chain_cache[oldest]
    return df
def definitions_from_production(
    chain_df: pd.DataFrame,
    *,
    day_str: str | None = None,
) -> pd.DataFrame:
    """Convert a production chain DataFrame to Databento definitions shape.

    Extracts unique option symbols and maps them to the columns that
    ``build_instrument_map`` expects: ``instrument_id``, ``raw_symbol``,
    ``strike_price``, ``expiration``, ``instrument_class``.

    H5 fix: ``ts_recv`` is filled with a **deterministic sentinel** rather
    than ``Timestamp.now()``.  The downstream ``load_definitions`` caller
    sorts by ``ts_recv`` then dedups by ``instrument_id`` (keep="last").
    With wall-clock now, two runs over identical inputs would race against
    each other and could pick a different "winning" row when an
    instrument_id appears in both Databento .dbn.zst and the production
    parquet.  A constant sentinel removes that nondeterminism: the row
    order is decided by the upstream stable sort of ``chain_df``.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Raw production chain data from ``load_production_chain``.
    day_str : str | None
        Optional ISO ``YYYYMMDD`` for the trading day; when provided, the
        sentinel ``ts_recv`` is set to ``YYYY-MM-DDT00:00:00Z`` so the
        dedup tiebreak still favours Databento data (which has real
        intraday timestamps after market open).  Without ``day_str`` we
        fall back to ``1970-01-01T00:00:00Z``.

    Returns
    -------
    pd.DataFrame
        Definitions-like DataFrame compatible with ``build_instrument_map``.
        ``ts_recv`` is a deterministic constant; do **not** treat it as a
        wall-clock timestamp.
    """
    unique = chain_df.drop_duplicates("option_symbol")[
        ["option_symbol", "strike", "expiration", "option_right"]
    ].copy()
    unique["instrument_id"] = unique["option_symbol"].map(_symbol_to_iid)
    unique = unique.rename(columns={
        "option_symbol": "raw_symbol",
        "strike": "strike_price",
        "option_right": "instrument_class",
    })
    if day_str is not None and len(day_str) == 8 and day_str.isdigit():
        sentinel = pd.Timestamp(
            f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:]}T00:00:00",
            tz="UTC",
        )
    else:
        sentinel = pd.Timestamp("1970-01-01T00:00:00", tz="UTC")
    unique["ts_recv"] = sentinel
    return unique
def cbbo_from_production(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a production chain DataFrame to Databento CBBO shape.

    Renames columns to match the Databento cbbo-1m schema that downstream
    functions expect: ``instrument_id``, ``ts``, ``bid_px_00``, ``ask_px_00``.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Raw production chain data from ``load_production_chain``.

    Returns
    -------
    pd.DataFrame
        CBBO-like DataFrame usable by ``get_cbbo_snapshot_at``.
    """
    df = chain_df[["ts", "option_symbol", "bid", "ask"]].copy()
    df["instrument_id"] = df["option_symbol"].map(_symbol_to_iid)
    df = df.rename(columns={"bid": "bid_px_00", "ask": "ask_px_00"})
    return df[["instrument_id", "ts", "bid_px_00", "ask_px_00"]]
def statistics_from_production(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a production chain DataFrame to Databento statistics (OI) shape.

    Takes the latest open_interest per option symbol and returns a DataFrame
    with ``instrument_id`` and ``oi`` columns matching the Databento
    statistics loader output.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Raw production chain data from ``load_production_chain``.

    Returns
    -------
    pd.DataFrame | None
        OI DataFrame, or None if no open_interest data is available.
    """
    if "open_interest" not in chain_df.columns:
        return None
    df = chain_df.dropna(subset=["open_interest"])
    if df.empty:
        return None
    df = df.sort_values("ts").drop_duplicates("option_symbol", keep="last")
    df["instrument_id"] = df["option_symbol"].map(_symbol_to_iid)
    return df[["instrument_id", "open_interest"]].rename(
        columns={"open_interest": "oi"},
    )
def _load_dbn(path: Path) -> pd.DataFrame | None:
    """Read a .dbn.zst file via the ``databento`` client.

    Returns a DataFrame with all columns (index reset).  Returns ``None``
    on read errors so callers can skip bad files gracefully.
    """
    try:
        import databento as db
        store = db.DBNStore.from_file(str(path))
        return store.to_df().reset_index()
    except Exception as exc:
        logger.warning("Cannot load %s: %s", path.name, exc)
        return None
def load_definitions(day_str: str) -> pd.DataFrame | None:
    """Load SPXW + SPX instrument definitions for a single trading day.

    Merges definitions from both SPXW (daily/weekly) and SPX (monthly)
    sources.  Keeps only the latest definition per instrument_id
    (handles intra-day security_update_action MODIFY records).

    Falls back to production-exported Parquet when no Databento files
    exist for the requested day.
    """
    frames: list[pd.DataFrame] = []
    for defs_dir in (SPXW_DEFS, SPX_DEFS):
        path = defs_dir / f"{day_str}.dbn.zst"
        if path.exists():
            tmp = _load_dbn(path)
            if tmp is not None and not tmp.empty:
                frames.append(tmp)
    if not frames:
        chain_df = load_production_chain(day_str)
        if chain_df is not None and not chain_df.empty:
            return definitions_from_production(chain_df, day_str=day_str)
        return None
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "instrument_id" in df.columns:
        df = df.sort_values("ts_recv").drop_duplicates(
            "instrument_id", keep="last",
        )
    return df
def build_instrument_map(def_df: pd.DataFrame) -> dict[int, dict]:
    """Build instrument_id -> {strike, expiry, put_call, raw_symbol}.

    Parses put/call from ``instrument_class`` (preferred) and falls back
    to OCC raw_symbol position 12.  Strike comes from the ``strike_price``
    column; expiration is converted to a ``date``.

    Only includes SPX/SPXW instruments -- SPY and VIX option definitions
    (present in mixed-symbol Databento downloads) are filtered out.
    """
    result: dict[int, dict] = {}
    for _, row in def_df.iterrows():
        iid = int(row["instrument_id"])
        raw_sym = str(row.get("raw_symbol", ""))

        # Mixed-symbol downloads may contain SPY/VIX options; skip them.
        if not raw_sym.startswith("SPX"):
            continue

        put_call = None
        ic = row.get("instrument_class")
        if isinstance(ic, str) and ic in ("C", "P"):
            put_call = ic
        if put_call is None and len(raw_sym) >= 13 and raw_sym[12] in ("C", "P"):
            put_call = raw_sym[12]
        if put_call is None:
            continue

        strike = float(row["strike_price"]) if pd.notna(row.get("strike_price")) else None
        if strike is None or strike <= 0:
            continue

        exp_ts = row.get("expiration")
        if pd.notna(exp_ts):
            expiry = pd.Timestamp(exp_ts).date()
        elif len(raw_sym) >= 12:
            try:
                expiry = datetime.strptime(raw_sym[6:12], "%y%m%d").date()
            except ValueError:
                continue
        else:
            continue

        result[iid] = {
            "strike": strike,
            "expiry": expiry,
            "put_call": put_call,
            "raw_symbol": raw_sym,
        }
    return result
def load_cbbo(day_str: str) -> pd.DataFrame | None:
    """Load SPXW (and optionally SPX) cbbo-1m data for one trading day.

    Prefers SPXW data (daily/weekly expirations) and merges with SPX
    monthly data when available.  Uses ``ts_recv`` as the canonical
    timestamp because ``ts_event`` is mostly NaT in the Databento
    cbbo-1m schema.

    Falls back to production-exported Parquet when no Databento files
    exist for the requested day.
    """
    frames: list[pd.DataFrame] = []
    for cbbo_dir in (SPXW_CBBO, SPX_CBBO):
        path = cbbo_dir / f"{day_str}.dbn.zst"
        if path.exists():
            tmp = _load_dbn(path)
            if tmp is not None and not tmp.empty:
                frames.append(tmp)
    if not frames:
        chain_df = load_production_chain(day_str)
        if chain_df is not None and not chain_df.empty:
            return cbbo_from_production(chain_df)
        return None
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "ts_recv" in df.columns:
        df["ts"] = pd.to_datetime(df["ts_recv"], utc=True)
    return df
def load_statistics(day_str: str) -> pd.DataFrame | None:
    """Load SPXW + SPX open-interest from Databento ``statistics`` files.

    Filters to ``stat_type == 9`` (open interest) and returns a DataFrame
    with columns ``instrument_id`` and ``oi`` (quantity).  Returns ``None``
    when no statistics files exist for the requested day.

    Falls back to production-exported Parquet when no Databento files
    exist for the requested day.
    """
    OI_STAT_TYPE = 9
    frames: list[pd.DataFrame] = []
    for stats_dir in (SPXW_STATS, SPX_STATS):
        path = stats_dir / f"{day_str}.dbn.zst"
        if path.exists():
            tmp = _load_dbn(path)
            if tmp is not None and not tmp.empty:
                frames.append(tmp)
    if not frames:
        chain_df = load_production_chain(day_str)
        if chain_df is not None and not chain_df.empty:
            return statistics_from_production(chain_df)
        return None
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    oi_df = df[df["stat_type"] == OI_STAT_TYPE].copy()
    if oi_df.empty:
        return None
    oi_df = oi_df[["instrument_id", "quantity"]].rename(columns={"quantity": "oi"})
    oi_df = oi_df.drop_duplicates("instrument_id", keep="last")
    return oi_df
def load_spy_equity() -> pd.DataFrame:
    """Load SPY 1-minute equity bars, merging Databento and production sources.

    Loads from both the Databento ohlcv-1m parquet (historical) and the
    production-exported ``SPY_1min.parquet`` (recent), deduplicates by
    timestamp keeping Databento rows when both exist (higher 1-min
    resolution vs 5-min production snapshots).  Returns a DataFrame with
    ``ts`` (tz-aware UTC) and ``close`` columns.
    """
    frames: list[pd.DataFrame] = []

    if SPY_EQUITY_PATH.exists():
        df = pd.read_parquet(str(SPY_EQUITY_PATH))
        df = df.reset_index()
        df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
        frames.append(df[["ts", "close"]])

    prod_path = PRODUCTION_UNDERLYING_DIR / "SPY_1min.parquet"
    if prod_path.exists():
        pdf = pd.read_parquet(prod_path)
        pdf["ts"] = pd.to_datetime(pdf["ts"], utc=True)
        frames.append(pdf[["ts", "close"]])

    if not frames:
        return pd.DataFrame(columns=["ts", "close"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts"], keep="first")
    return combined.sort_values("ts").reset_index(drop=True)
def load_daily_parquet(parquet_path: Path) -> dict[date, float]:
    """Load a daily-granularity parquet and return a date -> close mapping.

    Reads ``ts`` and ``close`` columns, groups by calendar date, takes the
    last close per day.  Returns an empty dict when the file is missing.

    .. WARNING ::
        The returned mapping is **end-of-session** for every trading day.
        Looking up ``daily[D]`` at an intraday decision instant on day
        ``D`` is **same-day lookahead** -- you are reading a value that
        was not knowable at the decision time.  Use
        :func:`lag_daily_to_next_session` to convert into a
        decision-safe (point-in-time) mapping before per-candidate
        lookups.

    Parameters
    ----------
    parquet_path : Path
        Path to a parquet file with ``ts`` (tz-aware UTC) and ``close``
        columns where each row is one daily observation.

    Returns
    -------
    dict[date, float]
        Mapping from trading date to the day's closing value (EOD).
    """
    if not parquet_path.exists():
        return {}
    df = pd.read_parquet(parquet_path, columns=["ts", "close"])
    df["date"] = df["ts"].dt.date
    daily = df.groupby("date")["close"].last()
    return daily.to_dict()
def lag_daily_to_next_session(daily: dict[date, float]) -> dict[date, float]:
    """Shift an EOD daily-close mapping by one trading day so each entry
    becomes the value most recently observable **before** that date.

    H6 fix: the FRD SKEW parquet (and any production EOD aggregation)
    stamps each day's value at the session close.  Using ``daily[D]``
    during an intraday decision on day ``D`` therefore peeks at a value
    that is not yet known.  This helper rebuilds the mapping so that
    looking up day ``D`` returns the EOD value from the previous date in
    the source dataset (i.e. the previous trading day, since holidays
    are absent from the EOD dataset).

    Parameters
    ----------
    daily : dict[date, float]
        Output of :func:`load_daily_parquet` (EOD-stamped).

    Returns
    -------
    dict[date, float]
        Decision-safe mapping where ``out[D]`` equals
        ``daily[previous_trading_day]``.  The earliest source date is
        absent from the result because no prior value exists.
    """
    if not daily:
        return {}
    sorted_dates = sorted(daily.keys())
    return {sorted_dates[i]: daily[sorted_dates[i - 1]] for i in range(1, len(sorted_dates))}
def load_economic_calendar(csv_path: Path) -> dict[date, dict]:
    """Load the economic event calendar into a date -> flags mapping.

    Parses ``economic_calendar.csv`` which contains OPEX, FOMC, and triple-
    witching events.  Returns a mapping from event dates to boolean flags.
    Dates not present in the calendar have no entry (caller should default
    to all-False).

    Parameters
    ----------
    csv_path : Path
        Path to the economic calendar CSV with columns ``date``,
        ``event_type``, ``has_projections``, ``is_triple_witching``.

    Returns
    -------
    dict[date, dict]
        ``{date: {"is_opex": bool, "is_fomc": bool, "is_triple_witching": bool,
        "is_cpi": bool, "is_nfp": bool}}``.
        A single date may have multiple events if they coincide; the
        loader aggregates across all rows for a given date.
    """
    if not csv_path.exists():
        return {}
    df = pd.read_csv(str(csv_path))
    out: dict[date, dict] = {}
    for _, row in df.iterrows():
        try:
            d = pd.to_datetime(row["date"]).date()
        except Exception:
            continue
        entry = out.setdefault(d, {
            "is_opex": False, "is_fomc": False, "is_triple_witching": False,
            "is_cpi": False, "is_nfp": False,
        })
        evt = str(row.get("event_type", "")).upper()
        if evt == "OPEX":
            entry["is_opex"] = True
        elif evt == "FOMC":
            entry["is_fomc"] = True
        elif evt == "CPI":
            entry["is_cpi"] = True
        elif evt == "NFP":
            entry["is_nfp"] = True
        if row.get("is_triple_witching") in (True, "True", "true", 1):
            entry["is_triple_witching"] = True
    return out
def lookup_intraday_value(
    uq_df: pd.DataFrame, symbol: str, dt_utc: datetime, window_minutes: int = 5,
) -> float | None:
    """Find the latest intraday value for *symbol* within a window of *dt_utc*.

    Parameters
    ----------
    uq_df          : underlying-quotes DataFrame from ``load_underlying_quotes``.
    symbol         : e.g. ``"VIX"``, ``"VIX9D"``, ``"SPX"``.
    dt_utc         : Decision timestamp (tz-aware UTC).
    window_minutes : Lookback window size in minutes.

    Returns
    -------
    float | None   Latest ``last`` value, or None if no row matches.
    """
    if uq_df.empty:
        return None
    ts_start = dt_utc - timedelta(minutes=window_minutes)
    mask = (
        (uq_df["symbol"] == symbol)
        & (uq_df["ts"] >= ts_start)
        & (uq_df["ts"] <= dt_utc)
    )
    matched = uq_df.loc[mask]
    if matched.empty:
        return None
    return float(matched.iloc[-1]["last"])
def load_frd_quotes(parquet_path: Path, symbol: str) -> pd.DataFrame:
    """Load underlying quotes from FRD parquet or production export fallback.

    Tries the FirstRateData parquet first, then falls back to the
    production-exported ``{SYMBOL}_1min.parquet``.  Converts the OHLC
    ``close`` column to ``last`` so the output schema matches
    ``load_underlying_quotes`` (columns: ``ts``, ``symbol``, ``last``).

    Parameters
    ----------
    parquet_path : Path
        Path to a FirstRateData parquet file (e.g. ``vix_1min.parquet``).
    symbol : str
        Symbol label to assign (e.g. ``"VIX"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ts`` (tz-aware UTC), ``symbol``, and
        ``last``, sorted by timestamp.  Empty DataFrame if both sources
        are missing.
    """
    frames: list[pd.DataFrame] = []

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path, columns=["ts", "close"])
        df = df.rename(columns={"close": "last"})
        df["symbol"] = symbol
        frames.append(df[["ts", "symbol", "last"]])

    prod_path = PRODUCTION_UNDERLYING_DIR / f"{symbol}_1min.parquet"
    if prod_path.exists():
        pdf = pd.read_parquet(prod_path)
        pdf["ts"] = pd.to_datetime(pdf["ts"], utc=True)
        pdf = pdf.rename(columns={"close": "last"})
        pdf["symbol"] = symbol
        frames.append(pdf[["ts", "symbol", "last"]])

    if not frames:
        return pd.DataFrame(columns=["ts", "symbol", "last"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts"], keep="first")
    return combined.sort_values("ts").reset_index(drop=True)
def merge_underlying_quotes(
    production_df: pd.DataFrame,
    *frd_dfs: pd.DataFrame,
) -> pd.DataFrame:
    """Merge production DB quotes with FirstRateData quotes.

    Production rows take priority when timestamps overlap (within the same
    symbol).  FirstRateData fills the gaps -- typically all dates before the
    production DB started capturing data.

    Parameters
    ----------
    production_df : pd.DataFrame
        Underlying-quotes from ``load_underlying_quotes`` (may be empty).
    *frd_dfs : pd.DataFrame
        One or more DataFrames from ``load_frd_quotes``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns ``ts``, ``symbol``, ``last``.
    """
    parts = [d for d in (production_df, *frd_dfs) if not d.empty]
    if not parts:
        return pd.DataFrame(columns=["ts", "symbol", "last"])
    combined = pd.concat(parts, ignore_index=True)
    # Production rows appear first in concat, so keep="first" preserves them
    # when there are duplicate (symbol, ts) pairs.
    combined = combined.drop_duplicates(subset=["symbol", "ts"], keep="first")
    return combined.sort_values("ts").reset_index(drop=True)
def _load_gex_csv(csv_path: Path) -> pd.DataFrame:
    """Load a single GEX CSV (offline cache or production export).

    Returns a DataFrame with at least ``ts`` (tz-aware UTC), ``gex_net``,
    and ``zero_gamma_level`` columns, sorted by timestamp.  Returns an
    empty DataFrame when the file does not exist.
    """
    empty = pd.DataFrame(columns=["ts", "gex_net", "zero_gamma_level"])
    if not csv_path.exists():
        return empty
    df = pd.read_csv(str(csv_path))
    if df.empty:
        return empty
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    for col in ("gex_net", "zero_gamma_level"):
        if col not in df.columns:
            df[col] = None
    return df[["ts", "gex_net", "zero_gamma_level"]].sort_values("ts").reset_index(drop=True)
def _load_merged_gex(
    offline_path: Path = OFFLINE_GEX_CSV,
    production_path: Path = CONTEXT_SNAPSHOTS_CSV,
) -> pd.DataFrame:
    """Load and merge offline GEX cache with production context_snapshots.

    Production rows (from the live system) take priority when timestamps
    overlap with the offline Databento-computed cache.  The offline cache
    covers historical dates before the production system was running.

    Parameters
    ----------
    offline_path : Path
        CSV with columns ``ts, gex_net, zero_gamma_level`` (offline cache).
    production_path : Path
        CSV exported from the ``context_snapshots`` DB table via
        ``export_production_data.py``.

    Returns
    -------
    pd.DataFrame
        Merged GEX DataFrame with ``ts, gex_net, zero_gamma_level``.
    """
    offline = _load_gex_csv(offline_path)
    production = _load_gex_csv(production_path)

    prod_gex = production[production["gex_net"].notna()]

    if prod_gex.empty:
        return offline
    if offline.empty:
        return prod_gex

    # Production rows win on timestamp overlap (round to nearest minute)
    combined = pd.concat([prod_gex, offline], ignore_index=True)
    combined["_ts_rounded"] = combined["ts"].dt.round("min")
    combined = combined.drop_duplicates(subset=["_ts_rounded"], keep="first")
    combined = combined.drop(columns=["_ts_rounded"])
    return combined.sort_values("ts").reset_index(drop=True)
def lookup_gex_context(
    cs_df: pd.DataFrame, dt_utc: datetime, spread_side: str, spot: float | None,
    window_minutes: int = 15,
) -> tuple[list[str], float | None, float | None]:
    """Look up GEX context_flags from context_snapshots nearest to *dt_utc*.

    Mirrors production ``_context_score`` flag logic from ``decision_job.py``.

    Parameters
    ----------
    cs_df          : context-snapshots DataFrame from ``load_context_snapshots``.
    dt_utc         : Decision timestamp (tz-aware UTC).
    spread_side    : ``"put"`` or ``"call"``.
    spot           : Current SPX spot price.
    window_minutes : Lookback window for matching.

    Returns
    -------
    tuple[list[str], float | None, float | None]
        (context_flags, gex_net, zero_gamma_level)
    """
    flags: list[str] = []
    if cs_df.empty:
        return flags, None, None

    ts_start = dt_utc - timedelta(minutes=window_minutes)
    mask = (cs_df["ts"] >= ts_start) & (cs_df["ts"] <= dt_utc) & cs_df["gex_net"].notna()
    matched = cs_df.loc[mask]
    if matched.empty:
        return flags, None, None

    row = matched.iloc[-1]
    gex_net = float(row["gex_net"]) if pd.notna(row["gex_net"]) else None
    zero_gamma = float(row["zero_gamma_level"]) if pd.notna(row["zero_gamma_level"]) else None

    if gex_net is not None:
        if spread_side == "put":
            if gex_net >= 0:
                flags.append("gex_support")
            else:
                flags.append("gex_headwind")
        else:
            if gex_net <= 0:
                flags.append("gex_support")
            else:
                flags.append("gex_headwind")

    if spot is not None and zero_gamma is not None:
        if spread_side == "put":
            if spot >= zero_gamma:
                flags.append("spot_above_zero_gamma")
            else:
                flags.append("spot_below_zero_gamma")
        else:
            if spot <= zero_gamma:
                flags.append("spot_below_zero_gamma")
            else:
                flags.append("spot_above_zero_gamma")

    return flags, gex_net, zero_gamma
