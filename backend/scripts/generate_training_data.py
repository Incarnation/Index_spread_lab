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

_BACKEND = Path(__file__).resolve().parents[1]
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


# ===================================================================
# BLACK-SCHOLES (vectorised with numpy / scipy)
# ===================================================================

def _bs_d1(
    S: np.ndarray, K: np.ndarray, T: float, r: float, sigma: np.ndarray,
) -> np.ndarray:
    """Compute Black-Scholes d1 for arrays of spots, strikes, and vols.

    Parameters
    ----------
    S, K : array-like  spot and strike prices.
    T    : float        time to expiry in years.
    r    : float        risk-free rate.
    sigma: array-like   implied volatilities.

    Returns
    -------
    np.ndarray  d1 values.
    """
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_price_vec(
    S: np.ndarray, K: np.ndarray, T: float, r: float,
    sigma: np.ndarray, is_call: np.ndarray,
) -> np.ndarray:
    """Vectorised Black-Scholes European option price.

    Parameters
    ----------
    S, K, sigma : array-like  spot, strike, vol arrays.
    T, r        : float       time-to-expiry (years), risk-free rate.
    is_call     : bool array  True for calls, False for puts.

    Returns
    -------
    np.ndarray  theoretical prices.
    """
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(is_call, call, put)


def bs_delta_vec(
    S: np.ndarray, K: np.ndarray, T: float, r: float,
    sigma: np.ndarray, is_call: np.ndarray,
) -> np.ndarray:
    """Vectorised Black-Scholes delta.

    Returns call delta (0..1) for calls and put delta (-1..0) for puts.
    """
    d1 = _bs_d1(S, K, T, r, sigma)
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)


def implied_vol_vec(
    mid_prices: np.ndarray, S: np.ndarray, K: np.ndarray,
    T: float, r: float, is_call: np.ndarray,
) -> np.ndarray:
    """Vectorised IV via bisection bounded by [MIN_IV, MAX_IV].

    Uses 60 iterations of bisection which gives ~18 digits of precision
    on the vol axis -- far more than needed.
    """
    n = len(mid_prices)
    lo = np.full(n, MIN_IV)
    hi = np.full(n, MAX_IV)
    for _ in range(IV_BISECT_ITERS):
        mid_vol = 0.5 * (lo + hi)
        prices = bs_price_vec(S, K, T, r, mid_vol, is_call)
        too_low = prices < mid_prices
        lo = np.where(too_low, mid_vol, lo)
        hi = np.where(~too_low, mid_vol, hi)
    return 0.5 * (lo + hi)


# ===================================================================
# DATA LOADING
# ===================================================================

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


# ===================================================================
# PRODUCTION DATA ADAPTERS
# ===================================================================

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


GEX_MAX_DTE_DAYS = 10
GEX_STRIKE_RANGE_PCT = 0.30


def compute_offline_gex(
    snapshot: pd.DataFrame,
    inst_map: dict[int, dict],
    oi_df: pd.DataFrame,
    spot: float,
    day_date: date,
) -> tuple[float | None, float | None]:
    """Compute GEX from Databento CBBO + definitions + open interest.

    Mirrors the Tradier GEX job computation: for each instrument with
    valid mid-price and OI, compute Black-Scholes gamma, then aggregate
    GEX per strike as ``gamma * OI * spot^2 * multiplier / 100``.
    Calls contribute positive GEX, puts negative.

    Filters to options within ``GEX_MAX_DTE_DAYS`` of expiry and strikes
    within ``GEX_STRIKE_RANGE_PCT`` of spot, matching production behavior
    (``gex_max_dte_days=10``, ``snapshot_strikes_each_side=75``).

    Parameters
    ----------
    snapshot    : CBBO snapshot DataFrame at decision time (from ``get_cbbo_snapshot_at``).
    inst_map    : instrument_id -> {strike, expiry, put_call} from definitions.
    oi_df       : DataFrame with ``instrument_id`` and ``oi`` from ``load_statistics``.
    spot        : Current SPX spot price.
    day_date    : Trading date for time-to-expiry computation.

    Returns
    -------
    tuple[float | None, float | None]
        ``(gex_net, zero_gamma_level)``.  Both ``None`` when data is
        insufficient.
    """
    if snapshot.empty or oi_df is None or oi_df.empty or spot <= 0:
        return None, None

    oi_map = dict(zip(oi_df["instrument_id"].values, oi_df["oi"].values))
    max_expiry = date(
        day_date.year, day_date.month, day_date.day,
    ) + timedelta(days=GEX_MAX_DTE_DAYS)
    strike_lo = spot * (1 - GEX_STRIKE_RANGE_PCT)
    strike_hi = spot * (1 + GEX_STRIKE_RANGE_PCT)

    strikes_arr: list[float] = []
    mids_arr: list[float] = []
    is_call_arr: list[bool] = []
    oi_arr: list[int] = []
    expiry_arr: list[date] = []

    for _, row in snapshot.iterrows():
        iid = int(row["instrument_id"])
        info = inst_map.get(iid)
        if info is None:
            continue
        if info["expiry"] > max_expiry or info["expiry"] < day_date:
            continue
        if info["strike"] < strike_lo or info["strike"] > strike_hi:
            continue
        oi_val = oi_map.get(iid, 0)
        if oi_val <= 0:
            continue

        mid = mid_price(row.get("bid_px_00"), row.get("ask_px_00"))
        if mid is None or mid < MIN_MID_PRICE:
            continue

        strikes_arr.append(info["strike"])
        mids_arr.append(mid)
        is_call_arr.append(info["put_call"] == "C")
        oi_arr.append(int(oi_val))
        expiry_arr.append(info["expiry"])

    if len(strikes_arr) < 10:
        return None, None

    S = np.full(len(strikes_arr), spot)
    K = np.array(strikes_arr)
    is_call = np.array(is_call_arr)
    mid_prices = np.array(mids_arr)
    oi_values = np.array(oi_arr, dtype=float)

    unique_expiries = sorted(set(expiry_arr))
    gamma_all = np.zeros(len(strikes_arr))

    for exp in unique_expiries:
        T = _time_to_expiry_years(exp, day_date)
        mask_exp = np.array([e == exp for e in expiry_arr])
        idx = np.where(mask_exp)[0]
        if len(idx) == 0:
            continue

        iv = implied_vol_vec(
            mid_prices[idx], S[idx], K[idx], T, RISK_FREE_RATE, is_call[idx],
        )
        d1 = _bs_d1(S[idx], K[idx], T, RISK_FREE_RATE, iv)
        gamma = norm.pdf(d1) / (S[idx] * iv * np.sqrt(T))
        gamma_all[idx] = gamma

    gex_per_instrument = np.where(
        is_call,
        gamma_all * oi_values * spot * spot * CONTRACT_MULT / 100.0,
        -gamma_all * oi_values * spot * spot * CONTRACT_MULT / 100.0,
    )

    per_strike: dict[float, dict[str, float]] = {}
    for i in range(len(strikes_arr)):
        strike = strikes_arr[i]
        gex_val = float(gex_per_instrument[i])
        if math.isnan(gex_val) or math.isinf(gex_val):
            continue
        bucket = per_strike.setdefault(strike, {"gex_calls": 0.0, "gex_puts": 0.0})
        if is_call_arr[i]:
            bucket["gex_calls"] += gex_val
        else:
            bucket["gex_puts"] += gex_val

    if not per_strike:
        return None, None

    gex_net = sum(b["gex_calls"] + b["gex_puts"] for b in per_strike.values())

    # Zero-gamma level: interpolated strike where cumulative net GEX crosses zero
    zero_gamma_level: float | None = None
    sorted_strikes = sorted(per_strike.keys())
    cumulative = 0.0
    prev_cumulative = 0.0
    prev_strike: float | None = None
    for strike in sorted_strikes:
        prev_cumulative = cumulative
        cumulative += per_strike[strike]["gex_calls"] + per_strike[strike]["gex_puts"]
        if prev_strike is None:
            prev_strike = strike
            continue
        if prev_cumulative == 0.0:
            zero_gamma_level = prev_strike
            break
        crossed = (prev_cumulative < 0 and cumulative > 0) or (
            prev_cumulative > 0 and cumulative < 0
        )
        if crossed:
            denom = abs(prev_cumulative) + abs(cumulative)
            if denom == 0:
                zero_gamma_level = strike
            else:
                weight = abs(prev_cumulative) / denom
                zero_gamma_level = prev_strike + (strike - prev_strike) * weight
            break
        prev_strike = strike

    return gex_net, zero_gamma_level


# ===================================================================
# DTE HELPERS
# ===================================================================

def build_dte_lookup(expirations: list[date], as_of: date) -> dict[date, int]:
    """Assign a trading-DTE index to each expiration on or after *as_of*.

    DTE 0 means the option expires *today*; DTE 1 is the next expiration
    after today, etc.  Replicates production ``dte.trading_dte_lookup``.
    """
    future = sorted(e for e in expirations if e >= as_of)
    if not future:
        return {}
    base = 0 if as_of in future else 1
    return {exp: idx + base for idx, exp in enumerate(future)}


def find_expiry_for_dte(
    dte_map: dict[date, int], target: int, tolerance: int = 1,
) -> date | None:
    """Find the expiration with the closest DTE to *target* within tolerance."""
    best, best_diff = None, None
    for exp, dte in dte_map.items():
        diff = abs(dte - target)
        if diff <= tolerance and (best_diff is None or diff < best_diff):
            best, best_diff = exp, diff
    return best


# ===================================================================
# SPX SPOT VIA PUT-CALL PARITY
# ===================================================================

def derive_spx_from_parity(
    snapshot: pd.DataFrame,
    inst_map: dict[int, dict],
    day_date: date,
    spy_estimate: float,
) -> float:
    """Derive SPX index level from SPXW put-call parity.

    Uses matched call/put pairs at the same strike and nearest expiry to
    compute the implied forward:  S = C_mid - P_mid + K * exp(-r*T).
    Takes the median of near-ATM strikes (within 50 pts of the SPY-based
    estimate) to reject outliers.  Falls back to ``spy_estimate * SPY_SPX_RATIO``
    when fewer than 3 valid pairs are available.

    Parameters
    ----------
    snapshot    : CBBO snapshot DataFrame (must have instrument_id, bid_px_00,
                  ask_px_00 columns).
    inst_map    : instrument_id -> {strike, expiry, put_call} mapping.
    day_date    : Current trading date.
    spy_estimate: Latest SPY equity price (used for fallback and ATM filter).

    Returns
    -------
    float  Implied SPX index level.
    """
    fallback = spy_estimate * SPY_SPX_RATIO

    if snapshot.empty or "instrument_id" not in snapshot.columns:
        return fallback

    nearest_expiry = None
    for info in inst_map.values():
        exp = info["expiry"]
        if exp >= day_date and (nearest_expiry is None or exp < nearest_expiry):
            nearest_expiry = exp
    if nearest_expiry is None:
        return fallback

    T = _time_to_expiry_years(nearest_expiry, day_date)
    discount = np.exp(-RISK_FREE_RATE * T)

    calls: dict[float, float] = {}
    puts: dict[float, float] = {}
    for _, row in snapshot.iterrows():
        iid = int(row["instrument_id"])
        info = inst_map.get(iid)
        if info is None or info["expiry"] != nearest_expiry:
            continue
        mid = mid_price(row["bid_px_00"], row["ask_px_00"])
        if mid is None:
            continue
        strike = info["strike"]
        if info["put_call"] == "C":
            calls[strike] = mid
        elif info["put_call"] == "P":
            puts[strike] = mid

    atm_lo = fallback - 50.0
    atm_hi = fallback + 50.0
    implied_spots: list[float] = []

    for strike in sorted(set(calls) & set(puts)):
        if strike < atm_lo or strike > atm_hi:
            continue
        S_implied = calls[strike] - puts[strike] + strike * discount
        if S_implied > 0:
            implied_spots.append(S_implied)

    if len(implied_spots) < 3:
        return fallback

    return float(np.median(implied_spots))


# ===================================================================
# SPREAD CONSTRUCTION
# ===================================================================

def _time_to_expiry_years(expiry: date, as_of: date) -> float:
    """Time to expiry in years; floors near-zero to avoid division errors."""
    days = (expiry - as_of).days
    if days <= 0:
        return 1.0 / (365.0 * 24.0)
    return days / 365.0


def get_cbbo_snapshot_at(cbbo_df: pd.DataFrame, dt_utc: datetime) -> pd.DataFrame:
    """Latest CBBO per instrument within a 5-minute window ending at *dt_utc*."""
    window_start = dt_utc - timedelta(minutes=5)
    mask = (cbbo_df["ts"] >= window_start) & (cbbo_df["ts"] <= dt_utc)
    if mask.sum() == 0:
        return pd.DataFrame()
    return (
        cbbo_df[mask]
        .sort_values("ts")
        .groupby("instrument_id")
        .last()
        .reset_index()
    )


def build_candidates_for_snapshot(
    *,
    snapshot: pd.DataFrame,
    inst_map: dict[int, dict],
    spot: float,
    spy_price: float,
    vix: float | None,
    vix9d: float | None,
    term_structure: float | None,
    vvix: float | None = None,
    skew: float | None = None,
    is_opex_day: bool = False,
    is_fomc_day: bool = False,
    is_triple_witching: bool = False,
    is_cpi_day: bool = False,
    is_nfp_day: bool = False,
    decision_dt: datetime,
    day_date: date,
    dte_target: int,
    expiry: date,
    delta_target: float,
    side: str,
    width_points: float = WIDTH_POINTS,
) -> list[dict]:
    """Build vertical credit-spread candidates from a single CBBO snapshot.

    Finds the short leg with |delta| closest to *delta_target*, pairs it
    with a long leg at short_strike +/- *width_points*, and computes entry
    credit.  Returns an empty list when no valid spread can be formed.
    """
    is_call = side != "put"
    pc_char = "C" if is_call else "P"

    if snapshot.empty or "instrument_id" not in snapshot.columns:
        return []

    valid_iids = {
        iid for iid, info in inst_map.items()
        if info["expiry"] == expiry and info["put_call"] == pc_char
    }
    opts = snapshot[snapshot["instrument_id"].isin(valid_iids)].copy()
    if opts.empty:
        return []

    opts["bid"] = opts["bid_px_00"].astype(float)
    opts["ask"] = opts["ask_px_00"].astype(float)
    opts["mid"] = (opts["bid"] + opts["ask"]) / 2.0
    # Apply the same policy as utils.pricing.mid_price: strict positivity,
    # finite values, and crossed-book rejection.  Vectorised here for speed
    # rather than per-row mid_price() calls which would dominate runtime
    # over the multi-million-row offline training scan.
    opts = opts[
        (opts["bid"] > 0)
        & (opts["ask"] > 0)
        & np.isfinite(opts["bid"])
        & np.isfinite(opts["ask"])
        & (opts["bid"] <= opts["ask"])
        & (opts["mid"] >= MIN_MID_PRICE)
    ]
    if opts.empty:
        return []

    opts["strike"] = opts["instrument_id"].map(
        lambda iid: inst_map.get(iid, {}).get("strike", 0),
    )
    opts["raw_symbol"] = opts["instrument_id"].map(
        lambda iid: inst_map.get(iid, {}).get("raw_symbol", ""),
    )
    opts = opts[opts["strike"] > 0]
    if len(opts) < 2:
        return []

    # --- IV / delta computation ---
    T = _time_to_expiry_years(expiry, day_date)
    S = np.full(len(opts), spot)
    K = opts["strike"].values.astype(float)
    mids = opts["mid"].values.astype(float)
    is_call_arr = np.full(len(opts), is_call)

    with np.errstate(divide="ignore", invalid="ignore"):
        iv = implied_vol_vec(mids, S, K, T, RISK_FREE_RATE, is_call_arr)
        delta = bs_delta_vec(S, K, T, RISK_FREE_RATE, iv, is_call_arr)

    opts["iv"] = iv
    opts["delta"] = delta
    opts["abs_delta"] = np.abs(delta)

    # discard options where IV solver hit the boundary (bad convergence)
    opts = opts[(opts["iv"] > MIN_IV + 0.001) & (opts["iv"] < MAX_IV - 0.001)]
    if len(opts) < 2:
        return []

    # --- short leg: closest |delta| to target ---
    opts["delta_diff"] = (opts["abs_delta"] - delta_target).abs()
    short_idx = opts["delta_diff"].idxmin()
    short = opts.loc[short_idx]

    # --- long leg: strike closest to desired offset ---
    desired_long = (
        short["strike"] - width_points if side == "put"
        else short["strike"] + width_points
    )
    long_cands = opts[opts.index != short_idx].copy()
    if long_cands.empty:
        return []
    long_cands["strike_diff"] = (long_cands["strike"] - desired_long).abs()
    long_idx = long_cands["strike_diff"].idxmin()
    lleg = long_cands.loc[long_idx]

    credit = float(short["mid"]) - float(lleg["mid"])
    if credit <= 0:
        return []

    actual_width = abs(float(short["strike"]) - float(lleg["strike"]))
    if actual_width <= 0 or abs(actual_width - width_points) > 0.01:
        return []

    return [{
        "entry_dt": decision_dt.isoformat(),
        "day": day_date.isoformat(),
        "dte_target": dte_target,
        "expiry": expiry.isoformat(),
        "spread_side": side,
        "delta_target": delta_target,
        "short_instrument_id": int(short["instrument_id"]),
        "long_instrument_id": int(lleg["instrument_id"]),
        "short_symbol": str(short["raw_symbol"]),
        "long_symbol": str(lleg["raw_symbol"]),
        "short_strike": float(short["strike"]),
        "long_strike": float(lleg["strike"]),
        "short_bid": float(short["bid"]),
        "short_ask": float(short["ask"]),
        "short_mid": float(short["mid"]),
        "short_delta": float(short["delta"]),
        "short_iv": float(short["iv"]),
        "long_bid": float(lleg["bid"]),
        "long_ask": float(lleg["ask"]),
        "long_mid": float(lleg["mid"]),
        "long_delta": float(lleg["delta"]),
        "long_iv": float(lleg["iv"]),
        "entry_credit": credit,
        "width_points": actual_width,
        "max_loss": max(actual_width - credit, 0.0),
        "credit_to_width": credit / actual_width,
        "spot": spot,
        "spy_price": spy_price,
        "vix": vix,
        "vix9d": vix9d,
        "term_structure": term_structure,
        "vvix": vvix,
        "skew": skew,
        "is_opex_day": is_opex_day,
        "is_fomc_day": is_fomc_day,
        "is_triple_witching": is_triple_witching,
        "is_cpi_day": is_cpi_day,
        "is_nfp_day": is_nfp_day,
        "contracts": CONTRACTS,
    }]


# ===================================================================
# LABEL RESOLUTION
# ===================================================================

# Note: the local ``_mid`` helper has been removed in favour of the
# canonical :func:`spx_backend.utils.pricing.mid_price`.  ``mid_price`` is
# strictly stricter than the original helper: in addition to rejecting
# NaN and non-positive quotes, it also rejects non-finite values (Inf)
# and crossed books (bid > ask).  These additional rejections are
# intentional data-quality gates and any historically-valid two-sided
# quote (positive, finite, non-crossed) yields the identical midpoint.


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


# -- keep old name accessible as fallback alias --
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


# ===================================================================
# LABEL CACHE (incremental re-labeling)
# ===================================================================

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


# ===================================================================
# FEATURE ENGINEERING
# ===================================================================

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


# ===================================================================
# TRAINING + VALIDATION
# ===================================================================

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


# ===================================================================
# DEPLOY
# ===================================================================

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


# ===================================================================
# PARALLEL CANDIDATE GENERATION
# ===================================================================

_WORKER_REF: dict[str, Any] = {}


def _init_candidate_worker(
    spy_df_bytes: bytes,
    uq_df_bytes: bytes,
    skew_daily_dict: dict,
    cal_map_dict: dict,
    active_grid: dict[str, Any] | None = None,
) -> None:
    """Initializer for candidate-generation worker processes.

    Deserializes shared read-only reference data into module-level storage
    so each worker can access it without re-reading from disk.
    On macOS (spawn-based multiprocessing), the parent's _ACTIVE_GRID is
    NOT inherited, so it must be passed explicitly here.
    """
    global _ACTIVE_GRID
    import pickle
    _WORKER_REF["spy_df"] = pickle.loads(spy_df_bytes)
    _WORKER_REF["uq_df"] = pickle.loads(uq_df_bytes)
    _WORKER_REF["skew_daily"] = skew_daily_dict
    _WORKER_REF["cal_map"] = cal_map_dict
    if active_grid is not None:
        _ACTIVE_GRID = active_grid


def _generate_candidates_for_day(day_str: str) -> list[dict]:
    """Generate all candidates for a single trading day (worker function).

    Top-level function so it can be pickled by multiprocessing.Pool.
    Reads Databento/production chain data for the day, constructs spread
    candidates at every decision time x DTE x delta x width combination,
    and returns the flat list of candidate dicts.
    """
    spy_df = _WORKER_REF["spy_df"]
    uq_df = _WORKER_REF["uq_df"]
    skew_daily = _WORKER_REF["skew_daily"]
    cal_map = _WORKER_REF["cal_map"]

    day_date = date(int(day_str[:4]), int(day_str[4:6]), int(day_str[6:]))

    def_df = load_definitions(day_str)
    if def_df is None:
        return []
    inst_map = build_instrument_map(def_df)
    del def_df
    if not inst_map:
        return []

    cbbo_df = load_cbbo(day_str)
    if cbbo_df is None or cbbo_df.empty:
        return []

    oi_df = load_statistics(day_str)

    all_expiries = sorted({info["expiry"] for info in inst_map.values()})
    dte_map = build_dte_lookup(all_expiries, day_date)

    grid_times = _get_grid_param("DECISION_MINUTES_ET")
    grid_dtes = _get_grid_param("DTE_TARGETS")
    grid_sides = _get_grid_param("SPREAD_SIDES")
    grid_deltas = _get_grid_param("DELTA_TARGETS")
    grid_widths = _get_grid_param("WIDTH_TARGETS")

    candidates: list[dict] = []
    for hour, minute in grid_times:
        dec_et = datetime(
            day_date.year, day_date.month, day_date.day,
            hour, minute, tzinfo=ET,
        )
        dec_utc = dec_et.astimezone(timezone.utc)

        snapshot = get_cbbo_snapshot_at(cbbo_df, dec_utc)
        if snapshot.empty:
            continue

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
            spx_spot = derive_spx_from_parity(
                snapshot, inst_map, day_date, spy_price,
            )

        vix = lookup_intraday_value(uq_df, "VIX", dec_utc)
        vix9d = lookup_intraday_value(uq_df, "VIX9D", dec_utc)
        term_structure = (vix9d / vix) if vix and vix > 0 and vix9d else None
        vvix = lookup_intraday_value(uq_df, "VVIX", dec_utc)
        skew = skew_daily.get(day_date)

        cal = cal_map.get(day_date, {})
        is_opex = cal.get("is_opex", False)
        is_fomc = cal.get("is_fomc", False)
        is_tw = cal.get("is_triple_witching", False)
        is_cpi = cal.get("is_cpi", False)
        is_nfp = cal.get("is_nfp", False)

        offline_gex_net: float | None = None
        offline_zero_gamma: float | None = None
        if oi_df is not None:
            offline_gex_net, offline_zero_gamma = compute_offline_gex(
                snapshot, inst_map, oi_df, spx_spot, day_date,
            )

        for dte_target in grid_dtes:
            expiry = find_expiry_for_dte(dte_map, dte_target)
            if expiry is None:
                continue
            for side in grid_sides:
                for delta_target in grid_deltas:
                    for width in grid_widths:
                        cands = build_candidates_for_snapshot(
                            snapshot=snapshot,
                            inst_map=inst_map,
                            spot=spx_spot,
                            spy_price=spy_price,
                            vix=vix,
                            vix9d=vix9d,
                            term_structure=term_structure,
                            vvix=vvix,
                            skew=skew,
                            is_opex_day=is_opex,
                            is_fomc_day=is_fomc,
                            is_triple_witching=is_tw,
                            is_cpi_day=is_cpi,
                            is_nfp_day=is_nfp,
                            decision_dt=dec_utc,
                            day_date=day_date,
                            dte_target=dte_target,
                            expiry=expiry,
                            delta_target=delta_target,
                            side=side,
                            width_points=width,
                        )
                        for cand in cands:
                            cand["offline_gex_net"] = offline_gex_net
                            cand["offline_zero_gamma"] = offline_zero_gamma
                        candidates.extend(cands)

    del cbbo_df, inst_map, oi_df
    gc.collect()
    return candidates


# ===================================================================
# INCREMENTAL CANDIDATE CACHING
# ===================================================================


def _compute_code_version() -> str:
    """SHA-256 of this script and its key dependencies for cache invalidation.

    Includes _constants.py (CONTRACT_MULT, CONTRACTS, MARGIN_PER_LOT) and
    regime_utils.py so that changes to contract economics or regime logic
    also invalidate the candidate and label caches.
    """
    hasher = hashlib.sha256(Path(__file__).read_bytes())
    for dep in ("_constants.py", "regime_utils.py"):
        dep_path = Path(__file__).resolve().parent / dep
        if dep_path.exists():
            hasher.update(dep_path.read_bytes())
    return hasher.hexdigest()[:16]


def _atomic_write_csv(df: "pd.DataFrame", dest: Path) -> None:
    """Write a DataFrame to ``dest`` atomically (L9 fix).

    Pandas' ``to_csv`` writes incrementally to the destination path; a
    KeyboardInterrupt or OOM halfway through leaves a truncated file
    that downstream consumers (the run_pipeline orchestrator,
    backtest_strategy.py) treat as complete and silently train/backtest
    on partial data.

    The helper writes to ``dest.with_suffix(dest.suffix + ".tmp")``
    first and then ``os.replace()``s into place, which is atomic on
    POSIX/macOS/NTFS — readers either see the old file or the new
    one, never a half-written file.

    The caller is responsible for ensuring ``dest`` lives in a writable
    directory; the temp lives next to ``dest`` so the rename is
    same-filesystem (a cross-filesystem ``os.replace()`` would fall
    back to copy+remove and lose atomicity).
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(str(tmp), index=False)
    os.replace(str(tmp), str(dest))


def _input_data_fingerprint(day_str: str) -> dict[str, list[float | int]]:
    """Return ``{input_file_name: [mtime_seconds, size_bytes]}`` for every
    input file ``_generate_candidates_for_day(day_str)`` may consume.

    H3 fix: the candidate cache previously only invalidated on
    ``code_version`` and ``grid_hash`` changes.  If a day's underlying
    .dbn.zst (or production parquet) was silently corrected or
    re-issued (Databento sometimes re-publishes corrected daily files,
    and the production parquet is rerun ad hoc), the cached candidates
    persisted into training rows.  Storing per-day file fingerprints
    in the manifest lets us detect content changes and re-generate just
    the affected days.

    The fingerprint is intentionally lightweight: ``(mtime, size)`` is
    cheap to compute and adequate for catching the failure modes we
    care about (file replaced, file truncated, file extended).  If
    bit-identical re-saves become a concern in the future, the helper
    can be extended to a full content hash without changing callers.

    Parameters
    ----------
    day_str : str
        Compact ``YYYYMMDD`` trading day.

    Returns
    -------
    dict[str, list[float | int]]
        Mapping from a stable identifier (``"<source>/<filename>"``) to
        ``[mtime_rounded, size_bytes]``.  Files that don't exist are
        omitted; an empty dict therefore means "no inputs found", which
        is a legitimate cache state for days where generation produces
        zero candidates.
    """
    candidates = [
        ("databento_spxw_def", SPXW_DEFS / f"{day_str}.dbn.zst"),
        ("databento_spx_def", SPX_DEFS / f"{day_str}.dbn.zst"),
        ("databento_spxw_cbbo", SPXW_CBBO / f"{day_str}.dbn.zst"),
        ("databento_spx_cbbo", SPX_CBBO / f"{day_str}.dbn.zst"),
        ("databento_spxw_stats", SPXW_STATS / f"{day_str}.dbn.zst"),
        ("databento_spx_stats", SPX_STATS / f"{day_str}.dbn.zst"),
        ("production_chain", PRODUCTION_CHAINS_DIR / f"{day_str}.parquet"),
    ]
    fp: dict[str, list[float | int]] = {}
    for tag, p in candidates:
        if p.exists():
            st = p.stat()
            # Round mtime to milliseconds so trivial filesystem-induced
            # nanosecond drift does not invalidate caches.
            fp[tag] = [round(st.st_mtime, 3), int(st.st_size)]
    return fp


def _load_cache_manifest(cache_dir: Path) -> dict:
    """Load the cache manifest file, or return an empty dict."""
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Corrupt cache manifest, ignoring: %s", exc)
    return {}


def _save_cache_manifest(cache_dir: Path, manifest: dict) -> None:
    """Write the cache manifest file atomically."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    tmp.rename(manifest_path)


def _cache_day_candidates(
    cache_dir: Path,
    day_str: str,
    candidates: list[dict],
) -> int:
    """Save one day's candidates as a Parquet file. Returns row count."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        return 0
    df = pd.DataFrame(candidates)
    parquet_path = cache_dir / f"{day_str}.parquet"
    df.to_parquet(parquet_path, index=False)
    return len(df)


def _load_cached_day(cache_dir: Path, day_str: str) -> list[dict]:
    """Load cached candidates for a single day from Parquet.

    Returns an empty list if the file is missing or corrupt so callers
    can fall back to regeneration.
    """
    parquet_path = cache_dir / f"{day_str}.parquet"
    if not parquet_path.exists():
        return []
    try:
        return pd.read_parquet(parquet_path).to_dict("records")
    except Exception:
        logger.warning("Corrupt candidate cache %s — will regenerate", parquet_path)
        return []


# ===================================================================
# MAIN PIPELINE ORCHESTRATION
# ===================================================================

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


# ===================================================================
# PRECOMPUTE OFFLINE GEX
# ===================================================================

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


# ===================================================================
# RELABEL EXISTING CSV WITH TRAJECTORY DATA
# ===================================================================

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


# ===================================================================
# CLI
# ===================================================================

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
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
