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
import json
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

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

from spx_backend.jobs.modeling import (
    extract_candidate_features,
    predict_with_bucket_model,
    summarize_strategy_quality,
    train_bucket_model,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = _BACKEND.parent / "data"
DATABENTO_DIR = DATA_DIR / "databento"
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
DECISION_MINUTES_ET = [(9, 45), (10, 15), (10, 45), (11, 15), (11, 45), (12, 15)]
DTE_TARGETS = [0, 1, 2, 3, 5, 7]
DELTA_TARGETS = [0.05, 0.10, 0.15, 0.20, 0.25]
SPREAD_SIDES = ["put", "call"]
WIDTH_POINTS = 10.0
TAKE_PROFIT_PCT = 0.50
STOP_LOSS_PCT = 2.00
LABEL_MARK_INTERVAL_MINUTES = 5
CONTRACT_MULT = 100
CONTRACTS = 1
MIN_MID_PRICE = 0.05
MAX_IV = 5.0
MIN_IV = 0.01
IV_BISECT_ITERS = 60


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
    """Return sorted YYYYMMDD strings for .dbn.zst files in *directory*."""
    results = []
    for f in directory.glob("*.dbn.zst"):
        day_str = f.name.split(".")[0]
        if day_str.isdigit() and len(day_str) == 8:
            results.append(day_str)
    return sorted(results)


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
        print(f"  [WARN] Cannot load {path.name}: {exc}")
        return None


def load_definitions(day_str: str) -> pd.DataFrame | None:
    """Load SPXW + SPX instrument definitions for a single trading day.

    Merges definitions from both SPXW (daily/weekly) and SPX (monthly)
    sources.  Keeps only the latest definition per instrument_id
    (handles intra-day security_update_action MODIFY records).
    """
    frames: list[pd.DataFrame] = []
    for defs_dir in (SPXW_DEFS, SPX_DEFS):
        path = defs_dir / f"{day_str}.dbn.zst"
        if path.exists():
            tmp = _load_dbn(path)
            if tmp is not None and not tmp.empty:
                frames.append(tmp)
    if not frames:
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
    """
    frames: list[pd.DataFrame] = []
    for cbbo_dir in (SPXW_CBBO, SPX_CBBO):
        path = cbbo_dir / f"{day_str}.dbn.zst"
        if path.exists():
            tmp = _load_dbn(path)
            if tmp is not None and not tmp.empty:
                frames.append(tmp)
    if not frames:
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
        return None
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    oi_df = df[df["stat_type"] == OI_STAT_TYPE].copy()
    if oi_df.empty:
        return None
    oi_df = oi_df[["instrument_id", "quantity"]].rename(columns={"quantity": "oi"})
    oi_df = oi_df.drop_duplicates("instrument_id", keep="last")
    return oi_df


def load_spy_equity() -> pd.DataFrame:
    """Load SPY 1-minute equity bars (parquet from Databento ohlcv-1m).

    Returns a DataFrame with ``ts`` (tz-aware UTC) and ``close`` columns.
    """
    df = pd.read_parquet(str(SPY_EQUITY_PATH))
    df = df.reset_index()
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    return df[["ts", "close"]].sort_values("ts").reset_index(drop=True)


def load_daily_parquet(parquet_path: Path) -> dict[date, float]:
    """Load a daily-granularity parquet and return a date -> close mapping.

    Reads ``ts`` and ``close`` columns, groups by calendar date, takes the
    last close per day.  Returns an empty dict when the file is missing.

    Parameters
    ----------
    parquet_path : Path
        Path to a parquet file with ``ts`` (tz-aware UTC) and ``close``
        columns where each row is one daily observation.

    Returns
    -------
    dict[date, float]
        Mapping from trading date to the day's closing value.
    """
    if not parquet_path.exists():
        return {}
    df = pd.read_parquet(parquet_path, columns=["ts", "close"])
    df["date"] = df["ts"].dt.date
    daily = df.groupby("date")["close"].last()
    return daily.to_dict()


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
    """Load a FirstRateData 1-min parquet as an underlying-quotes DataFrame.

    Converts the OHLC ``close`` column to ``last`` so the output schema
    matches ``load_underlying_quotes`` (columns: ``ts``, ``symbol``,
    ``last``).  This allows seamless merging with the production DB export.

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
        ``last``, sorted by timestamp.  Empty DataFrame if file missing.
    """
    if not parquet_path.exists():
        return pd.DataFrame(columns=["ts", "symbol", "last"])
    df = pd.read_parquet(parquet_path, columns=["ts", "close"])
    df = df.rename(columns={"close": "last"})
    df["symbol"] = symbol
    return df[["ts", "symbol", "last"]].sort_values("ts").reset_index(drop=True)


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

        mid = _mid(row.get("bid_px_00"), row.get("ask_px_00"))
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
        bid = float(row["bid_px_00"])
        ask = float(row["ask_px_00"])
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2.0
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
) -> list[dict]:
    """Build vertical credit-spread candidates from a single CBBO snapshot.

    Finds the short leg with |delta| closest to *delta_target*, pairs it
    with a long leg at short_strike ± WIDTH_POINTS, and computes entry
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
    opts = opts[(opts["bid"] > 0) & (opts["ask"] > 0) & (opts["mid"] >= MIN_MID_PRICE)]
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
        short["strike"] - WIDTH_POINTS if side == "put"
        else short["strike"] + WIDTH_POINTS
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
    if actual_width <= 0 or abs(actual_width - WIDTH_POINTS) > 0.01:
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

def _mid(bid: float | None, ask: float | None) -> float | None:
    """Mid-price when both sides are positive and finite; None otherwise."""
    if bid is None or ask is None:
        return None
    b, a = float(bid), float(ask)
    if math.isnan(b) or math.isnan(a) or b <= 0 or a <= 0:
        return None
    return (b + a) / 2.0


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


def _evaluate_outcome(entry_credit: float, marks: list[dict]) -> dict:
    """Evaluate credit-spread outcome from forward marks with full trajectory.

    Iterates ALL marks without early-returning at stop-loss, capturing
    trajectory data that enables analysis of any SL level (1x-5x) and any
    exit rule (TP50 vs hold-to-expiry) from a single pipeline run.

    Backward-compatible fields (``hit_tp50``, ``realized_pnl``,
    ``exit_reason``) preserve the original with-SL semantics.  New fields
    capture the full PnL trajectory and hold-through outcomes.

    Parameters
    ----------
    entry_credit : float
        Credit received at entry (positive).
    marks : list[dict]
        Forward CBBO marks with short/long bid/ask.

    Returns
    -------
    dict
        Outcome fields including backward-compatible (with-SL) columns and
        new trajectory / hold-through columns.

        Trajectory columns
        ------------------
        max_adverse_pnl       : worst PnL seen across all marks.
        max_adverse_multiple  : max_adverse_pnl / max_profit (e.g. -2.3).
        min_pnl_before_tp50   : worst PnL before TP50 first triggered; used
                                to determine which SL levels would fire
                                before TP50 for the sweep analysis.
        first_tp50_pnl        : PnL at the first TP50 trigger (None if never).
        final_pnl_at_expiry   : PnL at the very last mark.
        final_is_tp100        : True if last mark PnL >= full credit.
        hit_stop_loss         : True if PnL breached -2x credit at any point.
        recovered_after_sl    : True if 2x SL breached, then TP50 fired later.

        Hold-through columns (no-SL + close-at-TP50 strategy)
        ------------------------------------------------------
        hold_hit_tp50            : TP50 fires at any point ignoring SL.
        hold_realized_pnl        : PnL under hold-to-TP50 or hold-to-expiry.
        hold_exit_reason         : exit type under hold strategy.
        hold_hit_tp100_at_expiry : True if final PnL >= full credit.
    """
    _trajectory_nulls = {
        "max_adverse_pnl": None, "max_adverse_multiple": None,
        "min_pnl_before_tp50": None, "first_tp50_pnl": None,
        "final_pnl_at_expiry": None, "final_is_tp100": False,
        "hit_stop_loss": False, "recovered_after_sl": False,
        "hold_hit_tp50": False, "hold_realized_pnl": None,
        "hold_exit_reason": "NO_MARKS",
        "hold_hit_tp100_at_expiry": False,
    }

    if not marks:
        return {
            "resolved": False, "hit_tp50": False,
            "hit_tp100_at_expiry": False,
            "realized_pnl": None, "exit_reason": "NO_MARKS",
            **_trajectory_nulls,
        }

    max_profit = entry_credit * CONTRACT_MULT * CONTRACTS
    tp_thr = max_profit * TAKE_PROFIT_PCT
    sl_thr = max_profit * STOP_LOSS_PCT
    tp100_thr = max_profit

    max_adverse_pnl: float = 0.0
    min_pnl_before_tp50: float = 0.0
    first_tp50_pnl: float | None = None
    sl_breach_pnl: float | None = None
    sl_breached_before_tp50: bool = False
    last_pnl: float | None = None

    for m in marks:
        s_mid = _mid(m["short_bid"], m["short_ask"])
        l_mid = _mid(m["long_bid"], m["long_ask"])
        if s_mid is None or l_mid is None:
            continue
        exit_cost = s_mid - l_mid
        pnl = (entry_credit - exit_cost) * CONTRACT_MULT * CONTRACTS
        last_pnl = pnl

        max_adverse_pnl = min(max_adverse_pnl, pnl)

        if first_tp50_pnl is None:
            min_pnl_before_tp50 = min(min_pnl_before_tp50, pnl)

        if sl_breach_pnl is None and pnl <= -sl_thr:
            sl_breach_pnl = pnl
            if first_tp50_pnl is None:
                sl_breached_before_tp50 = True

        if first_tp50_pnl is None and pnl >= tp_thr:
            first_tp50_pnl = pnl

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
    final_is_tp100 = bool(last_pnl >= tp100_thr)
    hit_stop_loss = sl_breach_pnl is not None
    recovered_after_sl = sl_breached_before_tp50 and first_tp50_pnl is not None

    # --- Original outcome (backward-compatible with-SL) ---
    if sl_breached_before_tp50:
        orig_exit = "STOP_LOSS"
        orig_pnl = sl_breach_pnl
        orig_tp50 = False
        orig_tp100 = False
    elif first_tp50_pnl is not None:
        orig_exit = "TAKE_PROFIT_50"
        orig_pnl = first_tp50_pnl
        orig_tp50 = True
        orig_tp100 = final_is_tp100
    else:
        orig_exit = "EXPIRY_OR_LAST_MARK"
        orig_pnl = last_pnl
        orig_tp50 = False
        orig_tp100 = final_is_tp100

    # --- Hold-through outcome (ignore SL, still close at TP50) ---
    if first_tp50_pnl is not None:
        hold_exit = "TAKE_PROFIT_50"
        hold_pnl = first_tp50_pnl
        hold_tp50 = True
    else:
        hold_exit = "EXPIRY_OR_LAST_MARK"
        hold_pnl = last_pnl
        hold_tp50 = False

    return {
        "resolved": True,
        "hit_tp50": orig_tp50,
        "hit_tp100_at_expiry": orig_tp100,
        "realized_pnl": orig_pnl,
        "exit_reason": orig_exit,
        "max_adverse_pnl": max_adverse_pnl,
        "max_adverse_multiple": max_adverse_multiple,
        "min_pnl_before_tp50": min_pnl_before_tp50,
        "first_tp50_pnl": first_tp50_pnl,
        "final_pnl_at_expiry": last_pnl,
        "final_is_tp100": final_is_tp100,
        "hit_stop_loss": hit_stop_loss,
        "recovered_after_sl": recovered_after_sl,
        "hold_hit_tp50": hold_tp50,
        "hold_realized_pnl": hold_pnl,
        "hold_exit_reason": hold_exit,
        "hold_hit_tp100_at_expiry": final_is_tp100,
    }


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
        marks = _downsample_marks(marks)

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

    # Phase 2: extract marks per day (parallel or sequential)
    all_marks: list[list[dict]] = [[] for _ in candidates]
    day_args = sorted(plan.items())

    if workers > 1 and n_forward_days > 1:
        actual_workers = min(workers, n_forward_days)
        print(f"  Using {actual_workers} workers for mark extraction", flush=True)
        with Pool(actual_workers) as pool:
            for di, day_results in enumerate(
                pool.imap_unordered(_extract_marks_for_day, day_args),
            ):
                for ci, marks in day_results:
                    all_marks[ci].extend(marks)
                if (di + 1) % 10 == 0 or di + 1 == n_forward_days:
                    print(
                        f"  Extracted marks: {di + 1}/{n_forward_days} forward days",
                        flush=True,
                    )
    else:
        for di, day_item in enumerate(day_args):
            day_results = _extract_marks_for_day(day_item)
            for ci, marks in day_results:
                all_marks[ci].extend(marks)
            if (di + 1) % 10 == 0 or di + 1 == n_forward_days:
                print(
                    f"  Extracted marks: {di + 1}/{n_forward_days} forward days",
                    flush=True,
                )

    # Phase 3: sort, downsample, evaluate outcomes
    print("  Evaluating outcomes ...", flush=True)
    for ci, c in enumerate(candidates):
        marks = sorted(all_marks[ci], key=lambda m: m["ts"])
        marks = _downsample_marks(marks)
        c.update(_evaluate_outcome(c["entry_credit"], marks))
        if (ci + 1) % 5000 == 0:
            print(f"  Evaluated {ci + 1}/{len(candidates)} candidates", flush=True)

    print(f"  Labeling complete: {len(candidates)} candidates", flush=True)
    return candidates


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
    """Walk-forward validation: train on earliest *train_ratio*, test rest.

    Uses production ``train_bucket_model`` and ``predict_with_bucket_model``
    to produce out-of-sample quality metrics via ``summarize_strategy_quality``.
    """
    sorted_rows = sorted(rows, key=lambda r: r["day"])
    split_idx = int(len(sorted_rows) * train_ratio)
    train_rows = sorted_rows[:split_idx]
    test_rows = sorted_rows[split_idx:]

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

    from dotenv import load_dotenv
    load_dotenv(_BACKEND / ".env")

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("[ERROR] DATABASE_URL not set; cannot deploy model.")
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
# MAIN PIPELINE ORCHESTRATION
# ===================================================================

def run_pipeline(
    *,
    max_days: int | None = None,
    deploy: bool = False,
    verbose: bool = True,
    workers: int = 8,
) -> None:
    """Execute the full offline training pipeline end-to-end.

    Steps:
      1. Discover available trading days from Databento files.
      2. Load reference data (SPY equity bars, VIX / VIX9D CSVs).
      3. For each day × decision time × DTE × delta: construct candidates.
      4. Label all candidates with forward-looking cbbo-1m marks.
      5. Build production-compatible feature vectors.
      6. Walk-forward validation and final model training.
      7. (Optional) insert model into production DB as shadow.
    """
    t0 = time.time()

    # -- 1. Discover trading days --
    spxw_days = set(_available_day_files(SPXW_CBBO))
    spx_days = set(_available_day_files(SPX_CBBO))
    trading_days = sorted(spxw_days | spx_days)
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

    # Daily SKEW from parquet (one value per trading day)
    skew_daily = load_daily_parquet(FRD_SKEW)

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

    # -- 3. Generate candidates --
    print("[PIPELINE] Generating candidates ...", flush=True)
    all_candidates: list[dict] = []

    for day_str in trading_days:
        day_date = date(int(day_str[:4]), int(day_str[4:6]), int(day_str[6:]))

        def_df = load_definitions(day_str)
        if def_df is None:
            if verbose:
                print(f"  [{day_str}] no definitions -- skip")
            continue
        inst_map = build_instrument_map(def_df)
        del def_df
        if not inst_map:
            continue

        cbbo_df = load_cbbo(day_str)
        if cbbo_df is None or cbbo_df.empty:
            if verbose:
                print(f"  [{day_str}] no cbbo data -- skip")
            continue

        oi_df = load_statistics(day_str)

        all_expiries = sorted({info["expiry"] for info in inst_map.values()})
        dte_map = build_dte_lookup(all_expiries, day_date)

        day_count = 0
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
                spx_spot = derive_spx_from_parity(
                    snapshot, inst_map, day_date, spy_price,
                )

            vix = lookup_intraday_value(uq_df, "VIX", dec_utc)
            vix9d = lookup_intraday_value(uq_df, "VIX9D", dec_utc)
            term_structure = (vix9d / vix) if vix and vix > 0 and vix9d else None
            vvix = lookup_intraday_value(uq_df, "VVIX", dec_utc)
            skew = skew_daily.get(day_date)

            # Calendar event flags for this trading day
            cal = cal_map.get(day_date, {})
            is_opex = cal.get("is_opex", False)
            is_fomc = cal.get("is_fomc", False)
            is_tw = cal.get("is_triple_witching", False)
            is_cpi = cal.get("is_cpi", False)
            is_nfp = cal.get("is_nfp", False)

            # Precompute offline GEX for this decision time
            offline_gex_net: float | None = None
            offline_zero_gamma: float | None = None
            if oi_df is not None:
                offline_gex_net, offline_zero_gamma = compute_offline_gex(
                    snapshot, inst_map, oi_df, spx_spot, day_date,
                )

            for dte_target in DTE_TARGETS:
                expiry = find_expiry_for_dte(dte_map, dte_target)
                if expiry is None:
                    continue
                for side in SPREAD_SIDES:
                    for delta_target in DELTA_TARGETS:
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
                        )
                        for cand in cands:
                            cand["offline_gex_net"] = offline_gex_net
                            cand["offline_zero_gamma"] = offline_zero_gamma
                        all_candidates.extend(cands)
                        day_count += len(cands)

        print(
            f"  [{day_str}] {day_count} candidates "
            f"({len(inst_map)} instruments)", flush=True,
        )
        del cbbo_df, inst_map, oi_df
        gc.collect()

    print(f"[PIPELINE] Total candidates: {len(all_candidates)}", flush=True)
    if not all_candidates:
        print("[PIPELINE] No candidates generated. Check data paths.")
        return

    # -- 4. Label candidates --
    print(f"[PIPELINE] Labeling candidates (forward-looking, workers={workers}) ...", flush=True)
    labeled = label_candidates_fast(all_candidates, trading_days, workers=workers)
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
    pd.DataFrame(flat_records).to_csv(str(OUTPUT_CSV), index=False)

    # -- 7. Walk-forward validation --
    print("[PIPELINE] Walk-forward validation ...", flush=True)
    results = walk_forward_validate(training_rows)

    if "error" in results:
        print(f"  [WARN] {results['error']}")
        print("[PIPELINE] Training on all data instead ...")
        final_model = train_bucket_model(rows=training_rows)
    else:
        ts = results["test_summary"]
        print(f"\n{'=' * 60}")
        print("WALK-FORWARD RESULTS")
        print(f"{'=' * 60}")
        print(f"  Train : {results['train_count']} rows ({results['train_days']})")
        print(f"  Test  : {results['test_count']} rows ({results['test_days']})")

        def _fmt(label: str, key: str, fmt: str = ".1%", prefix: str = "") -> None:
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
    "max_adverse_pnl", "max_adverse_multiple", "min_pnl_before_tp50",
    "first_tp50_pnl", "final_pnl_at_expiry", "final_is_tp100",
    "hit_stop_loss", "recovered_after_sl",
    "hold_hit_tp50", "hold_realized_pnl", "hold_exit_reason",
    "hold_hit_tp100_at_expiry",
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
    # TP50 trades closed before expiry, so final_pnl is unknown; leave None
    # max_adverse_pnl / min_pnl_before_tp50 also unknown without marks

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
    args = parser.parse_args()

    if args.precompute_gex:
        precompute_offline_gex_to_csv(max_days=args.max_days)
        return

    if args.relabel:
        relabel_from_csv()
        return

    if args.model == "xgboost":
        from xgb_model import main as xgb_main
        xgb_main()
        return

    run_pipeline(
        max_days=args.max_days,
        deploy=args.deploy,
        verbose=not args.quiet,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
