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
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
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
VIX_CSV = DATA_DIR / "vix_history.csv"
VIX9D_CSV = DATA_DIR / "vix9d_history.csv"
UNDERLYING_QUOTES_CSV = DATA_DIR / "underlying_quotes_export.csv"
CONTEXT_SNAPSHOTS_CSV = DATA_DIR / "context_snapshots_export.csv"
OFFLINE_GEX_CSV = DATA_DIR / "offline_gex_cache.csv"
OUTPUT_CSV = DATA_DIR / "training_candidates.csv"

# FirstRateData 1-min index parquets (converted from raw .txt downloads)
FRD_DIR = DATA_DIR / "firstratedata"
FRD_SPX = FRD_DIR / "spx_1min.parquet"
FRD_VIX = FRD_DIR / "vix_1min.parquet"
FRD_VIX9D = FRD_DIR / "vix9d_1min.parquet"

# ---------------------------------------------------------------------------
# Pipeline constants (match production config defaults)
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
SPY_SPX_RATIO = 10.024
RISK_FREE_RATE = 0.043
DECISION_MINUTES_ET = [(10, 2), (11, 2), (12, 2)]
DTE_TARGETS = [0, 3, 5, 7, 10]
DELTA_TARGETS = [0.10, 0.20]
SPREAD_SIDES = ["put", "call"]
WIDTH_POINTS = 10.0
TAKE_PROFIT_PCT = 0.50
STOP_LOSS_PCT = 2.00
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


def load_vix_csv(csv_path: Path) -> dict[date, float]:
    """Load a CBOE daily CSV (VIX or VIX9D) into a date -> close mapping."""
    out: dict[date, float] = {}
    if not csv_path.exists():
        return out
    df = pd.read_csv(str(csv_path))
    for _, row in df.iterrows():
        try:
            d = pd.to_datetime(row["DATE"]).date()
            out[d] = float(row["CLOSE"])
        except Exception:
            continue
    return out


def load_underlying_quotes(csv_path: Path) -> pd.DataFrame:
    """Load the production DB underlying_quotes export for intraday VIX/SPX.

    Returns a DataFrame with columns ``ts`` (tz-aware UTC), ``symbol``,
    and ``last``, sorted by timestamp.  Returns an empty DataFrame when
    the file does not exist.
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["ts", "symbol", "last"])
    df = pd.read_csv(str(csv_path))
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df[["ts", "symbol", "last"]].dropna(subset=["last"])
    return df.sort_values("ts").reset_index(drop=True)


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


def load_context_snapshots(csv_path: Path) -> pd.DataFrame:
    """Load the production DB context_snapshots export for GEX data.

    Returns a DataFrame with columns ``ts`` (tz-aware UTC), ``gex_net``,
    and ``zero_gamma_level``, sorted by timestamp.  Returns an empty
    DataFrame when the file does not exist.
    """
    if not csv_path.exists():
        return pd.DataFrame(columns=["ts", "gex_net", "zero_gamma_level"])
    df = pd.read_csv(str(csv_path))
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return df.sort_values("ts").reset_index(drop=True)


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


def _evaluate_outcome(entry_credit: float, marks: list[dict]) -> dict:
    """Evaluate credit-spread outcome from forward marks.

    Mirrors production ``trade_pnl_job`` close logic:
    - Stop-loss fires when PnL <= -(STOP_LOSS_PCT * max_profit)
    - TP50 hit when PnL >= TAKE_PROFIT_PCT * max_profit
    - Stop-loss is checked first each bar (if both trigger in the same
      minute, the stop-loss takes precedence)
    - If neither fires, the last mark's PnL is used as the realized outcome

    Parameters
    ----------
    entry_credit : float      Credit received at entry (positive).
    marks        : list[dict] Forward CBBO marks with short/long bid/ask.

    Returns
    -------
    dict  Outcome fields to merge into the candidate.
    """
    if not marks:
        return {
            "resolved": False, "hit_tp50": False,
            "hit_tp100_at_expiry": False,
            "realized_pnl": None, "exit_reason": "NO_MARKS",
        }

    max_profit = entry_credit * CONTRACT_MULT * CONTRACTS
    tp_thr = max_profit * TAKE_PROFIT_PCT
    sl_thr = max_profit * STOP_LOSS_PCT
    tp100_thr = max_profit

    first_tp50_pnl: float | None = None
    last_pnl: float | None = None

    for m in marks:
        s_mid = _mid(m["short_bid"], m["short_ask"])
        l_mid = _mid(m["long_bid"], m["long_ask"])
        if s_mid is None or l_mid is None:
            continue
        exit_cost = s_mid - l_mid
        pnl = (entry_credit - exit_cost) * CONTRACT_MULT * CONTRACTS
        last_pnl = pnl

        if pnl <= -sl_thr:
            return {
                "resolved": True, "hit_tp50": False,
                "hit_tp100_at_expiry": False,
                "realized_pnl": pnl, "exit_reason": "STOP_LOSS",
            }

        if first_tp50_pnl is None and pnl >= tp_thr:
            first_tp50_pnl = pnl

    if last_pnl is None:
        return {
            "resolved": False, "hit_tp50": False,
            "hit_tp100_at_expiry": False,
            "realized_pnl": None, "exit_reason": "NO_VALID_MARKS",
        }

    hit_tp100 = bool(last_pnl >= tp100_thr)

    if first_tp50_pnl is not None:
        return {
            "resolved": True, "hit_tp50": True,
            "hit_tp100_at_expiry": hit_tp100,
            "realized_pnl": first_tp50_pnl, "exit_reason": "TAKE_PROFIT_50",
        }
    return {
        "resolved": True, "hit_tp50": False,
        "hit_tp100_at_expiry": hit_tp100,
        "realized_pnl": last_pnl, "exit_reason": "EXPIRY_OR_LAST_MARK",
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
        c.update(_evaluate_outcome(c["entry_credit"], marks))
        labeled.append(c)

        if (ci + 1) % 50 == 0:
            print(f"  Labeled {ci + 1}/{len(candidates)} candidates", flush=True)

    return labeled


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
    *, max_days: int | None = None, deploy: bool = False, verbose: bool = True,
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
    vix_daily = load_vix_csv(VIX_CSV)
    vix9d_daily = load_vix_csv(VIX9D_CSV)
    prod_uq = load_underlying_quotes(UNDERLYING_QUOTES_CSV)

    # FirstRateData 1-min parquets provide full-history intraday coverage
    # for VIX (since 2008) and VIX9D (since 2013), plus SPX as a reference.
    # Production DB rows take priority for any overlapping timestamps.
    frd_spx = load_frd_quotes(FRD_SPX, "SPX")
    frd_vix = load_frd_quotes(FRD_VIX, "VIX")
    frd_vix9d = load_frd_quotes(FRD_VIX9D, "VIX9D")
    uq_df = merge_underlying_quotes(prod_uq, frd_spx, frd_vix, frd_vix9d)
    frd_total = len(frd_spx) + len(frd_vix) + len(frd_vix9d)

    cs_df = load_context_snapshots(CONTEXT_SNAPSHOTS_CSV)

    # Merge offline GEX cache into context snapshots (fills gaps before prod DB)
    if OFFLINE_GEX_CSV.exists():
        gex_cache = pd.read_csv(str(OFFLINE_GEX_CSV))
        gex_cache["ts"] = pd.to_datetime(gex_cache["ts"], utc=True, format="ISO8601")
        if not gex_cache.empty:
            needed_cols = ["ts", "gex_net", "zero_gamma_level"]
            for col in needed_cols:
                if col not in gex_cache.columns:
                    gex_cache[col] = None
            cache_subset = gex_cache[needed_cols].copy()
            if cs_df.empty:
                cs_df = cache_subset
            else:
                existing_ts = set(cs_df["ts"].dropna())
                new_rows = cache_subset[~cache_subset["ts"].isin(existing_ts)]
                if not new_rows.empty:
                    cs_df = pd.concat([cs_df, new_rows], ignore_index=True)
                    cs_df = cs_df.sort_values("ts").reset_index(drop=True)

    gex_count = cs_df["gex_net"].notna().sum() if not cs_df.empty else 0
    print(f"  SPY equity rows : {len(spy_df)}")
    print(f"  VIX daily dates : {len(vix_daily)}")
    print(f"  VIX9D daily dates: {len(vix9d_daily)}")
    print(f"  Underlying quotes: {len(uq_df)} rows (prod={len(prod_uq)}, FRD={frd_total:,})")
    print(f"  Context snapshots: {len(cs_df)} rows ({gex_count} with GEX)", flush=True)

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

            # SPY price at decision time
            spy_mask = (
                (spy_df["ts"] >= dec_utc - timedelta(minutes=5))
                & (spy_df["ts"] <= dec_utc)
            )
            spy_rows = spy_df[spy_mask]
            if spy_rows.empty:
                continue
            spy_price = float(spy_rows.iloc[-1]["close"])

            # SPX spot: prefer put-call parity, fall back to SPY * ratio
            spx_spot = derive_spx_from_parity(
                snapshot, inst_map, day_date, spy_price,
            )

            # VIX / VIX9D: prefer production DB intraday, fall back to daily
            vix = lookup_intraday_value(uq_df, "VIX", dec_utc)
            if vix is None:
                vix = vix_daily.get(day_date)
            vix9d = lookup_intraday_value(uq_df, "VIX9D", dec_utc)
            if vix9d is None:
                vix9d = vix9d_daily.get(day_date)
            term_structure = (vix9d / vix) if vix and vix > 0 and vix9d else None

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
    print("[PIPELINE] Labeling candidates (forward-looking) ...", flush=True)
    labeled = label_candidates(all_candidates, trading_days)
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
    print(f"[GEX-PRECOMPUTE] {len(trading_days)} trading days, "
          f"SPY equity: {len(spy_df)} rows", flush=True)

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
    args = parser.parse_args()

    if args.precompute_gex:
        precompute_offline_gex_to_csv(max_days=args.max_days)
        return

    run_pipeline(
        max_days=args.max_days,
        deploy=args.deploy,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
