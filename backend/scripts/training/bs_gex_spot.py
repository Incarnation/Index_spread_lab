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
TAKE_PROFIT_PCT = 0.60
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
