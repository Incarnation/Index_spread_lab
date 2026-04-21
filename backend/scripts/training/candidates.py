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

from .bs_gex_spot import (
    _time_to_expiry_years,
    bs_delta_vec,
    build_dte_lookup,
    compute_offline_gex,
    derive_spx_from_parity,
    find_expiry_for_dte,
    get_cbbo_snapshot_at,
    implied_vol_vec,
)
from .io_loaders import (
    _available_day_files,
    build_instrument_map,
    load_cbbo,
    load_definitions,
    load_economic_calendar,
    load_frd_quotes,
    load_spy_equity,
    load_statistics,
    lookup_gex_context,
    lookup_intraday_value,
    merge_underlying_quotes,
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
def _compute_code_version() -> str:
    """SHA-256 of this script and its key dependencies for cache invalidation.

    The candidate cache (and, transitively, the label cache) is keyed on
    this hash.  ANY file whose contents materially change the bytes of
    a generated candidate row MUST be included here, otherwise stale
    cache entries will silently survive a code change and pollute the
    training data.

    Files hashed (Tier 1 BLOCKER fix from E2E pipeline review):

    * ``candidates.py`` (this module) – candidate construction logic
    * ``bs_gex_spot.py`` – GEX/spot computations called per-candidate
    * ``io_loaders.py`` – the loaders that provide the inputs we feed
      into the candidate logic (a change to lookup/merge logic can
      shift candidate values even without touching this module)
    * ``_constants.py`` – CONTRACT_MULT, CONTRACTS, MARGIN_PER_LOT
    * ``regime_utils.py`` – regime classification used in candidate
      enrichment

    Anything else (CLI parsing, label evaluation, output formatting)
    does NOT belong here.  Label-cache-specific code lives in
    ``_compute_label_code_hash`` in labeling.py.
    """
    hasher = hashlib.sha256(Path(__file__).read_bytes())
    here = Path(__file__).resolve().parent
    # Same-package deps (sibling modules in scripts/training/).
    for dep in ("bs_gex_spot.py", "io_loaders.py"):
        dep_path = here / dep
        if dep_path.exists():
            hasher.update(dep_path.read_bytes())
    # Parent-package deps living in scripts/.
    for dep in ("_constants.py", "regime_utils.py"):
        dep_path = here / dep
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
