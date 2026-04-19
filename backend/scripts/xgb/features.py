"""XGBoost model training, prediction, and walk-forward validation.

Trains dual models on the offline training CSV produced by
``generate_training_data.py``:

* **Classifier** — ``XGBClassifier`` predicting ``P(hit_tp50)``.
* **Regressor** — ``XGBRegressor`` predicting ``realized_pnl``.

Both consume continuous features directly (VIX as a float, not a regime
bucket), letting the tree learner discover optimal splits instead of
relying on hand-coded thresholds.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
from sklearn.metrics import brier_score_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor


CONTINUOUS_FEATURES = [
    "vix", "vix9d", "term_structure", "vvix", "skew",
    "delta_target", "credit_to_width", "entry_credit", "width_points",
    "spot", "spy_price",
    "short_iv", "long_iv", "short_delta", "long_delta",
    "offline_gex_net", "offline_zero_gamma",
    "max_loss",
]
ORDINAL_FEATURES = ["dte_target", "entry_hour"]
BINARY_FEATURES = ["is_opex_day", "is_fomc_day", "is_triple_witching", "is_cpi_day", "is_nfp_day"]
CATEGORICAL_FEATURES = ["spread_side"]
TARGET_CLS = "hit_tp50"
TARGET_REG = "realized_pnl"
def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convert the raw training CSV into a typed feature matrix + targets.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training CSV as loaded by ``pd.read_csv()``.  Must contain
        columns listed in ``CONTINUOUS_FEATURES``, ``ORDINAL_FEATURES``,
        ``BINARY_FEATURES``, ``CATEGORICAL_FEATURES``, and both targets.

    Returns
    -------
    X : pd.DataFrame   Feature matrix (float / int columns, NaN for missing).
    y_tp50 : pd.Series  Binary target (1 = hit TP50).
    y_pnl : pd.Series   Continuous target (realized PnL in dollars).
    """
    X = pd.DataFrame()

    # Derive entry_hour (Eastern Time) from entry_dt if available
    if "entry_hour" not in df.columns and "entry_dt" in df.columns:
        dt = pd.to_datetime(df["entry_dt"], utc=True, errors="coerce")
        df = df.copy()
        df["entry_hour"] = dt.dt.tz_convert("America/New_York").dt.hour

    for col in CONTINUOUS_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    for col in ORDINAL_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64") if col in df.columns else pd.NA

    for col in BINARY_FEATURES:
        if col in df.columns:
            X[col] = df[col].astype(int)
        else:
            X[col] = 0

    # One-hot encode spread_side (put -> 1, call -> 0)
    if "spread_side" in df.columns:
        X["is_put"] = (df["spread_side"].str.lower() == "put").astype(int)
    else:
        X["is_put"] = 0

    # Engineered interaction features
    X["vix_x_delta"] = X["vix"] * X["delta_target"]
    X["dte_x_credit"] = X["dte_target"].astype(float) * X["credit_to_width"]
    X["gex_sign"] = np.sign(X["offline_gex_net"])

    # Prefer hold-based labels (no SL) when available; fall back to originals
    cls_col = "hold_hit_tp50" if "hold_hit_tp50" in df.columns else TARGET_CLS
    reg_col = "hold_realized_pnl" if "hold_realized_pnl" in df.columns else TARGET_REG

    y_tp50 = df[cls_col].astype(int) if cls_col in df.columns else pd.Series(0, index=df.index)
    y_pnl = pd.to_numeric(df[reg_col], errors="coerce") if reg_col in df.columns else pd.Series(0.0, index=df.index)

    return X, y_tp50, y_pnl


def get_feature_names(X: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature column names."""
    return list(X.columns)
HOLD_TARGET_CLS = "hold_hit_tp50"
HOLD_TARGET_REG = "hold_realized_pnl"
def _resolve_targets(df: pd.DataFrame) -> tuple[str, str]:
    """Pick hold-based targets if available, else fall back to originals.

    Returns (cls_col, reg_col) column names present in *df*.
    """
    cls_col = HOLD_TARGET_CLS if HOLD_TARGET_CLS in df.columns else TARGET_CLS
    reg_col = HOLD_TARGET_REG if HOLD_TARGET_REG in df.columns else TARGET_REG
    return cls_col, reg_col


def build_entry_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build feature matrix using hold-based labels for the entry model.

    Identical feature engineering to ``build_feature_matrix`` but uses
    ``hold_hit_tp50`` / ``hold_realized_pnl`` when available, falling
    back to original columns for backward compatibility.

    Parameters
    ----------
    df : Raw training CSV DataFrame.

    Returns
    -------
    X, y_cls, y_pnl
    """
    X = pd.DataFrame()

    if "entry_hour" not in df.columns and "entry_dt" in df.columns:
        dt = pd.to_datetime(df["entry_dt"], utc=True, errors="coerce")
        df = df.copy()
        df["entry_hour"] = dt.dt.tz_convert("America/New_York").dt.hour

    for col in CONTINUOUS_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    for col in ORDINAL_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64") if col in df.columns else pd.NA

    for col in BINARY_FEATURES:
        X[col] = df[col].astype(int) if col in df.columns else 0

    X["is_put"] = (df["spread_side"].str.lower() == "put").astype(int) if "spread_side" in df.columns else 0

    X["vix_x_delta"] = X["vix"] * X["delta_target"]
    X["dte_x_credit"] = X["dte_target"].astype(float) * X["credit_to_width"]
    X["gex_sign"] = np.sign(X["offline_gex_net"])

    cls_col, reg_col = _resolve_targets(df)
    y_cls = df[cls_col].astype(int) if cls_col in df.columns else pd.Series(0, index=df.index)
    y_pnl = pd.to_numeric(df[reg_col], errors="coerce") if reg_col in df.columns else pd.Series(0.0, index=df.index)

    return X, y_cls, y_pnl
BIG_LOSS_THRESHOLD = -200.0
V2_EXTRA_FEATURES = [
    "is_call", "iv_skew_ratio", "credit_to_max_loss", "gex_abs",
    "vix_change_1d", "recent_loss_rate_5d",
]
def _add_v2_features(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Append loss-avoidance features to an existing feature matrix.

    Parameters
    ----------
    X  : Feature matrix from ``build_entry_feature_matrix``.
    df : Original DataFrame (same index as X) with raw columns.

    Returns
    -------
    X with extra columns appended in-place.
    """
    X["is_call"] = 1 - X["is_put"]

    short_iv = pd.to_numeric(df["short_iv"], errors="coerce") if "short_iv" in df.columns else pd.Series(np.nan, index=df.index)
    long_iv = pd.to_numeric(df["long_iv"], errors="coerce") if "long_iv" in df.columns else pd.Series(np.nan, index=df.index)
    X["iv_skew_ratio"] = short_iv / long_iv.replace(0, np.nan)

    ec = pd.to_numeric(df["entry_credit"], errors="coerce") if "entry_credit" in df.columns else pd.Series(np.nan, index=df.index)
    ml = pd.to_numeric(df["max_loss"], errors="coerce") if "max_loss" in df.columns else pd.Series(np.nan, index=df.index)
    X["credit_to_max_loss"] = ec / ml.replace(0, np.nan)

    gex = pd.to_numeric(df["offline_gex_net"], errors="coerce") if "offline_gex_net" in df.columns else pd.Series(0.0, index=df.index)
    X["gex_abs"] = gex.abs()

    # vix_change_1d: requires day-level VIX. Group by day, compute diff, merge.
    if "vix" in df.columns and "day" in df.columns:
        day_vix = df.groupby("day")["vix"].first().sort_index()
        day_vix_change = day_vix.astype(float).diff().rename("vix_change_1d")
        X["vix_change_1d"] = df["day"].map(day_vix_change).astype(float)
    else:
        X["vix_change_1d"] = np.nan

    # recent_loss_rate_5d: rolling 5-day loss rate from prior trades.
    # Uses hold_realized_pnl to define losers; computed per-day then mapped.
    reg_col = "hold_realized_pnl" if "hold_realized_pnl" in df.columns else "realized_pnl"
    if reg_col in df.columns and "day" in df.columns:
        pnl_col = pd.to_numeric(df[reg_col], errors="coerce")
        day_loss_rate = df.assign(_loss=(pnl_col < BIG_LOSS_THRESHOLD).astype(int)).groupby("day")["_loss"].mean().sort_index()
        rolling_5d = day_loss_rate.rolling(5, min_periods=1).mean().shift(1).rename("recent_loss_rate_5d")
        X["recent_loss_rate_5d"] = df["day"].map(rolling_5d).astype(float)
    else:
        X["recent_loss_rate_5d"] = np.nan

    return X


def build_entry_v2_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build V2 feature matrix targeting loss avoidance.

    Adds extra features on top of the V1 matrix and uses
    ``is_big_loss`` as the classifier target (inverted: 1 = big loss)
    and ``hold_realized_pnl`` as the regressor target.

    Parameters
    ----------
    df : Raw training CSV DataFrame.

    Returns
    -------
    X, y_is_big_loss, y_pnl
    """
    X, _, y_pnl = build_entry_feature_matrix(df)
    X = _add_v2_features(X, df.reset_index(drop=True))

    reg_col = "hold_realized_pnl" if "hold_realized_pnl" in df.columns else "realized_pnl"
    pnl = pd.to_numeric(df[reg_col], errors="coerce")
    y_big_loss = (pnl < BIG_LOSS_THRESHOLD).astype(int)

    return X, y_big_loss, y_pnl
HOLD_VS_CLOSE_TARGET = "hold_is_better"
def build_hold_vs_close_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build feature matrix and targets for hold-vs-close analysis.

    Creates three targets:
    * ``hit_stop_loss`` (binary): did the trade ever breach 2x SL?
    * ``hold_is_better`` (binary): is hold PnL > SL PnL? (only for SL trades)
    * ``hold_realized_pnl`` (regression): PnL under hold-through strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced CSV with trajectory columns (from ``--relabel``).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (same features as ``build_feature_matrix``).
    y_sl : pd.Series
        Binary target: 1 = trade hits 2x SL at any point.
    y_hold_better : pd.Series
        Binary target: 1 = holding produces better PnL than closing at SL.
        NaN for non-SL trades (they don't face the hold/close decision).
    y_hold_pnl : pd.Series
        Regression target: PnL under hold-through strategy.
    """
    X, _, _ = build_feature_matrix(df)

    y_sl = df["hit_stop_loss"].astype(int) if "hit_stop_loss" in df.columns else pd.Series(0, index=df.index)

    y_hold_pnl = (
        pd.to_numeric(df["hold_realized_pnl"], errors="coerce")
        if "hold_realized_pnl" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    # hold_is_better: for SL trades, is the hold PnL better than the SL PnL?
    sl_mask = df["exit_reason"] == "STOP_LOSS"
    y_hold_better = pd.Series(np.nan, index=df.index)
    if sl_mask.any() and "hold_realized_pnl" in df.columns:
        hold_pnl = pd.to_numeric(df.loc[sl_mask, "hold_realized_pnl"], errors="coerce")
        sl_pnl = pd.to_numeric(df.loc[sl_mask, "realized_pnl"], errors="coerce")
        y_hold_better.loc[sl_mask] = (hold_pnl > sl_pnl).astype(int)

    return X, y_sl, y_hold_better, y_hold_pnl
