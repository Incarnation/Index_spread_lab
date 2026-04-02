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
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

CONTINUOUS_FEATURES = [
    "vix", "vix9d", "term_structure", "vvix", "skew",
    "delta_target", "credit_to_width", "entry_credit", "width_points",
    "spot", "spy_price",
    "short_iv", "long_iv", "short_delta", "long_delta",
    "offline_gex_net", "offline_zero_gamma",
    "max_loss",
]

ORDINAL_FEATURES = ["dte_target", "entry_hour"]

BINARY_FEATURES = ["is_opex_day", "is_fomc_day", "is_triple_witching"]

CATEGORICAL_FEATURES = ["spread_side"]

TARGET_CLS = "hit_tp50"
TARGET_REG = "realized_pnl"

# Default XGBoost hyperparameters (reasonable starting point for ~14k rows)
DEFAULT_CLS_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

DEFAULT_REG_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


# ===================================================================
# FEATURE MATRIX
# ===================================================================

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


# ===================================================================
# TRAINING
# ===================================================================

def train_xgb_models(
    X_train: pd.DataFrame,
    y_tp50_train: pd.Series,
    y_pnl_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_tp50_val: pd.Series | None = None,
    y_pnl_val: pd.Series | None = None,
    cls_params: dict | None = None,
    reg_params: dict | None = None,
) -> dict[str, Any]:
    """Train XGBClassifier (TP50) and XGBRegressor (PnL).

    Parameters
    ----------
    X_train, y_tp50_train, y_pnl_train : training data.
    X_val, y_tp50_val, y_pnl_val : optional validation set for early stopping.
    cls_params, reg_params : override default hyperparameters.

    Returns
    -------
    dict with keys:
        classifier  : fitted XGBClassifier
        regressor   : fitted XGBRegressor
        feature_names : list[str]
        cls_params  : dict of params used
        reg_params  : dict of params used
    """
    cp = {**DEFAULT_CLS_PARAMS, **(cls_params or {})}
    rp = {**DEFAULT_REG_PARAMS, **(reg_params or {})}

    es_cls = cp.pop("early_stopping_rounds", None)
    es_reg = rp.pop("early_stopping_rounds", None)

    clf = XGBClassifier(**cp)
    reg = XGBRegressor(**rp)

    fit_cls_kw: dict[str, Any] = {}
    fit_reg_kw: dict[str, Any] = {}

    if X_val is not None and y_tp50_val is not None and es_cls:
        fit_cls_kw["eval_set"] = [(X_val, y_tp50_val)]
        fit_cls_kw["verbose"] = False
    if X_val is not None and y_pnl_val is not None and es_reg:
        fit_reg_kw["eval_set"] = [(X_val, y_pnl_val)]
        fit_reg_kw["verbose"] = False

    # XGBoost >= 2.0 uses callbacks for early stopping
    if es_cls and fit_cls_kw.get("eval_set"):
        from xgboost.callback import EarlyStopping
        clf.set_params(callbacks=[EarlyStopping(rounds=es_cls, save_best=True)])
    if es_reg and fit_reg_kw.get("eval_set"):
        from xgboost.callback import EarlyStopping
        reg.set_params(callbacks=[EarlyStopping(rounds=es_reg, save_best=True)])

    clf.fit(X_train, y_tp50_train, **fit_cls_kw)
    reg.fit(X_train, y_pnl_train, **fit_reg_kw)

    return {
        "classifier": clf,
        "regressor": reg,
        "feature_names": get_feature_names(X_train),
        "cls_params": cp,
        "reg_params": rp,
    }


# ===================================================================
# PREDICTION
# ===================================================================

def predict_xgb(
    models: dict[str, Any],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions from trained XGBoost models.

    Handles both sklearn wrappers (from ``train_xgb_models``) and raw
    Boosters (from ``load_model``).

    Parameters
    ----------
    models : dict returned by ``train_xgb_models`` or ``load_model``.
    X : feature matrix with same columns as training.

    Returns
    -------
    pd.DataFrame with columns ``prob_tp50`` and ``predicted_pnl``.
    """
    import xgboost as xgb

    clf = models["classifier"]
    reg = models["regressor"]

    if isinstance(clf, xgb.Booster):
        dm = xgb.DMatrix(X, feature_names=models.get("feature_names"))
        prob = clf.predict(dm)
        pnl = reg.predict(dm)
    else:
        prob = clf.predict_proba(X)[:, 1]
        pnl = reg.predict(X)

    return pd.DataFrame({
        "prob_tp50": prob,
        "predicted_pnl": pnl,
    }, index=X.index)


# ===================================================================
# SERIALIZATION
# ===================================================================

def save_model(models: dict[str, Any], path: Path) -> None:
    """Save XGBoost models to a directory (two .json booster files + metadata).

    Uses the native Booster serialization to avoid sklearn wrapper issues.

    Parameters
    ----------
    models : dict returned by ``train_xgb_models``.
    path   : directory to save into (created if absent).
    """
    path.mkdir(parents=True, exist_ok=True)
    models["classifier"].get_booster().save_model(str(path / "classifier.json"))
    models["regressor"].get_booster().save_model(str(path / "regressor.json"))
    meta = {
        "feature_names": models["feature_names"],
        "cls_params": models["cls_params"],
        "reg_params": models["reg_params"],
    }
    (path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(path: Path) -> dict[str, Any]:
    """Load XGBoost models from a directory saved by ``save_model``.

    Returns raw ``xgb.Booster`` objects which ``predict_xgb`` handles
    transparently alongside sklearn wrappers.

    Parameters
    ----------
    path : directory containing classifier.json, regressor.json, metadata.json.

    Returns
    -------
    dict compatible with ``predict_xgb``.
    """
    import xgboost as xgb

    meta = json.loads((path / "metadata.json").read_text())

    cls_booster = xgb.Booster()
    cls_booster.load_model(str(path / "classifier.json"))

    reg_booster = xgb.Booster()
    reg_booster.load_model(str(path / "regressor.json"))

    return {
        "classifier": cls_booster,
        "regressor": reg_booster,
        "feature_names": meta["feature_names"],
        "cls_params": meta.get("cls_params", {}),
        "reg_params": meta.get("reg_params", {}),
    }


# ===================================================================
# WALK-FORWARD VALIDATION
# ===================================================================

def _calibration_by_decile(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    """Compute predicted-vs-actual TP50 rate by probability decile.

    Parameters
    ----------
    y_true : binary ground truth (0/1).
    y_prob : predicted probabilities.

    Returns
    -------
    list of dicts with decile bounds, mean predicted prob, actual rate, count.
    """
    order = np.argsort(y_prob)
    y_true_s = y_true[order]
    y_prob_s = y_prob[order]

    n = len(y_prob_s)
    decile_size = max(n // 10, 1)
    rows = []
    for i in range(10):
        lo = i * decile_size
        hi = (i + 1) * decile_size if i < 9 else n
        if lo >= n:
            break
        chunk_true = y_true_s[lo:hi]
        chunk_prob = y_prob_s[lo:hi]
        rows.append({
            "decile": i + 1,
            "prob_lo": float(chunk_prob[0]),
            "prob_hi": float(chunk_prob[-1]),
            "mean_predicted": float(chunk_prob.mean()),
            "actual_rate": float(chunk_true.mean()),
            "count": int(hi - lo),
        })
    return rows


def walk_forward_validate_xgb(
    df: pd.DataFrame,
    train_ratio: float = 0.67,
) -> dict[str, Any]:
    """Walk-forward validation with XGBoost on the training CSV.

    Time-sorts by ``day``, splits at ``train_ratio``, trains on the earlier
    portion, and evaluates on the later portion.  Uses a 10 % holdout from
    the training split as the early-stopping validation set.

    Parameters
    ----------
    df          : raw training CSV DataFrame.
    train_ratio : fraction of rows (by time) used for training.

    Returns
    -------
    dict with train/test metadata, XGBoost metrics (Brier, MAE, calibration),
    and realized-outcome summary for comparison with the bucket model.
    """
    df_sorted = df.sort_values("day").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    if len(train_df) < 100 or len(test_df) < 50:
        return {"error": "Insufficient data for walk-forward split"}

    X_train_full, y_tp50_train_full, y_pnl_train_full = build_feature_matrix(train_df)
    X_test, y_tp50_test, y_pnl_test = build_feature_matrix(test_df)

    # Hold out 10% of training data for early stopping
    val_split = int(len(X_train_full) * 0.9)
    X_train = X_train_full.iloc[:val_split]
    y_tp50_train = y_tp50_train_full.iloc[:val_split]
    y_pnl_train = y_pnl_train_full.iloc[:val_split]
    X_val = X_train_full.iloc[val_split:]
    y_tp50_val = y_tp50_train_full.iloc[val_split:]
    y_pnl_val = y_pnl_train_full.iloc[val_split:]

    t0 = time.time()
    models = train_xgb_models(
        X_train, y_tp50_train, y_pnl_train,
        X_val, y_tp50_val, y_pnl_val,
    )
    train_time = time.time() - t0

    preds = predict_xgb(models, X_test)

    y_true_cls = y_tp50_test.values
    y_pred_prob = preds["prob_tp50"].values
    y_true_pnl = y_pnl_test.values
    y_pred_pnl = preds["predicted_pnl"].values

    brier = brier_score_loss(y_true_cls, y_pred_prob)
    mae = mean_absolute_error(y_true_pnl, y_pred_pnl)
    calibration = _calibration_by_decile(y_true_cls, y_pred_prob)

    # Realized outcome summary (for comparison with bucket model baseline)
    resolved_pnls = y_true_pnl.tolist()
    tp50_count = int(y_true_cls.sum())
    tp100_count = int(test_df["hit_tp100_at_expiry"].sum()) if "hit_tp100_at_expiry" in test_df.columns else 0
    total = len(resolved_pnls) or 1

    # Feature importance (top 15)
    clf = models["classifier"]
    feat_names = models["feature_names"]
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    feature_importance = [
        {"feature": feat_names[i], "importance": float(importances[i])}
        for i in top_idx
    ]

    return {
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_days": f"{train_df['day'].iloc[0]} .. {train_df['day'].iloc[-1]}",
        "test_days": f"{test_df['day'].iloc[0]} .. {test_df['day'].iloc[-1]}",
        "train_time_s": train_time,
        "models": models,
        "xgb_metrics": {
            "brier_score_tp50": brier,
            "mae_pnl": mae,
            "calibration_by_decile": calibration,
        },
        "test_summary": {
            "tp50_rate": tp50_count / total,
            "tp100_at_expiry_rate": tp100_count / total,
            "expectancy": sum(resolved_pnls) / total,
            "max_drawdown": _max_drawdown(resolved_pnls),
        },
        "feature_importance": feature_importance,
    }


def _max_drawdown(pnls: list[float]) -> float:
    """Compute max drawdown from a PnL series."""
    if not pnls:
        return 0.0
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    return float(dd.max())


# ===================================================================
# TRAIN FINAL MODEL (on all data)
# ===================================================================

def train_final_model(df: pd.DataFrame) -> dict[str, Any]:
    """Train XGBoost on the full dataset (no holdout).

    Uses a 90/10 split internally for early stopping only.

    Parameters
    ----------
    df : full training CSV DataFrame.

    Returns
    -------
    dict with trained models and metadata.
    """
    df_sorted = df.sort_values("day").reset_index(drop=True)
    X_full, y_tp50, y_pnl = build_feature_matrix(df_sorted)

    val_split = int(len(X_full) * 0.9)
    X_train = X_full.iloc[:val_split]
    y_tp50_train = y_tp50.iloc[:val_split]
    y_pnl_train = y_pnl.iloc[:val_split]
    X_val = X_full.iloc[val_split:]
    y_tp50_val = y_tp50.iloc[val_split:]
    y_pnl_val = y_pnl.iloc[val_split:]

    models = train_xgb_models(
        X_train, y_tp50_train, y_pnl_train,
        X_val, y_tp50_val, y_pnl_val,
    )
    return models


# ===================================================================
# ENTRY MODEL (rolling walk-forward, hold-based labels)
# ===================================================================

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


# ===================================================================
# ENTRY V2: loss-avoidance features & targets
# ===================================================================

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


def walk_forward_rolling(
    df: pd.DataFrame,
    train_months: int = 8,
    test_months: int = 2,
    step_months: int = 1,
    build_features_fn=None,
    cls_params: dict | None = None,
) -> dict[str, Any]:
    """Rolling walk-forward validation with overlapping windows.

    Sorts by ``day``, then slides a (train_months, test_months) window
    forward by step_months.  Aggregates metrics across all windows.

    Parameters
    ----------
    df            : Raw training CSV DataFrame.
    train_months  : Length of each training window in months.
    test_months   : Length of each test window in months.
    step_months   : How far to advance the window each step.
    build_features_fn : Feature builder (defaults to build_entry_feature_matrix).
    cls_params    : Override default classifier hyperparameters.

    Returns
    -------
    dict with per-window results and aggregated metrics.
    """
    from dateutil.relativedelta import relativedelta

    df_sorted = df.sort_values("day").reset_index(drop=True)
    df_sorted["_day_dt"] = pd.to_datetime(df_sorted["day"])

    first_day = df_sorted["_day_dt"].iloc[0]
    last_day = df_sorted["_day_dt"].iloc[-1]

    if build_features_fn is None:
        build_features_fn = build_entry_feature_matrix

    windows: list[dict[str, Any]] = []
    all_test_preds: list[pd.DataFrame] = []
    all_test_labels: list[pd.Series] = []
    all_test_pnls: list[pd.Series] = []
    win_idx = 0

    train_start = first_day
    while True:
        train_end = train_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)
        if train_end > last_day:
            break

        train_mask = (df_sorted["_day_dt"] >= train_start) & (df_sorted["_day_dt"] < train_end)
        test_mask = (df_sorted["_day_dt"] >= train_end) & (df_sorted["_day_dt"] < test_end)
        train_df = df_sorted[train_mask]
        test_df = df_sorted[test_mask]

        if len(train_df) < 100 or len(test_df) < 30:
            train_start += relativedelta(months=step_months)
            continue

        X_train_full, y_cls_train_full, y_pnl_train_full = build_features_fn(train_df)
        X_test, y_cls_test, y_pnl_test = build_features_fn(test_df)

        val_split = int(len(X_train_full) * 0.9)
        X_train = X_train_full.iloc[:val_split]
        y_cls_train = y_cls_train_full.iloc[:val_split]
        y_pnl_train = y_pnl_train_full.iloc[:val_split]
        X_val = X_train_full.iloc[val_split:]
        y_cls_val = y_cls_train_full.iloc[val_split:]
        y_pnl_val = y_pnl_train_full.iloc[val_split:]

        t0 = time.time()
        models = train_xgb_models(
            X_train, y_cls_train, y_pnl_train,
            X_val, y_cls_val, y_pnl_val,
            cls_params=cls_params,
        )
        elapsed = time.time() - t0

        preds = predict_xgb(models, X_test)
        y_pred_prob = preds["prob_tp50"].values
        y_pred_pnl = preds["predicted_pnl"].values

        brier = brier_score_loss(y_cls_test.values, y_pred_prob)
        mae = mean_absolute_error(y_pnl_test.values, y_pred_pnl)

        clf = models["classifier"]
        feat_names = models["feature_names"]
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        fi = [{"feature": feat_names[i], "importance": float(importances[i])} for i in top_idx]

        tp50_count = int(y_cls_test.sum())
        total = len(y_cls_test)
        resolved_pnls = y_pnl_test.values.tolist()

        win_result = {
            "window": win_idx,
            "train_count": len(train_df),
            "test_count": len(test_df),
            "train_days": f"{train_df['day'].iloc[0]} .. {train_df['day'].iloc[-1]}",
            "test_days": f"{test_df['day'].iloc[0]} .. {test_df['day'].iloc[-1]}",
            "train_time_s": elapsed,
            "brier": brier,
            "mae": mae,
            "tp50_rate": tp50_count / max(total, 1),
            "expectancy": sum(resolved_pnls) / max(total, 1),
            "max_drawdown": _max_drawdown(resolved_pnls),
            "feature_importance": fi,
        }
        windows.append(win_result)
        all_test_preds.append(preds)
        all_test_labels.append(y_cls_test)
        all_test_pnls.append(y_pnl_test)
        win_idx += 1
        train_start += relativedelta(months=step_months)

    if not windows:
        return {"error": "No valid walk-forward windows (insufficient data)"}

    pooled_probs = np.concatenate([p["prob_tp50"].values for p in all_test_preds])
    pooled_labels = np.concatenate([l.values for l in all_test_labels])
    pooled_pnls = np.concatenate([p.values for p in all_test_pnls])

    agg_brier = brier_score_loss(pooled_labels, pooled_probs)
    agg_mae = float(np.mean(np.abs(pooled_pnls - np.concatenate([p["predicted_pnl"].values for p in all_test_preds]))))
    agg_calibration = _calibration_by_decile(pooled_labels, pooled_probs)

    # Threshold tuning on pooled test predictions.
    # For V1 (TP50 classifier): take when prob >= thr (higher = better).
    # For V2 (loss classifier): take when prob < thr (lower = safer).
    # Detect direction from mean label: if > 0.5, classifier predicts "good" (V1 style).
    label_mean = float(pooled_labels.mean())
    higher_is_better = label_mean > 0.5

    if higher_is_better:
        thr_range = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    else:
        thr_range = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

    threshold_results = []
    for thr in thr_range:
        mask = pooled_probs >= thr if higher_is_better else pooled_probs < thr
        n_taken = int(mask.sum())
        if n_taken == 0:
            threshold_results.append({"threshold": thr, "trades_taken": 0, "win_rate": 0.0,
                                      "avg_pnl": 0.0, "total_pnl": 0.0})
            continue
        win_rate = float((pooled_pnls[mask] > 0).mean())
        avg_pnl = float(pooled_pnls[mask].mean())
        total_pnl = float(pooled_pnls[mask].sum())
        threshold_results.append({"threshold": thr, "trades_taken": n_taken, "win_rate": win_rate,
                                  "avg_pnl": avg_pnl, "total_pnl": total_pnl})

    best_thr = max(threshold_results, key=lambda x: x["total_pnl"])

    return {
        "windows": windows,
        "n_windows": len(windows),
        "pooled_test_count": len(pooled_labels),
        "aggregated_metrics": {
            "brier_score": agg_brier,
            "mae_pnl": agg_mae,
            "tp50_rate": float(pooled_labels.mean()),
            "expectancy": float(pooled_pnls.mean()),
            "max_drawdown": _max_drawdown(pooled_pnls.tolist()),
            "calibration_by_decile": agg_calibration,
        },
        "threshold_tuning": threshold_results,
        "recommended_threshold": best_thr["threshold"],
    }


def _extract_entry_rules(
    test_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Derive human-readable ENTER/SKIP rules from dimensional slices.

    For each slice, computes hold-label TP50 rate and avg PnL, then
    compares to the overall baseline.

    Parameters
    ----------
    test_df : Test-set rows (must include hold label columns and features).

    Returns
    -------
    List of rule dicts sorted by avg_pnl descending.
    """
    cls_col, reg_col = _resolve_targets(test_df)
    if cls_col not in test_df.columns or reg_col not in test_df.columns:
        return []

    baseline_tp50 = test_df[cls_col].mean()
    baseline_pnl = pd.to_numeric(test_df[reg_col], errors="coerce").mean()
    rules: list[dict[str, Any]] = []

    def _add(condition: str, sub: pd.DataFrame) -> None:
        if len(sub) < 5:
            return
        tp50_rate = float(sub[cls_col].mean())
        avg_pnl = float(pd.to_numeric(sub[reg_col], errors="coerce").mean())
        action = "ENTER" if avg_pnl > baseline_pnl else "SKIP"
        rules.append({
            "condition": condition,
            "action": action,
            "tp50_rate": tp50_rate,
            "avg_pnl": avg_pnl,
            "lift_vs_baseline": tp50_rate - baseline_tp50,
            "pnl_vs_baseline": avg_pnl - baseline_pnl,
            "n_trades": len(sub),
        })

    # DTE
    for dte in sorted(test_df["dte_target"].dropna().unique()):
        _add(f"dte={int(dte)}", test_df[test_df["dte_target"] == dte])

    # Spread side
    if "spread_side" in test_df.columns:
        for side in test_df["spread_side"].dropna().unique():
            _add(f"side={side}", test_df[test_df["spread_side"] == side])

    # Delta
    for delta in sorted(test_df["delta_target"].dropna().unique()):
        _add(f"delta={delta:.2f}", test_df[test_df["delta_target"] == delta])

    # VIX buckets
    if "vix" in test_df.columns:
        vix = pd.to_numeric(test_df["vix"], errors="coerce")
        for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 100)]:
            mask = (vix >= lo) & (vix < hi)
            _add(f"vix=[{lo},{hi})", test_df[mask])

    # Entry hour (ET)
    if "entry_hour" not in test_df.columns and "entry_dt" in test_df.columns:
        dt = pd.to_datetime(test_df["entry_dt"], utc=True, errors="coerce")
        test_df = test_df.copy()
        test_df["entry_hour"] = dt.dt.tz_convert("America/New_York").dt.hour
    if "entry_hour" in test_df.columns:
        for hr in sorted(test_df["entry_hour"].dropna().unique()):
            _add(f"hour_et={int(hr)}", test_df[test_df["entry_hour"] == hr])

    # Calendar flags
    for flag in BINARY_FEATURES:
        if flag in test_df.columns:
            mask = test_df[flag].astype(int) == 1
            if mask.sum() >= 5:
                _add(f"{flag}=True", test_df[mask])
                _add(f"{flag}=False", test_df[~mask])

    # GEX sign
    if "offline_gex_net" in test_df.columns:
        gex = pd.to_numeric(test_df["offline_gex_net"], errors="coerce")
        _add("gex>0", test_df[gex > 0])
        _add("gex<=0", test_df[gex <= 0])

    rules.sort(key=lambda r: r["avg_pnl"], reverse=True)
    return rules


# ===================================================================
# HOLD-VS-CLOSE MODEL (requires trajectory columns from --relabel)
# ===================================================================

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


def walk_forward_hold_vs_close(
    df: pd.DataFrame,
    train_ratio: float = 0.67,
) -> dict[str, Any]:
    """Walk-forward validation for the hold-vs-close decision model.

    Trains three models:
    1. SL risk classifier (on all trades)
    2. Hold-is-better classifier (on SL trades only)
    3. Hold PnL regressor (on all trades with hold_realized_pnl)

    Then evaluates on the test period and computes the strategy improvement
    from the model-driven hold/close decision.

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced CSV with trajectory columns.
    train_ratio : float
        Fraction of data (by time) for training.

    Returns
    -------
    dict
        Metrics, feature importances, and strategy comparison.
    """
    if "hold_realized_pnl" not in df.columns:
        return {"error": "CSV missing trajectory columns. Run --relabel first."}

    df_sorted = df.sort_values("day").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    if len(train_df) < 100 or len(test_df) < 50:
        return {"error": "Insufficient data for walk-forward split"}

    X_train_full, y_sl_train_full, y_hb_train_full, y_hpnl_train_full = (
        build_hold_vs_close_targets(train_df)
    )
    X_test, y_sl_test, y_hb_test, y_hpnl_test = (
        build_hold_vs_close_targets(test_df)
    )

    val_split = int(len(X_train_full) * 0.9)

    t0 = time.time()

    # --- 1. SL risk classifier (all trades) ---
    from xgboost.callback import EarlyStopping

    sl_clf = XGBClassifier(**{
        k: v for k, v in DEFAULT_CLS_PARAMS.items()
        if k != "early_stopping_rounds"
    })
    sl_clf.set_params(callbacks=[EarlyStopping(rounds=30, save_best=True)])
    sl_clf.fit(
        X_train_full.iloc[:val_split], y_sl_train_full.iloc[:val_split],
        eval_set=[(X_train_full.iloc[val_split:], y_sl_train_full.iloc[val_split:])],
        verbose=False,
    )

    # --- 2. Hold-is-better classifier (SL trades only) ---
    sl_train_mask = (train_df["exit_reason"] == "STOP_LOSS") & y_hb_train_full.notna()
    hb_clf = None
    hb_importance = []
    if sl_train_mask.sum() >= 50:
        hb_clf = XGBClassifier(**{
            k: v for k, v in DEFAULT_CLS_PARAMS.items()
            if k != "early_stopping_rounds"
        })
        hb_split = int(sl_train_mask.sum() * 0.85)
        sl_idx = sl_train_mask[sl_train_mask].index
        hb_clf.fit(
            X_train_full.loc[sl_idx[:hb_split]],
            y_hb_train_full.loc[sl_idx[:hb_split]].astype(int),
            eval_set=[(
                X_train_full.loc[sl_idx[hb_split:]],
                y_hb_train_full.loc[sl_idx[hb_split:]].astype(int),
            )] if len(sl_idx) > hb_split else None,
            verbose=False,
        )

        feat_names = list(X_train_full.columns)
        imps = hb_clf.feature_importances_
        top_idx = np.argsort(imps)[::-1][:15]
        hb_importance = [
            {"feature": feat_names[i], "importance": float(imps[i])}
            for i in top_idx
        ]

    train_time = time.time() - t0

    # --- Evaluate on test set ---
    sl_probs = sl_clf.predict_proba(X_test)[:, 1]

    # SL risk metrics
    sl_brier = brier_score_loss(y_sl_test, sl_probs)

    # Feature importance for SL classifier
    feat_names = list(X_test.columns)
    sl_imps = sl_clf.feature_importances_
    sl_top = np.argsort(sl_imps)[::-1][:15]
    sl_importance = [
        {"feature": feat_names[i], "importance": float(sl_imps[i])}
        for i in sl_top
    ]

    # Hold-vs-close strategy evaluation on test SL trades
    test_sl_mask = test_df["exit_reason"] == "STOP_LOSS"
    strategy_result = {}
    if test_sl_mask.any():
        sl_test_orig_pnl = test_df.loc[test_sl_mask, "realized_pnl"].sum()
        sl_test_hold_pnl = pd.to_numeric(
            test_df.loc[test_sl_mask, "hold_realized_pnl"], errors="coerce"
        ).sum()

        strategy_result = {
            "test_sl_trades": int(test_sl_mask.sum()),
            "close_at_sl_total_pnl": float(sl_test_orig_pnl),
            "hold_through_total_pnl": float(sl_test_hold_pnl),
            "improvement": float(sl_test_hold_pnl - sl_test_orig_pnl),
        }

        # Model-driven strategy: hold if model says P(hold_is_better) > 0.5
        if hb_clf is not None:
            test_sl_idx = test_sl_mask[test_sl_mask].index
            hb_probs = hb_clf.predict_proba(X_test.loc[test_sl_idx])[:, 1]
            hold_mask_model = hb_probs > 0.5

            model_pnl = 0.0
            for i, idx in enumerate(test_sl_idx):
                if hold_mask_model[i]:
                    model_pnl += float(
                        pd.to_numeric(test_df.at[idx, "hold_realized_pnl"], errors="coerce") or 0
                    )
                else:
                    model_pnl += float(test_df.at[idx, "realized_pnl"] or 0)

            strategy_result["model_driven_total_pnl"] = model_pnl
            strategy_result["model_hold_count"] = int(hold_mask_model.sum())
            strategy_result["model_close_count"] = int((~hold_mask_model).sum())

    # --- Extract human-readable rules ---
    rules = _extract_rules(train_df, test_df)

    return {
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_days": f"{train_df['day'].iloc[0]} .. {train_df['day'].iloc[-1]}",
        "test_days": f"{test_df['day'].iloc[0]} .. {test_df['day'].iloc[-1]}",
        "train_time_s": train_time,
        "sl_risk": {
            "brier_score": sl_brier,
            "sl_rate_actual": float(y_sl_test.mean()),
            "feature_importance": sl_importance,
        },
        "hold_vs_close": {
            "feature_importance": hb_importance,
            "strategy_comparison": strategy_result,
        },
        "rules": rules,
    }


def _extract_rules(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict]:
    """Extract human-readable trading rules by slicing test data.

    Computes recovery rate and PnL improvement for each dimension/bucket
    combination to produce rules like "For call spreads DTE>=5 with VIX<20,
    HOLD through SL (recovery 65%)".

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (for calibrating thresholds).
    test_df : pd.DataFrame
        Test data (for evaluating rules out-of-sample).

    Returns
    -------
    list[dict]
        Each dict has ``condition``, ``action``, ``recovery_rate``,
        ``avg_improvement``, ``n_trades``.
    """
    rules = []
    sl_test = test_df[test_df["exit_reason"] == "STOP_LOSS"].copy()
    if sl_test.empty or "recovered_after_sl" not in sl_test.columns:
        return rules

    dimensions = [
        ("dte_target", sorted(sl_test["dte_target"].unique())),
        ("spread_side", sorted(sl_test["spread_side"].unique())),
        ("delta_target", sorted(sl_test["delta_target"].unique())),
    ]

    for col, values in dimensions:
        for val in values:
            sub = sl_test[sl_test[col] == val]
            if len(sub) < 5:
                continue
            rec_rate = (sub["recovered_after_sl"] == True).mean()  # noqa: E712
            hold_pnl = pd.to_numeric(sub["hold_realized_pnl"], errors="coerce").mean()
            sl_pnl = sub["realized_pnl"].mean()
            improvement = hold_pnl - sl_pnl
            action = "HOLD" if improvement > 0 else "CLOSE at SL"
            rules.append({
                "condition": f"{col} = {val}",
                "action": action,
                "recovery_rate": float(rec_rate),
                "avg_improvement": float(improvement),
                "n_trades": len(sub),
            })

    # VIX buckets
    if "vix" in sl_test.columns:
        vix_cuts = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 100)]
        for lo, hi in vix_cuts:
            sub = sl_test[(sl_test["vix"] >= lo) & (sl_test["vix"] < hi)]
            if len(sub) < 5:
                continue
            rec_rate = (sub["recovered_after_sl"] == True).mean()  # noqa: E712
            hold_pnl = pd.to_numeric(sub["hold_realized_pnl"], errors="coerce").mean()
            sl_pnl = sub["realized_pnl"].mean()
            improvement = hold_pnl - sl_pnl
            action = "HOLD" if improvement > 0 else "CLOSE at SL"
            rules.append({
                "condition": f"VIX in [{lo}, {hi})",
                "action": action,
                "recovery_rate": float(rec_rate),
                "avg_improvement": float(improvement),
                "n_trades": len(sub),
            })

    rules.sort(key=lambda r: r["avg_improvement"], reverse=True)
    return rules


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main() -> None:
    """Train and evaluate XGBoost on the existing training_candidates.csv."""
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost model training & evaluation")
    parser.add_argument(
        "--csv", type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "training_candidates.csv"),
        help="Path to training_candidates.csv",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save trained model (default: data/xgb_model/)",
    )
    parser.add_argument(
        "--mode", type=str, default="tp50",
        choices=["tp50", "hold-vs-close", "entry", "entry-v2"],
        help="'tp50' trains TP50 + PnL models (default). "
             "'hold-vs-close' trains SL risk + hold decision models. "
             "'entry' trains entry classifier with rolling walk-forward. "
             "'entry-v2' loss-avoidance model with extra features and "
             "asymmetric weighting (requires --relabel trajectory data).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    print(f"[XGB] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[XGB] {len(df)} rows, {len(df.columns)} columns")

    if args.mode == "hold-vs-close":
        _run_hold_vs_close(df)
        return
    if args.mode == "entry":
        _run_entry(df, csv_path, args.save_dir)
        return
    if args.mode == "entry-v2":
        _run_entry_v2(df, csv_path, args.save_dir)
        return

    _run_tp50(df, csv_path, args.save_dir)


def _run_tp50(df: pd.DataFrame, csv_path: Path, save_dir_arg: str | None) -> None:
    """Original TP50 + PnL model training path."""
    print("[XGB] Walk-forward validation ...")
    results = walk_forward_validate_xgb(df)

    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return

    xm = results["xgb_metrics"]
    ts = results["test_summary"]
    fi = results["feature_importance"]

    print(f"\n{'=' * 60}")
    print("XGBOOST WALK-FORWARD RESULTS")
    print(f"{'=' * 60}")
    print(f"  Train     : {results['train_count']} rows ({results['train_days']})")
    print(f"  Test      : {results['test_count']} rows ({results['test_days']})")
    print(f"  Train time: {results['train_time_s']:.1f}s")
    print(f"\n  --- XGBoost Metrics ---")
    print(f"  Brier score (TP50) : {xm['brier_score_tp50']:.4f}")
    print(f"  MAE (PnL)          : ${xm['mae_pnl']:.2f}")
    print(f"\n  --- Realized Outcomes (test set) ---")
    print(f"  TP50 rate          : {ts['tp50_rate']:.1%}")
    print(f"  TP100 rate         : {ts['tp100_at_expiry_rate']:.1%}")
    print(f"  Expectancy         : ${ts['expectancy']:.2f}")
    print(f"  Max drawdown       : ${ts['max_drawdown']:.2f}")

    print(f"\n  --- Calibration (predicted vs actual TP50 by decile) ---")
    print(f"  {'Decile':>6} {'Pred Lo':>8} {'Pred Hi':>8} {'Mean Pred':>10} {'Actual':>8} {'Count':>6}")
    for row in xm["calibration_by_decile"]:
        print(f"  {row['decile']:>6d} {row['prob_lo']:>8.3f} {row['prob_hi']:>8.3f} "
              f"{row['mean_predicted']:>10.3f} {row['actual_rate']:>8.3f} {row['count']:>6d}")

    print(f"\n  --- Top Feature Importances ---")
    for i, f in enumerate(fi[:15], 1):
        print(f"  {i:>2}. {f['feature']:<25s} {f['importance']:.4f}")

    print(f"{'=' * 60}")

    print("\n[XGB] Training final model on all data ...")
    final = train_final_model(df)

    save_dir = Path(save_dir_arg) if save_dir_arg else csv_path.parent / "xgb_model"
    save_model(final, save_dir)
    print(f"[XGB] Model saved to {save_dir}/")
    print("[XGB] Done.")


def _run_hold_vs_close(df: pd.DataFrame) -> None:
    """Hold-vs-close model training and rule extraction."""
    print("[XGB] Hold-vs-close walk-forward validation ...")
    results = walk_forward_hold_vs_close(df)

    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return

    slr = results["sl_risk"]
    hvc = results["hold_vs_close"]
    strat = hvc.get("strategy_comparison", {})
    rules = results.get("rules", [])

    print(f"\n{'=' * 60}")
    print("HOLD-VS-CLOSE WALK-FORWARD RESULTS")
    print(f"{'=' * 60}")
    print(f"  Train : {results['train_count']} rows ({results['train_days']})")
    print(f"  Test  : {results['test_count']} rows ({results['test_days']})")
    print(f"  Time  : {results['train_time_s']:.1f}s")

    print(f"\n  --- SL Risk Classifier ---")
    print(f"  Actual SL rate (test) : {slr['sl_rate_actual']:.1%}")
    print(f"  Brier score           : {slr['brier_score']:.4f}")
    print(f"  Top features:")
    for i, f in enumerate(slr["feature_importance"][:10], 1):
        print(f"    {i:>2}. {f['feature']:<25s} {f['importance']:.4f}")

    if hvc["feature_importance"]:
        print(f"\n  --- Hold-is-Better Classifier (SL trades) ---")
        print(f"  Top features:")
        for i, f in enumerate(hvc["feature_importance"][:10], 1):
            print(f"    {i:>2}. {f['feature']:<25s} {f['importance']:.4f}")

    if strat:
        print(f"\n  --- Strategy Comparison (test SL trades: {strat['test_sl_trades']}) ---")
        print(f"  Close at SL (current):  ${strat['close_at_sl_total_pnl']:>10,.0f}")
        print(f"  Hold through (always):  ${strat['hold_through_total_pnl']:>10,.0f}")
        if "model_driven_total_pnl" in strat:
            print(f"  Model-driven hold:      ${strat['model_driven_total_pnl']:>10,.0f} "
                  f"(hold {strat['model_hold_count']}, close {strat['model_close_count']})")
        print(f"  Hold improvement:       ${strat['improvement']:>+10,.0f}")

    if rules:
        print(f"\n  --- Human-Readable Rules (sorted by improvement) ---")
        for r in rules[:20]:
            print(f"  {r['condition']:<30s} → {r['action']:<15s} "
                  f"recovery {r['recovery_rate']:.0%}, "
                  f"${r['avg_improvement']:>+7,.0f}/trade, "
                  f"n={r['n_trades']}")

    print(f"\n{'=' * 60}")
    print("[XGB] Done.")


def _run_entry(df: pd.DataFrame, csv_path: Path, save_dir_arg: str | None) -> None:
    """Entry model: rolling walk-forward with hold labels + threshold tuning."""
    cls_col, reg_col = _resolve_targets(df)
    print(f"[ENTRY] Labels: cls={cls_col}, reg={reg_col}")
    print("[ENTRY] Rolling walk-forward validation (8-month train / 2-month test / 1-month step) ...")

    results = walk_forward_rolling(df)
    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return

    am = results["aggregated_metrics"]
    windows = results["windows"]
    thr_results = results["threshold_tuning"]

    print(f"\n{'=' * 70}")
    print("ENTRY MODEL — ROLLING WALK-FORWARD RESULTS")
    print(f"{'=' * 70}")
    print(f"  Windows      : {results['n_windows']}")
    print(f"  Pooled test  : {results['pooled_test_count']} rows")

    for w in windows:
        print(f"\n  Window {w['window']}: train={w['train_count']} ({w['train_days']}), "
              f"test={w['test_count']} ({w['test_days']})")
        print(f"    Brier={w['brier']:.4f}  MAE=${w['mae']:.2f}  "
              f"TP50={w['tp50_rate']:.1%}  E[PnL]=${w['expectancy']:.2f}  "
              f"MaxDD=${w['max_drawdown']:.0f}  time={w['train_time_s']:.1f}s")

    print(f"\n  --- Aggregated Metrics (pooled across all windows) ---")
    print(f"  Brier score   : {am['brier_score']:.4f}")
    print(f"  MAE (PnL)     : ${am['mae_pnl']:.2f}")
    print(f"  TP50 rate     : {am['tp50_rate']:.1%}")
    print(f"  Expectancy    : ${am['expectancy']:.2f}")
    print(f"  Max drawdown  : ${am['max_drawdown']:.0f}")

    print(f"\n  --- Calibration (predicted vs actual TP50 by decile) ---")
    print(f"  {'Decile':>6} {'Pred Lo':>8} {'Pred Hi':>8} {'Mean Pred':>10} {'Actual':>8} {'Count':>6}")
    for row in am["calibration_by_decile"]:
        print(f"  {row['decile']:>6d} {row['prob_lo']:>8.3f} {row['prob_hi']:>8.3f} "
              f"{row['mean_predicted']:>10.3f} {row['actual_rate']:>8.3f} {row['count']:>6d}")

    print(f"\n  --- Threshold Tuning (pooled test set) ---")
    print(f"  {'Threshold':>10} {'Trades':>8} {'Win Rate':>10} {'Avg PnL':>10} {'Total PnL':>12}")
    for t in thr_results:
        marker = " <-- best" if t["threshold"] == results["recommended_threshold"] else ""
        print(f"  {t['threshold']:>10.2f} {t['trades_taken']:>8d} {t['win_rate']:>10.1%} "
              f"${t['avg_pnl']:>9.2f} ${t['total_pnl']:>11,.0f}{marker}")
    print(f"\n  Recommended decision_hybrid_min_probability = {results['recommended_threshold']:.2f}")

    # Feature importances (average across windows)
    all_fi: dict[str, list[float]] = {}
    for w in windows:
        for f in w["feature_importance"]:
            all_fi.setdefault(f["feature"], []).append(f["importance"])
    avg_fi = sorted(
        [{"feature": k, "importance": float(np.mean(v))} for k, v in all_fi.items()],
        key=lambda x: x["importance"], reverse=True,
    )
    print(f"\n  --- Top Feature Importances (avg across windows) ---")
    for i, f in enumerate(avg_fi[:15], 1):
        print(f"  {i:>2}. {f['feature']:<25s} {f['importance']:.4f}")

    # Entry rules from the last window's test data
    df_sorted = df.sort_values("day").reset_index(drop=True)
    rules = _extract_entry_rules(df_sorted.tail(int(len(df_sorted) * 0.25)))
    if rules:
        print(f"\n  --- Entry Rules (sorted by avg PnL) ---")
        baseline_pnl = pd.to_numeric(df_sorted.tail(int(len(df_sorted) * 0.25))[_resolve_targets(df_sorted)[1]],
                                      errors="coerce").mean()
        print(f"  Baseline avg PnL: ${baseline_pnl:.2f}")
        for r in rules[:25]:
            print(f"  {r['condition']:<30s} → {r['action']:<6s} "
                  f"TP50={r['tp50_rate']:.0%}, "
                  f"${r['avg_pnl']:>+8,.0f}/trade, "
                  f"lift={r['lift_vs_baseline']:>+.1%}, "
                  f"n={r['n_trades']}")

    print(f"\n{'=' * 70}")

    # Train final model on all data and save
    print("\n[ENTRY] Training final model on all data ...")
    df_sorted_all = df.sort_values("day").reset_index(drop=True)
    X_full, y_cls, y_pnl = build_entry_feature_matrix(df_sorted_all)
    val_split = int(len(X_full) * 0.9)
    models = train_xgb_models(
        X_full.iloc[:val_split], y_cls.iloc[:val_split], y_pnl.iloc[:val_split],
        X_full.iloc[val_split:], y_cls.iloc[val_split:], y_pnl.iloc[val_split:],
    )
    save_dir = Path(save_dir_arg) if save_dir_arg else csv_path.parent / "xgb_entry_model"
    save_model(models, save_dir)
    print(f"[ENTRY] Model saved to {save_dir}/")
    print("[ENTRY] Done.")


def _run_entry_v2(df: pd.DataFrame, csv_path: Path, save_dir_arg: str | None) -> None:
    """Loss-avoidance entry model: predict P(big_loss) with asymmetric weighting."""
    reg_col = "hold_realized_pnl" if "hold_realized_pnl" in df.columns else "realized_pnl"
    pnl = pd.to_numeric(df[reg_col], errors="coerce")
    big_loss_rate = (pnl < BIG_LOSS_THRESHOLD).mean()
    print(f"[ENTRY-V2] Big loss (< ${BIG_LOSS_THRESHOLD:.0f}) rate: {big_loss_rate:.1%}")

    # Asymmetric weighting: missed loser costs ~$592, skipped winner costs ~$78
    # scale_pos_weight = cost_of_FN / cost_of_FP ≈ 592/78 ≈ 7.6
    pos_weight = 7.0
    v2_cls_params = {
        **DEFAULT_CLS_PARAMS,
        "scale_pos_weight": pos_weight,
    }
    print(f"[ENTRY-V2] scale_pos_weight={pos_weight:.1f}")
    print("[ENTRY-V2] Rolling walk-forward (8/2/1 months) ...")

    results = walk_forward_rolling(
        df,
        build_features_fn=build_entry_v2_feature_matrix,
        cls_params=v2_cls_params,
    )
    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return

    am = results["aggregated_metrics"]
    windows = results["windows"]

    print(f"\n{'=' * 70}")
    print("ENTRY-V2 — LOSS-AVOIDANCE MODEL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Windows      : {results['n_windows']}")
    print(f"  Pooled test  : {results['pooled_test_count']} rows")

    for w in windows:
        print(f"\n  Window {w['window']}: train={w['train_count']} ({w['train_days']}), "
              f"test={w['test_count']} ({w['test_days']})")
        print(f"    Brier={w['brier']:.4f}  MAE=${w['mae']:.2f}  "
              f"BigLoss={w['tp50_rate']:.1%}  E[PnL]=${w['expectancy']:.2f}  "
              f"MaxDD=${w['max_drawdown']:.0f}  time={w['train_time_s']:.1f}s")

    print(f"\n  --- Aggregated Metrics ---")
    print(f"  Brier score    : {am['brier_score']:.4f}")
    print(f"  MAE (PnL)      : ${am['mae_pnl']:.2f}")
    print(f"  Big-loss rate  : {am['tp50_rate']:.1%}")
    print(f"  Expectancy     : ${am['expectancy']:.2f}")
    print(f"  Max drawdown   : ${am['max_drawdown']:.0f}")

    # For V2, threshold tuning means: SKIP trade when P(big_loss) > threshold
    # Lower threshold = more aggressive filtering (skip more)
    pooled_probs = np.concatenate([
        np.array(results["_all_test_probs"]) if "_all_test_probs" in results else []
    ]) if "_all_test_probs" in results else None

    # Re-derive pooled predictions from threshold_tuning for the report
    thr_results = results["threshold_tuning"]
    print(f"\n  --- Loss-Avoidance Threshold Tuning ---")
    print(f"  (Classifier predicts P(big_loss); SKIP when P > threshold)")
    print(f"  {'Threshold':>10} {'Trades':>8} {'Win Rate':>10} {'Avg PnL':>10} {'Total PnL':>12}")
    for t in thr_results:
        marker = " <-- best" if t["threshold"] == results["recommended_threshold"] else ""
        print(f"  {t['threshold']:>10.2f} {t['trades_taken']:>8d} {t['win_rate']:>10.1%} "
              f"${t['avg_pnl']:>9.2f} ${t['total_pnl']:>11,.0f}{marker}")

    # Feature importances
    all_fi: dict[str, list[float]] = {}
    for w in windows:
        for f in w["feature_importance"]:
            all_fi.setdefault(f["feature"], []).append(f["importance"])
    avg_fi = sorted(
        [{"feature": k, "importance": float(np.mean(v))} for k, v in all_fi.items()],
        key=lambda x: x["importance"], reverse=True,
    )
    print(f"\n  --- Top Feature Importances (loss-avoidance model) ---")
    for i, f in enumerate(avg_fi[:20], 1):
        print(f"  {i:>2}. {f['feature']:<25s} {f['importance']:.4f}")

    print(f"\n{'=' * 70}")

    # Train final model and save
    print("\n[ENTRY-V2] Training final model on all data ...")
    df_sorted = df.sort_values("day").reset_index(drop=True)
    X_full, y_cls, y_pnl = build_entry_v2_feature_matrix(df_sorted)
    val_split = int(len(X_full) * 0.9)
    models = train_xgb_models(
        X_full.iloc[:val_split], y_cls.iloc[:val_split], y_pnl.iloc[:val_split],
        X_full.iloc[val_split:], y_cls.iloc[val_split:], y_pnl.iloc[val_split:],
        cls_params=v2_cls_params,
    )
    save_dir = Path(save_dir_arg) if save_dir_arg else csv_path.parent / "xgb_entry_v2_model"
    save_model(models, save_dir)
    print(f"[ENTRY-V2] Model saved to {save_dir}/")
    print("[ENTRY-V2] Done.")


if __name__ == "__main__":
    main()
