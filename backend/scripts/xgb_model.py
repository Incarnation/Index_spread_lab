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

ORDINAL_FEATURES = ["dte_target"]

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

    y_tp50 = df[TARGET_CLS].astype(int) if TARGET_CLS in df.columns else pd.Series(0, index=df.index)
    y_pnl = pd.to_numeric(df[TARGET_REG], errors="coerce") if TARGET_REG in df.columns else pd.Series(0.0, index=df.index)

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
        choices=["tp50", "hold-vs-close"],
        help="'tp50' trains TP50 + PnL models (default). "
             "'hold-vs-close' trains SL risk + hold decision models "
             "(requires --relabel trajectory data).",
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


if __name__ == "__main__":
    main()
