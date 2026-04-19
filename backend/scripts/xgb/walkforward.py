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

from .features import (
    BINARY_FEATURES,
    build_entry_feature_matrix,
    build_feature_matrix,
    build_hold_vs_close_targets,
    _resolve_targets,
)
from .training import (
    DEFAULT_CLS_PARAMS,
    predict_xgb,
    train_xgb_models,
)


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
    # Pool VAL predictions across windows; threshold tuning operates on
    # val data only so the chosen threshold is independent of any test
    # row.  See C3 in OFFLINE_PIPELINE_AUDIT.md -- the prior implementation
    # picked the threshold that maximised pooled OOS test PnL, which is a
    # textbook test-set selection bias.
    all_val_preds: list[pd.DataFrame] = []
    all_val_labels: list[pd.Series] = []
    all_val_pnls: list[pd.Series] = []
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

        # Predict on both val (for threshold tuning) and test (for OOS metrics).
        val_preds = predict_xgb(models, X_val)
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
            "val_count": len(X_val),
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
        all_val_preds.append(val_preds)
        all_val_labels.append(y_cls_val)
        all_val_pnls.append(y_pnl_val)
        win_idx += 1
        train_start += relativedelta(months=step_months)

    if not windows:
        return {"error": "No valid walk-forward windows (insufficient data)"}

    pooled_probs = np.concatenate([p["prob_tp50"].values for p in all_test_preds])
    pooled_labels = np.concatenate([l.values for l in all_test_labels])
    pooled_pnls = np.concatenate([p.values for p in all_test_pnls])

    pooled_val_probs = np.concatenate([p["prob_tp50"].values for p in all_val_preds])
    pooled_val_labels = np.concatenate([l.values for l in all_val_labels])
    pooled_val_pnls = np.concatenate([p.values for p in all_val_pnls])

    agg_brier = brier_score_loss(pooled_labels, pooled_probs)
    agg_mae = float(np.mean(np.abs(pooled_pnls - np.concatenate([p["predicted_pnl"].values for p in all_test_preds]))))
    agg_calibration = _calibration_by_decile(pooled_labels, pooled_probs)

    # Threshold tuning runs on pooled VAL predictions only.
    # For V1 (TP50 classifier): take when prob >= thr (higher = better).
    # For V2 (loss classifier): take when prob < thr (lower = safer).
    # Detect direction from val labels (not test) so that the entire
    # threshold-selection pipeline is a function of train/val data only.
    label_mean = float(pooled_val_labels.mean()) if len(pooled_val_labels) else 0.5
    higher_is_better = label_mean > 0.5

    if higher_is_better:
        thr_range = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    else:
        thr_range = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

    threshold_results: list[dict[str, Any]] = []
    for thr in thr_range:
        # Per-threshold val statistics drive selection (NO test data here).
        val_mask = pooled_val_probs >= thr if higher_is_better else pooled_val_probs < thr
        n_val_taken = int(val_mask.sum())
        if n_val_taken == 0:
            val_total_pnl = 0.0
            val_avg_pnl = 0.0
            val_win_rate = 0.0
        else:
            val_pnl_subset = pooled_val_pnls[val_mask]
            val_total_pnl = float(val_pnl_subset.sum())
            val_avg_pnl = float(val_pnl_subset.mean())
            val_win_rate = float((val_pnl_subset > 0).mean())

        # Diagnostic test-side counters at the same threshold so operators
        # can see how the locked threshold performs OOS.  These are NOT
        # used for selection.
        test_mask = pooled_probs >= thr if higher_is_better else pooled_probs < thr
        n_test_taken = int(test_mask.sum())
        if n_test_taken == 0:
            test_avg_pnl = 0.0
            test_total_pnl = 0.0
            test_win_rate = 0.0
        else:
            test_pnl_subset = pooled_pnls[test_mask]
            test_total_pnl = float(test_pnl_subset.sum())
            test_avg_pnl = float(test_pnl_subset.mean())
            test_win_rate = float((test_pnl_subset > 0).mean())

        threshold_results.append({
            "threshold": thr,
            # Selection-relevant (val-only) numbers.
            "val_trades_taken": n_val_taken,
            "val_win_rate": val_win_rate,
            "val_avg_pnl": val_avg_pnl,
            "val_total_pnl": val_total_pnl,
            # Diagnostic test-side numbers reported back at the same threshold.
            "trades_taken": n_test_taken,
            "win_rate": test_win_rate,
            "avg_pnl": test_avg_pnl,
            "total_pnl": test_total_pnl,
        })

    # Selection: maximise val total PnL (NOT test).  Ties go to the lower
    # threshold for determinism.
    best_thr = max(
        threshold_results,
        key=lambda x: (x["val_total_pnl"], -x["threshold"]),
    )

    return {
        "windows": windows,
        "n_windows": len(windows),
        "pooled_test_count": len(pooled_labels),
        "pooled_val_count": len(pooled_val_labels),
        "aggregated_metrics": {
            "brier_score": agg_brier,
            "mae_pnl": agg_mae,
            "tp50_rate": float(pooled_labels.mean()) if len(pooled_labels) else 0.0,
            "expectancy": float(pooled_pnls.mean()) if len(pooled_pnls) else 0.0,
            "max_drawdown": _max_drawdown(pooled_pnls.tolist()),
            "calibration_by_decile": agg_calibration,
        },
        "threshold_tuning": threshold_results,
        "recommended_threshold": best_thr["threshold"],
        "recommended_threshold_source": "val_pool",
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
        """Append a rule entry to *rules* if the subset is large enough."""
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
