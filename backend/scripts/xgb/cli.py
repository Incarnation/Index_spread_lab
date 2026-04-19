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
    BIG_LOSS_THRESHOLD,
    build_entry_feature_matrix,
    build_entry_v2_feature_matrix,
    _resolve_targets,
)
from .training import (
    DEFAULT_CLS_PARAMS,
    save_model,
    train_final_model,
    _fit_full_after_early_stopping,
)
from .walkforward import (
    walk_forward_hold_vs_close,
    walk_forward_rolling,
    walk_forward_validate_xgb,
    _extract_entry_rules,
)


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
        logger.error("CSV not found: %s", csv_path)
        sys.exit(2)

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
        logger.error("%s", results["error"])
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
    save_model(final, save_dir, model_type="xgb_v1")
    print(f"[XGB] Model saved to {save_dir}/")
    print("[XGB] Done.")
def _run_hold_vs_close(df: pd.DataFrame) -> None:
    """Hold-vs-close model training and rule extraction."""
    print("[XGB] Hold-vs-close walk-forward validation ...")
    results = walk_forward_hold_vs_close(df)

    if "error" in results:
        logger.error("%s", results["error"])
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
        logger.error("%s", results["error"])
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

    # Train final model on all data and save.  Two-step recipe (see
    # train_final_model docstring): early-stopping calibration on 90/10 to
    # discover best_iteration, then refit on the full 100% with the round
    # count locked.  Previously this fit only on the first 90%, leaving the
    # most-recent decile out of the shipped booster (see C2 in
    # OFFLINE_PIPELINE_AUDIT.md).
    print("\n[ENTRY] Training final model on all data ...")
    df_sorted_all = df.sort_values("day").reset_index(drop=True)
    X_full, y_cls, y_pnl = build_entry_feature_matrix(df_sorted_all)
    models = _fit_full_after_early_stopping(
        X_full=X_full,
        y_cls_full=y_cls,
        y_pnl_full=y_pnl,
        cls_params=None,
        reg_params=None,
    )
    save_dir = Path(save_dir_arg) if save_dir_arg else csv_path.parent / "xgb_entry_model"
    save_model(models, save_dir, model_type="xgb_entry_v1")
    print(f"[ENTRY] Model saved to {save_dir}/ "
          f"(trained_rows={models['trained_rows']}, "
          f"best_cls={models['best_iteration_classifier']}, "
          f"best_reg={models['best_iteration_regressor']})")
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
        logger.error("%s", results["error"])
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

    # Train final model and save using the two-step recipe (see C2 in
    # OFFLINE_PIPELINE_AUDIT.md).  v2_cls_params is forwarded so the
    # asymmetric scale_pos_weight is preserved on the refit.
    print("\n[ENTRY-V2] Training final model on all data ...")
    df_sorted = df.sort_values("day").reset_index(drop=True)
    X_full, y_cls, y_pnl = build_entry_v2_feature_matrix(df_sorted)
    models = _fit_full_after_early_stopping(
        X_full=X_full,
        y_cls_full=y_cls,
        y_pnl_full=y_pnl,
        cls_params=v2_cls_params,
        reg_params=None,
    )
    save_dir = Path(save_dir_arg) if save_dir_arg else csv_path.parent / "xgb_entry_v2_model"
    save_model(models, save_dir, model_type="xgb_entry_v2")
    print(f"[ENTRY-V2] Model saved to {save_dir}/ "
          f"(trained_rows={models['trained_rows']}, "
          f"best_cls={models['best_iteration_classifier']}, "
          f"best_reg={models['best_iteration_regressor']})")
    print("[ENTRY-V2] Done.")


if __name__ == "__main__":
    main()
