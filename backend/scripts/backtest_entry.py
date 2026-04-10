"""Backtest framework for SPX credit spread entry strategies.

Runs walk-forward validation to collect per-trade out-of-sample
predictions, then simulates multiple strategies (all-trades, model-filtered,
rule-based) and reports equity curves, Sharpe, drawdown, and profit factor.

Usage::

    python scripts/backtest_entry.py --csv ../data/training_candidates.csv
    python scripts/backtest_entry.py --csv ../data/training_candidates.csv --v2
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xgb_model import (
    build_entry_feature_matrix,
    predict_xgb,
    train_xgb_models,
)


# -------------------------------------------------------------------
# Strategy definitions
# -------------------------------------------------------------------

@dataclass
class StrategyResult:
    """Aggregated metrics for a single strategy run."""

    name: str
    trades_taken: int
    trades_skipped: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    monthly: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


def _compute_metrics(
    pnls: np.ndarray,
    days: np.ndarray,
    name: str,
    total_candidates: int,
) -> StrategyResult:
    """Compute backtest metrics from an array of per-trade PnLs.

    Parameters
    ----------
    pnls  : PnL for each trade taken.
    days  : Corresponding trade day (YYYY-MM-DD string) per trade.
    name  : Strategy label.
    total_candidates : Total candidates (taken + skipped).

    Returns
    -------
    StrategyResult with all metrics populated.
    """
    n = len(pnls)
    if n == 0:
        return StrategyResult(name=name, trades_taken=0, trades_skipped=total_candidates,
                              total_pnl=0, avg_pnl=0, win_rate=0, profit_factor=0,
                              sharpe=0, max_drawdown=0)

    total = float(pnls.sum())
    avg = float(pnls.mean())
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float((pnls > 0).sum() / n)
    pf = float(abs(wins.sum() / losses.sum())) if losses.sum() != 0 else float("inf")

    # Daily PnL for Sharpe
    df_daily = pd.DataFrame({"day": days, "pnl": pnls})
    daily_pnl = df_daily.groupby("day")["pnl"].sum()
    sharpe = 0.0
    if daily_pnl.std() > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))

    # Equity curve and max drawdown
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    # Monthly breakdown
    df_daily["month"] = pd.to_datetime(df_daily["day"]).dt.to_period("M")
    monthly = []
    for m, grp in df_daily.groupby("month"):
        monthly.append({
            "month": str(m),
            "trades": len(grp),
            "pnl": float(grp["pnl"].sum()),
            "avg_pnl": float(grp["pnl"].mean()),
            "win_rate": float((grp["pnl"] > 0).sum() / len(grp)),
        })

    return StrategyResult(
        name=name,
        trades_taken=n,
        trades_skipped=total_candidates - n,
        total_pnl=total,
        avg_pnl=avg,
        win_rate=win_rate,
        profit_factor=pf,
        sharpe=sharpe,
        max_drawdown=max_dd,
        monthly=monthly,
        equity_curve=cum.tolist(),
    )


def _apply_strategy(
    pool: pd.DataFrame,
    mask: np.ndarray,
    name: str,
) -> StrategyResult:
    """Apply a boolean mask to the pooled predictions and compute metrics.

    Parameters
    ----------
    pool : DataFrame with columns ``day``, ``hold_realized_pnl``, ``prob_tp50``, etc.
    mask : Boolean array — True means TAKE the trade.
    name : Strategy label.
    """
    taken = pool[mask]
    pnls = taken["hold_realized_pnl"].values.astype(float)
    days = taken["day"].values
    return _compute_metrics(pnls, days, name, len(pool))


# -------------------------------------------------------------------
# Walk-forward to collect per-trade OOS predictions
# -------------------------------------------------------------------

def collect_oos_predictions(
    df: pd.DataFrame,
    train_months: int = 8,
    test_months: int = 2,
    step_months: int = 1,
    build_features_fn=None,
) -> pd.DataFrame:
    """Run walk-forward and return per-trade out-of-sample predictions.

    Preserves original row metadata (day, dte_target, spread_side, vix, etc.)
    alongside model predictions so strategies can filter on any column.

    Parameters
    ----------
    df : Full training CSV DataFrame.
    train_months, test_months, step_months : Window parameters.
    build_features_fn : Feature matrix builder (defaults to build_entry_feature_matrix).

    Returns
    -------
    DataFrame with one row per OOS trade, including predictions and original columns.
    """
    from dateutil.relativedelta import relativedelta

    if build_features_fn is None:
        build_features_fn = build_entry_feature_matrix

    df_sorted = df.sort_values("day").reset_index(drop=True)
    df_sorted["_day_dt"] = pd.to_datetime(df_sorted["day"])
    first_day = df_sorted["_day_dt"].iloc[0]
    last_day = df_sorted["_day_dt"].iloc[-1]

    all_chunks: list[pd.DataFrame] = []
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

        X_train_full, y_cls_train, y_pnl_train = build_features_fn(train_df)
        X_test, y_cls_test, y_pnl_test = build_features_fn(test_df)

        val_split = int(len(X_train_full) * 0.9)
        models = train_xgb_models(
            X_train_full.iloc[:val_split], y_cls_train.iloc[:val_split], y_pnl_train.iloc[:val_split],
            X_train_full.iloc[val_split:], y_cls_train.iloc[val_split:], y_pnl_train.iloc[val_split:],
        )

        preds = predict_xgb(models, X_test)
        chunk = test_df.copy()
        chunk["prob_tp50"] = preds["prob_tp50"].values
        chunk["predicted_pnl"] = preds["predicted_pnl"].values
        chunk["_window_train"] = f"{train_df['day'].iloc[0]}..{train_df['day'].iloc[-1]}"
        all_chunks.append(chunk)
        train_start += relativedelta(months=step_months)

    if not all_chunks:
        return pd.DataFrame()

    pooled = pd.concat(all_chunks, ignore_index=True)
    # Deduplicate: a trade may appear in overlapping test windows; keep the
    # prediction from the latest (most data) training window.
    pooled = pooled.sort_values("_window_train", ascending=False).drop_duplicates(
        subset=["entry_dt", "spread_side", "dte_target", "delta_target", "short_strike", "long_strike"],
        keep="first",
    ).sort_values("day").reset_index(drop=True)
    return pooled


# -------------------------------------------------------------------
# Strategy runner
# -------------------------------------------------------------------

def run_strategies(pool: pd.DataFrame, is_v2: bool = False) -> list[StrategyResult]:
    """Run all strategies on the pooled OOS predictions.

    Parameters
    ----------
    pool  : DataFrame from ``collect_oos_predictions`` with predictions and metadata.
    is_v2 : If True, ``prob_tp50`` is actually P(big_loss) — lower is better.

    Returns
    -------
    List of StrategyResult, one per strategy.
    """
    vix = pd.to_numeric(pool["vix"], errors="coerce").values
    dte = pool["dte_target"].values.astype(int)
    side = pool["spread_side"].str.lower().values
    prob = pool["prob_tp50"].values
    pred_pnl = pool["predicted_pnl"].values

    results: list[StrategyResult] = []

    # 1. Baseline: take all trades
    results.append(_apply_strategy(pool, np.ones(len(pool), dtype=bool), "All trades"))

    if is_v2:
        # V2: prob = P(big_loss). Take trades where prob < threshold.
        for thr in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
            results.append(_apply_strategy(pool, prob < thr, f"V2 P(loss)<{thr:.2f}"))

        # V2 + rule combos
        for thr in [0.10, 0.15, 0.20]:
            results.append(_apply_strategy(
                pool, (prob < thr) & (dte > 0),
                f"V2 P<{thr:.2f} + no 0DTE",
            ))
    else:
        # V1: prob = P(TP50). Take trades where prob >= threshold.
        for thr in [0.30, 0.50, 0.70]:
            results.append(_apply_strategy(pool, prob >= thr, f"Model P>=.{int(thr*100)}"))

    # Rule-based (same for both)
    results.append(_apply_strategy(pool, dte > 0, "Skip 0-DTE"))
    results.append(_apply_strategy(pool, ~((side == "call") & (vix > 25)), "Skip calls VIX>25"))
    results.append(_apply_strategy(pool, ~((side == "call") & (dte == 0)), "Skip calls 0-DTE"))
    results.append(_apply_strategy(
        pool, (dte > 0) & ~((side == "call") & (vix > 25)),
        "Skip 0DTE + calls VIX>25",
    ))

    # Predicted PnL filter
    results.append(_apply_strategy(pool, pred_pnl > 0, "E[PnL] > 0"))

    return results


def print_results(results: list[StrategyResult]) -> None:
    """Print a formatted comparison table of all strategies."""
    print(f"\n{'=' * 100}")
    print("STRATEGY BACKTEST COMPARISON (out-of-sample)")
    print(f"{'=' * 100}")
    print(f"{'Strategy':<28s} {'Trades':>7s} {'Skip':>5s} {'Total PnL':>12s} {'Avg PnL':>9s} "
          f"{'Win%':>6s} {'PF':>6s} {'Sharpe':>7s} {'MaxDD':>9s}")
    print("-" * 100)

    best = max(results, key=lambda r: r.sharpe)
    for r in results:
        marker = " ***" if r is best else ""
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 1000 else "inf"
        print(f"{r.name:<28s} {r.trades_taken:>7,d} {r.trades_skipped:>5,d} "
              f"${r.total_pnl:>11,.0f} ${r.avg_pnl:>8.2f} "
              f"{r.win_rate:>5.1%} {pf_str:>6s} {r.sharpe:>7.2f} ${r.max_drawdown:>8,.0f}{marker}")
    print(f"\n  *** = best Sharpe ratio")

    # Monthly detail for top 3 strategies by Sharpe
    top3 = sorted(results, key=lambda r: r.sharpe, reverse=True)[:3]
    for r in top3:
        print(f"\n  --- Monthly: {r.name} ---")
        print(f"  {'Month':>8s} {'Trades':>7s} {'PnL':>10s} {'Avg':>9s} {'Win%':>6s}")
        for m in r.monthly:
            print(f"  {m['month']:>8s} {m['trades']:>7d} ${m['pnl']:>9,.0f} "
                  f"${m['avg_pnl']:>8.2f} {m['win_rate']:>5.1%}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(description="Backtest entry strategies")
    parser.add_argument("--csv", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "training_candidates.csv"))
    parser.add_argument("--train-months", type=int, default=8)
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--v2", action="store_true", help="Use entry-v2 feature builder (loss-avoidance model)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        sys.exit(2)

    print(f"[BACKTEST] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[BACKTEST] {len(df)} rows, {len(df.columns)} columns")

    if "hold_realized_pnl" not in df.columns:
        logger.error("hold_realized_pnl column missing -- run --relabel first")
        sys.exit(2)

    build_fn = None
    if args.v2:
        from xgb_model import build_entry_v2_feature_matrix
        build_fn = build_entry_v2_feature_matrix
        print("[BACKTEST] Using entry-v2 features (loss-avoidance model)")

    print(f"[BACKTEST] Collecting OOS predictions ({args.train_months}/{args.test_months}/{args.step_months} months) ...")
    t0 = time.time()
    pool = collect_oos_predictions(
        df,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        build_features_fn=build_fn,
    )
    elapsed = time.time() - t0
    print(f"[BACKTEST] Collected {len(pool)} OOS predictions in {elapsed:.1f}s")

    if pool.empty:
        logger.error("No OOS predictions generated")
        sys.exit(1)

    results = run_strategies(pool, is_v2=args.v2)
    print_results(results)
    print(f"\n{'=' * 100}")
    print("[BACKTEST] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
