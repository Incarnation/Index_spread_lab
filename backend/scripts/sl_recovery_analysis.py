"""Stop-loss recovery analysis and strategy sweep.

Reads the enhanced ``training_candidates.csv`` (with trajectory columns from
``--relabel``) and evaluates every combination of SL level and exit rule.

Usage::

    python sl_recovery_analysis.py                     # default CSV path
    python sl_recovery_analysis.py --csv path/to.csv   # custom path

Outputs a leaderboard of strategy combinations and per-dimension breakdowns
of the top performers.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND / "scripts"))

from _constants import CONTRACT_MULT, CONTRACTS

DATA_DIR = _BACKEND.parent / "data"
DEFAULT_CSV = DATA_DIR / "training_candidates.csv"

# Strategy sweep grid
SL_LEVELS: list[float | None] = [
    None, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
]

# TP level is parameterized via ``--tp`` (audit M8).  The legacy default
# is TP50; TP60 and TP75 are supported when the candidate CSV carries
# the matching ``min_pnl_before_tp{N}`` / ``first_tp{N}_pnl`` columns
# from a fresh ``generate_training_data.py --relabel`` run.
SUPPORTED_TP_LEVELS: tuple[int, ...] = (50, 60, 75)
DEFAULT_TP_LEVEL: int = 50


def _exit_rules_for_tp(tp_level: int) -> list[str]:
    """Return the (TP-aware, expiry) exit rule pair for *tp_level*.

    Kept as a helper rather than a module constant so the rule label
    stays consistent with the trajectory columns the rest of the
    pipeline expects (``tp50`` / ``tp60`` / ``tp75``).
    """
    return [f"tp{tp_level}", "expiry"]


# -------------------------------------------------------------------
# PnL reconstruction from trajectory fields
# -------------------------------------------------------------------

def compute_trade_pnl(
    row: pd.Series,
    sl_mult: float | None,
    exit_rule: str,
    tp_level: int = DEFAULT_TP_LEVEL,
) -> float | None:
    """Derive a single trade's PnL under a given (SL level, exit rule) policy.

    Parameters
    ----------
    row : pd.Series
        One row of the enhanced CSV with trajectory columns.
    sl_mult : float | None
        SL threshold as a multiple of credit (e.g. 2.0 = -200%).
        None means no stop-loss.
    exit_rule : str
        ``"tp{tp_level}"`` closes at the TP_level take-profit;
        ``"expiry"`` holds to the last mark (collecting TP100 if the
        options expire worthless).
    tp_level : int, default 50
        Take-profit level to model. Must be one of
        :data:`SUPPORTED_TP_LEVELS`. Reads ``min_pnl_before_tp{N}`` and
        ``first_tp{N}_pnl`` from the row -- so a fresh ``--relabel``
        run is required when changing this. Audit M8 in
        ``backend/scripts/OFFLINE_PIPELINE_AUDIT.md``.

    Returns
    -------
    float | None
        Trade PnL in dollars, or None if the trade is unresolved.
    """
    if not row.get("resolved", False):
        return None

    max_profit = row["entry_credit"] * CONTRACT_MULT * CONTRACTS
    if max_profit <= 0:
        return None

    # TP-level-specific trajectory columns (audit M8).
    min_pnl_before_tp = row.get(f"min_pnl_before_tp{tp_level}")
    first_tp_pnl = row.get(f"first_tp{tp_level}_pnl")
    final_pnl = row.get("final_pnl_at_expiry")

    # All trajectory columns must be present for the full SL/TP path
    # analysis. If any core field is missing/NaN, fall back to the
    # backward-compatible labels. Note: first_tp{N}_pnl and final_pnl
    # may legitimately be NaN (no TP{N} hit, or no expiry mark), so we
    # only require min_pnl_before_tp{N} plus the boolean/categorical
    # columns that are always set during relabeling.
    traj_vals = [min_pnl_before_tp, row.get("recovered_after_sl"),
                 row.get("hold_hit_tp50"), row.get("exit_reason")]
    has_trajectory = all(v is not None and not _isnan(v) for v in traj_vals)

    if not has_trajectory:
        return _fallback_pnl(row, sl_mult, exit_rule, max_profit, tp_level)

    sl_threshold = max_profit * sl_mult if sl_mult is not None else float("inf")

    # Does SL fire before TP_level?
    sl_fires_before_tp = min_pnl_before_tp <= -sl_threshold

    if sl_fires_before_tp and sl_mult is not None:
        # SL fires before any TP_level hit -> trade closed at SL.
        return -sl_threshold
    if first_tp_pnl is not None and not _isnan(first_tp_pnl):
        # TP_level fires (SL either didn't fire or fires after TP).
        if exit_rule == f"tp{tp_level}":
            return first_tp_pnl
        return final_pnl if (final_pnl is not None and not _isnan(final_pnl)) else first_tp_pnl
    if exit_rule == "expiry" and final_pnl is not None and not _isnan(final_pnl):
        return final_pnl
    return row.get("realized_pnl")


def _fallback_pnl(
    row: pd.Series,
    sl_mult: float | None,
    exit_rule: str,
    max_profit: float,
    tp_level: int = DEFAULT_TP_LEVEL,
) -> float | None:
    """Compute PnL for non-SL trades where trajectory fields are missing.

    For TP/EXPIRY trades the backward-compatible columns are sufficient
    because SL was never breached before the TP target. The ``tp_level``
    parameter only affects the matching exit rule label (e.g. ``tp60``).
    """
    orig_exit = row.get("exit_reason", "")
    orig_pnl = row.get("realized_pnl")

    if orig_exit == "TAKE_PROFIT_50":
        if exit_rule == f"tp{tp_level}":
            return orig_pnl
        # hold-to-expiry: we don't have final_pnl for TP-closed trades (closed early)
        return orig_pnl
    if orig_exit == "STOP_LOSS":
        if sl_mult is None:
            hold_pnl = row.get("hold_realized_pnl")
            return hold_pnl if hold_pnl is not None else orig_pnl
        # Counterfactual: if an SL was applied, the trade would have
        # been stopped at -sl_threshold (trajectory is missing so we
        # approximate).
        sl_threshold = max_profit * sl_mult
        return -sl_threshold
    return orig_pnl


def _isnan(val) -> bool:
    """Check if a value is NaN (safe for non-float types)."""
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


# -------------------------------------------------------------------
# Strategy evaluation
# -------------------------------------------------------------------

def evaluate_strategy(
    df: pd.DataFrame,
    sl_mult: float | None,
    exit_rule: str,
    tp_level: int = DEFAULT_TP_LEVEL,
) -> dict:
    """Evaluate a single strategy across all trades.

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced CSV with trajectory columns.
    sl_mult : float | None
        SL level (None = no SL).
    exit_rule : str
        ``"tp{tp_level}"`` or ``"expiry"``.
    tp_level : int, default 50
        Take-profit level to model (audit M8). Forwarded to
        :func:`compute_trade_pnl`.

    Returns
    -------
    dict
        Strategy name, total PnL, avg PnL, win rate, max drawdown,
        Sharpe proxy, and tail loss.
    """
    pnls = []
    for _, row in df.iterrows():
        pnl = compute_trade_pnl(row, sl_mult, exit_rule, tp_level)
        if pnl is not None:
            pnls.append(pnl)

    if not pnls:
        return {"sl_mult": sl_mult, "exit_rule": exit_rule, "n_trades": 0}

    pnls_arr = np.array(pnls)
    total_pnl = float(np.sum(pnls_arr))
    avg_pnl = float(np.mean(pnls_arr))
    std_pnl = float(np.std(pnls_arr)) if len(pnls_arr) > 1 else 0.0
    win_rate = float(np.mean(pnls_arr > 0))
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
    tail_5 = float(np.mean(np.sort(pnls_arr)[: max(1, len(pnls_arr) // 20)]))
    max_dd = _max_drawdown(pnls_arr)

    sl_label = f"{sl_mult:.1f}x" if sl_mult is not None else "none"

    return {
        "strategy": f"SL={sl_label} | exit={exit_rule}",
        "sl_mult": sl_mult,
        "exit_rule": exit_rule,
        "n_trades": len(pnls),
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "tail_5_pct": tail_5,
    }


def _max_drawdown(pnls: np.ndarray) -> float:
    """Compute max drawdown from a sequence of trade PnLs.

    Parameters
    ----------
    pnls : np.ndarray
        Array of per-trade PnLs (not cumulative).

    Returns
    -------
    float
        Maximum peak-to-trough drawdown (negative number).
    """
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0


# -------------------------------------------------------------------
# Dimensional slicing
# -------------------------------------------------------------------

def slice_by_dimension(
    df: pd.DataFrame,
    sl_mult: float | None,
    exit_rule: str,
    dim_col: str,
    buckets: dict[str, any] | None = None,
    tp_level: int = DEFAULT_TP_LEVEL,
) -> list[dict]:
    """Evaluate a strategy broken down by a single dimension.

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced CSV.
    sl_mult, exit_rule
        Strategy parameters.
    dim_col : str
        Column name to slice by (e.g. ``"dte_target"``).
    buckets : dict | None
        Optional mapping of bucket_name -> boolean mask.  If None,
        slices by unique values of ``dim_col``.
    tp_level : int, default 50
        Take-profit level to model (audit M8). Forwarded to
        :func:`evaluate_strategy`.

    Returns
    -------
    list[dict]
        One result dict per slice.
    """
    results = []
    if buckets is not None:
        for name, mask in buckets.items():
            sub = df[mask]
            if sub.empty:
                continue
            r = evaluate_strategy(sub, sl_mult, exit_rule, tp_level)
            r["dimension"] = dim_col
            r["slice"] = name
            results.append(r)
    else:
        for val in sorted(df[dim_col].dropna().unique()):
            sub = df[df[dim_col] == val]
            r = evaluate_strategy(sub, sl_mult, exit_rule, tp_level)
            r["dimension"] = dim_col
            r["slice"] = val
            results.append(r)
    return results


def build_vix_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Create VIX bucket masks."""
    return {
        "VIX<15": df["vix"] < 15,
        "15<=VIX<20": (df["vix"] >= 15) & (df["vix"] < 20),
        "20<=VIX<25": (df["vix"] >= 20) & (df["vix"] < 25),
        "25<=VIX<30": (df["vix"] >= 25) & (df["vix"] < 30),
        "VIX>=30": df["vix"] >= 30,
    }


def build_gex_buckets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Create GEX sign bucket masks."""
    gex = df.get("offline_gex_net")
    if gex is None:
        return {}
    return {
        "GEX>0": gex > 0,
        "GEX<=0": gex <= 0,
    }


def extract_entry_hour(df: pd.DataFrame) -> pd.Series:
    """Extract Eastern Time hour from entry_dt for slicing."""
    dt = pd.to_datetime(df["entry_dt"], utc=True)
    return dt.dt.tz_convert("America/New_York").dt.hour


# -------------------------------------------------------------------
# Recovery analysis (SL-specific)
# -------------------------------------------------------------------

def recovery_analysis(df: pd.DataFrame) -> None:
    """Print detailed recovery statistics for SL trades.

    Parameters
    ----------
    df : pd.DataFrame
        Enhanced CSV with trajectory columns.
    """
    sl = df[df["exit_reason"] == "STOP_LOSS"].copy()
    if sl.empty:
        print("\nNo STOP_LOSS trades found.")
        return

    print(f"\n{'='*70}")
    print("STOP-LOSS RECOVERY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total SL trades: {len(sl)}")

    recovered = sl["recovered_after_sl"] == True  # noqa: E712
    hold_tp50 = sl["hold_hit_tp50"] == True  # noqa: E712
    print(f"Recovered (SL then TP50): {recovered.sum()}/{len(sl)} "
          f"({recovered.mean()*100:.1f}%)")
    print(f"Hold hits TP50 (any time): {hold_tp50.sum()}/{len(sl)} "
          f"({hold_tp50.mean()*100:.1f}%)")

    # PnL comparison
    orig_mean = sl["realized_pnl"].mean()
    hold_mean = sl["hold_realized_pnl"].mean()
    final_mean = sl["final_pnl_at_expiry"].mean()
    print(f"\nAvg PnL at SL close:     ${orig_mean:>8.0f}")
    print(f"Avg PnL hold-to-TP50:    ${hold_mean:>8.0f}")
    print(f"Avg PnL hold-to-expiry:  ${final_mean:>8.0f}")
    print(f"Improvement (hold-TP50): ${hold_mean - orig_mean:>+8.0f}/trade")

    # By DTE
    print("\n--- Recovery by DTE ---")
    for dte in sorted(sl["dte_target"].unique()):
        sub = sl[sl["dte_target"] == dte]
        rec = (sub["recovered_after_sl"] == True).mean()  # noqa: E712
        hold = sub["hold_realized_pnl"].mean()
        orig = sub["realized_pnl"].mean()
        print(f"  DTE {dte:>2}: {len(sub):>4} trades, "
              f"recovery {rec*100:>5.1f}%, "
              f"SL ${orig:>7.0f} → hold ${hold:>7.0f} "
              f"(${hold-orig:>+7.0f})")

    # By side
    print("\n--- Recovery by Side ---")
    for side in sorted(sl["spread_side"].unique()):
        sub = sl[sl["spread_side"] == side]
        rec = (sub["recovered_after_sl"] == True).mean()  # noqa: E712
        hold = sub["hold_realized_pnl"].mean()
        orig = sub["realized_pnl"].mean()
        print(f"  {side:>5}: {len(sub):>4} trades, "
              f"recovery {rec*100:>5.1f}%, "
              f"SL ${orig:>7.0f} → hold ${hold:>7.0f}")

    # By VIX bucket
    if "vix" in sl.columns:
        print("\n--- Recovery by VIX ---")
        for label, mask in build_vix_buckets(sl).items():
            sub = sl[mask]
            if sub.empty:
                continue
            rec = (sub["recovered_after_sl"] == True).mean()  # noqa: E712
            hold = sub["hold_realized_pnl"].mean()
            orig = sub["realized_pnl"].mean()
            print(f"  {label:>12}: {len(sub):>4} trades, "
                  f"recovery {rec*100:>5.1f}%, "
                  f"SL ${orig:>7.0f} → hold ${hold:>7.0f}")

    # By delta
    print("\n--- Recovery by Delta ---")
    for delta in sorted(sl["delta_target"].unique()):
        sub = sl[sl["delta_target"] == delta]
        rec = (sub["recovered_after_sl"] == True).mean()  # noqa: E712
        hold = sub["hold_realized_pnl"].mean()
        orig = sub["realized_pnl"].mean()
        print(f"  delta {delta:.2f}: {len(sub):>4} trades, "
              f"recovery {rec*100:>5.1f}%, "
              f"SL ${orig:>7.0f} → hold ${hold:>7.0f}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Run the full strategy sweep and recovery analysis."""
    parser = argparse.ArgumentParser(description="SL recovery + strategy sweep")
    parser.add_argument(
        "--csv", type=str, default=str(DEFAULT_CSV),
        help="Path to the enhanced training_candidates.csv",
    )
    parser.add_argument(
        "--tp",
        type=int,
        choices=list(SUPPORTED_TP_LEVELS),
        default=DEFAULT_TP_LEVEL,
        help=(
            "Take-profit level to model (50, 60, or 75; default 50). "
            "Selects which `min_pnl_before_tp{N}` / `first_tp{N}_pnl` "
            "trajectory columns drive the SL counterfactual. Audit M8."
        ),
    )
    args = parser.parse_args()
    tp_level: int = args.tp

    csv_path = Path(args.csv)
    df = pd.read_csv(str(csv_path))
    print(f"Loaded {len(df)} rows from {csv_path}")

    has_trajectory = "hold_realized_pnl" in df.columns
    if not has_trajectory:
        logger.warning(
            "CSV does not have trajectory columns. "
            "Run `--relabel` first for full analysis. "
            "Falling back to backward-compatible columns only."
        )

    # Verify the TP-level-specific trajectory columns are present, else
    # the analysis silently degrades to fallback PnL for everything.
    tp_col = f"min_pnl_before_tp{tp_level}"
    if has_trajectory and tp_col not in df.columns:
        logger.warning(
            "CSV is missing the `%s` trajectory column required for "
            "--tp %d analysis; results will use the fallback PnL path "
            "instead. Re-run `generate_training_data.py --relabel` "
            "with TP%d enabled to populate the missing column.",
            tp_col, tp_level, tp_level,
        )

    # --- Recovery analysis (SL-specific) ---
    if has_trajectory:
        recovery_analysis(df)

    # --- Strategy sweep ---
    print(f"\n{'='*70}")
    print(f"STRATEGY SWEEP: SL level x Exit rule (TP={tp_level})")
    print(f"{'='*70}")

    results = []
    exit_rules = _exit_rules_for_tp(tp_level)
    for sl_mult in SL_LEVELS:
        for exit_rule in exit_rules:
            r = evaluate_strategy(df, sl_mult, exit_rule, tp_level)
            results.append(r)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("total_pnl", ascending=False)

    print(f"\n{'Strategy':<30} {'Trades':>6} {'Total PnL':>12} "
          f"{'Avg PnL':>9} {'Win%':>6} {'Sharpe':>7} {'MaxDD':>10}")
    print("-" * 90)
    for _, r in results_df.iterrows():
        print(f"{r['strategy']:<30} {r['n_trades']:>6} "
              f"${r['total_pnl']:>11,.0f} "
              f"${r['avg_pnl']:>8,.0f} "
              f"{r['win_rate']*100:>5.1f}% "
              f"{r['sharpe']:>7.3f} "
              f"${r['max_drawdown']:>9,.0f}")

    # --- Dimensional slicing for top 3 strategies ---
    if has_trajectory and len(results_df) >= 3:
        top3 = results_df.head(3)
        print(f"\n{'='*70}")
        print("DIMENSIONAL BREAKDOWN (top 3 strategies)")
        print(f"{'='*70}")

        for _, strat in top3.iterrows():
            sl_m = strat["sl_mult"]
            er = strat["exit_rule"]
            print(f"\n--- {strat['strategy']} ---")

            for dim_col in ["dte_target", "spread_side", "delta_target"]:
                slices = slice_by_dimension(df, sl_m, er, dim_col, tp_level=tp_level)
                if slices:
                    print(f"\n  By {dim_col}:")
                    for s in slices:
                        print(f"    {str(s['slice']):>8}: "
                              f"{s['n_trades']:>5} trades, "
                              f"${s['avg_pnl']:>7,.0f} avg, "
                              f"{s['win_rate']*100:>5.1f}% win")

            # VIX buckets
            vix_slices = slice_by_dimension(
                df, sl_m, er, "vix", build_vix_buckets(df), tp_level=tp_level,
            )
            if vix_slices:
                print("\n  By VIX:")
                for s in vix_slices:
                    print(f"    {str(s['slice']):>12}: "
                          f"{s['n_trades']:>5} trades, "
                          f"${s['avg_pnl']:>7,.0f} avg, "
                          f"{s['win_rate']*100:>5.1f}% win")

            # Entry hour
            if "entry_dt" in df.columns:
                df["_entry_hour"] = extract_entry_hour(df)
                hour_slices = slice_by_dimension(
                    df, sl_m, er, "_entry_hour", tp_level=tp_level,
                )
                if hour_slices:
                    print("\n  By entry hour (ET):")
                    for s in hour_slices:
                        print(f"    {int(s['slice']):>5}:00: "
                              f"{s['n_trades']:>5} trades, "
                              f"${s['avg_pnl']:>7,.0f} avg, "
                              f"{s['win_rate']*100:>5.1f}% win")
                df.drop(columns=["_entry_hour"], inplace=True)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
