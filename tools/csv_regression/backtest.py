"""CSV regression check for the backtest_strategy split (Phase 3.2).

Runs the same backtest configuration four times -- twice via the
back-compat shim ``backtest_strategy`` and twice via the new
``backtest`` package -- on a pinned slice of ``training_candidates.csv``,
then verifies the equity-curve + summary CSV is byte-identical across
both paths and across runs (i.e. proves both runtime determinism and
shim/package parity).

Usage::

    python tools/csv_regression/backtest.py \\
        --candidates data/training_candidates.csv \\
        --start 2025-01-01 --end 2025-03-31

Operator-driven; not part of the pytest suite (would require
``data/training_candidates.csv``, which isn't shipped).  Exits 0 on
PASS, 1 on FAIL, 2 on missing input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# `_common` lives next to this file under tools/csv_regression/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    df_to_canonical_csv_bytes,
    fresh_import,
    report_parity,
    require_input,
    setup_import_paths,
)


def _build_pinned_config(mod):
    """Construct the locked PortfolioConfig + EventConfig + FullConfig.

    Hard-coded values are intentional: this is a *regression* check so
    every input must be byte-stable across runs.  Update the constants
    in lock-step on both shim and package paths if a future tweak
    materially changes the optimizer surface.
    """
    pc = mod.PortfolioConfig(
        starting_capital=20_000,
        max_trades_per_day=2,
        monthly_drawdown_limit=0.15,
        lot_per_equity=10_000,
        calls_only=True,
        min_dte=2,
        max_delta=0.20,
    )
    evc = mod.EventConfig(
        enabled=True,
        budget_mode="shared",
        max_event_trades=1,
        spx_drop_threshold=-0.01,
        side_preference="puts",
        min_dte=5,
        max_dte=7,
        min_delta=0.15,
        max_delta=0.25,
        rally_avoidance=True,
        rally_threshold=0.01,
    )
    return mod.FullConfig(portfolio=pc, event=evc)


def _run(via: str, csv_path: Path, start: str, end: str) -> bytes:
    """Run the backtest via ``via`` and return canonical CSV bytes.

    ``via`` selects ``shim`` (``import backtest_strategy``) or
    ``package`` (``import backtest``).  Returns the equity-curve CSV +
    a summary metrics CSV concatenated with a sentinel separator so a
    single SHA captures drift in either accounting or selection paths.
    """
    if via == "shim":
        mod = fresh_import("backtest_strategy")
    elif via == "package":
        mod = fresh_import("backtest")
    else:
        raise ValueError(f"unknown via: {via!r}")

    df_full = pd.read_csv(csv_path)
    df = df_full[(df_full["day"] >= start) & (df_full["day"] <= end)].copy()
    if df.empty:
        raise SystemExit(
            f"FAIL: no rows in {csv_path} between {start} and {end}"
        )

    # The two precompute helpers mutate `df` in place; the operator's
    # canonical configuration sweeps TP {50, 60, 75} and SL
    # {None, 1.0, 1.5, 2.0}.
    mod.precompute_pnl_columns(
        df,
        tp_values=[0.5, 0.6, 0.75],
        sl_values=[None, 1.0, 1.5, 2.0],
    )
    daily_signals = mod.precompute_daily_signals(df)
    cfg = _build_pinned_config(mod)
    result = mod.run_backtest(df, daily_signals, cfg, "regression-check")

    summary = pd.DataFrame([{
        "final_equity": round(result.final_equity, 6),
        "total_return_pct": round(result.total_return_pct, 6),
        "max_drawdown_pct": round(result.max_drawdown_pct, 6),
        "total_trades": result.total_trades,
        "days_traded": result.days_traded,
        "win_days": result.win_days,
        "sharpe": round(result.sharpe, 6),
    }])
    curve = pd.DataFrame([
        {
            "day": str(r.day),
            "equity": round(r.equity, 6),
            "daily_pnl": round(r.daily_pnl, 6),
            "n_trades": r.n_trades,
            "lots": r.lots,
            "status": r.status,
            "event_signals": r.event_signals,
        }
        for r in result.curve
    ])
    return (
        df_to_canonical_csv_bytes(curve)
        + b"---SUMMARY---\n"
        + df_to_canonical_csv_bytes(summary)
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/training_candidates.csv"),
        help="Path to training_candidates.csv (default: data/...)",
    )
    parser.add_argument(
        "--start", default="2025-01-01",
        help="First day (inclusive) of the regression slice (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end", default="2025-03-31",
        help="Last day (inclusive) of the regression slice (YYYY-MM-DD).",
    )
    args = parser.parse_args(argv)

    setup_import_paths()
    csv_path = require_input(
        args.candidates,
        "run backend/scripts/generate_training_data.py to produce it",
    )

    print(
        f"Backtest CSV regression: {csv_path.name} "
        f"window {args.start} .. {args.end}"
    )

    shim_runs = [_run("shim", csv_path, args.start, args.end) for _ in range(2)]
    pkg_runs = [_run("package", csv_path, args.start, args.end) for _ in range(2)]
    return report_parity("backtest_strategy", shim_runs, pkg_runs)


if __name__ == "__main__":
    sys.exit(main())
