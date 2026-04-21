#!/usr/bin/env python3
"""End-to-end pipeline orchestrator for the backtest optimization workflow.

Chains all phases (export, generate, optimize, walk-forward, analyze, ingest)
with proper file path passing, timing, and status logging. Each phase calls
the underlying script via subprocess to maintain isolation.

Usage::

    # Full pipeline run
    python run_pipeline.py --run-name "2026-04-regen" \\
        --optimizer-config configs/optimizer/event_only_v2_explore.yaml

    # Skip export, use narrow training grid
    python run_pipeline.py --run-name "quick-test" \\
        --training-config configs/training/narrow.yaml \\
        --optimizer-config configs/optimizer/event_only_v2.yaml \\
        --skip-export --max-days 30

    # Single phase only
    python run_pipeline.py --run-name "v3" --phase generate --force-regen
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPTS_DIR.parent
_DATA_DIR = _BACKEND_DIR.parent / "data"


def _run_phase(
    name: str,
    cmd: list[str],
    expected_output: Path | None = None,
    log: dict | None = None,
) -> bool:
    """Execute a pipeline phase via subprocess.

    Parameters
    ----------
    name : Human-readable phase name.
    cmd : Command and arguments to run.
    expected_output : If set, verify this file exists and is non-empty after the command.
    log : Pipeline log dict to record timing and status.

    Returns
    -------
    True if the phase succeeded, False otherwise.
    """
    print(f"\n{'=' * 80}")
    print(f"  PHASE: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n", flush=True)

    phase_info = {"phase": name, "command": " ".join(cmd), "started_at": _now()}
    t0 = time.time()

    result = subprocess.run(cmd, cwd=str(_BACKEND_DIR))

    elapsed = time.time() - t0
    phase_info["elapsed_seconds"] = round(elapsed, 1)
    phase_info["exit_code"] = result.returncode

    if result.returncode != 0:
        phase_info["status"] = "FAILED"
        if log is not None:
            log.setdefault("phases", []).append(phase_info)
        print(f"\n  FAILED: {name} exited with code {result.returncode} "
              f"after {elapsed:.0f}s", flush=True)
        return False

    if expected_output is not None:
        if not expected_output.exists() or expected_output.stat().st_size == 0:
            phase_info["status"] = "FAILED"
            phase_info["error"] = f"Expected output missing: {expected_output}"
            if log is not None:
                log.setdefault("phases", []).append(phase_info)
            print(f"\n  FAILED: expected output {expected_output} not found or empty",
                  flush=True)
            return False
        phase_info["output_file"] = str(expected_output)
        phase_info["output_size_mb"] = round(expected_output.stat().st_size / 1e6, 1)

    phase_info["status"] = "OK"
    if log is not None:
        log.setdefault("phases", []).append(phase_info)
    print(f"\n  OK: {name} completed in {elapsed:.0f}s", flush=True)
    return True


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def main() -> None:
    """Parse arguments and run the selected pipeline phases."""
    parser = argparse.ArgumentParser(
        description="End-to-end backtest optimization pipeline orchestrator",
    )
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name for this pipeline run (used for output file naming)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "export", "generate", "optimize",
                                 "walkforward", "analyze", "ingest"],
                        help="Which phase(s) to run (default: all)")
    parser.add_argument("--training-config", type=str, default=None,
                        help="Path to training grid YAML (for generate phase)")
    parser.add_argument("--optimizer-config", type=str, default=None,
                        help="Path to optimizer grid YAML (for optimize/walkforward)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip the production data export phase")
    parser.add_argument("--force-regen", action="store_true",
                        help="Ignore candidate and label caches")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Custom candidate cache directory (forwarded to generate phase)")
    parser.add_argument("--label-cache-dir", type=str, default=None,
                        help="Custom label cache directory (forwarded to generate phase)")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limit training data to first N days (for testing)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for export (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for export (YYYY-MM-DD)")
    parser.add_argument("--holdout-months", type=int, default=0,
                        help="Reserve N months for holdout evaluation (default: 0)")
    parser.add_argument(
        "--holdout-min-pass-windows", type=int, default=None,
        help="Override the cli.py cross-window consistency filter for "
             "the holdout picker.  When unset, cli.py uses its phase-"
             "aware default (2 for walk-forward, 1 for plain optimize).  "
             "Set to 1 to disable the S2 regularizer when operating on "
             "a pre-filtered grid (empirical D.b finding: tight grids "
             "transfer better with the legacy top-N-by-test_sharpe "
             "picker).  Set higher to demand more cross-window "
             "consistency on noisy/wide grids like v3_explore.",
    )
    parser.add_argument("--ingest-mode", type=str, default="yaml-config",
                        choices=["staged", "event-only", "selective",
                                 "exhaustive", "yaml-config"],
                        help="Ingest mode for optimizer results (default: yaml-config)")
    args = parser.parse_args()

    run_name = args.run_name
    phase = args.phase

    results_csv = _DATA_DIR / f"{run_name}_results.csv"
    # H4 fix: walkforward_results.csv used to be a single global file shared
    # across runs; concurrent or sequential pipeline runs clobbered each
    # other's WF outputs while their _results.csv files were correctly
    # per-run.  We now scope WF per-run too and pass an explicit
    # --walkforward-output-csv to backtest_strategy.py.
    walkforward_csv = _DATA_DIR / f"{run_name}_walkforward.csv"
    # S1b fix (Tier-2 follow-up): holdout_results.csv had the same global-
    # single-file problem as walkforward_results.csv, and when A1/A2 were
    # run back-to-back the second run silently overwrote the first run's
    # holdout history.  Every pipeline run now gets a per-run holdout CSV
    # so we can compare runs side-by-side.
    holdout_csv = _DATA_DIR / f"{run_name}_holdout.csv"

    log: dict = {
        "run_name": run_name,
        "started_at": _now(),
        "args": vars(args),
        "phases": [],
    }

    phases_to_run = (
        ["export", "generate", "optimize", "walkforward", "analyze", "ingest"]
        if phase == "all"
        else [phase]
    )

    t0_total = time.time()
    python = sys.executable

    for p in phases_to_run:

        # -- EXPORT --
        if p == "export":
            if args.skip_export:
                print("\n  Skipping export phase (--skip-export)", flush=True)
                continue
            cmd = [python, "scripts/export_production_data.py", "--tables", "chains"]
            if args.start_date:
                cmd.extend(["--start", args.start_date])
            if args.end_date:
                cmd.extend(["--end", args.end_date])
            if not _run_phase("Export production data", cmd, log=log):
                break

            export_ok = True
            for table in ["underlying_parquet", "context_snapshots", "underlying_quotes"]:
                cmd2 = [python, "scripts/export_production_data.py", "--tables", table]
                if args.start_date:
                    cmd2.extend(["--start", args.start_date])
                if args.end_date:
                    cmd2.extend(["--end", args.end_date])
                if not _run_phase(f"Export {table}", cmd2, log=log):
                    export_ok = False
                    break
            if not export_ok:
                break

        # -- GENERATE --
        elif p == "generate":
            cmd = [
                python, "scripts/generate_training_data.py",
                "--workers", str(args.workers),
                "--quiet",
            ]
            if args.force_regen:
                cmd.append("--force-regen")
            if args.max_days:
                cmd.extend(["--max-days", str(args.max_days)])
            if args.training_config:
                cmd.extend(["--config", args.training_config])
            if args.cache_dir:
                cmd.extend(["--cache-dir", args.cache_dir])
            if args.label_cache_dir:
                cmd.extend(["--label-cache-dir", args.label_cache_dir])
            if args.start_date:
                cmd.extend(["--start-date", args.start_date])
            if args.end_date:
                cmd.extend(["--end-date", args.end_date])

            if not _run_phase(
                "Generate training candidates", cmd,
                expected_output=_DATA_DIR / "training_candidates.csv",
                log=log,
            ):
                break

        # -- OPTIMIZE --
        elif p == "optimize":
            if not args.optimizer_config and phase == "all":
                print("\n  WARNING: no --optimizer-config specified, "
                      "using exhaustive grid", flush=True)

            cmd = [
                python, "scripts/backtest_strategy.py",
                "--optimize",
                "--output-csv", str(results_csv),
                "--backtest-workers", str(args.workers),
            ]
            if args.optimizer_config:
                cmd.extend(["--config", args.optimizer_config])
            if args.holdout_months > 0:
                # Per-run holdout CSV (S1b).  For plain --optimize the
                # cross-window filter is a no-op (min_pass_windows=1 by
                # default), so we just need the per-run path to avoid
                # stomping on previous runs' holdout history.
                cmd.extend([
                    "--holdout-months", str(args.holdout_months),
                    "--holdout-output-csv", str(holdout_csv),
                ])
                # Track B: forward the override only when the operator
                # set one.  Forwarding a sentinel when unset would
                # override cli.py's phase-aware default (1 here), so
                # we deliberately only extend the argv when requested.
                if args.holdout_min_pass_windows is not None:
                    cmd.extend([
                        "--holdout-min-pass-windows",
                        str(args.holdout_min_pass_windows),
                    ])

            if not _run_phase(
                "Run optimizer", cmd,
                expected_output=results_csv,
                log=log,
            ):
                break

        # -- WALKFORWARD --
        elif p == "walkforward":
            cmd = [
                python, "scripts/backtest_strategy.py",
                "--walkforward", "--wf-auto",
                "--output-csv", str(results_csv),
                "--walkforward-output-csv", str(walkforward_csv),
                "--backtest-workers", str(args.workers),
            ]
            if args.optimizer_config:
                cmd.extend(["--config", args.optimizer_config])
            if args.holdout_months > 0:
                # Per-run holdout CSV (S1b).  cli.py's
                # --holdout-min-pass-windows defaults to 2 in the
                # walkforward branch, activating the S2 cross-window
                # consistency picker automatically so the pipeline never
                # silently reverts to single-window-by-sharpe.
                cmd.extend([
                    "--holdout-months", str(args.holdout_months),
                    "--holdout-output-csv", str(holdout_csv),
                ])
                # Track B: forward the override only when explicitly
                # set.  The D.b experiment (v2-puts, min_pass_windows=3
                # fell back to legacy picker and outperformed mpw=2 on
                # holdout) showed that tight pre-filtered grids can
                # benefit from dialing the S2 regularizer back.  Keep
                # the default as 2 (preserves the v3-overfit safety
                # rail) and make operators opt-in to 1 or 3+ per run.
                if args.holdout_min_pass_windows is not None:
                    cmd.extend([
                        "--holdout-min-pass-windows",
                        str(args.holdout_min_pass_windows),
                    ])

            if not _run_phase(
                "Walk-forward validation", cmd,
                expected_output=walkforward_csv,
                log=log,
            ):
                break

        # -- ANALYZE --
        elif p == "analyze":
            cmd = [
                python, "scripts/backtest_strategy.py",
                "--analyze",
                "--csv", str(_DATA_DIR / "training_candidates.csv"),
                "--output-csv", str(results_csv),
            ]
            if not _run_phase("Analyze results", cmd, log=log):
                break

        # -- INGEST --
        elif p == "ingest":
            cmd = [
                python, "scripts/ingest_optimizer_results.py",
                "--run-name", run_name,
                "--mode", args.ingest_mode,
                "--results-csv", str(results_csv),
            ]
            if walkforward_csv.exists():
                # Only pass walkforward CSV if it has data rows (not just headers)
                with open(walkforward_csv) as _wf:
                    _has_data = sum(1 for _ in _wf) > 1
                if _has_data:
                    cmd.extend(["--walkforward-csv", str(walkforward_csv)])
                else:
                    print("  ⚠ Walkforward CSV has no data rows — skipping ingest for it")

            if not _run_phase("Ingest results to DB", cmd, log=log):
                break

    total_elapsed = time.time() - t0_total
    log["completed_at"] = _now()
    log["total_elapsed_seconds"] = round(total_elapsed, 1)

    executed = log["phases"]
    all_ok = all(p.get("status") == "OK" for p in executed) if executed else True
    log["status"] = "COMPLETED" if all_ok else "FAILED"
    if not executed:
        log["note"] = "No phases were executed (all skipped or single phase not matched)"

    log_path = _DATA_DIR / f"pipeline_log_{run_name}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"  PIPELINE {'COMPLETED' if all_ok else 'FAILED'} in {total_elapsed:.0f}s")
    print(f"  Log: {log_path}")
    print(f"{'=' * 80}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
