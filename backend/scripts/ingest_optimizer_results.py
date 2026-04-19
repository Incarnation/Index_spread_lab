#!/usr/bin/env python3
"""Ingest optimizer CSV results into PostgreSQL for the interactive dashboard.

Reads ``backtest_results.csv`` (and optionally ``walkforward_results.csv``),
computes the Pareto frontier, and upserts rows into the ``optimizer_runs``,
``optimizer_results``, and ``optimizer_walkforward`` tables.

Usage::

    python ingest_optimizer_results.py --run-name "staged-v2"
    python ingest_optimizer_results.py --results-csv data/backtest_results.csv
    python ingest_optimizer_results.py --walkforward-csv data/walkforward_results.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPTS_DIR.parents[1] / "data"
DEFAULT_RESULTS_CSV = _DATA_DIR / "backtest_results.csv"
DEFAULT_WF_CSV = _DATA_DIR / "walkforward_results.csv"

# Make the local scripts dir importable so we can pull in the shared
# Pareto helper (M6 in OFFLINE_PIPELINE_AUDIT.md). This script can be
# invoked either as ``python -m scripts.ingest_optimizer_results`` (in
# which case ``_pareto`` is already on the package path) or as a plain
# script, which is why we add the directory explicitly.
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _pareto import compute_pareto_mask  # noqa: E402  (path setup above)


def _to_sync_url(url: str) -> str:
    """Convert an async DATABASE_URL to a sync one for SQLAlchemy.

    Replaces ``postgresql+asyncpg://`` with ``postgresql://`` so that
    ``create_engine`` works without greenlet/async context.
    """
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "")
    return url


def _get_database_url() -> str:
    """Read DATABASE_URL from environment or .env file.

    Automatically converts async driver URLs (``postgresql+asyncpg://``)
    to sync (``postgresql://``) so ``create_engine`` works without greenlet.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return _to_sync_url(url)

    env_file = _SCRIPTS_DIR.parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return _to_sync_url(line.split("=", 1)[1].strip().strip('"').strip("'"))

    backend_env = _SCRIPTS_DIR.parent / ".env"
    if backend_env.exists():
        for line in backend_env.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return _to_sync_url(line.split("=", 1)[1].strip().strip('"').strip("'"))

    raise RuntimeError(
        "DATABASE_URL not found. Set it as an environment variable "
        "or in a .env file."
    )


def _compute_pareto(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series marking Pareto-optimal rows (Sharpe vs max DD).

    Thin wrapper around :func:`_pareto.compute_pareto_mask` (the single
    canonical implementation, see M6 in ``OFFLINE_PIPELINE_AUDIT.md``).
    The wrapper is preserved so existing callers and tests that import
    ``_compute_pareto`` continue to work.
    """
    return compute_pareto_mask(df)


# Column name mappings from CSV to DB
_CONFIG_COLUMNS = [
    "p_starting_capital", "p_max_trades_per_day", "p_monthly_drawdown_limit",
    "p_lot_per_equity", "p_max_equity_risk_pct", "p_max_margin_pct",
    "p_calls_only", "p_min_dte", "p_max_delta",
    "t_tp_pct", "t_sl_mult", "t_max_vix", "t_max_term_structure",
    "t_avoid_opex", "t_prefer_event_days", "t_width_filter", "t_entry_count",
    "e_enabled", "e_signal_mode", "e_budget_mode", "e_max_event_trades",
    "e_spx_drop_threshold", "e_spx_drop_2d_threshold", "e_spx_drop_min",
    "e_spx_drop_max", "e_vix_spike_threshold", "e_vix_elevated_threshold",
    "e_term_inversion_threshold", "e_side_preference", "e_min_dte", "e_max_dte",
    "e_min_delta", "e_max_delta", "e_rally_avoidance", "e_rally_threshold",
    "e_event_only",
    "r_enabled", "r_high_vix_threshold", "r_high_vix_multiplier",
    "r_extreme_vix_threshold", "r_big_drop_threshold", "r_big_drop_multiplier",
    "r_consecutive_loss_days", "r_consecutive_loss_multiplier",
]

_METRIC_COLUMNS = [
    "final_equity", "return_pct", "ann_return_pct", "max_dd_pct",
    "trough", "sharpe", "total_trades", "days_traded", "days_stopped",
    "win_days", "win_rate",
]


def ingest_results(
    results_csv: Path,
    run_name: str,
    optimizer_mode: str = "staged",
    database_url: str | None = None,
    walkforward_csv: Path | None = None,
) -> str:
    """Ingest optimizer results CSV into PostgreSQL.

    Parameters
    ----------
    results_csv : Path to the backtest results CSV.
    run_name : Human-readable name for this optimizer run.
    optimizer_mode : One of 'staged', 'event-only', 'selective', 'exhaustive', 'yaml-config'.
    database_url : PostgreSQL connection string; read from env if None.
    walkforward_csv : Optional path to walk-forward results CSV.

    Returns
    -------
    The generated run_id string.
    """
    if database_url is None:
        database_url = _get_database_url()

    engine = create_engine(database_url)
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df):,} results from {results_csv}", flush=True)

    # Compute a content hash for duplicate detection
    csv_bytes = results_csv.read_bytes()
    content_hash = hashlib.sha256(csv_bytes).hexdigest()[:16]

    with engine.connect() as conn:
        dup = conn.execute(
            text("SELECT run_id FROM optimizer_runs WHERE metadata->>'content_hash' = :h"),
            {"h": content_hash},
        ).first()
    if dup:
        existing_id = dup[0]
        print(f"Duplicate detected: CSV already ingested as {existing_id}. Skipping.", flush=True)
        return existing_id

    # Compute Pareto frontier
    print("Computing Pareto frontier ...", flush=True)
    df["is_pareto"] = _compute_pareto(df)
    n_pareto = df["is_pareto"].sum()
    print(f"  {n_pareto} Pareto-optimal configs", flush=True)

    # Generate run ID
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{ts}_{uuid.uuid4().hex[:6]}"

    # Prepare results rows up-front so a parsing error aborts BEFORE we
    # write anything to PostgreSQL.
    all_cols = _CONFIG_COLUMNS + _METRIC_COLUMNS + ["is_pareto"]
    present_cols = [c for c in all_cols if c in df.columns]
    insert_df = df[present_cols].copy()
    insert_df["run_id"] = run_id
    insert_df = insert_df.where(pd.notna(insert_df), None)

    # Optionally prepare walk-forward rows BEFORE opening the transaction
    # so any parsing failure aborts cleanly without touching the DB.
    wf_df_to_insert: pd.DataFrame | None = None
    if walkforward_csv and Path(walkforward_csv).exists():
        wf_df = pd.read_csv(walkforward_csv)
        print(f"Ingesting {len(wf_df):,} walk-forward results ...", flush=True)
        wf_df["run_id"] = run_id

        # Map CSV column names to DB column names
        wf_rename = {
            "window": "window_label",
            "train_return_pct": "train_return",
            "test_return_pct": "test_return",
        }
        wf_df.rename(columns=wf_rename, inplace=True)

        # Build a config_key from param columns to identify each config
        param_cols = [c for c in _CONFIG_COLUMNS if c in wf_df.columns]
        if param_cols:
            wf_df["config_key"] = wf_df[param_cols].fillna("").astype(str).agg("|".join, axis=1).apply(
                lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
            )

        # Compute decay_ratio = test_sharpe / train_sharpe when both are available
        if "train_sharpe" in wf_df.columns and "test_sharpe" in wf_df.columns:
            wf_df["decay_ratio"] = wf_df.apply(
                lambda r: r["test_sharpe"] / r["train_sharpe"]
                if pd.notna(r["train_sharpe"]) and r["train_sharpe"] != 0
                else None,
                axis=1,
            )

        wf_df = wf_df.where(pd.notna(wf_df), None)

        wf_db_cols = [c for c in wf_df.columns if c in {
            "run_id", "config_key", "window_label",
            "train_start", "train_end", "test_start", "test_end",
            "train_sharpe", "test_sharpe", "train_return", "test_return",
            "train_trades", "test_trades", "decay_ratio",
        }]
        if wf_db_cols:
            wf_df_to_insert = wf_df[wf_db_cols]

    # H2 fix: wrap optimizer_runs INSERT, optimizer_results bulk insert,
    # and optional optimizer_walkforward insert in a SINGLE transaction so
    # we never end up with a parent run row whose children are missing.
    # If any of the three writes raises, the whole tree is rolled back and
    # the duplicate-detection guard at the top of this function will not
    # block a retry (no run row exists).
    print(f"Inserting {len(insert_df):,} results into optimizer_results ...", flush=True)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO optimizer_runs
                (run_id, run_name, optimizer_mode, started_at, finished_at,
                 num_configs, status, metadata)
            VALUES
                (:run_id, :run_name, :mode, now(), now(),
                 :num_configs, 'completed', :metadata)
        """), {
            "run_id": run_id,
            "run_name": run_name,
            "mode": optimizer_mode,
            "num_configs": len(df),
            "metadata": json.dumps({"content_hash": content_hash}),
        })

        # to_sql accepts a SQLAlchemy Connection so the inserts share the
        # same in-flight transaction as the optimizer_runs INSERT above.
        insert_df.to_sql(
            "optimizer_results",
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

        if wf_df_to_insert is not None:
            wf_df_to_insert.to_sql(
                "optimizer_walkforward",
                conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )

    print(f"Ingestion complete. Run ID: {run_id}", flush=True)
    return run_id


def main() -> None:
    """CLI entry point for ingesting optimizer results."""
    parser = argparse.ArgumentParser(
        description="Ingest optimizer results CSV into PostgreSQL",
    )
    parser.add_argument(
        "--results-csv", type=str, default=str(DEFAULT_RESULTS_CSV),
        help=f"Path to backtest results CSV (default: {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--walkforward-csv", type=str, default=None,
        help="Optional path to walk-forward results CSV",
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Human-readable name for this optimizer run",
    )
    parser.add_argument(
        "--mode", type=str, default="staged",
        choices=["staged", "event-only", "selective", "exhaustive", "yaml-config"],
        help="Optimizer mode used to generate the results",
    )
    parser.add_argument(
        "--database-url", type=str, default=None,
        help="PostgreSQL URL (reads from DATABASE_URL env var if not set)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    if not results_path.exists():
        logger.error("Results CSV not found: %s", results_path)
        sys.exit(1)

    wf_path = Path(args.walkforward_csv) if args.walkforward_csv else None

    ingest_results(
        results_csv=results_path,
        run_name=args.run_name,
        optimizer_mode=args.mode,
        database_url=args.database_url,
        walkforward_csv=wf_path,
    )


if __name__ == "__main__":
    main()
