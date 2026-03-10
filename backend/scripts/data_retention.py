#!/usr/bin/env python3
"""Export old data to local CSV files, then purge from the database.

Usage
-----
    python backend/scripts/data_retention.py --days 90 --export-dir ./exports
    python backend/scripts/data_retention.py --days 90 --export-dir ./exports --dry-run

The script connects to the production database (via DATABASE_URL in the
environment or .env file), exports rows older than ``--days`` from the
heaviest tables, saves them as gzipped CSV files, then deletes the
exported rows.  Chain snapshot deletions cascade to option_chain_rows,
gex_by_strike, gex_by_expiry_strike, and gex_snapshots via FK ON DELETE
CASCADE.

Requirements: psycopg2-binary (or psycopg2), pandas
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import pandas as pd
    import psycopg2
except ImportError:
    sys.exit(
        "Missing dependencies.  Install with:\n"
        "  pip install psycopg2-binary pandas"
    )


def _sync_dsn() -> str:
    """Resolve a synchronous psycopg2-compatible DSN from the environment.

    Returns
    -------
    str
        PostgreSQL connection string with asyncpg:// replaced by postgresql://.
    """
    raw = os.getenv("DATABASE_URL", "")
    if not raw:
        dotenv = Path(__file__).resolve().parents[2] / ".env"
        if dotenv.exists():
            for line in dotenv.read_text().splitlines():
                if line.startswith("DATABASE_URL="):
                    raw = line.split("=", 1)[1].strip()
                    break
    if not raw:
        sys.exit("DATABASE_URL not set and no .env file found.")
    return raw.replace("postgresql+asyncpg://", "postgresql://")


def _export_and_purge(
    conn,
    *,
    cutoff: datetime,
    export_dir: Path,
    dry_run: bool,
) -> None:
    """Export chain_snapshots (and cascade-dependent rows) older than cutoff.

    Parameters
    ----------
    conn:
        psycopg2 connection with autocommit=False.
    cutoff:
        UTC timestamp; rows with ts/fetched_at before this are exported.
    export_dir:
        Directory to write gzipped CSV files into.
    dry_run:
        When True, only count rows without exporting or deleting.
    """
    cur = conn.cursor()

    cur.execute(
        "SELECT count(*) FROM chain_snapshots WHERE ts < %s",
        (cutoff,),
    )
    snap_count = cur.fetchone()[0]
    print(f"Chain snapshots older than {cutoff.date()}: {snap_count}")

    if snap_count == 0:
        print("Nothing to purge.")
        return

    cur.execute(
        "SELECT snapshot_id FROM chain_snapshots WHERE ts < %s",
        (cutoff,),
    )
    old_ids = [r[0] for r in cur.fetchall()]
    print(f"  Snapshot IDs to purge: {len(old_ids)}")

    tables = [
        ("option_chain_rows", "snapshot_id"),
        ("gex_by_strike", "snapshot_id"),
        ("gex_by_expiry_strike", "snapshot_id"),
        ("gex_snapshots", "snapshot_id"),
        ("trade_marks", "snapshot_id"),
    ]

    for table, fk_col in tables:
        cur.execute(
            f"SELECT count(*) FROM {table} WHERE {fk_col} = ANY(%s)",
            (old_ids,),
        )
        row_count = cur.fetchone()[0]
        print(f"  {table}: {row_count} rows")

        if dry_run or row_count == 0:
            continue

        export_path = export_dir / f"{table}_{cutoff.date()}.csv.gz"
        print(f"    Exporting to {export_path} ...")
        query = f"SELECT * FROM {table} WHERE {fk_col} = ANY(%s)"
        df = pd.read_sql_query(query, conn, params=(old_ids,))
        df.to_csv(export_path, index=False, compression="gzip")
        print(f"    Exported {len(df)} rows.")

    if not dry_run:
        snap_path = export_dir / f"chain_snapshots_{cutoff.date()}.csv.gz"
        print(f"  Exporting chain_snapshots to {snap_path} ...")
        df_snap = pd.read_sql_query(
            "SELECT snapshot_id, ts, underlying, source, target_dte, expiration, checksum "
            "FROM chain_snapshots WHERE ts < %s",
            conn,
            params=(cutoff,),
        )
        df_snap.to_csv(snap_path, index=False, compression="gzip")

        print("  Deleting chain_snapshots (cascades to child tables) ...")
        cur.execute(
            "DELETE FROM chain_snapshots WHERE ts < %s",
            (cutoff,),
        )
        deleted = cur.rowcount
        conn.commit()
        print(f"  Deleted {deleted} chain_snapshots (+ cascaded children).")

        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        print(f"  Current DB size: {cur.fetchone()[0]}")
    else:
        print("\n  [DRY RUN] No data was exported or deleted.")

    cur.close()


def main() -> None:
    """CLI entry point for the data retention export + purge script."""
    parser = argparse.ArgumentParser(
        description="Export old chain/GEX data to CSV, then purge from DB.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Delete data older than this many days (default: 90).",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./exports",
        help="Directory to save exported CSV files (default: ./exports).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows only; do not export or delete.",
    )
    args = parser.parse_args()

    dsn = _sync_dsn()
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
    print(f"Retention cutoff: {cutoff.date()} ({args.days} days ago)")
    print(f"Export directory:  {export_dir.resolve()}")
    print(f"Dry run:          {args.dry_run}\n")

    conn = psycopg2.connect(dsn)
    try:
        _export_and_purge(conn, cutoff=cutoff, export_dir=export_dir, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
