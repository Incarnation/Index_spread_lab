"""Export production DB tables to CSV for offline training pipeline.

Connects to the production PostgreSQL database using DATABASE_URL from
the .env file, exports ``underlying_quotes``, ``context_snapshots``, and
``economic_events`` to CSV files that the offline training pipeline can load.

Usage:
    python scripts/export_production_data.py [--start DATE] [--end DATE]

Defaults to exporting all available data when --start/--end are omitted.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


_BACKEND = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND.parent / "data"

UNDERLYING_QUOTES_CSV = DATA_DIR / "underlying_quotes_export.csv"
CONTEXT_SNAPSHOTS_CSV = DATA_DIR / "context_snapshots_export.csv"
ECONOMIC_EVENTS_CSV = DATA_DIR / "economic_events_export.csv"


def _sync_url(async_url: str) -> str:
    """Convert an asyncpg DATABASE_URL to a synchronous psycopg2 URL.

    The production .env uses ``postgresql+asyncpg://...``.  This helper
    replaces the driver portion so we can use a blocking engine in this
    standalone script.
    """
    return async_url.replace("postgresql+asyncpg://", "postgresql://")


def _load_env() -> str:
    """Load .env and return the DATABASE_URL.

    Searches backend/.env and repo-root/.env (matching Settings lookup
    order in config.py).
    """
    for candidate in [_BACKEND / ".env", _BACKEND.parent / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)
            break

    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not found in environment or .env", file=sys.stderr)
        sys.exit(1)
    return url


def export_underlying_quotes(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = UNDERLYING_QUOTES_CSV,
) -> int:
    """Export underlying_quotes (SPX, SPY, VIX, VIX9D) to CSV.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output:
        Destination CSV path.

    Returns
    -------
    int
        Number of rows exported.
    """
    clauses = ["symbol IN ('SPX', 'SPY', 'VIX', 'VIX9D', 'VVIX', 'SKEW')"]
    params: dict = {}
    if start:
        clauses.append("ts >= :start")
        params["start"] = start
    if end:
        clauses.append("ts < :end")
        params["end"] = end

    where = " AND ".join(clauses)
    query = text(
        f"SELECT ts, symbol, last, bid, ask, volume "
        f"FROM underlying_quotes WHERE {where} ORDER BY ts"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return len(df)


def export_context_snapshots(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = CONTEXT_SNAPSHOTS_CSV,
) -> int:
    """Export context_snapshots to CSV.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output:
        Destination CSV path.

    Returns
    -------
    int
        Number of rows exported.
    """
    clauses = ["1=1"]
    params: dict = {}
    if start:
        clauses.append("ts >= :start")
        params["start"] = start
    if end:
        clauses.append("ts < :end")
        params["end"] = end

    where = " AND ".join(clauses)
    query = text(
        f"SELECT ts, spx_price, spy_price, vix, vix9d, "
        f"term_structure, gex_net, zero_gamma_level "
        f"FROM context_snapshots WHERE {where} ORDER BY ts"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return len(df)


def export_economic_events(
    engine,
    *,
    output: Path = ECONOMIC_EVENTS_CSV,
) -> int:
    """Export economic_events to CSV.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    output:
        Destination CSV path.

    Returns
    -------
    int
        Number of rows exported.
    """
    query = text(
        "SELECT date, event_type, has_projections, is_triple_witching "
        "FROM economic_events ORDER BY date"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return len(df)


def main() -> None:
    """CLI entry point for production data export."""
    parser = argparse.ArgumentParser(
        description="Export production DB tables to CSV for offline training."
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date (inclusive, ISO format, e.g. 2026-01-01)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (exclusive, ISO format, e.g. 2026-04-01)",
    )
    parser.add_argument(
        "--tables", default="all",
        choices=["all", "underlying_quotes", "context_snapshots", "economic_events"],
        help="Which table(s) to export (default: all)",
    )
    parser.add_argument(
        "--check-gex", action="store_true",
        help="Only check if GEX data exists in context_snapshots (no export)",
    )
    args = parser.parse_args()

    db_url = _load_env()
    sync_url = _sync_url(db_url)
    engine = create_engine(sync_url)

    if args.check_gex:
        print("\n" + "=" * 60)
        print("GEX DATA CHECK")
        print("=" * 60)
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) AS total, "
                "COUNT(gex_net) AS gex_rows, "
                "MIN(ts) FILTER (WHERE gex_net IS NOT NULL) AS gex_first, "
                "MAX(ts) FILTER (WHERE gex_net IS NOT NULL) AS gex_last "
                "FROM context_snapshots"
            ))
            row = result.fetchone()
            print(f"  Total context_snapshots rows: {row[0]}")
            print(f"  Rows with gex_net non-null:   {row[1]}")
            if row[1] and row[1] > 0:
                print(f"  GEX data range: {row[2]} to {row[3]}")
                print("\n  GEX data IS available in production DB.")
            else:
                print("\n  GEX data is NOT available (all gex_net values are NULL).")
        print("=" * 60)
        return

    date_msg = ""
    if args.start or args.end:
        date_msg = f" [{args.start or '...'} to {args.end or '...'}]"

    print(f"\nExporting production data{date_msg}")
    print("=" * 60)

    if args.tables in ("all", "underlying_quotes"):
        n = export_underlying_quotes(engine, start=args.start, end=args.end)
        print(f"  underlying_quotes: {n} rows -> {UNDERLYING_QUOTES_CSV}")

    if args.tables in ("all", "context_snapshots"):
        n = export_context_snapshots(engine, start=args.start, end=args.end)
        print(f"  context_snapshots: {n} rows -> {CONTEXT_SNAPSHOTS_CSV}")

    if args.tables in ("all", "economic_events"):
        n = export_economic_events(engine)
        print(f"  economic_events: {n} rows -> {ECONOMIC_EVENTS_CSV}")

    print("=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
