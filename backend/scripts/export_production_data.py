"""Export production DB tables to CSV/Parquet for offline training pipeline.

Connects to the production PostgreSQL database using DATABASE_URL from
the .env file and exports ``underlying_quotes``, ``context_snapshots``,
``economic_events``, ``chains`` (option chain snapshots), and
``underlying_parquet`` (per-symbol parquet) to files that the offline
training pipeline can load.

The legacy ``trade_candidates`` exporter was removed in Track A.7 along
with the underlying table (migration 015).  Future ML re-entry will
synthesize training rows directly from ``trade_decisions`` + ``trades``
+ ``trade_marks`` instead.

Usage:
    python scripts/export_production_data.py [--start DATE] [--end DATE]
    python scripts/export_production_data.py --tables chains
    python scripts/export_production_data.py --tables underlying_parquet

Defaults to exporting all available data when --start/--end are omitted.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import load_project_env

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


_BACKEND = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND.parent / "data"

UNDERLYING_QUOTES_CSV = DATA_DIR / "underlying_quotes_export.csv"
CONTEXT_SNAPSHOTS_CSV = DATA_DIR / "context_snapshots_export.csv"
ECONOMIC_EVENTS_CSV = DATA_DIR / "economic_events_export.csv"
PRODUCTION_CHAINS_DIR = DATA_DIR / "production_exports" / "chains"
PRODUCTION_UNDERLYING_DIR = DATA_DIR / "production_exports" / "underlying"
ECONOMIC_CALENDAR_CSV = DATA_DIR / "economic_calendar.csv"


def _sync_url(async_url: str) -> str:
    """Convert an asyncpg DATABASE_URL to a synchronous psycopg2 URL.

    The production .env uses ``postgresql+asyncpg://...``.  This helper
    replaces the driver portion so we can use a blocking engine in this
    standalone script.
    """
    return async_url.replace("postgresql+asyncpg://", "postgresql://")


def _load_env() -> str:
    """Load .env and return the DATABASE_URL.

    Delegates to the shared ``_env.load_project_env()`` helper, then
    reads DATABASE_URL from the environment.

    Returns:
        The DATABASE_URL string.

    Raises:
        SystemExit: If DATABASE_URL is not set after loading .env.
    """
    load_project_env()

    url = os.getenv("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not found in environment or .env")
        sys.exit(1)
    return url


def export_underlying_quotes(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = UNDERLYING_QUOTES_CSV,
) -> int:
    """Export underlying_quotes (SPX, SPY, VIX, VIX9D, VVIX, SKEW) to CSV.

    When a date filter is applied, new rows are merged with the existing
    CSV (if present) and deduplicated by (ts, symbol) so incremental
    exports don't destroy previously exported data.

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
        Total number of rows in the final output.
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

    if output.exists() and (start or end):
        existing = pd.read_csv(str(output))
        df = pd.concat([existing, df], ignore_index=True)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
        df = df.drop_duplicates(subset=["ts", "symbol"], keep="last")
        df = df.sort_values("ts").reset_index(drop=True)

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
    """Export context_snapshots to CSV, merging with any existing export.

    When a date filter is applied, new rows are merged with the existing
    CSV (if present) and deduplicated by timestamp so incremental exports
    don't destroy previously exported data.

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
        Total number of rows in the final output.
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
        f"SELECT ts, underlying, spx_price, spy_price, vix, vix9d, "
        f"term_structure, vvix, skew, gex_net, zero_gamma_level "
        f"FROM context_snapshots WHERE {where} ORDER BY ts"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if output.exists() and (start or end):
        existing = pd.read_csv(str(output))
        df = pd.concat([existing, df], ignore_index=True)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
        df = df.drop_duplicates(subset=["ts"], keep="last")
        df = df.sort_values("ts").reset_index(drop=True)

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


def export_chain_data(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = PRODUCTION_CHAINS_DIR,
) -> tuple[int, int]:
    """Export chain_snapshots + option_chain_rows as per-day Parquet files.

    Joins the two tables and writes one Parquet file per trading day to
    ``output_dir/{YYYYMMDD}.parquet``.  Each file contains the columns
    needed by the offline training pipeline's production adapters.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output_dir:
        Destination directory for per-day Parquet files.

    Returns
    -------
    tuple[int, int]
        (number of days exported, total rows across all days).
    """
    clauses = ["cs.underlying = 'SPX'"]
    params: dict = {}
    if start:
        clauses.append("cs.ts >= :start")
        params["start"] = start
    if end:
        clauses.append("cs.ts < :end")
        params["end"] = end

    where = " AND ".join(clauses)
    query = text(
        f"SELECT cs.ts, ocr.option_symbol, ocr.expiration, ocr.strike, "
        f"ocr.option_right, ocr.bid, ocr.ask, ocr.open_interest, "
        f"ocr.delta, ocr.gamma "
        f"FROM chain_snapshots cs "
        f"JOIN option_chain_rows ocr ON cs.snapshot_id = ocr.snapshot_id "
        f"WHERE {where} "
        f"ORDER BY cs.ts, ocr.option_symbol"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return 0, 0

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["day"] = df["ts"].dt.strftime("%Y%m%d")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    days_exported = 0

    for day_str, day_df in df.groupby("day"):
        out_path = output_dir / f"{day_str}.parquet"
        day_df.drop(columns=["day"]).to_parquet(out_path, index=False)
        total_rows += len(day_df)
        days_exported += 1

    return days_exported, total_rows


def export_underlying_parquet(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = PRODUCTION_UNDERLYING_DIR,
) -> dict[str, int]:
    """Export underlying_quotes as per-symbol Parquet files.

    Writes one Parquet file per symbol to ``output_dir/{SYMBOL}_1min.parquet``
    with columns ``[ts, close]``, matching the FRD parquet format so the
    training pipeline's ``load_frd_quotes`` can read them directly.

    When a date filter is applied, new rows are merged with the existing
    parquet (if present) and deduplicated by timestamp so incremental
    exports don't destroy previously exported data.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output_dir:
        Destination directory for per-symbol Parquet files.

    Returns
    -------
    dict[str, int]
        Mapping of symbol name to total number of rows in each output file.
    """
    symbols = ("SPX", "SPY", "VIX", "VIX9D", "VVIX", "SKEW")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for symbol in symbols:
        clauses = [f"symbol = '{symbol}'"]
        params: dict = {}
        if start:
            clauses.append("ts >= :start")
            params["start"] = start
        if end:
            clauses.append("ts < :end")
            params["end"] = end

        where = " AND ".join(clauses)
        query = text(
            f"SELECT ts, last AS close FROM underlying_quotes "
            f"WHERE {where} ORDER BY ts"
        )

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            results[symbol] = 0
            continue

        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        out_path = output_dir / f"{symbol}_1min.parquet"
        if out_path.exists() and (start or end):
            existing = pd.read_parquet(out_path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["ts"], keep="last")
            df = df.sort_values("ts").reset_index(drop=True)

        df.to_parquet(out_path, index=False)
        results[symbol] = len(df)

    return results


def export_economic_calendar_merged(
    engine,
    *,
    output: Path = ECONOMIC_CALENDAR_CSV,
) -> int:
    """Export economic_events and merge with existing economic_calendar.csv.

    Reads the production ``economic_events`` table and the existing
    ``economic_calendar.csv`` (historical events), deduplicates, and writes
    the merged result back.  This ensures the training pipeline always has
    a complete calendar covering both historical and production periods.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    output:
        Destination CSV path (typically ``data/economic_calendar.csv``).

    Returns
    -------
    int
        Total number of rows in the merged output.
    """
    query = text(
        "SELECT date, event_type, has_projections, is_triple_witching "
        "FROM economic_events ORDER BY date"
    )
    with engine.connect() as conn:
        prod_df = pd.read_sql(query, conn)

    # Load existing historical calendar if present
    if output.exists():
        hist_df = pd.read_csv(str(output))
    else:
        hist_df = pd.DataFrame(columns=["date", "event_type", "has_projections", "is_triple_witching"])

    merged = pd.concat([hist_df, prod_df], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates(subset=["date", "event_type"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return len(merged)


def main() -> None:
    """CLI entry point for production data export."""
    parser = argparse.ArgumentParser(
        description="Export production DB tables to CSV/Parquet for offline training."
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
        choices=[
            "all", "underlying_quotes", "context_snapshots",
            "economic_events",
            "chains", "underlying_parquet", "calendar_merge",
        ],
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

    if args.tables in ("all", "chains"):
        days, rows = export_chain_data(engine, start=args.start, end=args.end)
        print(f"  chains: {rows} rows across {days} days -> {PRODUCTION_CHAINS_DIR}/")

    if args.tables in ("all", "underlying_parquet"):
        sym_counts = export_underlying_parquet(engine, start=args.start, end=args.end)
        for sym, n in sym_counts.items():
            if n > 0:
                print(f"  underlying_parquet/{sym}: {n} rows -> {PRODUCTION_UNDERLYING_DIR}/{sym}_1min.parquet")

    if args.tables in ("all", "calendar_merge"):
        n = export_economic_calendar_merged(engine)
        print(f"  economic_calendar (merged): {n} rows -> {ECONOMIC_CALENDAR_CSV}")

    print("=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
