"""Export production DB tables to CSV for offline training pipeline.

Connects to the production PostgreSQL database using DATABASE_URL from
the .env file, exports ``underlying_quotes``, ``context_snapshots``,
``economic_events``, and ``trade_candidates`` to CSV files that the
offline training pipeline can load.

Usage:
    python scripts/export_production_data.py [--start DATE] [--end DATE]

Defaults to exporting all available data when --start/--end are omitted.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import json

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


_BACKEND = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND.parent / "data"

UNDERLYING_QUOTES_CSV = DATA_DIR / "underlying_quotes_export.csv"
CONTEXT_SNAPSHOTS_CSV = DATA_DIR / "context_snapshots_export.csv"
ECONOMIC_EVENTS_CSV = DATA_DIR / "economic_events_export.csv"
TRADE_CANDIDATES_CSV = DATA_DIR / "trade_candidates_export.csv"


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
        f"term_structure, vvix, skew, gex_net, zero_gamma_level "
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


def _flatten_candidate(row) -> dict:
    """Flatten a trade_candidates DB row into a training-compatible dict.

    Maps the nested ``candidate_json`` (with ``legs.short``, ``legs.long``,
    ``context``, label columns) into the flat column layout expected by the
    offline training CSV (``training_candidates.csv``).

    Parameters
    ----------
    row:
        A SQLAlchemy Row with ``ts``, ``candidate_json``, ``label_json``,
        ``realized_pnl``, ``hit_tp50_before_sl_or_expiry``, and
        ``hit_tp100_at_expiry``.

    Returns
    -------
    dict
        Flat dictionary matching the training CSV column schema.
    """
    cj = row.candidate_json if isinstance(row.candidate_json, dict) else json.loads(row.candidate_json)
    lj = row.label_json if isinstance(row.label_json, dict) else (json.loads(row.label_json) if row.label_json else {})

    legs = cj.get("legs") or {}
    short = legs.get("short") or {}
    long_leg = legs.get("long") or {}

    ctx = cj.get("context") or {}
    cboe = cj.get("cboe_context") or {}

    width = cj.get("width_points")
    credit = cj.get("entry_credit")
    credit_to_width = credit / width if credit and width and width > 0 else None
    max_loss = (width - credit) * (cj.get("contracts") or 1) * 100 if credit and width else None

    return {
        "entry_dt": str(row.ts),
        "day": str(row.ts.date()) if hasattr(row.ts, "date") else None,
        "dte_target": cj.get("target_dte"),
        "expiry": cj.get("expiration"),
        "spread_side": cj.get("spread_side"),
        "delta_target": cj.get("delta_target"),
        "short_instrument_id": None,
        "long_instrument_id": None,
        "short_symbol": short.get("symbol"),
        "long_symbol": long_leg.get("symbol"),
        "short_strike": short.get("strike"),
        "long_strike": long_leg.get("strike"),
        "short_bid": short.get("bid"),
        "short_ask": short.get("ask"),
        "short_mid": short.get("mid"),
        "short_delta": short.get("delta"),
        "short_iv": short.get("iv"),
        "long_bid": long_leg.get("bid"),
        "long_ask": long_leg.get("ask"),
        "long_mid": long_leg.get("mid"),
        "long_delta": long_leg.get("delta"),
        "long_iv": long_leg.get("iv"),
        "entry_credit": credit,
        "width_points": width,
        "max_loss": max_loss,
        "credit_to_width": credit_to_width,
        "spot": cj.get("spot") or cj.get("spx_price"),
        "spy_price": cj.get("spy_price"),
        "vix": cj.get("vix") or ctx.get("vix"),
        "vix9d": cj.get("vix9d") or ctx.get("vix9d"),
        "term_structure": cj.get("term_structure") or ctx.get("term_structure"),
        "vvix": cj.get("vvix") or ctx.get("vvix"),
        "skew": cj.get("skew") or ctx.get("skew"),
        "is_opex_day": cj.get("is_opex_day", False),
        "is_fomc_day": cj.get("is_fomc_day", False),
        "is_triple_witching": cj.get("is_triple_witching", False),
        "is_cpi_day": cj.get("is_cpi_day", False),
        "is_nfp_day": cj.get("is_nfp_day", False),
        "contracts": cj.get("contracts"),
        "offline_gex_net": cboe.get("expiry_gex_net") or ctx.get("gex_net"),
        "offline_zero_gamma": ctx.get("zero_gamma_level"),
        "resolved": True,
        "hit_tp50": row.hit_tp50_before_sl_or_expiry,
        "hit_tp100_at_expiry": row.hit_tp100_at_expiry,
        "realized_pnl": row.realized_pnl,
        "exit_reason": lj.get("exit_reason"),
    }


def export_trade_candidates(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = TRADE_CANDIDATES_CSV,
) -> int:
    """Export resolved trade_candidates to a flat CSV matching the training schema.

    Each row's ``candidate_json`` is flattened so the output is directly
    compatible with the offline training CSV (``training_candidates.csv``).

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
    clauses = ["label_status = 'resolved'"]
    params: dict = {}
    if start:
        clauses.append("ts >= :start")
        params["start"] = start
    if end:
        clauses.append("ts < :end")
        params["end"] = end

    where = " AND ".join(clauses)
    query = text(
        f"SELECT ts, candidate_json, label_json, "
        f"realized_pnl, hit_tp50_before_sl_or_expiry, hit_tp100_at_expiry "
        f"FROM trade_candidates WHERE {where} ORDER BY ts"
    )

    with engine.connect() as conn:
        rows = conn.execute(query, params).fetchall()

    flat = [_flatten_candidate(r) for r in rows]
    df = pd.DataFrame(flat)
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
        choices=["all", "underlying_quotes", "context_snapshots", "economic_events", "trade_candidates"],
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

    if args.tables in ("all", "trade_candidates"):
        n = export_trade_candidates(engine, start=args.start, end=args.end)
        print(f"  trade_candidates: {n} rows -> {TRADE_CANDIDATES_CSV}")

    print("=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
