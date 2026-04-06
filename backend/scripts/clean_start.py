#!/usr/bin/env python3
"""One-time database cleanup for a fresh $20k portfolio start.

Truncates all trade, decision, ML pipeline, performance analytics, and
portfolio tables while preserving market data ingestion, auth, and
economic event tables.

Usage
-----
    python backend/scripts/clean_start.py              # dry-run (shows counts only)
    python backend/scripts/clean_start.py --execute    # actually truncate
    python backend/scripts/clean_start.py --execute -v # verbose output
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

_backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_backend_dir))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

TABLES_TO_TRUNCATE = [
    "portfolio_trades",
    "portfolio_state",
    "trade_performance_equity_curve",
    "trade_performance_breakdowns",
    "trade_performance_snapshots",
    "trade_marks",
    "trade_legs",
    "fills",
    "orders",
    "trades",
    "trade_decisions",
    "model_predictions",
    "strategy_recommendations",
    "trade_candidates",
    "feature_snapshots",
    "backtest_runs",
    "training_runs",
    "model_versions",
    "strategy_versions",
]

TABLES_TO_PRESERVE = [
    "underlying_quotes",
    "chain_snapshots",
    "option_chain_rows",
    "gex_snapshots",
    "gex_by_strike",
    "gex_by_expiry_strike",
    "context_snapshots",
    "option_instruments",
    "market_clock_audit",
    "users",
    "auth_audit_log",
    "economic_events",
]


async def _get_counts(engine, tables: list[str]) -> dict[str, int]:
    """Return row counts for each table.

    Parameters
    ----------
    engine : Async SQLAlchemy engine.
    tables : Table names to count.

    Returns
    -------
    Dict mapping table name to row count.
    """
    counts: dict[str, int] = {}
    async with engine.connect() as conn:
        for tbl in tables:
            try:
                row = await conn.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
                counts[tbl] = row.scalar_one()
            except Exception:
                counts[tbl] = -1
    return counts


def _print_counts(title: str, counts: dict[str, int]) -> None:
    """Pretty-print table row counts.

    Parameters
    ----------
    title : Section header to display.
    counts : Dict of table name to row count.
    """
    max_name = max(len(t) for t in counts) if counts else 20
    print(f"\n  {title}")
    print(f"  {'Table':<{max_name}}  Rows")
    print(f"  {'-' * (max_name + 12)}")
    for tbl, cnt in counts.items():
        label = str(cnt) if cnt >= 0 else "ERROR"
        print(f"  {tbl:<{max_name}}  {label:>8}")


async def main() -> None:
    """Run the clean-start process: audit, truncate, and verify."""
    parser = argparse.ArgumentParser(description="Clean start: truncate trade/ML/portfolio tables")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually execute the TRUNCATE. Without this flag, only a dry-run audit is shown.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("FATAL: DATABASE_URL not set")
        sys.exit(1)

    engine = create_async_engine(db_url, pool_pre_ping=True, pool_size=2)

    print("=" * 60)
    print("  Index Spread Lab -- Clean Start Utility")
    print("=" * 60)

    before_truncate = await _get_counts(engine, TABLES_TO_TRUNCATE)
    preserved = await _get_counts(engine, TABLES_TO_PRESERVE)

    _print_counts("TABLES TO TRUNCATE (before):", before_truncate)
    _print_counts("TABLES TO PRESERVE (unchanged):", preserved)

    total_rows = sum(v for v in before_truncate.values() if v > 0)
    print(f"\n  Total rows to delete: {total_rows:,}")

    if not args.execute:
        print("\n  DRY RUN -- no changes made. Pass --execute to truncate.\n")
        await engine.dispose()
        return

    table_list = ", ".join(TABLES_TO_TRUNCATE)
    sql = f"TRUNCATE {table_list} RESTART IDENTITY CASCADE"

    print(f"\n  Executing TRUNCATE on {len(TABLES_TO_TRUNCATE)} tables ...")
    async with engine.begin() as conn:
        await conn.execute(text(sql))
    print("  TRUNCATE complete.")

    after_truncate = await _get_counts(engine, TABLES_TO_TRUNCATE)
    preserved_after = await _get_counts(engine, TABLES_TO_PRESERVE)

    _print_counts("TRUNCATED TABLES (after):", after_truncate)

    all_zero = all(v == 0 for v in after_truncate.values())
    preserved_ok = all(
        preserved_after.get(t, -1) == preserved.get(t, -1)
        for t in TABLES_TO_PRESERVE
    )

    if all_zero:
        print("\n  [OK] All truncated tables are empty.")
    else:
        print("\n  [WARN] Some tables still have rows -- check above.")

    if preserved_ok:
        print("  [OK] Preserved tables are untouched.")
    else:
        print("  [WARN] Preserved table counts changed -- investigate!")
        if args.verbose:
            _print_counts("PRESERVED TABLES (after):", preserved_after)

    print("\n  Clean start complete. Portfolio will begin fresh at $20,000.\n")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
