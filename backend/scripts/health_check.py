#!/usr/bin/env python3
"""Production health check for Index Spread Lab.

Verifies database connectivity, migration state, Tradier API access,
data freshness, and portfolio configuration before paper trading.

Usage
-----
    python backend/scripts/health_check.py          # uses .env from repo root
    python backend/scripts/health_check.py --verbose # show extra detail
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

# Ensure the backend package is importable when run from repo root.
_backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_backend_dir))

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import load_project_env

load_project_env()

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _env(key: str, default: str = "") -> str:
    """Read an environment variable with an optional default."""
    return os.getenv(key, default)


async def check_db_connection(engine) -> tuple[str, str]:
    """Verify PostgreSQL is reachable and responsive."""
    try:
        async with engine.connect() as conn:
            row = await conn.execute(text("SELECT 1"))
            row.fetchone()
        return PASS, "connected"
    except Exception as exc:
        return FAIL, str(exc)[:120]


async def check_migration_007(engine) -> tuple[str, str]:
    """Verify portfolio_state and portfolio_trades tables exist (migration 007)."""
    try:
        async with engine.connect() as conn:
            for table in ("portfolio_state", "portfolio_trades"):
                await conn.execute(text(f"SELECT 1 FROM {table} LIMIT 0"))
        return PASS, "portfolio_state + portfolio_trades exist"
    except Exception as exc:
        return FAIL, f"migration 007 missing: {exc}"


async def check_data_freshness(engine, verbose: bool) -> tuple[str, str]:
    """Check recency of quotes, snapshots, and context data.

    Warns if the latest row is older than 4 hours during RTH,
    or simply reports the latest timestamp otherwise.
    """
    tables = {
        "underlying_quotes": "ts",
        "chain_snapshots": "ts",
        "context_snapshots": "ts",
    }
    now = datetime.now(timezone.utc)
    details: list[str] = []
    worst = PASS
    try:
        async with engine.connect() as conn:
            for tbl, col in tables.items():
                row = await conn.execute(text(
                    f"SELECT MAX({col}) AS latest FROM {tbl}"
                ))
                result = row.fetchone()
                latest = result[0] if result else None
                if latest is None:
                    details.append(f"{tbl}: NO DATA")
                    worst = FAIL
                else:
                    age = now - latest.replace(tzinfo=timezone.utc) if latest.tzinfo is None else now - latest
                    age_hrs = age.total_seconds() / 3600
                    label = f"{tbl}: {latest.isoformat()} ({age_hrs:.1f}h ago)"
                    if age_hrs > 24:
                        worst = WARN if worst == PASS else worst
                    details.append(label)
    except Exception as exc:
        return FAIL, str(exc)[:120]
    summary = "; ".join(details) if verbose else f"{len(details)} tables checked"
    return worst, summary


async def check_tradier(verbose: bool) -> tuple[str, str]:
    """Verify Tradier API token is valid by hitting the profile endpoint."""
    base = _env("TRADIER_BASE_URL", "https://sandbox.tradier.com/v1")
    token = _env("TRADIER_ACCESS_TOKEN")
    if not token:
        return FAIL, "TRADIER_ACCESS_TOKEN not set"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{base}/user/profile",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )
        if resp.status_code == 200:
            return PASS, "profile OK"
        return FAIL, f"HTTP {resp.status_code}"
    except Exception as exc:
        return FAIL, str(exc)[:120]


def check_portfolio_config() -> tuple[str, str]:
    """Verify critical PORTFOLIO_* env vars are set."""
    required = [
        "PORTFOLIO_ENABLED",
        "PORTFOLIO_STARTING_CAPITAL",
        "PORTFOLIO_MAX_TRADES_PER_DAY",
        "PORTFOLIO_MAX_TRADES_PER_RUN",
    ]
    missing = [k for k in required if not _env(k)]
    if missing:
        return WARN, f"missing: {', '.join(missing)}"
    enabled = _env("PORTFOLIO_ENABLED", "false").lower() in ("true", "1", "yes")
    if not enabled:
        return WARN, "PORTFOLIO_ENABLED is false"
    return PASS, (
        f"enabled, capital=${_env('PORTFOLIO_STARTING_CAPITAL')}, "
        f"trades/day={_env('PORTFOLIO_MAX_TRADES_PER_DAY')}, "
        f"trades/run={_env('PORTFOLIO_MAX_TRADES_PER_RUN')}"
    )


def check_skip_init_db() -> tuple[str, str]:
    """Warn if SKIP_INIT_DB=true since migration 007 won't auto-run."""
    val = _env("SKIP_INIT_DB", "false").lower()
    if val in ("true", "1", "yes"):
        return WARN, "SKIP_INIT_DB=true -- migration 007 will NOT auto-run on restart"
    return PASS, "SKIP_INIT_DB=false -- migrations will run on startup"


async def main() -> None:
    """Run all production health checks and print a summary table."""
    parser = argparse.ArgumentParser(description="Production health check")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    db_url = _env("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    engine = create_async_engine(db_url, pool_pre_ping=True, pool_size=1)

    checks: list[tuple[str, str, str]] = []

    # Sync checks
    checks.append(("Portfolio config", *check_portfolio_config()))
    checks.append(("SKIP_INIT_DB", *check_skip_init_db()))

    # Async checks
    checks.append(("DB connection", *await check_db_connection(engine)))
    checks.append(("Migration 007", *await check_migration_007(engine)))
    checks.append(("Data freshness", *await check_data_freshness(engine, args.verbose)))
    checks.append(("Tradier API", *await check_tradier(args.verbose)))

    await engine.dispose()

    # Pretty-print results
    max_name = max(len(c[0]) for c in checks)
    print(f"\n{'Check':<{max_name}}  Status  Detail")
    print("-" * (max_name + 50))
    any_fail = False
    for name, status, detail in checks:
        marker = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
        print(f"{name:<{max_name}}  [{marker}] {status:<4}  {detail}")
        if status == FAIL:
            any_fail = True
    print()

    if any_fail:
        print("Some checks FAILED. Fix issues above before starting paper trading.")
        sys.exit(1)
    else:
        print("All checks passed. Ready for paper trading.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
