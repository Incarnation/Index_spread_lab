from __future__ import annotations

import re
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from spx_backend.database.connection import engine

# All application tables (order matches db_schema / reset scripts).
ALL_APP_TABLES = [
    "users",
    "auth_audit_log",
    "option_instruments",
    "chain_snapshots",
    "option_chain_rows",
    "gex_snapshots",
    "gex_by_strike",
    "gex_by_expiry_strike",
    "context_snapshots",
    "underlying_quotes",
    "market_clock_audit",
    "strategy_versions",
    "model_versions",
    "training_runs",
    "feature_snapshots",
    "trade_candidates",
    "model_predictions",
    "backtest_runs",
    "strategy_recommendations",
    "trade_decisions",
    "orders",
    "fills",
    "trades",
    "trade_legs",
    "trade_marks",
    "trade_performance_snapshots",
    "trade_performance_breakdowns",
    "trade_performance_equity_curve",
    "economic_events",
    "portfolio_state",
    "portfolio_trades",
    "schema_migrations",
]

# Tables dropped by ML-only reset (db_reset_ml_tables.sql).
ML_RESET_TABLES = [
    "strategy_versions",
    "model_versions",
    "training_runs",
    "feature_snapshots",
    "trade_candidates",
    "model_predictions",
    "backtest_runs",
    "strategy_recommendations",
    "trade_decisions",
    "orders",
    "fills",
    "trades",
    "trade_legs",
    "trade_marks",
    "trade_performance_snapshots",
    "trade_performance_breakdowns",
    "trade_performance_equity_curve",
]


def _sql_dir() -> Path:
    """
    Return SQL directory.

    SQL files live under `spx_backend/database/sql/`.
    """
    return Path(__file__).resolve().parent / "sql"


def _strip_sql_comments(sql: str) -> str:
    """Remove ``--`` line comments and ``/* */`` block comments from raw SQL.

    This prevents asyncpg from choking on comment-only fragments that survive
    the semicolon split in :func:`_execute_sql_file`.  String literals
    containing ``--`` are not used in our migration files so the simple regex
    approach is safe here.
    """
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", "", sql)
    return sql


async def _execute_sql_file(path: Path) -> None:
    """Execute SQL file statement-by-statement for asyncpg compatibility.

    Parameters
    ----------
    path:
        Filesystem path to a ``.sql`` file.  Comments are stripped before
        splitting on ``;`` so comment-only fragments never reach the driver.
    """
    raw_sql = path.read_text(encoding="utf-8")
    sql = _strip_sql_comments(raw_sql)
    async with engine.begin() as conn:
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                await conn.exec_driver_sql(stmt)
            except Exception:
                logger.error("sql_exec_failed file={} statement={!r}", path.name, stmt[:120])
                raise


def _migrations_dir() -> Path:
    """Return path to sql/migrations/."""
    return _sql_dir() / "migrations"


async def _ensure_migration_table() -> None:
    """Create the ``schema_migrations`` tracking table if it doesn't exist.

    Called before every migration run so the tracker is always available,
    even on databases created before this feature was added.
    """
    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  version TEXT PRIMARY KEY,"
            "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )


async def _applied_versions() -> set[str]:
    """Return the set of migration versions already recorded as applied."""
    async with engine.connect() as conn:
        result = await conn.exec_driver_sql(
            "SELECT version FROM schema_migrations"
        )
        return {row[0] for row in result}


async def _run_migrations() -> None:
    """Run only *pending* migration SQL files, recording each on success.

    On the very first run (empty ``schema_migrations`` table) all existing
    migration files are seeded as already-applied.  This is safe because
    production has already executed them, and on a fresh database the base
    schema DDL in ``db_schema.sql`` makes the data-modifying migrations
    no-ops (no rows to update/delete).
    """
    mig_dir = _migrations_dir()
    if not mig_dir.exists():
        return

    await _ensure_migration_table()
    applied = await _applied_versions()
    paths = sorted(mig_dir.glob("*.sql"))

    if not applied and paths:
        async with engine.begin() as conn:
            for p in paths:
                await conn.exec_driver_sql(
                    "INSERT INTO schema_migrations (version) VALUES ($1) "
                    "ON CONFLICT DO NOTHING",
                    (p.stem,),
                )
        logger.info("migration_seed count={}", len(paths))
        return

    for path in paths:
        version = path.stem
        if version in applied:
            logger.debug("migration_skip version={}", version)
            continue
        logger.info("migration_run version={}", version)
        await _execute_sql_file(path)
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                (version,),
            )
        logger.info("migration_done version={}", version)


async def init_db() -> None:
    """Initialize full schema (idempotent creates) and run migrations."""
    schema_path = _sql_dir() / "db_schema.sql"
    await _execute_sql_file(schema_path)
    await _run_migrations()


async def reset_ml_tables() -> None:
    """Drop and recreate ML/decision/trade tables."""
    reset_path = _sql_dir() / "db_reset_ml_tables.sql"
    await _execute_sql_file(reset_path)
    await init_db()


async def reset_all_tables() -> None:
    """Drop and recreate all application tables."""
    reset_path = _sql_dir() / "db_reset_all_tables.sql"
    await _execute_sql_file(reset_path)
    await init_db()


async def verify_table_counts(
    table_names: list[str] | None = None,
) -> dict[str, int]:
    """
    Query row counts for the given tables (or all app tables if None).
    Returns a dict of table_name -> count. Used after reset to confirm no leftover data.
    """
    tables = table_names if table_names is not None else ALL_APP_TABLES
    counts: dict[str, int] = {}
    async with engine.connect() as conn:
        for name in tables:
            # Table names are from our constants only (no user input).
            stmt = text("SELECT count(*) FROM " + name)
            result = await conn.execute(stmt)
            counts[name] = result.scalar_one()
    return counts

