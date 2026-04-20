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
    "alert_cooldowns",
    "portfolio_state",
    "portfolio_trades",
    "optimizer_runs",
    "optimizer_results",
    "optimizer_walkforward",
    "schema_migrations",
]
# Note: the online-ML tables (`training_runs`, `feature_snapshots`,
# `trade_candidates`, `model_predictions`) were dropped by migration 015
# (Track A.7).  `model_versions` is retained for offline ML re-entry.

# Tables dropped by ML-only reset (db_reset_ml_tables.sql).
ML_RESET_TABLES = [
    "strategy_versions",
    "model_versions",
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


# Header marker that opts a migration into a non-transactional autocommit
# code path.  Required for DDL statements like ``CREATE INDEX CONCURRENTLY``
# which PostgreSQL forbids inside a transaction block.  The marker MUST
# appear within the first ``_NON_TRANSACTIONAL_HEADER_LINES`` lines of the
# SQL file (i.e. as a top-of-file directive); markers buried mid-file are
# ignored so a stray reference inside a comment block does not flip the
# whole runner.
_NON_TRANSACTIONAL_MARKER = "+migrate-no-transaction"
_NON_TRANSACTIONAL_HEADER_LINES = 5


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


def _has_non_transactional_marker(raw_sql: str) -> bool:
    """Return True iff the SQL file opts into the autocommit runner path.

    The marker is recognised only in the first
    :data:`_NON_TRANSACTIONAL_HEADER_LINES` lines so a stray mention of the
    string inside a documentation comment block further down cannot flip
    the runner mode for the whole file.
    """
    header = raw_sql.splitlines()[:_NON_TRANSACTIONAL_HEADER_LINES]
    return any(_NON_TRANSACTIONAL_MARKER in line for line in header)


async def _execute_sql_file(path: Path) -> None:
    """Execute SQL file statement-by-statement for asyncpg compatibility.

    Parameters
    ----------
    path:
        Filesystem path to a ``.sql`` file.  Comments are stripped before
        splitting on ``;`` so comment-only fragments never reach the driver.

    Notes
    -----
    Most migrations run inside a single ``engine.begin()`` transaction so
    a partial failure rolls the file back as one unit.  Files whose
    *header* (first :data:`_NON_TRANSACTIONAL_HEADER_LINES` lines) contains
    the :data:`_NON_TRANSACTIONAL_MARKER` directive are routed through an
    autocommit connection instead.  This is required for statements like
    ``CREATE INDEX CONCURRENTLY`` that PostgreSQL refuses to run inside a
    transaction (see migration 024 for an example).  In that mode each
    statement commits independently, so a failure leaves any earlier
    statements applied -- the caller (``_run_migrations``) records the
    file as applied only after the whole body succeeds.
    """
    raw_sql = path.read_text(encoding="utf-8")
    sql = _strip_sql_comments(raw_sql)
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if _has_non_transactional_marker(raw_sql):
        # Autocommit mode: open a single connection, set isolation level
        # to AUTOCOMMIT, execute each statement standalone.  No
        # ``engine.begin()`` wrapper because that would start a tx and
        # invalidate ``CREATE INDEX CONCURRENTLY``.
        logger.info("sql_exec_autocommit file={} statements={}", path.name, len(statements))
        async with engine.connect() as conn:
            ac_conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
            for stmt in statements:
                try:
                    await ac_conn.exec_driver_sql(stmt)
                except Exception:
                    logger.error(
                        "sql_exec_failed file={} statement={!r} (autocommit mode)",
                        path.name, stmt[:120],
                    )
                    raise
        return
    async with engine.begin() as conn:
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

    Behaviour matrix:

    * Production / populated DB: each numbered ``.sql`` file under
      ``database/sql/migrations/`` is executed exactly once and recorded
      in ``schema_migrations``.  Already-applied versions are skipped.
    * Fresh DB built from ``db_schema.sql`` (empty ``schema_migrations``):
      all numbered files are **seeded as already-applied without being
      executed**.  This is required because ``db_schema.sql`` is kept
      in sync with the post-migration schema, so re-running historical
      DDL migrations (e.g. ``ALTER TABLE`` on a column that the base
      schema already defines, or DROP statements like migration 015
      that target tables no longer present) would error rather than
      no-op.  The seed path therefore records intent ("treat these as
      done") rather than relying on idempotent SQL.
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

