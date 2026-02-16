from __future__ import annotations

from pathlib import Path

from spx_backend.database.connection import engine


def _sql_dir() -> Path:
    """
    Return SQL directory.

    SQL files live under `spx_backend/database/sql/`.
    """
    return Path(__file__).resolve().parent / "sql"


async def _execute_sql_file(path: Path) -> None:
    """Execute SQL file statement-by-statement for asyncpg compatibility."""
    sql = path.read_text(encoding="utf-8")
    async with engine.begin() as conn:
        # Execute statement-by-statement (asyncpg won't reliably accept multi-statement executes).
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            await conn.exec_driver_sql(stmt)


async def init_db() -> None:
    """Initialize full schema (idempotent creates)."""
    schema_path = _sql_dir() / "db_schema.sql"
    await _execute_sql_file(schema_path)


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

