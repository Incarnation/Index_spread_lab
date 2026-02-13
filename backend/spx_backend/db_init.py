from __future__ import annotations

from pathlib import Path

from spx_backend.db import engine


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
    schema_path = Path(__file__).with_name("db_schema.sql")
    await _execute_sql_file(schema_path)


async def reset_ml_tables() -> None:
    """Drop and recreate ML/decision/trade tables."""
    reset_path = Path(__file__).with_name("db_reset_ml_tables.sql")
    await _execute_sql_file(reset_path)
    await init_db()

