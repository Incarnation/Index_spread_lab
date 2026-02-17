"""
Integration tests for DB schema reset and post-reset table count verification.

Verifies that reset_all_tables() and verify_table_counts() behave correctly
against the test database.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from spx_backend.database import schema


async def _execute_sql_file(engine, path: Path) -> None:
    """Execute SQL file statement-by-statement for asyncpg compatibility."""
    sql = path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.exec_driver_sql(stmt)


@pytest.fixture
async def fresh_test_engine(database_url_test: str):  # noqa: ANN201
    """Create engine for test DB and apply full reset + schema so all tables exist and are empty."""
    engine = create_async_engine(
        database_url_test, pool_pre_ping=True, pool_size=2, max_overflow=2
    )
    sql_dir = Path(schema.__file__).resolve().parent / "sql"
    await _execute_sql_file(engine, sql_dir / "db_reset_all_tables.sql")
    await _execute_sql_file(engine, sql_dir / "db_schema.sql")
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_verify_table_counts_returns_zero_for_all_after_reset(
    database_url_test: str, fresh_test_engine
):
    """
    After a full reset + init_db, verify_table_counts() should return 0 for every app table
    when run against that database.
    """
    import spx_backend.database.schema as schema_mod

    with patch.object(schema_mod, "engine", fresh_test_engine):
        counts = await schema.verify_table_counts(schema.ALL_APP_TABLES)

    assert list(counts.keys()) == schema.ALL_APP_TABLES
    for name, n in counts.items():
        assert n == 0, f"Table {name} should be empty after reset, got {n}"
