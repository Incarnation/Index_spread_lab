from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _ensure_safe_test_database_url(url: str) -> None:
    """Fail fast when DATABASE_URL_TEST does not look like a local test database."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    db_name = parsed.path.lstrip("/").lower()
    if host not in {"localhost", "127.0.0.1"}:
        raise RuntimeError(
            f"Unsafe DATABASE_URL_TEST host '{host}'. Only localhost/127.0.0.1 are allowed for integration tests."
        )
    if "test" not in db_name:
        raise RuntimeError(
            f"Unsafe DATABASE_URL_TEST database '{db_name}'. Database name must include 'test'."
        )


async def _execute_sql_file(engine, path: Path) -> None:  # noqa: ANN001
    """Execute SQL file statement-by-statement for asyncpg compatibility."""
    sql = path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.exec_driver_sql(stmt)


@pytest.fixture(scope="session")
def database_url_test() -> str:
    """Load DATABASE_URL_TEST for DB-backed integration tests."""
    url = os.getenv("DATABASE_URL_TEST")
    if not url:
        pytest.skip("DATABASE_URL_TEST is not set; skipping DB-backed integration tests")
    _ensure_safe_test_database_url(url)
    return url


@pytest.fixture
async def integration_db_session(database_url_test: str) -> AsyncSession:
    """Yield DB session bound to an isolated, freshly reset test database."""
    engine = create_async_engine(database_url_test, pool_pre_ping=True, pool_size=2, max_overflow=2)
    sql_dir = Path(__file__).resolve().parents[2] / "spx_backend" / "database" / "sql"
    await _execute_sql_file(engine, sql_dir / "db_reset_all_tables.sql")
    await _execute_sql_file(engine, sql_dir / "db_schema.sql")
    mig_dir = sql_dir / "migrations"
    if mig_dir.exists():
        for path in sorted(mig_dir.glob("*.sql")):
            await _execute_sql_file(engine, path)

    session_factory = async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


@pytest.fixture
def admin_headers() -> dict[str, str]:
    """Return optional admin header used by integration request helpers."""
    return {"X-API-Key": "test-key"}
