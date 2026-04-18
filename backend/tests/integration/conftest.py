from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from spx_backend.database.schema import _strip_sql_comments


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
    """Execute SQL file statement-by-statement for asyncpg compatibility.

    Mirrors :func:`spx_backend.database.schema._execute_sql_file` so that
    test DB setup tokenizes migrations identically to production.  In
    particular, ``--`` and ``/* */`` comments are stripped before splitting
    on ``;`` so an embedded ``;`` inside a comment cannot truncate a
    statement.  The production helper binds to the module-level engine,
    which is unsuitable here because each integration-test fixture builds
    its own per-test engine; hence the parameter rather than re-export.
    """
    raw = path.read_text(encoding="utf-8")
    sql = _strip_sql_comments(raw)
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.exec_driver_sql(stmt)


async def _seed_migrations_as_applied(engine, mig_dir: Path) -> None:  # noqa: ANN001
    """Record every migration in ``mig_dir`` as already-applied without running it.

    Mirrors the fresh-DB branch of
    :func:`spx_backend.database.schema._run_migrations` (the ``if not applied
    and paths`` block).  ``db_schema.sql`` is kept in sync with the
    post-migration schema, so re-executing historical DDL on a fresh DB
    would error rather than no-op -- e.g. migration 008 ALTERs
    ``feature_snapshots`` and migration 011 ALTERs ``trade_candidates``,
    both of which were dropped by migration 015 and are no longer created
    by ``db_schema.sql``.

    The seed path therefore records intent ("treat these as done"), exactly
    matching what production does the first time it boots against an empty
    ``schema_migrations`` table.
    """
    paths = sorted(mig_dir.glob("*.sql"))
    if not paths:
        return
    async with engine.begin() as conn:
        # db_schema.sql already creates schema_migrations; clear any rows that
        # survived a prior test run on the same database before reseeding.
        await conn.exec_driver_sql("DELETE FROM schema_migrations")
        for p in paths:
            await conn.exec_driver_sql(
                "INSERT INTO schema_migrations (version) VALUES ($1) "
                "ON CONFLICT DO NOTHING",
                (p.stem,),
            )


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
        # Mirror production's fresh-DB behaviour: db_schema.sql is the
        # post-migration source of truth, so historical DDL files are
        # recorded as applied rather than re-executed.
        await _seed_migrations_as_applied(engine, mig_dir)

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
