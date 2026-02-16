from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from httpx import ASGITransport, AsyncClient
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from spx_backend.config import settings
from spx_backend.web.routers import admin, public


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

    session_factory = async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


@pytest.fixture
def admin_headers(monkeypatch) -> dict[str, str]:
    """Enable admin API key auth in test app and return valid header."""
    monkeypatch.setattr(settings, "admin_api_key", "test-key")
    return {"X-API-Key": "test-key"}


@pytest.fixture
async def integration_client(integration_db_session: AsyncSession):
    """Build FastAPI test client with DB dependency overrides."""
    app = FastAPI()
    app.include_router(public.router)
    app.include_router(admin.router)

    async def _override_db():
        yield integration_db_session

    app.dependency_overrides[public.get_db_session] = _override_db
    app.dependency_overrides[admin.get_db_session] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


async def seed_core_records(session: AsyncSession) -> dict[str, int]:
    """Insert minimum viable rows for DB-backed API smoke tests."""
    snapshot_row = await session.execute(
        text(
            """
            INSERT INTO chain_snapshots (ts, underlying, target_dte, expiration, payload_json, checksum)
            VALUES (now(), 'SPX', 3, CURRENT_DATE + INTERVAL '3 day', '{}'::jsonb, 'seed_chk')
            RETURNING snapshot_id
            """
        )
    )
    snapshot_id = int(snapshot_row.scalar_one())

    await session.execute(
        text(
            """
            INSERT INTO option_chain_rows (
              snapshot_id, option_symbol, underlying, expiration, strike, option_right,
              bid, ask, last, volume, open_interest, delta, raw_json
            )
            VALUES (
              :snapshot_id, 'SPX_SEED_OPT', 'SPX', CURRENT_DATE + INTERVAL '3 day', 6000.0, 'P',
              1.0, 1.2, 1.1, 10, 100, -0.2, '{}'::jsonb
            )
            """
        ),
        {"snapshot_id": snapshot_id},
    )

    await session.execute(
        text(
            """
            INSERT INTO gex_snapshots (
              snapshot_id, ts, underlying, spot_price, gex_net, gex_calls, gex_puts, gex_abs, zero_gamma_level, method
            )
            VALUES (:snapshot_id, now(), 'SPX', 6000.0, 100.0, 150.0, -50.0, 200.0, 5980.0, 'oi_gamma_spot')
            """
        ),
        {"snapshot_id": snapshot_id},
    )

    decision_row = await session.execute(
        text(
            """
            INSERT INTO trade_decisions (
              ts, target_dte, entry_slot, delta_target, decision, reason, score, chain_snapshot_id,
              ruleset_version, decision_source, chosen_legs_json, strategy_params_json
            )
            VALUES (
              now(), 3, 10, 0.2, 'TRADE', NULL, 1.0, :snapshot_id,
              'rules_v1', 'rules', '{}'::jsonb, '{}'::jsonb
            )
            RETURNING decision_id
            """
        ),
        {"snapshot_id": snapshot_id},
    )
    decision_id = int(decision_row.scalar_one())
    await session.commit()
    return {"snapshot_id": snapshot_id, "decision_id": decision_id}


@pytest.fixture
def seed_core_records_fn():
    """Expose async seeding helper to integration tests."""
    return seed_core_records
