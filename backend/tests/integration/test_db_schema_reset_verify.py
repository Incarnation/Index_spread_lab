"""
Integration tests for DB schema reset and post-reset table count verification.

Verifies that reset_all_tables() and verify_table_counts() behave correctly
against the test database.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from spx_backend.database import schema


async def _execute_sql_file(engine, path: Path) -> None:
    """Execute SQL file statement-by-statement for asyncpg compatibility."""
    sql = path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.exec_driver_sql(stmt)


async def _insert_cboe_rows_for_dte_cleanup(engine) -> None:
    """Seed CBOE snapshots spanning trading-slot DTE 0..12 with stale labels.

    This fixture data validates that migration 004 recomputes CBOE DTE fields
    using trading-slot semantics and prunes rows outside 0..10.
    """
    batch_ts = datetime(2026, 2, 20, 20, 59, 40, tzinfo=timezone.utc)
    as_of = date(2026, 2, 20)
    expirations = [as_of + timedelta(days=offset) for offset in range(13)]
    async with engine.begin() as conn:
        for idx, expiration in enumerate(expirations):
            snapshot_id = (
                await conn.execute(
                    text(
                        """
                        INSERT INTO chain_snapshots (ts, underlying, source, target_dte, expiration, payload_json, checksum)
                        VALUES (:ts, :underlying, :source, :target_dte, :expiration, CAST(:payload_json AS jsonb), :checksum)
                        RETURNING snapshot_id
                        """
                    ),
                    {
                        "ts": batch_ts,
                        "underlying": "SPX",
                        "source": "CBOE",
                        "target_dte": 999,
                        "expiration": expiration,
                        "payload_json": "{}",
                        "checksum": f"cboe_cleanup_seed_{idx}",
                    },
                )
            ).scalar_one()
            await conn.execute(
                text(
                    """
                    INSERT INTO gex_snapshots (
                      snapshot_id, ts, underlying, source, spot_price, gex_net, gex_calls, gex_puts, gex_abs, zero_gamma_level, method
                    )
                    VALUES (
                      :snapshot_id, :ts, :underlying, :source, :spot_price, :gex_net, :gex_calls, :gex_puts, :gex_abs, :zero_gamma_level, :method
                    )
                    """
                ),
                {
                    "snapshot_id": snapshot_id,
                    "ts": batch_ts,
                    "underlying": "SPX",
                    "source": "CBOE",
                    "spot_price": 6000.0,
                    "gex_net": 10.0,
                    "gex_calls": 12.0,
                    "gex_puts": -2.0,
                    "gex_abs": 14.0,
                    "zero_gamma_level": 6000.0,
                    "method": "cboe_seed",
                },
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO gex_by_expiry_strike (
                      snapshot_id, expiration, dte_days, strike, gex_net, gex_calls, gex_puts, oi_total, method
                    )
                    VALUES (
                      :snapshot_id, :expiration, :dte_days, :strike, :gex_net, :gex_calls, :gex_puts, :oi_total, :method
                    )
                    """
                ),
                {
                    "snapshot_id": snapshot_id,
                    "expiration": expiration,
                    "dte_days": 999,
                    "strike": float(6000 + idx),
                    "gex_net": 10.0,
                    "gex_calls": 12.0,
                    "gex_puts": -2.0,
                    "oi_total": 100,
                    "method": "cboe_seed",
                },
            )


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


@pytest.mark.asyncio
async def test_migration_004_cboe_dte_cleanup_is_idempotent(database_url_test: str, fresh_test_engine) -> None:
    """Migration 004 should recompute CBOE DTE and prune out-of-range rows safely.

    The migration is expected to keep only DTE 0..10 CBOE snapshots and to be
    safe when executed multiple times.
    """
    sql_dir = Path(schema.__file__).resolve().parent / "sql"
    migration_path = sql_dir / "migrations" / "004_cboe_trading_slot_dte_cleanup.sql"
    await _insert_cboe_rows_for_dte_cleanup(fresh_test_engine)

    await _execute_sql_file(fresh_test_engine, migration_path)
    await _execute_sql_file(fresh_test_engine, migration_path)

    async with fresh_test_engine.connect() as conn:
        remaining_count = (
            await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM chain_snapshots
                    WHERE source = 'CBOE'
                    """
                )
            )
        ).scalar_one()
        dte_bounds = (
            await conn.execute(
                text(
                    """
                    SELECT MIN(target_dte) AS min_dte, MAX(target_dte) AS max_dte
                    FROM chain_snapshots
                    WHERE source = 'CBOE'
                    """
                )
            )
        ).one()
        distinct_dtes = [
            row.target_dte
            for row in (
                await conn.execute(
                    text(
                        """
                        SELECT DISTINCT target_dte
                        FROM chain_snapshots
                        WHERE source = 'CBOE'
                        ORDER BY target_dte
                        """
                    )
                )
            ).fetchall()
        ]
        mismatch_count = (
            await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM gex_by_expiry_strike gbes
                    JOIN chain_snapshots cs ON cs.snapshot_id = gbes.snapshot_id
                    WHERE cs.source = 'CBOE'
                      AND gbes.dte_days <> cs.target_dte
                    """
                )
            )
        ).scalar_one()

    assert remaining_count == 11
    assert dte_bounds.min_dte == 0
    assert dte_bounds.max_dte == 10
    assert distinct_dtes == list(range(11))
    assert mismatch_count == 0
