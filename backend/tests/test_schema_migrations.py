"""Unit tests for schema_migrations version tracking in schema.py.

Validates that _run_migrations() seeds existing migrations on first run,
skips already-applied migrations on subsequent runs, and executes + records
new migration files.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import spx_backend.database.schema as schema_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_migration_dir(tmp_path: Path, names: list[str]) -> Path:
    """Create a temp migrations directory with empty .sql stub files."""
    mig_dir = tmp_path / "sql" / "migrations"
    mig_dir.mkdir(parents=True)
    for name in names:
        (mig_dir / f"{name}.sql").write_text(f"-- stub {name}\nSELECT 1;")
    return mig_dir


class _FakeResult:
    """Minimal stand-in for an asyncpg result set."""

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeConn:
    """In-memory fake async DB connection that tracks exec_driver_sql calls.

    Maintains a ``schema_migrations`` dict keyed by version to simulate the
    tracking table.
    """

    def __init__(self, migrations: dict[str, bool] | None = None) -> None:
        self.migrations: dict[str, bool] = migrations if migrations is not None else {}
        self.executed: list[str] = []

    async def exec_driver_sql(self, sql: str, params: tuple | None = None) -> _FakeResult:
        self.executed.append(sql)
        if "SELECT version FROM schema_migrations" in sql:
            return _FakeResult([(v,) for v in self.migrations])
        if "INSERT INTO schema_migrations" in sql and params:
            version = params[0]
            self.migrations[version] = True
        return _FakeResult([])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeEngine:
    """Minimal async engine wrapper around _FakeConn."""

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def begin(self):
        return self._conn

    def connect(self):
        return self._conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_run_seeds_all_existing_migrations(tmp_path: Path) -> None:
    """When schema_migrations is empty, all migration files should be seeded
    as already-applied and none of them should be actually executed."""
    mig_dir = _fake_migration_dir(
        tmp_path, ["001_alpha", "002_beta", "003_gamma"]
    )
    conn = _FakeConn(migrations={})
    fake_engine = _FakeEngine(conn)

    with (
        patch.object(schema_mod, "engine", fake_engine),
        patch.object(schema_mod, "_migrations_dir", return_value=mig_dir),
        patch.object(schema_mod, "_execute_sql_file", new_callable=AsyncMock) as mock_exec,
    ):
        await schema_mod._run_migrations()

    assert conn.migrations == {
        "001_alpha": True,
        "002_beta": True,
        "003_gamma": True,
    }
    mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_skips_already_applied_migrations(tmp_path: Path) -> None:
    """Migrations already in schema_migrations should be skipped entirely."""
    mig_dir = _fake_migration_dir(
        tmp_path, ["001_alpha", "002_beta", "003_gamma"]
    )
    conn = _FakeConn(migrations={
        "001_alpha": True,
        "002_beta": True,
        "003_gamma": True,
    })
    fake_engine = _FakeEngine(conn)

    with (
        patch.object(schema_mod, "engine", fake_engine),
        patch.object(schema_mod, "_migrations_dir", return_value=mig_dir),
        patch.object(schema_mod, "_execute_sql_file", new_callable=AsyncMock) as mock_exec,
    ):
        await schema_mod._run_migrations()

    mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_runs_only_new_migration(tmp_path: Path) -> None:
    """Only migration files not yet in schema_migrations should be executed."""
    mig_dir = _fake_migration_dir(
        tmp_path, ["001_alpha", "002_beta", "003_gamma"]
    )
    new_migration_path = mig_dir / "003_gamma.sql"

    conn = _FakeConn(migrations={
        "001_alpha": True,
        "002_beta": True,
    })
    fake_engine = _FakeEngine(conn)

    with (
        patch.object(schema_mod, "engine", fake_engine),
        patch.object(schema_mod, "_migrations_dir", return_value=mig_dir),
        patch.object(schema_mod, "_execute_sql_file", new_callable=AsyncMock) as mock_exec,
    ):
        await schema_mod._run_migrations()

    mock_exec.assert_called_once_with(new_migration_path)
    assert "003_gamma" in conn.migrations


@pytest.mark.asyncio
async def test_no_migration_dir_is_noop(tmp_path: Path) -> None:
    """If the migrations directory doesn't exist, nothing should happen."""
    missing_dir = tmp_path / "nonexistent"

    with (
        patch.object(schema_mod, "_migrations_dir", return_value=missing_dir),
        patch.object(schema_mod, "_ensure_migration_table", new_callable=AsyncMock) as mock_ensure,
    ):
        await schema_mod._run_migrations()

    mock_ensure.assert_not_called()


@pytest.mark.asyncio
async def test_empty_migrations_dir_is_noop(tmp_path: Path) -> None:
    """An empty migrations directory should seed nothing and run nothing."""
    mig_dir = tmp_path / "sql" / "migrations"
    mig_dir.mkdir(parents=True)

    conn = _FakeConn(migrations={})
    fake_engine = _FakeEngine(conn)

    with (
        patch.object(schema_mod, "engine", fake_engine),
        patch.object(schema_mod, "_migrations_dir", return_value=mig_dir),
        patch.object(schema_mod, "_execute_sql_file", new_callable=AsyncMock) as mock_exec,
    ):
        await schema_mod._run_migrations()

    mock_exec.assert_not_called()
    assert conn.migrations == {}
