"""Unit tests for the migration runner's transaction-mode dispatch.

These tests cover the audit follow-up that taught
``_execute_sql_file`` to recognise a ``-- +migrate-no-transaction``
header marker and route those files through an autocommit code path
(required for ``CREATE INDEX CONCURRENTLY``-style DDL).  The tests are
DB-free: they monkeypatch ``engine.connect`` / ``engine.begin`` and
assert which path the runner walked.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from spx_backend.database import schema as schema_mod


class _RecordingConn:
    """Minimal async connection stub that records every executed statement."""

    def __init__(self, store: list[str]):
        self._store = store

    async def exec_driver_sql(self, stmt: str, *_args, **_kwargs):
        """Capture the statement text for later assertions."""
        self._store.append(stmt)

    async def execution_options(self, **kwargs):
        """Mirror SQLAlchemy's ``execution_options``: returns a derived conn.

        The autocommit path calls this with ``isolation_level='AUTOCOMMIT'``.
        We record the kwargs on ``self`` so the test can assert what was
        requested, then return ``self`` so subsequent ``exec_driver_sql``
        calls keep landing in the same store.
        """
        self.last_options = kwargs
        return self


class _StubEngine:
    """Minimal stand-in for the SQLAlchemy ``AsyncEngine`` used by the runner.

    Holds two per-call recorders:

    * ``begin_stmts`` -- statements seen via ``engine.begin()``
      (the transactional code path).
    * ``connect_stmts`` -- statements seen via ``engine.connect()``
      (the autocommit code path used for ``CREATE INDEX CONCURRENTLY``).

    The stub is required because SQLAlchemy's real ``AsyncEngine`` blocks
    attribute assignment, so we can't ``monkeypatch.setattr`` ``engine.begin``
    directly; instead we replace the entire ``schema.engine`` symbol.
    """

    def __init__(self):
        self.begin_stmts: list[str] = []
        self.connect_stmts: list[str] = []
        # Exposed so tests can inspect what ``execution_options`` was called
        # with (the autocommit path expects ``isolation_level='AUTOCOMMIT'``).
        self.connect_conn = _RecordingConn(self.connect_stmts)

    def begin(self):
        # Returns a fresh async-context-manager each call (mirrors real engine).
        outer = self

        @asynccontextmanager
        async def _cm():
            yield _RecordingConn(outer.begin_stmts)
        return _cm()

    def connect(self):
        outer = self

        @asynccontextmanager
        async def _cm():
            yield outer.connect_conn
        return _cm()


def _make_engine_stubs(monkeypatch: pytest.MonkeyPatch) -> _StubEngine:
    """Replace ``schema.engine`` with a recording stub for the test scope."""
    stub = _StubEngine()
    monkeypatch.setattr(schema_mod, "engine", stub)
    return stub


@pytest.fixture()
def tmp_sql(tmp_path: Path) -> Path:
    """Provide a fresh temp directory for synthesised migration SQL files."""
    return tmp_path


def test_marker_detected_at_top_of_file_uses_autocommit(monkeypatch, tmp_sql):
    """Files whose header carries the marker run via engine.connect/AUTOCOMMIT.

    Confirms the autocommit path executes the statements and that the
    transactional path is NOT used.  This is the contract migration 024
    relies on for ``CREATE INDEX CONCURRENTLY``.
    """
    sql_path = tmp_sql / "test_marker.sql"
    sql_path.write_text(
        "-- +migrate-no-transaction\n"
        "-- A diagnostic index that requires CONCURRENTLY.\n"
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_test ON foo (bar);\n"
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_test2 ON foo (baz);\n"
    )
    stub = _make_engine_stubs(monkeypatch)

    asyncio.run(schema_mod._execute_sql_file(sql_path))

    assert stub.begin_stmts == [], "transactional path must NOT be used for marked files"
    assert len(stub.connect_stmts) == 2, "both CREATE INDEX statements should land via connect()"
    assert all("CREATE INDEX CONCURRENTLY" in s for s in stub.connect_stmts)
    # Verify the autocommit handshake actually happened.
    assert getattr(stub.connect_conn, "last_options", None) == {"isolation_level": "AUTOCOMMIT"}


def test_no_marker_uses_transactional_path(monkeypatch, tmp_sql):
    """Files without the marker keep using engine.begin() (existing behaviour).

    This is the regression guard: every existing migration relies on the
    transactional code path, so the dispatch must not flip for a normal
    migration.
    """
    sql_path = tmp_sql / "test_no_marker.sql"
    sql_path.write_text(
        "-- A normal migration with no special directive.\n"
        "ALTER TABLE foo ADD COLUMN bar INT;\n"
        "ALTER TABLE foo ADD COLUMN baz TEXT;\n"
    )
    stub = _make_engine_stubs(monkeypatch)

    asyncio.run(schema_mod._execute_sql_file(sql_path))

    assert stub.connect_stmts == [], "autocommit path must NOT be used for unmarked files"
    assert len(stub.begin_stmts) == 2, "both ALTER TABLE statements should land via begin()"
    assert all("ALTER TABLE" in s for s in stub.begin_stmts)


def test_marker_only_recognized_in_header_lines(monkeypatch, tmp_sql):
    """A marker buried mid-file (past the header window) is ignored.

    Defends against a documentation comment that happens to contain the
    string ``+migrate-no-transaction`` accidentally flipping a
    transactional migration into autocommit mode and leaving partial
    DDL applied on failure.
    """
    sql_path = tmp_sql / "test_buried_marker.sql"
    # Pad the header so the marker is well past _NON_TRANSACTIONAL_HEADER_LINES.
    header_padding = "\n".join(["-- pad"] * 20)
    sql_path.write_text(
        f"-- This is a normal migration.\n"
        f"{header_padding}\n"
        f"-- documentation note: see +migrate-no-transaction in 024 for the autocommit path\n"
        f"ALTER TABLE foo ADD COLUMN bar INT;\n"
    )
    stub = _make_engine_stubs(monkeypatch)

    asyncio.run(schema_mod._execute_sql_file(sql_path))

    assert stub.connect_stmts == []
    assert stub.begin_stmts == ["ALTER TABLE foo ADD COLUMN bar INT"]


def test_has_non_transactional_marker_is_header_only():
    """Direct unit test for the marker-detection helper."""
    assert schema_mod._has_non_transactional_marker(
        "-- +migrate-no-transaction\nCREATE INDEX CONCURRENTLY ...;\n"
    )
    # Marker on line 2 is still inside the header window, so accepted.
    assert schema_mod._has_non_transactional_marker(
        "-- file header\n-- +migrate-no-transaction\nCREATE INDEX ...;\n"
    )
    # Marker buried far past the header window is ignored.
    body = "\n".join(["-- pad"] * 50) + "\n-- +migrate-no-transaction\n"
    assert not schema_mod._has_non_transactional_marker(body)
    # Empty / no marker.
    assert not schema_mod._has_non_transactional_marker("ALTER TABLE foo ADD COLUMN bar INT;")
