"""Tests for the data retention job (retention_job.py).

Covers:
- Skipping when disabled
- Batch deletion with loop termination
- Safety exclusion of snapshots referenced by OPEN trades
- Per-batch transaction isolation
- L4/L5 (audit): post-purge ANALYZE + cascade approx-delta logging
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spx_backend.config import settings


# L4/L5 (audit) helpers: the production code now issues
#   1. four ``SELECT reltuples ... FROM pg_class`` rows (one per cascade
#      child) BEFORE the DELETE loop,
#   2. the DELETE loop itself,
#   3. four ``ANALYZE`` + ``SELECT reltuples`` round-trip pairs AFTER
#      the loop.
# The tests below only care about the DELETE rowcount behavior, so a
# small dispatcher returns a benign reltuples row for SELECT statements,
# a no-op result for ANALYZE, and routes DELETE calls to the per-test
# rowcount side-effect.


def _stat_row(approx: int = 0):
    """Return a minimal pg_class.reltuples-shaped row for SELECT mocks."""
    row = MagicMock()
    row.approx = approx
    return row


def _build_dispatching_execute(delete_rowcount_provider):
    """Build an execute mock that splits SELECT/ANALYZE/DELETE traffic.

    ``delete_rowcount_provider`` is called once per DELETE statement
    and must return the integer rowcount the production code should
    observe.
    """

    async def _execute(stmt, params=None):  # noqa: ANN001
        sql = str(stmt).strip().upper()
        if sql.startswith("SELECT") and "PG_CLASS" in sql:
            result = MagicMock()
            result.fetchone = MagicMock(return_value=_stat_row(0))
            return result
        if sql.startswith("ANALYZE"):
            return MagicMock()
        if sql.startswith("DELETE"):
            result = MagicMock()
            result.rowcount = delete_rowcount_provider()
            return result
        result = MagicMock()
        result.rowcount = 0
        result.fetchone = MagicMock(return_value=None)
        return result

    return _execute


@pytest.fixture(autouse=True)
def _pin_retention_settings(monkeypatch: pytest.MonkeyPatch):
    """Set retention config to known values for all tests."""
    monkeypatch.setattr(settings, "retention_enabled", False)
    monkeypatch.setattr(settings, "retention_days", 60)
    monkeypatch.setattr(settings, "retention_batch_size", 500)


def _make_engine_mock():
    """Build a mock engine whose .begin() returns an async context manager.

    Returns (engine_mock, begin_conn_mock) so callers can configure
    ``begin_conn_mock.execute`` with custom side effects.
    """
    engine_mock = MagicMock()
    begin_conn = AsyncMock()
    begin_ctx = AsyncMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=begin_conn)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    engine_mock.begin.return_value = begin_ctx
    return engine_mock, begin_conn


@pytest.mark.asyncio
async def test_run_once_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_once returns skipped=True when retention_enabled is False."""
    from spx_backend.jobs import retention_job

    result = await retention_job.run_once()
    assert result["skipped"] is True
    assert result["reason"] == "disabled"


@pytest.mark.asyncio
async def test_run_once_force_overrides_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """force=True bypasses the retention_enabled guard."""
    engine_mock, begin_conn = _make_engine_mock()

    begin_conn.execute = AsyncMock(side_effect=_build_dispatching_execute(lambda: 0))

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once(force=True)

    assert result["skipped"] is False
    assert result["deleted_snapshots"] == 0
    # L4/L5 (audit): cascade telemetry payload always populated.
    assert "cascade_pre_counts" in result
    assert "cascade_post_counts" in result
    assert "cascade_approx_deltas" in result


@pytest.mark.asyncio
async def test_batch_loop_terminates_when_fewer_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a batch deletes fewer rows than batch_size, the loop exits."""
    monkeypatch.setattr(settings, "retention_enabled", True)
    monkeypatch.setattr(settings, "retention_batch_size", 100)

    engine_mock, begin_conn = _make_engine_mock()

    delete_call_count = {"n": 0}

    def delete_rowcount_provider() -> int:
        delete_call_count["n"] += 1
        return 42 if delete_call_count["n"] == 1 else 0

    begin_conn.execute = AsyncMock(
        side_effect=_build_dispatching_execute(delete_rowcount_provider)
    )

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once()

    assert result["deleted_snapshots"] == 42
    # L4/L5 (audit): the small first batch (42 < batch_size=100) means
    # exactly one DELETE round trip should fire.
    assert delete_call_count["n"] == 1


@pytest.mark.asyncio
async def test_multi_batch_accumulates_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple full batches followed by a partial batch sums correctly."""
    monkeypatch.setattr(settings, "retention_enabled", True)
    monkeypatch.setattr(settings, "retention_batch_size", 100)

    engine_mock, begin_conn = _make_engine_mock()

    batches = iter([100, 100, 37])

    def delete_rowcount_provider() -> int:
        return next(batches, 0)

    begin_conn.execute = AsyncMock(
        side_effect=_build_dispatching_execute(delete_rowcount_provider)
    )

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once()

    assert result["deleted_snapshots"] == 237


@pytest.mark.asyncio
async def test_delete_query_excludes_open_trade_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DELETE SQL should contain a NOT IN clause referencing OPEN trades."""
    monkeypatch.setattr(settings, "retention_enabled", True)

    engine_mock, begin_conn = _make_engine_mock()

    captured_delete_sql: list[str] = []

    async def execute_side_effect(stmt, params=None):  # noqa: ANN001
        sql = str(stmt)
        upper = sql.strip().upper()
        if upper.startswith("SELECT") and "PG_CLASS" in upper:
            result = MagicMock()
            result.fetchone = MagicMock(return_value=_stat_row(0))
            return result
        if upper.startswith("ANALYZE"):
            return MagicMock()
        if upper.startswith("DELETE"):
            captured_delete_sql.append(sql)
        result = MagicMock()
        result.rowcount = 0
        return result

    begin_conn.execute = AsyncMock(side_effect=execute_side_effect)

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        await retention_job.run_once()

    assert len(captured_delete_sql) == 1
    sql = captured_delete_sql[0]
    assert "NOT IN" in sql
    assert "OPEN" in sql
    assert "entry_snapshot_id" in sql
    assert "last_snapshot_id" in sql


@pytest.mark.asyncio
async def test_each_batch_gets_own_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """engine.begin() should be called once per DELETE batch, plus the
    L4/L5 (audit) pre-count and post-count transactions (one each).
    """
    monkeypatch.setattr(settings, "retention_enabled", True)
    monkeypatch.setattr(settings, "retention_batch_size", 100)

    engine_mock = MagicMock()
    begin_call_count = {"n": 0}
    delete_calls = {"n": 0}

    def make_begin_ctx():
        begin_call_count["n"] += 1

        async def execute_side_effect(stmt, params=None):  # noqa: ANN001
            upper = str(stmt).strip().upper()
            if upper.startswith("SELECT") and "PG_CLASS" in upper:
                result = MagicMock()
                result.fetchone = MagicMock(return_value=_stat_row(0))
                return result
            if upper.startswith("ANALYZE"):
                return MagicMock()
            if upper.startswith("DELETE"):
                delete_calls["n"] += 1
                result = MagicMock()
                result.rowcount = 100 if delete_calls["n"] <= 2 else 15
                return result
            result = MagicMock()
            result.rowcount = 0
            return result

        conn = AsyncMock()
        conn.execute = AsyncMock(side_effect=execute_side_effect)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    engine_mock.begin = MagicMock(side_effect=make_begin_ctx)

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once()

    assert result["deleted_snapshots"] == 215
    # 1 pre-count tx + 3 DELETE batches + 1 post-count/ANALYZE tx = 5.
    assert engine_mock.begin.call_count == 5
