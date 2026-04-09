"""Tests for the data retention job (retention_job.py).

Covers:
- Skipping when disabled
- Batch deletion with loop termination
- Safety exclusion of snapshots referenced by OPEN trades
- Per-batch transaction isolation
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spx_backend.config import settings


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

    result_mock = MagicMock()
    result_mock.rowcount = 0
    begin_conn.execute = AsyncMock(return_value=result_mock)

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once(force=True)

    assert result["skipped"] is False
    assert result["deleted_snapshots"] == 0


@pytest.mark.asyncio
async def test_batch_loop_terminates_when_fewer_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a batch deletes fewer rows than batch_size, the loop exits."""
    monkeypatch.setattr(settings, "retention_enabled", True)
    monkeypatch.setattr(settings, "retention_batch_size", 100)

    engine_mock, begin_conn = _make_engine_mock()

    call_count = {"n": 0}

    async def execute_side_effect(*args, **kwargs):
        call_count["n"] += 1
        result = MagicMock()
        result.rowcount = 42 if call_count["n"] == 1 else 0
        return result

    begin_conn.execute = AsyncMock(side_effect=execute_side_effect)

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once()

    assert result["deleted_snapshots"] == 42
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_multi_batch_accumulates_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple full batches followed by a partial batch sums correctly."""
    monkeypatch.setattr(settings, "retention_enabled", True)
    monkeypatch.setattr(settings, "retention_batch_size", 100)

    engine_mock, begin_conn = _make_engine_mock()

    batches = iter([100, 100, 37])

    async def execute_side_effect(*args, **kwargs):
        result = MagicMock()
        result.rowcount = next(batches, 0)
        return result

    begin_conn.execute = AsyncMock(side_effect=execute_side_effect)

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once()

    assert result["deleted_snapshots"] == 237


@pytest.mark.asyncio
async def test_delete_query_excludes_open_trade_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DELETE SQL should contain a NOT IN clause referencing OPEN trades."""
    monkeypatch.setattr(settings, "retention_enabled", True)

    engine_mock, begin_conn = _make_engine_mock()

    captured_sql: list[str] = []

    async def execute_side_effect(stmt, params=None):
        captured_sql.append(str(stmt))
        result = MagicMock()
        result.rowcount = 0
        return result

    begin_conn.execute = AsyncMock(side_effect=execute_side_effect)

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        await retention_job.run_once()

    assert len(captured_sql) == 1
    sql = captured_sql[0]
    assert "NOT IN" in sql
    assert "OPEN" in sql
    assert "entry_snapshot_id" in sql
    assert "last_snapshot_id" in sql


@pytest.mark.asyncio
async def test_each_batch_gets_own_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """engine.begin() should be called once per batch, not once for all."""
    monkeypatch.setattr(settings, "retention_enabled", True)
    monkeypatch.setattr(settings, "retention_batch_size", 100)

    engine_mock = MagicMock()
    begin_call_count = {"n": 0}

    def make_begin_ctx():
        begin_call_count["n"] += 1
        n = begin_call_count["n"]
        conn = AsyncMock()
        result = MagicMock()
        result.rowcount = 100 if n <= 2 else 15
        conn.execute = AsyncMock(return_value=result)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    engine_mock.begin = MagicMock(side_effect=make_begin_ctx)

    with patch("spx_backend.jobs.retention_job.engine", engine_mock):
        from spx_backend.jobs import retention_job
        result = await retention_job.run_once()

    assert result["deleted_snapshots"] == 215
    assert engine_mock.begin.call_count == 3
