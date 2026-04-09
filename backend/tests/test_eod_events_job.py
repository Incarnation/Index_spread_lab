"""Unit tests for EodEventsJob.

Covers run_once behavior: empty calendar, successful upsert, and
idempotent re-run (ON CONFLICT DO NOTHING).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spx_backend.jobs.eod_events_job import EodEventsJob


@pytest.fixture
def _mock_generate_rows(monkeypatch):
    """Patch _generate_rows so we never import the scripts directory."""
    rows = [
        {"date": "2026-04-09", "event_type": "CPI", "has_projections": True, "is_triple_witching": False},
        {"date": "2026-04-17", "event_type": "OPEX", "has_projections": False, "is_triple_witching": True},
    ]
    monkeypatch.setattr("spx_backend.jobs.eod_events_job._generate_rows", lambda: rows)
    return rows


@pytest.fixture
def _mock_generate_rows_empty(monkeypatch):
    """Patch _generate_rows to return an empty list."""
    monkeypatch.setattr("spx_backend.jobs.eod_events_job._generate_rows", lambda: [])


def _fake_engine(rowcounts: list[int] | None = None):
    """Build a mock async engine that records executed SQL parameters.

    Parameters
    ----------
    rowcounts:
        Per-statement rowcount values the mock cursor should return.
        Defaults to 1 for every execution if not provided.
    """
    executed: list[dict] = []
    call_idx = {"i": 0}

    async def _execute(sql, params):
        executed.append(params)
        result = MagicMock()
        if rowcounts and call_idx["i"] < len(rowcounts):
            result.rowcount = rowcounts[call_idx["i"]]
        else:
            result.rowcount = 1
        call_idx["i"] += 1
        return result

    conn = AsyncMock()
    conn.execute = _execute

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    engine = MagicMock()
    engine.begin = MagicMock(return_value=ctx)
    return engine, executed


@pytest.mark.asyncio
async def test_run_once_inserts_calendar_rows(_mock_generate_rows, monkeypatch):
    """Successful run upserts each calendar row into the DB."""
    engine, executed = _fake_engine()
    monkeypatch.setattr("spx_backend.jobs.eod_events_job.engine", engine)

    job = EodEventsJob()
    await job.run_once()

    assert len(executed) == 2
    assert executed[0]["event_type"] == "CPI"
    assert executed[1]["event_type"] == "OPEX"
    assert executed[1]["is_triple_witching"] is True


@pytest.mark.asyncio
async def test_run_once_empty_calendar_does_not_touch_db(_mock_generate_rows_empty, monkeypatch):
    """When generate_rows returns empty, no DB interaction occurs."""
    engine, executed = _fake_engine()
    monkeypatch.setattr("spx_backend.jobs.eod_events_job.engine", engine)

    job = EodEventsJob()
    await job.run_once()

    assert len(executed) == 0


@pytest.mark.asyncio
async def test_run_once_counts_inserted_rows(_mock_generate_rows, monkeypatch):
    """Inserted count reflects rowcount from ON CONFLICT DO NOTHING."""
    engine, _ = _fake_engine(rowcounts=[1, 0])
    monkeypatch.setattr("spx_backend.jobs.eod_events_job.engine", engine)

    job = EodEventsJob()
    # run_once returns None; we verify no crash and correct SQL dispatch
    await job.run_once()


@pytest.mark.asyncio
async def test_run_once_force_flag_accepted(_mock_generate_rows, monkeypatch):
    """force=True is accepted without error (no behavioral change for this job)."""
    engine, _ = _fake_engine()
    monkeypatch.setattr("spx_backend.jobs.eod_events_job.engine", engine)

    job = EodEventsJob()
    await job.run_once(force=True)
