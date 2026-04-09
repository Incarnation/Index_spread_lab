"""Test the labeler orphan feature_snapshot cascade.

Verifies that feature_snapshots with label_status='pending', no linked
trade_candidates, and age > max_wait are expired with label_error='no_candidates'.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.config import settings
from spx_backend.jobs.labeler_job import LabelerJob


@pytest.fixture(autouse=True)
def _pin_labeler_settings(monkeypatch: pytest.MonkeyPatch):
    """Set labeler config to known values for all tests."""
    monkeypatch.setattr(settings, "labeler_enabled", True)
    monkeypatch.setattr(settings, "labeler_batch_limit", 100)
    monkeypatch.setattr(settings, "labeler_min_age_minutes", 5)
    monkeypatch.setattr(settings, "labeler_max_wait_hours", 24)
    monkeypatch.setattr(settings, "labeler_take_profit_pct", 0.50)
    monkeypatch.setattr(settings, "label_schema_version", "v2")
    monkeypatch.setattr(settings, "label_contract_multiplier", 100)
    monkeypatch.setattr(settings, "tz", "America/New_York")
    monkeypatch.setattr(settings, "trade_pnl_stop_loss_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_stop_loss_pct", 1.0)
    monkeypatch.setattr(settings, "trade_pnl_stop_loss_basis", "max_profit")


class _FakeResult:
    """Mock for a DB result with configurable fetchone/fetchall and rowcount."""

    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


@pytest.mark.asyncio
async def test_orphan_feature_snapshots_are_expired() -> None:
    """Orphan feature_snapshots (no candidates, old enough) get status='expired'."""
    captured_sql: list[tuple[str, dict]] = []
    call_idx = {"n": 0}

    async def mock_execute(stmt, params=None):
        """Track all SQL calls and return appropriate results.

        Call sequence for run_once with 0 pending candidates:
        1. _has_optional_columns -> SELECT column_name FROM information_schema
        2. pending_rows -> SELECT candidate_id FROM trade_candidates
        3. session.commit()
        4. orphan cascade UPDATE feature_snapshots
        """
        sql = str(stmt)
        captured_sql.append((sql, dict(params or {})))
        call_idx["n"] += 1

        if "information_schema" in sql:
            return _FakeResult(rows=[])

        if "UPDATE" in sql and "feature_snapshots" in sql:
            return _FakeResult(rowcount=3)

        if "SELECT" in sql and "trade_candidates" in sql:
            return _FakeResult(rows=[])

        return _FakeResult()

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(side_effect=mock_execute)
    mock_session.commit = AsyncMock()

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("spx_backend.jobs.labeler_job.SessionLocal", return_value=session_ctx):
        job = LabelerJob()
        result = await job.run_once()

    assert result["skipped"] is False
    assert result["resolved"] == 0
    assert result["pending"] == 0

    orphan_sqls = [
        (sql, params) for sql, params in captured_sql
        if "feature_snapshots" in sql and "expired" in sql
    ]
    assert len(orphan_sqls) == 1, "Expected exactly one orphan cascade UPDATE"

    sql, params = orphan_sqls[0]
    assert "label_status = 'expired'" in sql
    assert "label_error = 'no_candidates'" in sql
    assert "NOT EXISTS" in sql
    assert "cutoff" in params


@pytest.mark.asyncio
async def test_orphan_cascade_uses_max_wait_cutoff() -> None:
    """The orphan cascade cutoff should be now_utc minus max_wait."""
    captured_params: list[dict] = []

    async def mock_execute(stmt, params=None):
        sql = str(stmt)
        if params:
            captured_params.append({"sql": sql, **dict(params)})

        if "information_schema" in sql:
            return _FakeResult(rows=[])
        if "UPDATE" in sql and "feature_snapshots" in sql:
            return _FakeResult(rowcount=0)
        if "SELECT" in sql and "trade_candidates" in sql:
            return _FakeResult(rows=[])
        return _FakeResult()

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(side_effect=mock_execute)
    mock_session.commit = AsyncMock()

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("spx_backend.jobs.labeler_job.SessionLocal", return_value=session_ctx):
        job = LabelerJob()
        await job.run_once()

    orphan_params = [
        p for p in captured_params
        if "feature_snapshots" in p.get("sql", "") and "cutoff" in p
    ]
    assert len(orphan_params) == 1

    cutoff = orphan_params[0]["cutoff"]
    max_wait_hours = settings.labeler_max_wait_hours
    now_utc = datetime.now(ZoneInfo("UTC"))
    expected_cutoff = now_utc - timedelta(hours=max_wait_hours)

    diff = abs((cutoff - expected_cutoff).total_seconds())
    assert diff < 10, f"Cutoff should be ~{max_wait_hours}h ago, got {cutoff}"


@pytest.mark.asyncio
async def test_orphan_cascade_commits_on_updates() -> None:
    """When orphan rows are updated, the session should be committed."""
    async def mock_execute(stmt, params=None):
        sql = str(stmt)
        if "information_schema" in sql:
            return _FakeResult(rows=[])
        # The orphan SQL also contains "trade_candidates" in its NOT EXISTS,
        # so check for UPDATE to distinguish the SELECT from the UPDATE.
        if "UPDATE" in sql and "feature_snapshots" in sql:
            return _FakeResult(rowcount=5)
        if "SELECT" in sql and "trade_candidates" in sql:
            return _FakeResult(rows=[])
        return _FakeResult()

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(side_effect=mock_execute)
    mock_session.commit = AsyncMock()

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("spx_backend.jobs.labeler_job.SessionLocal", return_value=session_ctx):
        job = LabelerJob()
        await job.run_once()

    commit_count = mock_session.commit.await_count
    assert commit_count >= 2, (
        f"Expected at least 2 commits (main + orphan), got {commit_count}"
    )


@pytest.mark.asyncio
async def test_orphan_cascade_skips_commit_when_no_orphans() -> None:
    """When no orphan rows are found, the orphan commit is skipped."""
    async def mock_execute(stmt, params=None):
        sql = str(stmt)
        if "information_schema" in sql:
            return _FakeResult(rows=[])
        if "UPDATE" in sql and "feature_snapshots" in sql:
            return _FakeResult(rowcount=0)
        if "SELECT" in sql and "trade_candidates" in sql:
            return _FakeResult(rows=[])
        return _FakeResult()

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(side_effect=mock_execute)
    mock_session.commit = AsyncMock()

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("spx_backend.jobs.labeler_job.SessionLocal", return_value=session_ctx):
        job = LabelerJob()
        await job.run_once()

    commit_count = mock_session.commit.await_count
    assert commit_count == 1, (
        f"Expected exactly 1 commit (main only, no orphan commit), got {commit_count}"
    )
