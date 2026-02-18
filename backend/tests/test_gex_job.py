from __future__ import annotations

from datetime import datetime

import pytest

import spx_backend.jobs.gex_job as gex_module
from spx_backend.config import settings
from spx_backend.jobs.gex_job import GexJob


class _FakeExecResult:
    """Minimal SQLAlchemy-like execute result wrapper for GEX tests."""

    def __init__(self, *, rows: list[tuple] | None = None):
        """Store rows returned by a fake SELECT execution."""
        self._rows = rows or []

    def fetchall(self) -> list[tuple]:
        """Return all fake rows for query result iteration."""
        return self._rows


class _NoPendingSession:
    """Fake async DB session that reports no pending snapshots."""

    async def execute(self, stmt, params=None):  # noqa: ANN001 - SQLAlchemy text object in production
        """Return an empty candidate set for pending snapshot lookup."""
        return _FakeExecResult(rows=[])

    async def commit(self) -> None:
        """No-op commit hook for fake session compatibility."""
        return None

    async def rollback(self) -> None:
        """No-op rollback hook for fake session compatibility."""
        return None


class _SessionFactory:
    """Async context-manager factory that returns one fake session."""

    def __init__(self, session: _NoPendingSession):
        """Store the fake session returned by the context manager."""
        self._session = session

    def __call__(self):
        """Return self to mimic SessionLocal callable behavior."""
        return self

    async def __aenter__(self) -> _NoPendingSession:
        """Yield the fake session to GEX job code."""
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        """Do not suppress exceptions raised inside the async block."""
        return False


class _FailIfUsedSessionFactory:
    """SessionLocal substitute that fails if the job touches the database."""

    def __call__(self):
        """Raise immediately when database access is unexpectedly attempted."""
        raise AssertionError("SessionLocal should not be used when market is closed")


class _ClosedClockCache:
    """Clock-cache stub that always reports market-closed."""

    async def is_open(self, now_et: datetime) -> bool:  # noqa: ARG002
        """Return false so run_once should skip when not forced."""
        return False


class _ClockCacheShouldNotBeCalled:
    """Clock-cache stub that raises if market-open check is invoked."""

    async def is_open(self, now_et: datetime) -> bool:  # noqa: ARG002
        """Raise to prove force mode bypasses market-hour gate."""
        raise AssertionError("Clock cache should not be called in force mode")


def test_zero_gamma_level_interpolates_crossing() -> None:
    """Zero-gamma interpolation should return a strike between sign change points."""
    job = GexJob()
    per_strike = {
        6800.0: {"gex_calls": 100.0, "gex_puts": -300.0},  # cumulative: -200
        6825.0: {"gex_calls": 250.0, "gex_puts": -20.0},   # cumulative: +30 (cross)
    }

    zero = job._zero_gamma_level(per_strike)

    assert zero is not None
    assert 6800.0 < zero < 6825.0


def test_zero_gamma_level_returns_none_without_cross() -> None:
    """Zero-gamma should be None when cumulative net never crosses zero."""
    job = GexJob()
    per_strike = {
        6800.0: {"gex_calls": 200.0, "gex_puts": -50.0},
        6825.0: {"gex_calls": 150.0, "gex_puts": -40.0},
    }

    zero = job._zero_gamma_level(per_strike)

    assert zero is None


@pytest.mark.asyncio
async def test_run_once_skips_when_market_closed(monkeypatch) -> None:
    """GEX run should short-circuit before DB work when market is closed."""
    monkeypatch.setattr(settings, "gex_enabled", True)
    monkeypatch.setattr(settings, "gex_allow_outside_rth", False)
    monkeypatch.setattr(gex_module, "SessionLocal", _FailIfUsedSessionFactory())

    result = await GexJob(clock_cache=_ClosedClockCache()).run_once()

    assert result["skipped"] is True
    assert result["reason"] == "market_closed"
    assert result["computed_snapshots"] == 0
    assert result["skipped_snapshots"] == 0
    assert result["failed_snapshots"] == []
    parsed_now = datetime.fromisoformat(result["now_et"])
    assert parsed_now.tzinfo is not None


@pytest.mark.asyncio
async def test_run_once_force_bypasses_market_gate(monkeypatch) -> None:
    """Forced GEX run should bypass market clock gating and query pending snapshots."""
    monkeypatch.setattr(settings, "gex_enabled", True)
    monkeypatch.setattr(settings, "gex_allow_outside_rth", False)
    monkeypatch.setattr(gex_module, "SessionLocal", _SessionFactory(_NoPendingSession()))

    result = await GexJob(clock_cache=_ClockCacheShouldNotBeCalled()).run_once(force=True)

    assert result == {"skipped": True, "reason": "no_pending_snapshots"}
