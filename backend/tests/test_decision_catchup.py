"""Tests for the startup decision-job catch-up mechanism.

Validates that ``_maybe_decision_catchup`` fires the decision pipeline
exactly once when the service starts after all configured entry times have
passed, and correctly skips in every other scenario.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.config import settings
from spx_backend.web.app import _maybe_decision_catchup


ET = ZoneInfo("America/New_York")


# ── Helpers ───────────────────────────────────────────────────────


class _FakeClockCache:
    """Stub MarketClockCache where ``is_open`` returns a fixed value."""

    def __init__(self, *, is_open: bool):
        self._is_open = is_open

    async def is_open(self, now_et: datetime) -> bool:
        return self._is_open


class _FakeResult:
    """Minimal SQLAlchemy-like scalar result."""

    def __init__(self, value: Any = 0):
        self._value = value

    def scalar(self) -> Any:
        return self._value


class _FakeSession:
    """Async context-manager session returning a canned trade_decisions count."""

    def __init__(self, count: int = 0):
        self._count = count

    async def execute(self, stmt: Any, params: Any = None) -> _FakeResult:
        return _FakeResult(self._count)

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


def _session_factory(count: int = 0):
    """Build a callable matching ``SessionLocal()`` that yields a fake session."""
    @asynccontextmanager
    async def _factory():
        yield _FakeSession(count)
    return _factory


# ── Tests ─────────────────────────────────────────────────────────


class TestDecisionCatchup:
    """Cover all branches of ``_maybe_decision_catchup``."""

    @pytest.mark.asyncio
    async def test_fires_when_all_triggers_missed_and_no_trades_today(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Catch-up should run feature_builder + decision_job when all entry
        times have passed, market is open, and no trades exist for today."""
        monkeypatch.setattr(settings, "decision_startup_catchup_enabled", True)
        monkeypatch.setattr(settings, "decision_entry_times", "10:01,11:01,12:01")
        monkeypatch.setattr(settings, "feature_builder_enabled", True)
        monkeypatch.setattr(settings, "tz", "America/New_York")

        fake_now = datetime(2026, 4, 6, 14, 0, 0, tzinfo=ET)
        clock = _FakeClockCache(is_open=True)
        feature_job = SimpleNamespace(run_once=AsyncMock(return_value={}))
        decision_job = SimpleNamespace(run_once=AsyncMock(return_value={"skipped": False}))

        with (
            patch("spx_backend.web.app.datetime") as mock_dt,
            patch("spx_backend.web.app.SessionLocal", _session_factory(count=0)),
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            await _maybe_decision_catchup(
                clock_cache=clock,
                feature_builder_job=feature_job,
                decision_job=decision_job,
            )

        feature_job.run_once.assert_awaited_once_with(force=True)
        decision_job.run_once.assert_awaited_once_with(force=True)

    @pytest.mark.asyncio
    async def test_skips_when_trades_already_placed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Catch-up must not fire if trade_decisions already exist for today."""
        monkeypatch.setattr(settings, "decision_startup_catchup_enabled", True)
        monkeypatch.setattr(settings, "decision_entry_times", "10:01,11:01,12:01")
        monkeypatch.setattr(settings, "feature_builder_enabled", True)
        monkeypatch.setattr(settings, "tz", "America/New_York")

        fake_now = datetime(2026, 4, 6, 14, 0, 0, tzinfo=ET)
        clock = _FakeClockCache(is_open=True)
        decision_job = SimpleNamespace(run_once=AsyncMock())

        with (
            patch("spx_backend.web.app.datetime") as mock_dt,
            patch("spx_backend.web.app.SessionLocal", _session_factory(count=3)),
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            await _maybe_decision_catchup(
                clock_cache=clock,
                feature_builder_job=SimpleNamespace(run_once=AsyncMock()),
                decision_job=decision_job,
            )

        decision_job.run_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_future_trigger_remains(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No catch-up when at least one entry time is still ahead today."""
        monkeypatch.setattr(settings, "decision_startup_catchup_enabled", True)
        monkeypatch.setattr(settings, "decision_entry_times", "10:01,11:01,12:01")
        monkeypatch.setattr(settings, "tz", "America/New_York")

        fake_now = datetime(2026, 4, 6, 11, 30, 0, tzinfo=ET)
        clock = _FakeClockCache(is_open=True)
        decision_job = SimpleNamespace(run_once=AsyncMock())

        with patch("spx_backend.web.app.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            await _maybe_decision_catchup(
                clock_cache=clock,
                feature_builder_job=SimpleNamespace(run_once=AsyncMock()),
                decision_job=decision_job,
            )

        decision_job.run_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_outside_rth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No catch-up when the market is closed."""
        monkeypatch.setattr(settings, "decision_startup_catchup_enabled", True)
        monkeypatch.setattr(settings, "decision_entry_times", "10:01,11:01,12:01")
        monkeypatch.setattr(settings, "tz", "America/New_York")

        fake_now = datetime(2026, 4, 6, 17, 0, 0, tzinfo=ET)
        clock = _FakeClockCache(is_open=False)
        decision_job = SimpleNamespace(run_once=AsyncMock())

        with patch("spx_backend.web.app.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            await _maybe_decision_catchup(
                clock_cache=clock,
                feature_builder_job=SimpleNamespace(run_once=AsyncMock()),
                decision_job=decision_job,
            )

        decision_job.run_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_respects_disabled_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No catch-up when the feature flag is disabled."""
        monkeypatch.setattr(settings, "decision_startup_catchup_enabled", False)
        monkeypatch.setattr(settings, "tz", "America/New_York")

        clock = _FakeClockCache(is_open=True)
        decision_job = SimpleNamespace(run_once=AsyncMock())

        await _maybe_decision_catchup(
            clock_cache=clock,
            feature_builder_job=SimpleNamespace(run_once=AsyncMock()),
            decision_job=decision_job,
        )

        decision_job.run_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_feature_builder_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Feature builder should be skipped when feature_builder_enabled=False,
        but decision_job should still fire."""
        monkeypatch.setattr(settings, "decision_startup_catchup_enabled", True)
        monkeypatch.setattr(settings, "decision_entry_times", "10:01,11:01,12:01")
        monkeypatch.setattr(settings, "feature_builder_enabled", False)
        monkeypatch.setattr(settings, "tz", "America/New_York")

        fake_now = datetime(2026, 4, 6, 14, 0, 0, tzinfo=ET)
        clock = _FakeClockCache(is_open=True)
        feature_job = SimpleNamespace(run_once=AsyncMock())
        decision_job = SimpleNamespace(run_once=AsyncMock(return_value={"skipped": False}))

        with (
            patch("spx_backend.web.app.datetime") as mock_dt,
            patch("spx_backend.web.app.SessionLocal", _session_factory(count=0)),
        ):
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            await _maybe_decision_catchup(
                clock_cache=clock,
                feature_builder_job=feature_job,
                decision_job=decision_job,
            )

        feature_job.run_once.assert_not_awaited()
        decision_job.run_once.assert_awaited_once_with(force=True)
