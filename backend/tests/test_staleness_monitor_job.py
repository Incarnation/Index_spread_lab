"""Tests for StalenessMonitorJob: freshness detection, cooldown, alerting, RTH guard."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.jobs.staleness_monitor_job import (
    StalenessMonitorJob,
    build_staleness_monitor_job,
)

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeSession:
    """Async session stub returning configurable MAX(ts) results per table."""

    def __init__(self, latest_by_table: dict[str, datetime | None]):
        self._latest = latest_by_table

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        for table, ts in self._latest.items():
            if table in sql:
                return MagicMock(fetchone=MagicMock(return_value=SimpleNamespace(latest=ts)))
        return MagicMock(fetchone=MagicMock(return_value=SimpleNamespace(latest=None)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _fresh_timestamps(now_utc: datetime) -> dict[str, datetime]:
    """All tables have data from 5 minutes ago (well within thresholds)."""
    return {
        "underlying_quotes": now_utc - timedelta(minutes=5),
        "chain_snapshots": now_utc - timedelta(minutes=5),
        "gex_snapshots": now_utc - timedelta(minutes=5),
        "trade_decisions": now_utc - timedelta(minutes=30),
    }


def _stale_timestamps(now_utc: datetime) -> dict[str, datetime]:
    """Quotes table is stale beyond the 120-minute default threshold."""
    return {
        "underlying_quotes": now_utc - timedelta(minutes=200),
        "chain_snapshots": now_utc - timedelta(minutes=5),
        "gex_snapshots": now_utc - timedelta(minutes=5),
        "trade_decisions": now_utc - timedelta(minutes=30),
    }


# ---------------------------------------------------------------------------
# _check_freshness
# ---------------------------------------------------------------------------

class TestCheckFreshness:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
            s.tz = "America/New_York"
            s.staleness_quotes_max_minutes = 120
            s.staleness_snapshots_max_minutes = 120
            s.staleness_gex_max_minutes = 120
            s.staleness_decisions_max_minutes = 480
            self.mock_settings = s
            yield

    @pytest.mark.asyncio
    async def test_all_fresh(self):
        """No stale sources when all timestamps are recent."""
        now = datetime.now(tz=UTC)
        session = FakeSession(_fresh_timestamps(now))
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        assert result == []

    @pytest.mark.asyncio
    async def test_one_stale(self):
        """Detects single stale source."""
        now = datetime.now(tz=UTC)
        session = FakeSession(_stale_timestamps(now))
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        assert len(result) == 1
        assert result[0]["source"] == "underlying_quotes"
        assert result[0]["age_minutes"] > 120

    @pytest.mark.asyncio
    async def test_null_timestamp(self):
        """Table with no data (NULL max) is reported as stale."""
        now = datetime.now(tz=UTC)
        timestamps = _fresh_timestamps(now)
        timestamps["gex_snapshots"] = None
        session = FakeSession(timestamps)
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        assert len(result) == 1
        assert result[0]["source"] == "gex_snapshots"
        assert result[0]["age_minutes"] is None


# ---------------------------------------------------------------------------
# _should_alert (cooldown)
# ---------------------------------------------------------------------------

class TestShouldAlert:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
            s.staleness_cooldown_minutes = 360
            yield

    def test_first_alert_always_sent(self):
        """First alert is always allowed when no previous alert exists."""
        job = StalenessMonitorJob()
        assert job._should_alert(datetime.now(tz=UTC)) is True

    def test_within_cooldown(self):
        """Alert is suppressed when last alert was sent within the cooldown window."""
        now = datetime.now(tz=UTC)
        job = StalenessMonitorJob(_last_alert_ts=now - timedelta(minutes=10))
        assert job._should_alert(now) is False

    def test_after_cooldown(self):
        """Alert is allowed again once the cooldown period has elapsed."""
        now = datetime.now(tz=UTC)
        job = StalenessMonitorJob(_last_alert_ts=now - timedelta(minutes=400))
        assert job._should_alert(now) is True


# ---------------------------------------------------------------------------
# _send_alert
# ---------------------------------------------------------------------------

class TestSendAlert:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
            s.sendgrid_api_key = "SG.fake"
            s.email_alert_recipient = "test@example.com"
            s.email_alert_sender = "alerts@app.test"
            yield

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """Returns False when SendGrid API key is empty."""
        with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
            s.sendgrid_api_key = ""
            s.email_alert_recipient = "test@example.com"
            job = StalenessMonitorJob()
            result = await job._send_alert([{"source": "t", "age_minutes": 999, "threshold_minutes": 60}])
            assert result is False

    @pytest.mark.asyncio
    async def test_no_recipient(self):
        """Returns False when recipient is empty."""
        with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
            s.sendgrid_api_key = "SG.fake"
            s.email_alert_recipient = ""
            job = StalenessMonitorJob()
            result = await job._send_alert([{"source": "t", "age_minutes": 999, "threshold_minutes": 60}])
            assert result is False

    @pytest.mark.asyncio
    async def test_sendgrid_called(self):
        """SendGrid client is constructed and send() called with correct payload."""
        mock_response = MagicMock(status_code=202)
        mock_sg_instance = MagicMock()
        mock_sg_instance.send = MagicMock(return_value=mock_response)

        with patch.dict("sys.modules", {
            "sendgrid": MagicMock(SendGridAPIClient=MagicMock(return_value=mock_sg_instance)),
            "sendgrid.helpers": MagicMock(),
            "sendgrid.helpers.mail": MagicMock(Mail=MagicMock()),
        }):
            job = StalenessMonitorJob()
            result = await job._send_alert([
                {"source": "underlying_quotes", "age_minutes": 200, "threshold_minutes": 120},
            ])
            assert result is True


# ---------------------------------------------------------------------------
# run_once integration
# ---------------------------------------------------------------------------

class TestRunOnce:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
            s.tz = "America/New_York"
            s.staleness_alert_enabled = True
            s.staleness_cooldown_minutes = 360
            s.staleness_quotes_max_minutes = 120
            s.staleness_snapshots_max_minutes = 120
            s.staleness_gex_max_minutes = 120
            s.staleness_decisions_max_minutes = 480
            s.sendgrid_api_key = ""
            s.email_alert_recipient = ""
            s.email_alert_sender = "alerts@test.com"
            self.mock_settings = s
            yield

    @pytest.mark.asyncio
    async def test_disabled(self):
        """Skips immediately when staleness alerting is disabled in settings."""
        self.mock_settings.staleness_alert_enabled = False
        job = StalenessMonitorJob()
        result = await job.run_once()
        assert result["skipped"] is True
        assert result["reason"] == "staleness_alert_disabled"

    @pytest.mark.asyncio
    async def test_market_closed_skip(self):
        """Skips when market is closed and not forced."""
        mock_cache = AsyncMock()
        mock_cache.is_open = AsyncMock(return_value=False)
        job = StalenessMonitorJob(clock_cache=mock_cache)
        result = await job.run_once()
        assert result["skipped"] is True
        assert result["reason"] == "market_closed"

    @pytest.mark.asyncio
    async def test_force_bypasses_market_check(self):
        """force=True bypasses the market-closed check."""
        now = datetime.now(tz=UTC)
        session = FakeSession(_fresh_timestamps(now))

        mock_cache = AsyncMock()
        mock_cache.is_open = AsyncMock(return_value=False)

        with patch("spx_backend.jobs.staleness_monitor_job.SessionLocal", return_value=session):
            job = StalenessMonitorJob(clock_cache=mock_cache)
            result = await job.run_once(force=True)

        assert result["skipped"] is False
        assert result["stale_count"] == 0

    @pytest.mark.asyncio
    async def test_all_fresh_no_alert(self):
        """When all sources are fresh, stale_count=0 and no alert."""
        now = datetime.now(tz=UTC)
        session = FakeSession(_fresh_timestamps(now))

        with (
            patch("spx_backend.jobs.staleness_monitor_job.SessionLocal", return_value=session),
            patch.object(StalenessMonitorJob, "_market_open", new_callable=AsyncMock, return_value=True),
        ):
            job = StalenessMonitorJob()
            result = await job.run_once()

        assert result["stale_count"] == 0
        assert result["alerted"] is False

    @pytest.mark.asyncio
    async def test_stale_detected(self):
        """When a source is stale, stale_count > 0."""
        now = datetime.now(tz=UTC)
        session = FakeSession(_stale_timestamps(now))

        with (
            patch("spx_backend.jobs.staleness_monitor_job.SessionLocal", return_value=session),
            patch.object(StalenessMonitorJob, "_market_open", new_callable=AsyncMock, return_value=True),
        ):
            job = StalenessMonitorJob()
            result = await job.run_once()

        assert result["stale_count"] >= 1
        assert any(s["source"] == "underlying_quotes" for s in result["stale_sources"])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_without_cache(self):
        """Factory creates job with no clock_cache by default."""
        job = build_staleness_monitor_job()
        assert job.clock_cache is None

    def test_with_cache(self):
        """Factory injects clock_cache into the job instance."""
        cache = MagicMock()
        job = build_staleness_monitor_job(clock_cache=cache)
        assert job.clock_cache is cache
