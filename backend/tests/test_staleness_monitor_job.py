"""Tests for StalenessMonitorJob: per-bucket freshness, RTH guard, alert delegation.

Audit Wave 3 / 4 reshaped this job (see staleness_monitor_job docstring):
* H8: per-underlying / per-symbol GROUP BY in ``_check_freshness``.
* M12: CBOE-specific source filter (``gex_snapshots_cboe``).
* Refactor #3: alert dispatch delegated to ``services.alerts.send_alert``;
  the local ``_send_alert`` / ``_should_alert`` / ``_last_alert_ts`` paths
  no longer exist. Cooldown ownership lives in ``services.alerts``
  (DB-backed via H7).

These tests cover the new shape exclusively; the old shape's tests were
removed because the underlying methods no longer exist.
"""
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


class FakeSession:
    """Async session stub for the per-bucket GROUP BY freshness queries.

    ``per_bucket`` maps logical_name (matches first column of GROUP BY
    queries via the table name in the SQL) to a list of
    ``(bucket, latest_ts)`` tuples returned by ``fetchall``. Global
    queries (no GROUP BY -- e.g. trade_decisions) read ``global_max``
    keyed by table name.
    """

    def __init__(
        self,
        *,
        per_bucket: dict[str, list[tuple[str, datetime | None]]] | None = None,
        global_max: dict[str, datetime | None] | None = None,
    ):
        self._per_bucket = per_bucket or {}
        self._global = global_max or {}

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        # M12 (audit): CBOE-source variant runs against gex_snapshots
        # with a WHERE filter; route by the source filter present in
        # params so the test can supply distinct buckets per source.
        is_cboe_filtered = bool(params and params.get("src") == "CBOE")
        if "GROUP BY" in sql:
            for table, rows in self._per_bucket.items():
                if table in sql:
                    if is_cboe_filtered and table == "gex_snapshots":
                        # Look for a sentinel "gex_snapshots_cboe" key.
                        rows = self._per_bucket.get("gex_snapshots_cboe", rows)
                    fetchall = [
                        SimpleNamespace(bucket=b, latest=ts) for b, ts in rows
                    ]
                    return MagicMock(fetchall=MagicMock(return_value=fetchall))
            return MagicMock(fetchall=MagicMock(return_value=[]))
        # Global MAX(ts) (no GROUP BY).
        for table, ts in self._global.items():
            if table in sql:
                return MagicMock(
                    fetchone=MagicMock(return_value=SimpleNamespace(latest=ts))
                )
        return MagicMock(fetchone=MagicMock(return_value=SimpleNamespace(latest=None)))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture(autouse=True)
def _patch_settings():
    """Provide a stable settings stub for every test in this module."""
    with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
        s.tz = "America/New_York"
        s.staleness_alert_enabled = True
        s.staleness_cooldown_minutes = 360
        s.staleness_quotes_max_minutes = 120
        s.staleness_snapshots_max_minutes = 120
        s.staleness_gex_max_minutes = 120
        s.staleness_cboe_gex_max_minutes = 120
        s.staleness_decisions_max_minutes = 480
        s.snapshot_underlying = "SPX"
        s.cboe_gex_underlyings = "SPX,SPY"
        s.quote_symbols = "SPX,VIX,VIX9D,SPY,VVIX,SKEW"
        yield s


def _all_fresh_per_bucket(now_utc: datetime) -> dict[str, list[tuple[str, datetime]]]:
    """Build per-bucket fixture where every expected bucket is fresh."""
    fresh = now_utc - timedelta(minutes=5)
    return {
        "underlying_quotes": [
            (s, fresh) for s in ("SPX", "VIX", "VIX9D", "SPY", "VVIX", "SKEW")
        ],
        "chain_snapshots": [("SPX", fresh), ("SPY", fresh)],
        "gex_snapshots": [("SPX", fresh), ("SPY", fresh)],
        "gex_snapshots_cboe": [("SPX", fresh), ("SPY", fresh)],
    }


def _all_fresh_global(now_utc: datetime) -> dict[str, datetime]:
    """Trade decisions are fresh for tests that do not stale them."""
    return {"trade_decisions": now_utc - timedelta(minutes=30)}


class TestCheckFreshness:
    @pytest.mark.asyncio
    async def test_all_fresh(self):
        """No stale entries when every expected bucket is recent."""
        now = datetime.now(tz=UTC)
        session = FakeSession(
            per_bucket=_all_fresh_per_bucket(now),
            global_max=_all_fresh_global(now),
        )
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        assert result == []

    @pytest.mark.asyncio
    async def test_one_underlying_stale(self):
        """A single stale underlying surfaces; siblings remain hidden."""
        now = datetime.now(tz=UTC)
        per = _all_fresh_per_bucket(now)
        per["chain_snapshots"] = [
            ("SPX", now - timedelta(minutes=5)),
            ("SPY", now - timedelta(minutes=300)),
        ]
        session = FakeSession(per_bucket=per, global_max=_all_fresh_global(now))
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        stale_chain = [s for s in result if s["source"] == "chain_snapshots"]
        assert len(stale_chain) == 1
        assert stale_chain[0]["bucket"] == "SPY"
        assert stale_chain[0]["age_minutes"] > 120

    @pytest.mark.asyncio
    async def test_missing_bucket_reported(self):
        """A bucket the operator expects but never observed is reported as no-data."""
        now = datetime.now(tz=UTC)
        per = _all_fresh_per_bucket(now)
        per["gex_snapshots"] = [("SPX", now - timedelta(minutes=5))]  # SPY missing
        session = FakeSession(per_bucket=per, global_max=_all_fresh_global(now))
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        missing = [
            s for s in result
            if s["source"] == "gex_snapshots" and s["bucket"] == "SPY"
        ]
        assert len(missing) == 1
        assert missing[0]["age_minutes"] is None
        assert missing[0]["latest_ts"] is None

    @pytest.mark.asyncio
    async def test_cboe_specific_filter(self):
        """CBOE-source variant flags CBOE-only staleness (M12)."""
        now = datetime.now(tz=UTC)
        per = _all_fresh_per_bucket(now)
        # Tradier-side gex_snapshots is fresh; CBOE-source side is stale.
        per["gex_snapshots_cboe"] = [
            ("SPX", now - timedelta(minutes=300)),
            ("SPY", now - timedelta(minutes=300)),
        ]
        session = FakeSession(per_bucket=per, global_max=_all_fresh_global(now))
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        cboe = [s for s in result if s["source"] == "gex_snapshots_cboe"]
        assert {s["bucket"] for s in cboe} == {"SPX", "SPY"}
        # The Tradier-side gex_snapshots should still be fresh.
        assert not any(s["source"] == "gex_snapshots" for s in result)

    @pytest.mark.asyncio
    async def test_global_check_no_data(self):
        """Global MAX(ts) check (trade_decisions) reports stale when null."""
        now = datetime.now(tz=UTC)
        session = FakeSession(
            per_bucket=_all_fresh_per_bucket(now),
            global_max={"trade_decisions": None},
        )
        job = StalenessMonitorJob()
        result = await job._check_freshness(session)
        td = [s for s in result if s["source"] == "trade_decisions"]
        assert len(td) == 1
        assert td[0]["bucket"] == "*"
        assert td[0]["age_minutes"] is None


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_disabled(self):
        """Skips immediately when staleness alerting is disabled in settings."""
        with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
            s.tz = "America/New_York"
            s.staleness_alert_enabled = False
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
        """force=True bypasses the market-closed check; all-fresh returns no alerts."""
        now = datetime.now(tz=UTC)
        session = FakeSession(
            per_bucket=_all_fresh_per_bucket(now),
            global_max=_all_fresh_global(now),
        )
        with patch(
            "spx_backend.jobs.staleness_monitor_job.SessionLocal",
            return_value=session,
        ):
            job = StalenessMonitorJob()
            result = await job.run_once(force=True)
        assert result["skipped"] is False
        assert result["stale_count"] == 0

    @pytest.mark.asyncio
    async def test_stale_delegates_to_send_alert(self):
        """When stale rows exist, run_once calls services.alerts.send_alert
        with the configured cooldown_key + minutes; alert delivery is mocked."""
        now = datetime.now(tz=UTC)
        per = _all_fresh_per_bucket(now)
        per["chain_snapshots"] = [
            ("SPX", now - timedelta(minutes=5)),
            ("SPY", now - timedelta(minutes=300)),  # stale
        ]
        session = FakeSession(per_bucket=per, global_max=_all_fresh_global(now))

        mock_send = AsyncMock(return_value=True)
        with (
            patch(
                "spx_backend.jobs.staleness_monitor_job.SessionLocal",
                return_value=session,
            ),
            patch.object(
                StalenessMonitorJob,
                "_market_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "spx_backend.jobs.staleness_monitor_job.alerts.send_alert",
                new=mock_send,
            ),
        ):
            job = StalenessMonitorJob()
            result = await job.run_once()

        assert result["stale_count"] >= 1
        assert result["alerted"] is True
        mock_send.assert_awaited_once()
        kwargs = mock_send.await_args.kwargs
        assert kwargs["cooldown_key"] == "staleness:any"
        assert kwargs["cooldown_minutes"] == 360
        assert kwargs["subject"].startswith("[IndexSpreadLab]")

    @pytest.mark.asyncio
    async def test_all_fresh_no_alert(self):
        """When all sources are fresh, stale_count=0 and no alert is sent."""
        now = datetime.now(tz=UTC)
        session = FakeSession(
            per_bucket=_all_fresh_per_bucket(now),
            global_max=_all_fresh_global(now),
        )
        with (
            patch(
                "spx_backend.jobs.staleness_monitor_job.SessionLocal",
                return_value=session,
            ),
            patch.object(
                StalenessMonitorJob,
                "_market_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            job = StalenessMonitorJob()
            result = await job.run_once()
        assert result["stale_count"] == 0
        assert result["alerted"] is False


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
