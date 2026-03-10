"""Tests for market_clock.py: is_rth boundary cases and MarketClockCache."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.market_clock import MarketClockCache, is_rth

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# is_rth
# ---------------------------------------------------------------------------

class TestIsRth:
    """Boundary tests for the is_rth() helper."""

    def test_weekday_during_rth(self):
        """Monday 10:30 ET should be inside RTH."""
        dt = datetime(2025, 3, 3, 10, 30, tzinfo=ET)  # Monday
        assert is_rth(dt) is True

    def test_weekday_before_open(self):
        """Monday 09:00 ET (before 09:30) should be outside RTH."""
        dt = datetime(2025, 3, 3, 9, 0, tzinfo=ET)
        assert is_rth(dt) is False

    def test_weekday_at_open(self):
        """Exactly 09:30 ET should be inside RTH."""
        dt = datetime(2025, 3, 3, 9, 30, tzinfo=ET)
        assert is_rth(dt) is True

    def test_weekday_at_close(self):
        """Exactly 16:00 ET should be inside RTH (inclusive)."""
        dt = datetime(2025, 3, 3, 16, 0, tzinfo=ET)
        assert is_rth(dt) is True

    def test_weekday_after_close(self):
        """Monday 16:01 ET should be outside RTH."""
        dt = datetime(2025, 3, 3, 16, 1, tzinfo=ET)
        assert is_rth(dt) is False

    def test_saturday(self):
        """Saturday during normal hours should be outside RTH."""
        dt = datetime(2025, 3, 1, 12, 0, tzinfo=ET)  # Saturday
        assert is_rth(dt) is False

    def test_sunday(self):
        """Sunday should be outside RTH."""
        dt = datetime(2025, 3, 2, 12, 0, tzinfo=ET)  # Sunday
        assert is_rth(dt) is False

    def test_friday_afternoon(self):
        """Friday 15:00 ET should be inside RTH."""
        dt = datetime(2025, 2, 28, 15, 0, tzinfo=ET)  # Friday
        assert is_rth(dt) is True


# ---------------------------------------------------------------------------
# MarketClockCache
# ---------------------------------------------------------------------------

class TestMarketClockCache:
    """Tests for cache TTL, Tradier integration, and fallback behavior."""

    @pytest.mark.asyncio
    async def test_open_state(self):
        """When Tradier says 'open', is_open returns True."""
        tradier = AsyncMock()
        tradier.get_market_clock = AsyncMock(return_value={
            "clock": {"state": "open"},
        })
        cache = MarketClockCache(tradier=tradier, ttl_seconds=300)

        with patch.object(cache, "_record", new_callable=AsyncMock):
            result = await cache.is_open(datetime.now(tz=ET))
        assert result is True

    @pytest.mark.asyncio
    async def test_closed_state(self):
        """When Tradier says 'closed', is_open returns False."""
        tradier = AsyncMock()
        tradier.get_market_clock = AsyncMock(return_value={
            "clock": {"state": "closed"},
        })
        cache = MarketClockCache(tradier=tradier, ttl_seconds=300)

        with patch.object(cache, "_record", new_callable=AsyncMock):
            result = await cache.is_open(datetime.now(tz=ET))
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_reuse(self):
        """Within TTL, the cached value is returned without a new API call."""
        tradier = AsyncMock()
        tradier.get_market_clock = AsyncMock(return_value={
            "clock": {"state": "open"},
        })
        cache = MarketClockCache(tradier=tradier, ttl_seconds=300)
        now = datetime.now(tz=ET)

        with patch.object(cache, "_record", new_callable=AsyncMock):
            await cache.is_open(now)
            await cache.is_open(now)

        assert tradier.get_market_clock.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_to_is_rth_on_error(self):
        """On API failure with no cached value, falls back to is_rth."""
        tradier = AsyncMock()
        tradier.get_market_clock = AsyncMock(side_effect=Exception("network error"))
        cache = MarketClockCache(tradier=tradier, ttl_seconds=300)

        weekday_noon = datetime(2025, 3, 3, 12, 0, tzinfo=ET)
        with patch.object(cache, "_record", new_callable=AsyncMock):
            result = await cache.is_open(weekday_noon)
        assert result is True  # is_rth(Mon 12:00) = True

    @pytest.mark.asyncio
    async def test_fallback_uses_cache_when_available(self):
        """On API failure, uses cached value if available."""
        tradier = AsyncMock()
        tradier.get_market_clock = AsyncMock(return_value={
            "clock": {"state": "closed"},
        })
        cache = MarketClockCache(tradier=tradier, ttl_seconds=300)
        now = datetime.now(tz=ET)

        with patch.object(cache, "_record", new_callable=AsyncMock):
            await cache.is_open(now)

        # Now simulate failure
        tradier.get_market_clock = AsyncMock(side_effect=Exception("timeout"))
        cache._cached_at = None  # Force cache expiry

        with patch.object(cache, "_record", new_callable=AsyncMock):
            result = await cache.is_open(now)
        assert result is False  # Uses previously cached "closed"
