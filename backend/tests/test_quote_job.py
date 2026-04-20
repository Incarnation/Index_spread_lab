"""Tests for quote_job.py: market gate, symbol parsing, and run_once flows."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.jobs.quote_job import QuoteJob, _parse_quotes, _quotes_by_symbol, build_quote_job


# ---------------------------------------------------------------------------
# _parse_quotes
# ---------------------------------------------------------------------------

class TestParseQuotes:
    def test_list_payload(self):
        """Tradier returns a list of quote dicts."""
        resp = {"quotes": {"quote": [{"symbol": "SPX"}, {"symbol": "VIX"}]}}
        assert len(_parse_quotes(resp)) == 2

    def test_single_quote(self):
        """Tradier returns a single dict when only one symbol is requested."""
        resp = {"quotes": {"quote": {"symbol": "SPX"}}}
        result = _parse_quotes(resp)
        assert len(result) == 1
        assert result[0]["symbol"] == "SPX"

    def test_none_payload(self):
        """Returns empty list when quotes payload is None."""
        resp = {"quotes": {"quote": None}}
        assert _parse_quotes(resp) == []

    def test_missing_quotes(self):
        """Returns empty list when top-level 'quotes' key is missing."""
        assert _parse_quotes({}) == []


# ---------------------------------------------------------------------------
# _quotes_by_symbol
# ---------------------------------------------------------------------------

class TestQuotesBySymbol:
    def test_indexes_by_symbol(self):
        """Builds a dict keyed by symbol from a list of quote dicts."""
        quotes = [{"symbol": "SPX", "last": 5000}, {"symbol": "VIX", "last": 18}]
        result = _quotes_by_symbol(quotes)
        assert "SPX" in result
        assert result["VIX"]["last"] == 18

    def test_skips_missing_symbol(self):
        """Quotes without a 'symbol' key are excluded from the index."""
        quotes = [{"last": 100}]
        assert _quotes_by_symbol(quotes) == {}


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------

class TestRunOnce:
    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch("spx_backend.jobs.quote_job.settings") as s:
            s.tz = "America/New_York"
            s.allow_quotes_outside_rth = True
            s.quote_symbols = "SPX,VIX"
            s.quote_symbols_list = MagicMock(return_value=["SPX", "VIX"])
            # M13 (audit): explicit int so the consecutive-failure
            # threshold check uses a real comparison instead of a
            # MagicMock auto-attr that always evaluates truthy.
            s.quote_consecutive_failure_threshold = 3
            self.mock_settings = s
            # Reset module-level counter so per-test failure tallies
            # start from zero regardless of run order.
            from spx_backend.jobs.quote_job import reset_fetch_failure_counter
            reset_fetch_failure_counter()
            yield

    @pytest.mark.asyncio
    async def test_market_closed_skip(self):
        """Skips when market is closed and allow_quotes_outside_rth=False."""
        self.mock_settings.allow_quotes_outside_rth = False

        mock_cache = AsyncMock()
        mock_cache.is_open = AsyncMock(return_value=False)
        tradier = AsyncMock()

        job = QuoteJob(tradier=tradier, clock_cache=mock_cache)
        result = await job.run_once()

        assert result["skipped"] is True
        assert result["reason"] == "market_closed"

    @pytest.mark.asyncio
    async def test_no_symbols_skip(self):
        """Skips when no quote symbols are configured."""
        self.mock_settings.quote_symbols_list = MagicMock(return_value=[])
        tradier = AsyncMock()

        job = QuoteJob(tradier=tradier)
        result = await job.run_once()

        assert result["skipped"] is True
        assert result["reason"] == "no_symbols"

    @pytest.mark.asyncio
    async def test_fetch_failure(self):
        """Skips gracefully when Tradier API fails."""
        tradier = AsyncMock()
        tradier.get_quotes = AsyncMock(side_effect=Exception("timeout"))

        job = QuoteJob(tradier=tradier)
        result = await job.run_once()

        assert result["skipped"] is True
        assert result["reason"] == "quote_fetch_failed"

    @pytest.mark.asyncio
    async def test_empty_quotes(self):
        """Skips when Tradier returns an empty quotes payload."""
        tradier = AsyncMock()
        tradier.get_quotes = AsyncMock(return_value={"quotes": {"quote": None}})

        job = QuoteJob(tradier=tradier)
        result = await job.run_once()

        assert result["skipped"] is True
        assert result["reason"] == "no_quotes"

    @pytest.mark.asyncio
    async def test_successful_insertion(self):
        """Quotes are inserted and context snapshot is created."""
        tradier = AsyncMock()
        tradier.get_quotes = AsyncMock(return_value={
            "quotes": {"quote": [
                {"symbol": "SPX", "last": 5000, "bid": 4999, "ask": 5001},
                {"symbol": "VIX", "last": 18, "bid": 17.9, "ask": 18.1},
            ]},
        })

        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.begin_nested = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False),
        ))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("spx_backend.jobs.quote_job.SessionLocal", return_value=session):
            job = QuoteJob(tradier=tradier)
            result = await job.run_once()

        assert result["skipped"] is False
        assert result["quotes_inserted"] == 2


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBuildQuoteJob:
    def test_factory_with_cache(self):
        """Factory injects clock_cache and tradier into the job instance."""
        cache = MagicMock()
        tradier = MagicMock()
        job = build_quote_job(clock_cache=cache, tradier=tradier)
        assert job.clock_cache is cache
        assert job.tradier is tradier
