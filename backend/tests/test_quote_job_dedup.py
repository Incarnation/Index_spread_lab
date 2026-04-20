"""Wave 5 (audit) tests for quote_job's H5 minute-bucket dedup.

These tests pin the contract introduced in Wave 2 H5:

* The underlying_quotes INSERT carries
  ``ON CONFLICT (symbol, (date_bin('1 minute', ts, TIMESTAMPTZ '2000-01-01 00:00:00+00')))
  DO NOTHING`` and the ``RETURNING quote_id`` clause. (date_bin is used
  rather than date_trunc because PG requires unique-index expressions
  to be IMMUTABLE; date_trunc on timestamptz is STABLE.)
* When the conflict path fires (vendor-driven duplicate retry inside
  the same minute) the writer increments ``quotes_dedup_skipped``
  rather than ``quotes_inserted`` and DOES NOT raise.
* H6 (audit) ``vendor_ts`` is bound from the Tradier ``trade_date``
  ms-epoch via ``_parse_tradier_epoch_ms`` and threads through the
  INSERT params so as-of joins downstream can prefer it.

The tests are pure asyncio + AsyncMock; no DB is required.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spx_backend.jobs.quote_job import QuoteJob, _parse_tradier_epoch_ms, reset_fetch_failure_counter


def _make_session_mock(*, insert_returns_row: bool):
    """Build an async-context-manager-shaped session mock.

    Parameters
    ----------
    insert_returns_row:
        When ``True`` ``execute().fetchone()`` returns a stub row so the
        production code treats the INSERT as a fresh row. When ``False``
        ``fetchone()`` returns ``None`` so the production code treats
        the INSERT as a suppressed-by-conflict (dedup) write.
    """
    session = MagicMock()

    captured_executes: list[tuple[str, dict]] = []

    async def _execute(stmt, params=None):  # noqa: ANN001
        captured_executes.append((str(stmt), dict(params or {})))
        result = MagicMock()
        if "INSERT INTO underlying_quotes" in str(stmt):
            if insert_returns_row:
                row = MagicMock()
                row.quote_id = 12345
                result.fetchone = MagicMock(return_value=row)
            else:
                result.fetchone = MagicMock(return_value=None)
        else:
            result.fetchone = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.begin_nested = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        )
    )
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    return session, captured_executes


@pytest.fixture(autouse=True)
def _patch_settings():
    """Pin enough settings for the H5/H6/M13 paths to evaluate."""
    with patch("spx_backend.jobs.quote_job.settings") as s:
        s.tz = "America/New_York"
        s.allow_quotes_outside_rth = True
        s.quote_symbols = "SPX"
        s.quote_symbols_list = MagicMock(return_value=["SPX"])
        s.quote_consecutive_failure_threshold = 3
        reset_fetch_failure_counter()
        yield s


class TestParseTradierEpochMs:
    """Unit tests for the H6 ms-epoch helper."""

    def test_parses_positive_ms(self):
        """A positive ms-epoch returns a tz-aware UTC datetime."""
        # 2026-04-16T12:00:00Z = 1776000000 * 1000 ms.
        ms = int(datetime(2026, 4, 16, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        parsed = _parse_tradier_epoch_ms(ms)
        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed.year == 2026 and parsed.month == 4 and parsed.day == 16

    @pytest.mark.parametrize("bad", [None, "", "not_a_number", -1, 0])
    def test_returns_none_for_invalid(self, bad):
        """Missing / non-numeric / non-positive values yield None."""
        assert _parse_tradier_epoch_ms(bad) is None


class TestOnConflictPath:
    """H5 (audit) verifies the dedup path increments the right counter."""

    @pytest.mark.asyncio
    async def test_first_insert_increments_inserted(self):
        """RETURNING row -> ``quotes_inserted`` increments, not the dedup tally."""
        tradier = AsyncMock()
        tradier.get_quotes = AsyncMock(
            return_value={"quotes": {"quote": [
                {"symbol": "SPX", "last": 5000.0, "bid": 4999.0, "ask": 5001.0,
                 "trade_date": int(datetime(2026, 4, 16, 13, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)},
            ]}}
        )

        session, captured = _make_session_mock(insert_returns_row=True)

        with patch("spx_backend.jobs.quote_job.SessionLocal", return_value=session):
            job = QuoteJob(tradier=tradier)
            result = await job.run_once()

        assert result["skipped"] is False
        assert result["quotes_inserted"] == 1
        assert result["quotes_dedup_skipped"] == 0

    @pytest.mark.asyncio
    async def test_conflict_increments_dedup_skipped(self):
        """fetchone() returning None means the conflict suppressed the row."""
        tradier = AsyncMock()
        tradier.get_quotes = AsyncMock(
            return_value={"quotes": {"quote": [
                {"symbol": "SPX", "last": 5000.0, "bid": 4999.0, "ask": 5001.0},
            ]}}
        )

        session, captured = _make_session_mock(insert_returns_row=False)

        with patch("spx_backend.jobs.quote_job.SessionLocal", return_value=session):
            job = QuoteJob(tradier=tradier)
            result = await job.run_once()

        assert result["skipped"] is False
        assert result["quotes_inserted"] == 0
        assert result["quotes_dedup_skipped"] == 1

    @pytest.mark.asyncio
    async def test_insert_sql_carries_minute_bucket_on_conflict_clause(self):
        """The literal SQL must contain the H5 expression-index target."""
        tradier = AsyncMock()
        tradier.get_quotes = AsyncMock(
            return_value={"quotes": {"quote": [
                {"symbol": "SPX", "last": 5000.0},
            ]}}
        )

        session, captured = _make_session_mock(insert_returns_row=True)

        with patch("spx_backend.jobs.quote_job.SessionLocal", return_value=session):
            job = QuoteJob(tradier=tradier)
            await job.run_once()

        underlying_inserts = [sql for sql, _ in captured if "INSERT INTO underlying_quotes" in sql]
        assert underlying_inserts, "expected at least one INSERT INTO underlying_quotes"
        sql = underlying_inserts[0]
        # H5 (audit): ON CONFLICT must target the minute-bucket
        # expression index, not the legacy unique constraint. We use
        # date_bin (IMMUTABLE) rather than date_trunc (STABLE) because
        # PG requires unique-index expressions to be IMMUTABLE.
        assert "date_bin('1 minute', ts, TIMESTAMPTZ '2000-01-01 00:00:00+00')" in sql
        assert "DO NOTHING" in sql
        assert "RETURNING quote_id" in sql

    @pytest.mark.asyncio
    async def test_insert_params_bind_vendor_ts(self):
        """H6: trade_date should be parsed and bound as vendor_ts."""
        ms = int(datetime(2026, 4, 16, 14, 30, 12, tzinfo=timezone.utc).timestamp() * 1000)

        tradier = AsyncMock()
        tradier.get_quotes = AsyncMock(
            return_value={"quotes": {"quote": [
                {"symbol": "SPX", "last": 5000.0, "trade_date": ms},
            ]}}
        )

        session, captured = _make_session_mock(insert_returns_row=True)

        with patch("spx_backend.jobs.quote_job.SessionLocal", return_value=session):
            job = QuoteJob(tradier=tradier)
            await job.run_once()

        bound_params = [
            params for sql, params in captured
            if "INSERT INTO underlying_quotes" in sql
        ]
        assert bound_params, "expected at least one underlying_quotes insert with bound params"
        params = bound_params[0]
        assert "vendor_ts" in params
        assert params["vendor_ts"] is not None
        # Round-trip: same ms-epoch should reproduce the same UTC datetime.
        assert params["vendor_ts"].year == 2026
        assert params["vendor_ts"].minute == 30
