"""Comprehensive tests for TradePnlJob: mark-to-market, TP/SL, expiry, stale, no-legs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.jobs.trade_pnl_job import TradePnlJob, _mid, build_trade_pnl_job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _make_trade(
    trade_id=1,
    entry_credit=2.0,
    contracts=1,
    contract_multiplier=100,
    expiration=None,
    take_profit_target=None,
    stop_loss_target=None,
    max_profit=None,
    max_loss=None,
):
    """Create a fake trade row matching the column names from the SQL SELECT."""
    return SimpleNamespace(
        trade_id=trade_id,
        entry_credit=entry_credit,
        contracts=contracts,
        contract_multiplier=contract_multiplier,
        expiration=expiration,
        take_profit_target=take_profit_target,
        stop_loss_target=stop_loss_target,
        max_profit=max_profit,
        max_loss=max_loss,
    )


def _make_mark(short_bid=1.0, short_ask=1.2, long_bid=0.3, long_ask=0.5, snapshot_id=10, ts=None):
    """Create a fake spread mark dict."""
    return {
        "snapshot_id": snapshot_id,
        "ts": ts or datetime.now(tz=UTC),
        "short_bid": short_bid,
        "short_ask": short_ask,
        "long_bid": long_bid,
        "long_ask": long_ask,
    }


class FakeSession:
    """Minimal async session stub tracking execute calls and providing commit."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params or {}))
        return MagicMock(fetchall=MagicMock(return_value=[]))

    async def commit(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Unit: _mid
# ---------------------------------------------------------------------------


class TestMid:
    def test_both_present(self):
        assert _mid(1.0, 2.0) == 1.5

    def test_bid_none(self):
        assert _mid(None, 2.0) is None

    def test_ask_none(self):
        assert _mid(1.0, None) is None

    def test_both_none(self):
        assert _mid(None, None) is None


# ---------------------------------------------------------------------------
# Unit: _bulk_trade_legs
# ---------------------------------------------------------------------------


class TestBulkTradeLegs:
    @pytest.mark.asyncio
    async def test_empty_ids(self):
        """When no trade IDs are provided, return empty dict without querying."""
        job = TradePnlJob()
        result = await job._bulk_trade_legs(FakeSession(), [])
        assert result == {}

    @pytest.mark.asyncio
    async def test_parses_legs(self):
        """Correctly separates short and long legs from bulk query results."""
        rows = [
            SimpleNamespace(trade_id=1, leg_index=0, option_symbol="SPX230120P04000", side="STO", qty=1, entry_price=3.0),
            SimpleNamespace(trade_id=1, leg_index=1, option_symbol="SPX230120P03950", side="BTO", qty=1, entry_price=1.0),
            SimpleNamespace(trade_id=2, leg_index=0, option_symbol="SPX230120C04100", side="STO", qty=1, entry_price=2.0),
            SimpleNamespace(trade_id=2, leg_index=1, option_symbol="SPX230120C04150", side="BTO", qty=1, entry_price=0.5),
        ]

        class _Sess:
            async def execute(self, stmt, params=None):
                return MagicMock(fetchall=MagicMock(return_value=rows))

        job = TradePnlJob()
        result = await job._bulk_trade_legs(_Sess(), [1, 2])
        assert 1 in result
        assert 2 in result
        short_1, long_1 = result[1]
        assert short_1["side"] == "STO"
        assert long_1["side"] == "BTO"

    @pytest.mark.asyncio
    async def test_missing_leg_excluded(self):
        """Trades with only one leg are excluded from the result."""
        rows = [
            SimpleNamespace(trade_id=3, leg_index=0, option_symbol="SPX230120P04000", side="STO", qty=1, entry_price=3.0),
        ]

        class _Sess:
            async def execute(self, stmt, params=None):
                return MagicMock(fetchall=MagicMock(return_value=rows))

        job = TradePnlJob()
        result = await job._bulk_trade_legs(_Sess(), [3])
        assert 3 not in result


# ---------------------------------------------------------------------------
# Integration-style: run_once scenarios
# ---------------------------------------------------------------------------


class TestRunOnce:
    """Test the full run_once flow with mocked DB and settings."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch("spx_backend.jobs.trade_pnl_job.settings") as mock_settings:
            mock_settings.tz = "America/New_York"
            mock_settings.trade_pnl_enabled = True
            mock_settings.trade_pnl_allow_outside_rth = True
            mock_settings.trade_pnl_mark_max_age_minutes = 30
            mock_settings.trade_pnl_take_profit_pct = 0.50
            mock_settings.trade_pnl_stop_loss_pct = 1.00
            mock_settings.trade_pnl_contract_multiplier = 100
            mock_settings.decision_contracts = 1
            self.mock_settings = mock_settings
            yield

    @pytest.mark.asyncio
    async def test_disabled(self):
        """When trade_pnl_enabled is False, the job is skipped."""
        self.mock_settings.trade_pnl_enabled = False
        job = TradePnlJob()
        result = await job.run_once()
        assert result["skipped"] is True
        assert result["reason"] == "trade_pnl_disabled"

    def _make_session_with_trades(self, trades):
        """Create a FakeSession returning the given trades from the SELECT."""
        session = FakeSession()
        open_result = MagicMock()
        open_result.fetchall = MagicMock(return_value=trades)

        async def fake_execute(stmt, params=None):
            session.calls.append((str(stmt), params or {}))
            return open_result

        session.execute = fake_execute
        return session

    @pytest.mark.asyncio
    async def test_normal_mark_to_market(self):
        """Normal cycle: open trade with fresh mark gets PnL updated, not closed."""
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=1, entry_credit=2.0, max_profit=200.0,
            expiration=date.today() + timedelta(days=5),
        )
        # exit_cost = 2.1 - 0.5 = 1.6, pnl = (2.0 - 1.6)*100 = 40 < TP(100)
        fresh_mark = _make_mark(
            short_bid=2.0, short_ask=2.2, long_bid=0.4, long_ask=0.6,
            ts=now_utc - timedelta(minutes=2),
        )
        session = self._make_session_with_trades([trade])
        legs = {1: (
            {"leg_index": 0, "option_symbol": "STO_SYM", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "BTO_SYM", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=fresh_mark),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["skipped"] is False
        assert result["updated"] >= 1
        assert result["closed"] == 0

    @pytest.mark.asyncio
    async def test_take_profit_close(self):
        """Trade at significant profit closes with TAKE_PROFIT_50."""
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=2, entry_credit=2.0, max_profit=200.0,
            expiration=date.today() + timedelta(days=5),
        )
        fresh_mark = _make_mark(
            short_bid=0.01, short_ask=0.03, long_bid=0.01, long_ask=0.03,
            ts=now_utc - timedelta(minutes=1),
        )
        session = self._make_session_with_trades([trade])
        legs = {2: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=fresh_mark),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["closed"] >= 1
        tp_calls = [c for c in session.calls if "TAKE_PROFIT" in str(c[1].get("exit_reason", ""))]
        assert len(tp_calls) > 0

    @pytest.mark.asyncio
    async def test_stop_loss_close(self):
        """Trade in deep loss closes with STOP_LOSS."""
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=3, entry_credit=2.0, max_profit=200.0,
            expiration=date.today() + timedelta(days=5),
        )
        deep_loss_mark = _make_mark(
            short_bid=5.0, short_ask=5.2, long_bid=0.1, long_ask=0.3,
            ts=now_utc - timedelta(minutes=1),
        )
        session = self._make_session_with_trades([trade])
        legs = {3: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=deep_loss_mark),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["closed"] >= 1
        close_calls = [c for c in session.calls if "STOP_LOSS" in str(c[1].get("exit_reason", ""))]
        assert len(close_calls) > 0

    @pytest.mark.asyncio
    async def test_expired_trade_closes(self):
        """Trade past expiration date closes with EXPIRED, even without legs."""
        expired_trade = _make_trade(
            trade_id=4, entry_credit=2.0,
            expiration=date.today() - timedelta(days=2),
            max_loss=300.0,
        )
        session = self._make_session_with_trades([expired_trade])

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value={}),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["expired_closed"] == 1
        assert result["closed"] == 1
        close_calls = [c for c in session.calls if c[1].get("exit_reason") == "EXPIRED"]
        assert len(close_calls) > 0

    @pytest.mark.asyncio
    async def test_expired_trade_with_mark(self):
        """Expired trade with last available mark uses mark PnL, not max_loss."""
        now_utc = datetime.now(tz=UTC)
        expired_trade = _make_trade(
            trade_id=5, entry_credit=2.0,
            expiration=date.today() - timedelta(days=1),
            max_loss=300.0,
        )
        old_mark = _make_mark(
            short_bid=0.5, short_ask=0.7, long_bid=0.2, long_ask=0.4,
            ts=now_utc - timedelta(hours=24),
        )
        session = self._make_session_with_trades([expired_trade])
        legs = {5: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=old_mark),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["expired_closed"] == 1
        close_calls = [c for c in session.calls if c[1].get("exit_reason") == "EXPIRED"]
        assert len(close_calls) > 0
        pnl = close_calls[0][1].get("realized_pnl") or close_calls[0][1].get("current_pnl")
        assert pnl is not None
        assert pnl != -300.0

    @pytest.mark.asyncio
    async def test_stale_mark_skipped(self):
        """Trades with stale marks are skipped (not force mode)."""
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=6, entry_credit=2.0,
            expiration=date.today() + timedelta(days=5),
        )
        stale_mark = _make_mark(
            short_bid=1.0, short_ask=1.2, long_bid=0.3, long_ask=0.5,
            ts=now_utc - timedelta(minutes=120),
        )
        session = self._make_session_with_trades([trade])
        legs = {6: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=stale_mark),
        ):
            job = TradePnlJob()
            result = await job.run_once()

        assert result["skipped_stale"] == 1
        assert result["updated"] == 0

    @pytest.mark.asyncio
    async def test_no_legs_skipped(self):
        """Trades with missing legs are skipped."""
        trade = _make_trade(
            trade_id=7, entry_credit=2.0,
            expiration=date.today() + timedelta(days=5),
        )
        session = self._make_session_with_trades([trade])

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value={}),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["skipped_no_legs"] == 1
        assert result["updated"] == 0

    @pytest.mark.asyncio
    async def test_no_mark_skipped(self):
        """Trades where the mark query returns None are skipped."""
        trade = _make_trade(
            trade_id=8, entry_credit=2.0,
            expiration=date.today() + timedelta(days=5),
        )
        session = self._make_session_with_trades([trade])
        legs = {8: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=None),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["skipped_no_mark"] == 1
        assert result["updated"] == 0

    @pytest.mark.asyncio
    async def test_no_open_trades(self):
        """When there are no open trades, run_once completes with all counters at zero."""
        session = self._make_session_with_trades([])

        with patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["updated"] == 0
        assert result["closed"] == 0
        assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Unit: build_trade_pnl_job factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_no_cache(self):
        job = build_trade_pnl_job()
        assert job.clock_cache is None

    def test_factory_with_cache(self):
        mock_cache = MagicMock()
        job = build_trade_pnl_job(clock_cache=mock_cache)
        assert job.clock_cache is mock_cache
