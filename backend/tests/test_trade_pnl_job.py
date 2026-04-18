"""Comprehensive tests for TradePnlJob: mark-to-market, TP/SL, expiry, stale, no-legs."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.jobs.trade_pnl_job import TradePnlJob, build_trade_pnl_job
from spx_backend.services.portfolio_manager import PortfolioManager
from spx_backend.utils.pricing import mid_price


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _today_et() -> date:
    """Return today's date in ET, matching the timezone used by TradePnlJob.run_once."""
    return datetime.now(tz=ET).date()


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
    strategy_type="credit_vertical_put",
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
        strategy_type=strategy_type,
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
# Unit: mid_price (delegated to shared util; smoke tests only)
# ---------------------------------------------------------------------------


class TestMid:
    """Smoke checks that the trade_pnl_job uses the strict shared helper.

    Detailed acceptance/rejection scenarios live in
    ``test_utils_pricing.py``; these are kept here so the trade_pnl_job
    test module trips fast if the import wiring regresses (e.g. if a
    future refactor reintroduces a permissive local helper).
    """

    def test_both_present(self):
        assert mid_price(1.0, 2.0) == 1.5

    def test_bid_none(self):
        assert mid_price(None, 2.0) is None

    def test_ask_none(self):
        assert mid_price(1.0, None) is None

    def test_both_none(self):
        assert mid_price(None, None) is None


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
            SimpleNamespace(trade_id=1, leg_index=0, option_symbol="SPX230120P04000", side="STO", qty=1, entry_price=3.0, strike=4000.0),
            SimpleNamespace(trade_id=1, leg_index=1, option_symbol="SPX230120P03950", side="BTO", qty=1, entry_price=1.0, strike=3950.0),
            SimpleNamespace(trade_id=2, leg_index=0, option_symbol="SPX230120C04100", side="STO", qty=1, entry_price=2.0, strike=4100.0),
            SimpleNamespace(trade_id=2, leg_index=1, option_symbol="SPX230120C04150", side="BTO", qty=1, entry_price=0.5, strike=4150.0),
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
            SimpleNamespace(trade_id=3, leg_index=0, option_symbol="SPX230120P04000", side="STO", qty=1, entry_price=3.0, strike=4000.0),
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
    """Test the full run_once flow with mocked DB and settings.

    The class previously pinned ``portfolio_enabled=False`` to keep the
    closure path from instantiating a real ``PortfolioManager``.  After the
    flag was removed, the closure branch always runs, so we patch both
    ``_get_portfolio_manager`` (to avoid a live ``PortfolioManager()`` /
    ``begin_day`` DB hit) and ``_close_with_portfolio`` (to skip the
    ``record_closure`` write) for every test in the class.
    ``TestPortfolioClosure`` overrides these with its own real-pm mocks
    to assert ``record_closure`` is invoked.
    """

    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with (
            patch("spx_backend.jobs.trade_pnl_job.settings") as mock_settings,
            patch.object(TradePnlJob, "_get_portfolio_manager", new_callable=AsyncMock),
            patch.object(TradePnlJob, "_close_with_portfolio", new_callable=AsyncMock),
        ):
            mock_settings.tz = "America/New_York"
            mock_settings.trade_pnl_enabled = True
            mock_settings.trade_pnl_allow_outside_rth = True
            mock_settings.trade_pnl_mark_max_age_minutes = 30
            mock_settings.trade_pnl_take_profit_pct = 0.50
            mock_settings.trade_pnl_stop_loss_enabled = True
            mock_settings.trade_pnl_stop_loss_basis = "max_profit"
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
            expiration=_today_et() + timedelta(days=5),
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
            expiration=_today_et() + timedelta(days=5),
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
            expiration=_today_et() + timedelta(days=5),
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
    async def test_stop_loss_disabled_no_close(self):
        """When trade_pnl_stop_loss_enabled is False, deep-loss trade stays open."""
        self.mock_settings.trade_pnl_stop_loss_enabled = False
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=30, entry_credit=2.0, max_profit=200.0,
            expiration=_today_et() + timedelta(days=5),
        )
        deep_loss_mark = _make_mark(
            short_bid=5.0, short_ask=5.2, long_bid=0.1, long_ask=0.3,
            ts=now_utc - timedelta(minutes=1),
        )
        session = self._make_session_with_trades([trade])
        legs = {30: (
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

        assert result["closed"] == 0
        assert result["updated"] >= 1
        sl_calls = [c for c in session.calls if "STOP_LOSS" in str(c[1].get("exit_reason", ""))]
        assert len(sl_calls) == 0

    @pytest.mark.asyncio
    async def test_stop_loss_disabled_existing_target_ignored(self):
        """Trade with stop_loss_target already set in DB is not closed when flag is off."""
        self.mock_settings.trade_pnl_stop_loss_enabled = False
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=31, entry_credit=2.0, max_profit=200.0,
            stop_loss_target=200.0,
            expiration=_today_et() + timedelta(days=5),
        )
        deep_loss_mark = _make_mark(
            short_bid=5.0, short_ask=5.2, long_bid=0.1, long_ask=0.3,
            ts=now_utc - timedelta(minutes=1),
        )
        session = self._make_session_with_trades([trade])
        legs = {31: (
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

        assert result["closed"] == 0
        assert result["updated"] >= 1
        sl_calls = [c for c in session.calls if "STOP_LOSS" in str(c[1].get("exit_reason", ""))]
        assert len(sl_calls) == 0

    @pytest.mark.asyncio
    async def test_expired_trade_closes(self):
        """Trade past expiration date closes with EXPIRED, even without legs."""
        expired_trade = _make_trade(
            trade_id=4, entry_credit=2.0,
            expiration=_today_et() - timedelta(days=2),
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
            expiration=_today_et() - timedelta(days=1),
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
            expiration=_today_et() + timedelta(days=5),
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
            expiration=_today_et() + timedelta(days=5),
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
            expiration=_today_et() + timedelta(days=5),
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


# ---------------------------------------------------------------------------
# Expiration closes outside RTH (P2)
# ---------------------------------------------------------------------------


class TestExpirationOutsideRTH:
    """Verify that expired trades close even when market is closed.

    See ``TestRunOnce`` for why ``_get_portfolio_manager`` and
    ``_close_with_portfolio`` are patched at the fixture level:
    closure-flow tests in this class focus on the expiration branch and
    do not exercise the portfolio side, but the code path always invokes
    these helpers now that ``portfolio_enabled`` is gone.
    """

    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with (
            patch("spx_backend.jobs.trade_pnl_job.settings") as mock_settings,
            patch.object(TradePnlJob, "_get_portfolio_manager", new_callable=AsyncMock),
            patch.object(TradePnlJob, "_close_with_portfolio", new_callable=AsyncMock),
        ):
            mock_settings.tz = "America/New_York"
            mock_settings.trade_pnl_enabled = True
            mock_settings.trade_pnl_allow_outside_rth = False
            mock_settings.trade_pnl_mark_max_age_minutes = 30
            mock_settings.trade_pnl_take_profit_pct = 0.50
            mock_settings.trade_pnl_stop_loss_enabled = False
            mock_settings.trade_pnl_stop_loss_pct = 1.00
            mock_settings.trade_pnl_contract_multiplier = 100
            mock_settings.decision_contracts = 1
            self.mock_settings = mock_settings
            yield

    @pytest.mark.asyncio
    async def test_expired_trade_closes_when_market_closed(self):
        """Expired trades should close even when market is closed (weekend)."""
        expired_trade = _make_trade(
            trade_id=50, entry_credit=2.0,
            expiration=_today_et() - timedelta(days=2),
            max_loss=300.0,
        )
        session = FakeSession()
        open_result = MagicMock()
        open_result.fetchall = MagicMock(return_value=[expired_trade])

        async def fake_execute(stmt, params=None):
            session.calls.append((str(stmt), params or {}))
            return open_result

        session.execute = fake_execute

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value={}),
            patch.object(TradePnlJob, "_market_open", new_callable=AsyncMock, return_value=False),
        ):
            job = TradePnlJob()
            result = await job.run_once()

        assert result["expired_closed"] == 1
        assert result["closed"] == 1
        close_calls = [c for c in session.calls if c[1].get("exit_reason") == "EXPIRED"]
        assert len(close_calls) > 0

    @pytest.mark.asyncio
    async def test_active_trade_skipped_when_market_closed(self):
        """Non-expired trades should NOT be processed when market is closed."""
        active_trade = _make_trade(
            trade_id=51, entry_credit=2.0,
            expiration=_today_et() + timedelta(days=5),
        )
        session = FakeSession()
        open_result = MagicMock()
        open_result.fetchall = MagicMock(return_value=[active_trade])

        async def fake_execute(stmt, params=None):
            session.calls.append((str(stmt), params or {}))
            return open_result

        session.execute = fake_execute

        legs = {51: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_market_open", new_callable=AsyncMock, return_value=False),
        ):
            job = TradePnlJob()
            result = await job.run_once()

        assert result["skipped"] is True
        assert result["reason"] == "market_closed"
        assert result["updated"] == 0
        assert result["closed"] == 0

    @pytest.mark.asyncio
    async def test_mixed_expired_and_active_outside_rth(self):
        """Outside RTH: expired trades close, active trades deferred."""
        expired_trade = _make_trade(
            trade_id=52, entry_credit=2.0,
            expiration=_today_et() - timedelta(days=1),
            max_loss=300.0,
        )
        active_trade = _make_trade(
            trade_id=53, entry_credit=2.0,
            expiration=_today_et() + timedelta(days=5),
        )
        session = FakeSession()
        open_result = MagicMock()
        open_result.fetchall = MagicMock(return_value=[expired_trade, active_trade])

        async def fake_execute(stmt, params=None):
            session.calls.append((str(stmt), params or {}))
            return open_result

        session.execute = fake_execute

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value={}),
            patch.object(TradePnlJob, "_market_open", new_callable=AsyncMock, return_value=False),
        ):
            job = TradePnlJob()
            result = await job.run_once()

        assert result["expired_closed"] == 1
        assert result["closed"] == 1
        assert result["skipped"] is False
        assert result["updated"] == 0


# ---------------------------------------------------------------------------
# Portfolio closure integration (P0)
# ---------------------------------------------------------------------------


class TestPortfolioClosure:
    """Verify trade closures flow PnL back to PortfolioManager."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        with patch("spx_backend.jobs.trade_pnl_job.settings") as mock_settings:
            mock_settings.tz = "America/New_York"
            mock_settings.trade_pnl_enabled = True
            mock_settings.trade_pnl_allow_outside_rth = True
            mock_settings.trade_pnl_mark_max_age_minutes = 30
            mock_settings.trade_pnl_take_profit_pct = 0.50
            mock_settings.trade_pnl_stop_loss_enabled = False
            mock_settings.trade_pnl_stop_loss_pct = 1.00
            mock_settings.trade_pnl_contract_multiplier = 100
            mock_settings.decision_contracts = 1
            self.mock_settings = mock_settings
            yield

    def _make_session_with_trades(self, trades):
        """Create a FakeSession returning trades from SELECT, with rowcount support."""
        session = FakeSession()
        open_result = MagicMock()
        open_result.fetchall = MagicMock(return_value=trades)
        open_result.rowcount = 1

        async def fake_execute(stmt, params=None):
            session.calls.append((str(stmt), params or {}))
            return open_result

        session.execute = fake_execute
        return session

    @pytest.mark.asyncio
    async def test_tp_close_calls_record_closure(self):
        """Take-profit close should call PortfolioManager.record_closure."""
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=60, entry_credit=2.0, max_profit=200.0,
            expiration=_today_et() + timedelta(days=5),
        )
        fresh_mark = _make_mark(
            short_bid=0.01, short_ask=0.03, long_bid=0.01, long_ask=0.03,
            ts=now_utc - timedelta(minutes=1),
        )
        session = self._make_session_with_trades([trade])
        legs = {60: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1, "entry_price": 3.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1, "entry_price": 1.0},
        )}

        mock_pm = AsyncMock(spec=PortfolioManager)
        mock_pm.begin_day = AsyncMock()
        mock_pm.record_closure = AsyncMock()

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=fresh_mark),
            patch("spx_backend.jobs.trade_pnl_job.PortfolioManager", return_value=mock_pm),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["closed"] >= 1
        mock_pm.begin_day.assert_awaited_once()
        mock_pm.record_closure.assert_awaited_once()
        call_args = mock_pm.record_closure.await_args
        assert call_args[0][0] == 60
        assert call_args[0][1] > 0

    @pytest.mark.asyncio
    async def test_expired_close_calls_record_closure(self):
        """Expired trade close should call PortfolioManager.record_closure."""
        expired_trade = _make_trade(
            trade_id=61, entry_credit=2.0,
            expiration=_today_et() - timedelta(days=1),
            max_loss=300.0,
        )
        session = self._make_session_with_trades([expired_trade])

        mock_pm = AsyncMock(spec=PortfolioManager)
        mock_pm.begin_day = AsyncMock()
        mock_pm.record_closure = AsyncMock()

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value={}),
            patch("spx_backend.jobs.trade_pnl_job.PortfolioManager", return_value=mock_pm),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["expired_closed"] == 1
        mock_pm.begin_day.assert_awaited_once()
        mock_pm.record_closure.assert_awaited_once()
        call_args = mock_pm.record_closure.await_args
        assert call_args[0][0] == 61

# NOTE: ``test_portfolio_disabled_skips_record_closure`` was deleted along
# with the ``portfolio_enabled`` flag in the online-ML decommission -- the
# closure path is unconditional now and there is no "disabled" branch to
# assert against.


# ---------------------------------------------------------------------------
# Multi-contract PnL scaling
# ---------------------------------------------------------------------------


class TestMultiContractPnl(TestRunOnce):
    """Verify PnL scales correctly with contracts > 1 and varying multipliers."""

    @pytest.mark.asyncio
    async def test_mtm_scales_with_contracts(self):
        """MTM PnL = (entry_credit - exit_cost) * contracts * multiplier."""
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=80, entry_credit=1.5, contracts=5, contract_multiplier=100,
            expiration=_today_et() + timedelta(days=5),
        )
        mark = _make_mark(
            short_bid=0.90, short_ask=1.10, long_bid=0.40, long_ask=0.60,
            ts=now_utc - timedelta(minutes=1),
        )
        session = self._make_session_with_trades([trade])
        legs = {80: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 5, "entry_price": 2.0, "strike": 6000.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 5, "entry_price": 0.5, "strike": 5975.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=mark),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["updated"] >= 1
        # exit_cost = (0.90+1.10)/2 - (0.40+0.60)/2 = 1.0 - 0.5 = 0.5
        # pnl = (1.5 - 0.5) * 5 * 100 = 500.0
        pnl_calls = [c for c in session.calls if "current_pnl" in str(c[1])]
        assert len(pnl_calls) > 0
        actual_pnl = pnl_calls[0][1]["current_pnl"]
        assert abs(actual_pnl - 500.0) < 0.01

    @pytest.mark.asyncio
    async def test_tp_threshold_with_multi_contract(self):
        """TP fires when dollar PnL >= take_profit_target (also in multi-contract)."""
        now_utc = datetime.now(tz=UTC)
        trade = _make_trade(
            trade_id=81, entry_credit=2.0, contracts=3, contract_multiplier=100,
            expiration=_today_et() + timedelta(days=5),
            max_profit=600.0,
        )
        # Near-zero exit cost → pnl ≈ 2.0 * 3 * 100 = 600 (== max_profit)
        mark = _make_mark(
            short_bid=0.01, short_ask=0.03, long_bid=0.01, long_ask=0.03,
            ts=now_utc - timedelta(minutes=1),
        )
        session = self._make_session_with_trades([trade])
        legs = {81: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 3, "entry_price": 3.0, "strike": 6000.0},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 3, "entry_price": 1.0, "strike": 5975.0},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=mark),
        ):
            job = TradePnlJob()
            result = await job.run_once(force=True)

        assert result["closed"] >= 1
        tp_calls = [c for c in session.calls if "TAKE_PROFIT" in str(c[1].get("exit_reason", ""))]
        assert len(tp_calls) > 0


# ---------------------------------------------------------------------------
# Intrinsic settlement for expired trades
# ---------------------------------------------------------------------------


class TestIntrinsicSettlement:
    """Verify intrinsic-value settlement when marks are missing.

    The helper now takes ``option_right`` directly ("put" / "call" /
    None) instead of branching on ``strategy_type``; the live caller
    resolves the right via
    :func:`spx_backend.utils.options.resolve_option_right` and passes
    the resolved value in.  See ``test_utils_options.py`` for the
    full resolution-precedence test matrix.
    """

    def test_put_spread_expires_otm(self):
        """Put credit spread fully OTM at expiry: exit_cost = 0."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right="put",
            short_strike=5800.0,
            long_strike=5775.0,
            spot=5900.0,
        )
        assert result == 0.0

    def test_put_spread_expires_full_loss(self):
        """Put credit spread fully ITM at expiry: exit_cost = width."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right="put",
            short_strike=5800.0,
            long_strike=5775.0,
            spot=5700.0,
        )
        assert result == 25.0

    def test_put_spread_partial_intrinsic(self):
        """Put credit spread partially ITM: short has intrinsic, long worthless."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right="put",
            short_strike=5800.0,
            long_strike=5775.0,
            spot=5790.0,
        )
        # short intrinsic = max(5800-5790, 0) = 10, long = max(5775-5790, 0) = 0
        assert abs(result - 10.0) < 0.01

    def test_call_spread_expires_otm(self):
        """Call credit spread fully OTM at expiry: exit_cost = 0."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right="call",
            short_strike=5900.0,
            long_strike=5925.0,
            spot=5800.0,
        )
        assert result == 0.0

    def test_call_spread_expires_full_loss(self):
        """Call credit spread fully ITM at expiry: exit_cost = width."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right="call",
            short_strike=5900.0,
            long_strike=5925.0,
            spot=6000.0,
        )
        assert result == 25.0

    def test_call_spread_partial_intrinsic(self):
        """Call credit spread partially ITM: short has intrinsic, long worthless."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right="call",
            short_strike=5900.0,
            long_strike=5925.0,
            spot=5910.0,
        )
        # short intrinsic = max(5910-5900, 0) = 10, long = max(5910-5925, 0) = 0
        assert abs(result - 10.0) < 0.01

    def test_returns_none_when_spot_missing(self):
        """Returns None when spot is unavailable (triggers tier-3 fallback)."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right="put",
            short_strike=5800.0,
            long_strike=5775.0,
            spot=None,
        )
        assert result is None

    def test_returns_none_for_unknown_right(self):
        """Returns None when option_right cannot be resolved."""
        result = TradePnlJob._intrinsic_exit_cost(
            option_right=None,
            short_strike=5800.0,
            long_strike=5775.0,
            spot=5900.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tier-3 NULL max_loss hard-fail
# ---------------------------------------------------------------------------


class TestTier3NullMaxLoss:
    """Tier-3 fallback for expired trades hard-fails on NULL ``max_loss``.

    Previously the job silently booked PnL = 0 when both tier 1 (mark)
    and tier 2 (intrinsic from spot/strikes) failed AND ``max_loss`` was
    NULL.  That hides a data-integrity issue (every closed trade should
    have a known max_loss recorded at entry) and silently misstates
    portfolio equity.

    The new contract: raise ``RuntimeError`` so the surrounding session
    rolls back -- the trade stays OPEN and the next ``trade_pnl_job``
    run can retry once data has been backfilled."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self):
        # See ``TestRunOnce`` for why ``_get_portfolio_manager`` and
        # ``_close_with_portfolio`` are patched at the fixture level:
        # this class focuses on max-loss validation and does not exercise
        # the real portfolio writer.
        with (
            patch("spx_backend.jobs.trade_pnl_job.settings") as mock_settings,
            patch.object(TradePnlJob, "_get_portfolio_manager", new_callable=AsyncMock),
            patch.object(TradePnlJob, "_close_with_portfolio", new_callable=AsyncMock),
        ):
            mock_settings.tz = "America/New_York"
            mock_settings.trade_pnl_enabled = True
            mock_settings.trade_pnl_allow_outside_rth = True
            mock_settings.trade_pnl_mark_max_age_minutes = 30
            mock_settings.trade_pnl_take_profit_pct = 0.50
            mock_settings.trade_pnl_stop_loss_enabled = False
            mock_settings.trade_pnl_stop_loss_pct = 1.00
            mock_settings.trade_pnl_contract_multiplier = 100
            mock_settings.decision_contracts = 1
            self.mock_settings = mock_settings
            yield

    def _make_session_with_trades(self, trades):
        """Same FakeSession-with-rowcount pattern as TestPortfolioClosure."""
        session = FakeSession()
        open_result = MagicMock()
        open_result.fetchall = MagicMock(return_value=trades)
        open_result.rowcount = 1

        async def fake_execute(stmt, params=None):
            session.calls.append((str(stmt), params or {}))
            return open_result

        session.execute = fake_execute
        return session

    @pytest.mark.asyncio
    async def test_expired_trade_with_no_legs_and_null_max_loss_raises(self):
        """No legs at all + NULL max_loss -> RuntimeError, no closure."""
        trade = _make_trade(
            trade_id=900,
            entry_credit=2.0,
            expiration=_today_et() - timedelta(days=1),
            max_loss=None,
        )
        session = self._make_session_with_trades([trade])

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value={}),
            patch.object(TradePnlJob, "_market_open", new_callable=AsyncMock, return_value=True),
        ):
            job = TradePnlJob()
            # Note: force=True is intentionally omitted here -- Phase 1 of
            # run_once always processes _close_expired_trades regardless of
            # the force flag (force only gates Phase 2's RTH check, and we
            # already set trade_pnl_allow_outside_rth=True in the fixture).
            with pytest.raises(RuntimeError, match="trade.max_loss is NULL"):
                await job.run_once()

        # No close-trade UPDATE should have run with this trade_id.
        close_calls = [
            c for c in session.calls
            if c[1].get("trade_id") == 900
            and "EXPIRED" in str(c[1].get("exit_reason", ""))
        ]
        assert len(close_calls) == 0, (
            "trade with NULL max_loss must NOT be closed; the surrounding "
            "session should roll back and the trade stays OPEN for retry"
        )

    @pytest.mark.asyncio
    async def test_expired_trade_with_legs_no_spot_and_null_max_loss_raises(self):
        """Legs present but no spot/intrinsic AND NULL max_loss -> RuntimeError.

        This is the tier-2 -> tier-3 path: legs exist, but
        ``_intrinsic_exit_cost`` returns None because no spot is loaded
        (we patch ``_latest_spot`` to return None) and there's no last
        snapshot mark.  With NULL max_loss the tier-3 fallback must hard
        fail rather than book PnL = 0.
        """
        trade = _make_trade(
            trade_id=901,
            entry_credit=2.0,
            expiration=_today_et() - timedelta(days=1),
            max_loss=None,
        )
        session = self._make_session_with_trades([trade])
        legs = {901: (
            {"leg_index": 0, "option_symbol": "S", "side": "STO", "qty": 1,
             "entry_price": 3.0, "strike": 5800.0, "option_right": "put"},
            {"leg_index": 1, "option_symbol": "L", "side": "BTO", "qty": 1,
             "entry_price": 1.0, "strike": 5775.0, "option_right": "put"},
        )}

        with (
            patch("spx_backend.jobs.trade_pnl_job.SessionLocal", return_value=session),
            patch.object(TradePnlJob, "_bulk_trade_legs", new_callable=AsyncMock, return_value=legs),
            patch.object(TradePnlJob, "_market_open", new_callable=AsyncMock, return_value=True),
            patch.object(TradePnlJob, "_latest_spread_mark", new_callable=AsyncMock, return_value=None),
            patch.object(TradePnlJob, "_latest_spot", new_callable=AsyncMock, return_value=None),
        ):
            job = TradePnlJob()
            # See note in test_expired_trade_with_no_legs_and_null_max_loss_raises:
            # force=True is unnecessary because Phase 1 always runs.
            with pytest.raises(RuntimeError, match="trade.max_loss is NULL"):
                await job.run_once()

        # Mirror the close_calls assertion from the no-legs test: even when
        # legs exist but tier-2/tier-3 fail with NULL max_loss, no close
        # UPDATE should be issued for this trade.  The session's outer
        # transaction must roll back so the trade stays OPEN for retry.
        close_calls = [
            c for c in session.calls
            if c[1].get("trade_id") == 901
            and "EXPIRED" in str(c[1].get("exit_reason", ""))
        ]
        assert len(close_calls) == 0, (
            "trade with legs but NULL max_loss must NOT be closed; the "
            "surrounding session should roll back and the trade stays "
            "OPEN for retry once data has been backfilled"
        )


# ---------------------------------------------------------------------------
# _close_with_portfolio split-brain wrapper
# ---------------------------------------------------------------------------


class TestCloseWithPortfolio:
    """The ``_close_with_portfolio`` helper wraps ``pm.record_closure``
    so a ``PortfolioClosureSplitBrainError`` triggers an alert + re-raise
    (so the surrounding session rolls back and the trade stays OPEN for
    operator reconciliation).

    These tests exercise the wrapper in isolation rather than through
    the full ``run_once`` flow, so we can pin the alert-emission and
    re-raise semantics independent of the rest of the trade lifecycle.
    """

    @pytest.mark.asyncio
    async def test_happy_path_passthrough_no_alert(self):
        """No exception -> no alert fired, helper returns normally and
        forwards the exact session object it was given.

        We capture ``session`` in a local before the call so the assertion
        compares against a stable reference rather than reading
        ``mock_pm.record_closure.await_args.kwargs['session']`` (which is
        what we just wrote and would tautologically equal itself).
        """
        from spx_backend.services.portfolio_manager import PortfolioManager
        mock_pm = AsyncMock(spec=PortfolioManager)
        mock_pm.record_closure = AsyncMock()  # succeeds
        session = AsyncMock()

        with patch(
            "spx_backend.jobs.trade_pnl_job.send_alert", new_callable=AsyncMock,
        ) as mock_alert:
            job = TradePnlJob()
            await job._close_with_portfolio(
                pm=mock_pm, trade_id=10, pnl=42.5, session=session,
            )

        mock_pm.record_closure.assert_awaited_once_with(10, 42.5, session=session)
        mock_alert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_split_brain_sends_alert_and_reraises(self):
        """``PortfolioClosureSplitBrainError`` -> alert sent ONCE with the
        correct cooldown_key, then exception re-raised so the surrounding
        session rolls back."""
        from spx_backend.services.portfolio_manager import (
            PortfolioClosureSplitBrainError,
            PortfolioManager,
        )

        mock_pm = AsyncMock(spec=PortfolioManager)
        mock_pm.record_closure = AsyncMock(
            side_effect=PortfolioClosureSplitBrainError(
                trade_id=77, message="row missing",
            ),
        )

        with patch(
            "spx_backend.jobs.trade_pnl_job.send_alert", new_callable=AsyncMock,
        ) as mock_alert:
            job = TradePnlJob()
            with pytest.raises(PortfolioClosureSplitBrainError):
                await job._close_with_portfolio(
                    pm=mock_pm, trade_id=77, pnl=-50.0, session=AsyncMock(),
                )

        mock_alert.assert_awaited_once()
        # Verify the cooldown_key targets the specific trade so that
        # other trades' split-brain alerts aren't suppressed by this
        # trade's cooldown window.
        kwargs = mock_alert.await_args.kwargs
        assert kwargs["cooldown_key"] == "split_brain:trade_id=77"
        assert kwargs["cooldown_minutes"] == 60
        assert "trade_id=77" in kwargs["subject"]
        assert "split-brain" in kwargs["subject"].lower()

    @pytest.mark.asyncio
    async def test_split_brain_reraises_even_if_alert_returns_false(self):
        """Even if ``send_alert`` returns False (cooldown active, creds
        missing, SendGrid down), the original split-brain error must
        STILL propagate.  Otherwise an alert-channel outage would
        silently swallow a data-integrity bug."""
        from spx_backend.services.portfolio_manager import (
            PortfolioClosureSplitBrainError,
            PortfolioManager,
        )

        mock_pm = AsyncMock(spec=PortfolioManager)
        mock_pm.record_closure = AsyncMock(
            side_effect=PortfolioClosureSplitBrainError(
                trade_id=88, message="row missing",
            ),
        )

        with patch(
            "spx_backend.jobs.trade_pnl_job.send_alert",
            new_callable=AsyncMock,
            return_value=False,  # alert suppressed by cooldown
        ) as mock_alert:
            job = TradePnlJob()
            with pytest.raises(PortfolioClosureSplitBrainError):
                await job._close_with_portfolio(
                    pm=mock_pm, trade_id=88, pnl=0.0, session=AsyncMock(),
                )

        mock_alert.assert_awaited_once()
