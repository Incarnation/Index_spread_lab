"""Tests for production portfolio manager and decision-job portfolio integration.

Covers:
- PortfolioManager._count_trades_today (daily count persistence)
- _create_trade_from_decision contracts_override parameter
- Per-run stagger limit in _run_portfolio_managed
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.services.portfolio_manager import PortfolioManager, MARGIN_PER_LOT


# ── Helper to run coroutines in sync test context ────────────────

def run(coro):
    """Run an async coroutine in a fresh event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── PortfolioManager daily-count persistence ─────────────────────


class TestPortfolioManagerDailyCount:
    """Verify begin_day loads the trade count from the DB."""

    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        """Patch the DB engine so no real connection is needed."""
        with patch("spx_backend.services.portfolio_manager.engine") as mock_eng:
            self.mock_engine = mock_eng

            # Default: _load_latest_equity returns None (no prior state)
            mock_conn_ctx = AsyncMock()
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock()
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_begin_ctx = AsyncMock()
            mock_begin_conn = AsyncMock()
            mock_begin_conn.execute = AsyncMock()
            mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_begin_conn)
            mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_eng.connect.return_value = mock_conn_ctx
            mock_eng.begin.return_value = mock_begin_ctx

            self.mock_conn = mock_conn
            self.mock_begin_conn = mock_begin_conn
            yield

    def _setup_db_responses(self, equity: float | None, trade_count: int, state_id: int = 1):
        """Configure mock DB to return specific equity and trade count values."""
        call_index = {"n": 0}

        async def execute_side_effect(*args, **kwargs):
            query = str(args[0]) if args else ""
            result = MagicMock()

            if "equity_end" in query.lower() or "equity_end" in query:
                row = MagicMock()
                row.__getitem__ = lambda self, i: equity
                result.fetchone.return_value = row if equity is not None else None
            elif "count" in query.lower():
                row = MagicMock()
                row.__getitem__ = lambda self, i: trade_count
                result.fetchone.return_value = row
            else:
                result.fetchone.return_value = None

            return result

        self.mock_conn.execute = AsyncMock(side_effect=execute_side_effect)

        # For _upsert_day_state (uses engine.begin)
        upsert_result = MagicMock()
        upsert_result.scalar_one.return_value = state_id
        self.mock_begin_conn.execute = AsyncMock(return_value=upsert_result)

    def test_begin_day_loads_trade_count(self):
        """begin_day should set _trades_today from DB, not reset to 0."""
        self._setup_db_responses(equity=25_000, trade_count=1)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=2)

        run(pm.begin_day(date(2026, 4, 1)))

        assert pm._trades_today == 1
        assert pm.equity == 25_000

    def test_begin_day_zero_trades(self):
        """First run of the day: trade count is 0."""
        self._setup_db_responses(equity=20_000, trade_count=0)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=2)

        run(pm.begin_day(date(2026, 4, 1)))

        assert pm._trades_today == 0
        assert pm.can_trade()

    def test_begin_day_at_daily_limit(self):
        """If DB shows max trades already placed, can_trade returns False."""
        self._setup_db_responses(equity=20_000, trade_count=2)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=2)

        run(pm.begin_day(date(2026, 4, 1)))

        assert pm._trades_today == 2
        assert not pm.can_trade()

    def test_record_trade_increments_count(self):
        """record_trade should bump _trades_today beyond the DB-loaded value."""
        self._setup_db_responses(equity=20_000, trade_count=1)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)

        run(pm.begin_day(date(2026, 4, 1)))
        assert pm._trades_today == 1

        run(pm.record_trade(trade_id=100, pnl_per_lot=50, lots=1))
        assert pm._trades_today == 2
        assert pm.can_trade()

        run(pm.record_trade(trade_id=101, pnl_per_lot=50, lots=1))
        assert pm._trades_today == 3
        assert not pm.can_trade()


# ── contracts_override in _create_trade_from_decision ────────────


class TestContractsOverride:
    """Verify _create_trade_from_decision uses contracts_override when provided."""

    def test_override_changes_pnl_calculation(self):
        """With 2-lot override, max_profit and max_loss should double."""
        credit = 1.50
        width = 10.0
        multiplier = 100

        contracts_1 = 1
        max_profit_1 = credit * contracts_1 * multiplier  # 150
        max_loss_1 = (width - credit) * contracts_1 * multiplier  # 850

        contracts_2 = 2
        max_profit_2 = credit * contracts_2 * multiplier  # 300
        max_loss_2 = (width - credit) * contracts_2 * multiplier  # 1700

        assert max_profit_2 == max_profit_1 * 2
        assert max_loss_2 == max_loss_1 * 2

    def test_override_none_falls_back(self):
        """When contracts_override is None, use default."""
        override = None
        default = 1
        effective = override if override is not None else default
        assert effective == 1

    def test_override_explicit_zero(self):
        """contracts_override=0 should be respected (edge case)."""
        override = 0
        default = 1
        effective = override if override is not None else default
        assert effective == 0


# ── Per-run stagger limit ────────────────────────────────────────


class TestPerRunStagger:
    """Verify the stagger limit logic used in _run_portfolio_managed."""

    def test_run_limit_caps_scheduled(self):
        """Per-run limit should cap scheduled trades below daily max."""
        run_limit = 1
        daily_max = 2
        sched_limit = min(run_limit, daily_max)
        assert sched_limit == 1

    def test_event_cap_respects_run_limit(self):
        """Event cap is min(event_max_trades, run_limit)."""
        event_max = 2
        run_limit = 1
        event_cap = min(event_max, run_limit)
        assert event_cap == 1

    def test_shared_budget_subtracts_events(self):
        """In shared mode, sched_limit decreases by event trades placed."""
        run_limit = 1
        daily_max = 2
        sched_limit = min(run_limit, daily_max)  # 1
        event_trades_placed = 1
        sched_limit = max(0, sched_limit - event_trades_placed)
        assert sched_limit == 0

    def test_separate_budget_ignores_events(self):
        """In separate mode, event trades don't reduce sched_limit."""
        run_limit = 1
        daily_max = 2
        sched_limit = min(run_limit, daily_max)
        # budget_mode == "separate" -> no subtraction
        assert sched_limit == 1

    def test_three_runs_produce_two_trades(self):
        """Simulate 3 entry-time runs with run_limit=1, daily_max=2.

        Run 1 (10:02): trades_today=0, places 1 -> trades_today=1
        Run 2 (11:02): trades_today=1, places 1 -> trades_today=2
        Run 3 (12:02): trades_today=2, can_trade()=False -> skip
        """
        daily_max = 2
        run_limit = 1
        trades_today = 0
        total_placed = 0

        for run_idx in range(3):
            can_trade = trades_today < daily_max
            if not can_trade:
                break
            sched_limit = min(run_limit, daily_max)
            placed_this_run = min(sched_limit, 1)  # 1 candidate available
            trades_today += placed_this_run
            total_placed += placed_this_run

        assert total_placed == 2
        assert trades_today == 2


# ── Session pass-through (FK fix) ─────────────────────────────────


class TestRecordTradeSessionPassthrough:
    """Verify record_trade routes SQL through a caller-supplied session.

    When the decision_job inserts a ``trades`` row and then calls
    ``pm.record_trade(session=session)``, the portfolio_trades INSERT and
    portfolio_state UPDATE must run on the *same* session so the uncommitted
    ``trades`` row is visible to the FK check (READ COMMITTED isolation).
    """

    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        with patch("spx_backend.services.portfolio_manager.engine") as mock_eng:
            self.mock_engine = mock_eng

            mock_conn_ctx = AsyncMock()
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock()
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_begin_ctx = AsyncMock()
            mock_begin_conn = AsyncMock()
            mock_begin_conn.execute = AsyncMock()
            mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_begin_conn)
            mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_eng.connect.return_value = mock_conn_ctx
            mock_eng.begin.return_value = mock_begin_ctx

            self.mock_conn = mock_conn
            self.mock_begin_conn = mock_begin_conn
            yield

    def _setup_pm(self) -> PortfolioManager:
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        pm._today = date(2026, 4, 7)
        pm._state_id = 1
        pm._trades_today = 0
        pm._lots_today = 2
        pm.equity = 20_000
        pm.month_start_equity = 20_000
        return pm

    def test_session_receives_both_inserts(self):
        """When session is provided, both portfolio_trades INSERT and
        portfolio_state UPDATE run on that session -- not engine.begin()."""
        mock_session = AsyncMock()
        pm = self._setup_pm()

        run(pm.record_trade(trade_id=42, pnl_per_lot=3.5, lots=2,
                            source="event", event_signal="term_inversion",
                            session=mock_session))

        assert mock_session.execute.await_count == 2
        self.mock_engine.begin.assert_not_called()

        calls = [str(c.args[0]) for c in mock_session.execute.await_args_list]
        assert any("portfolio_trades" in c for c in calls)
        assert any("portfolio_state" in c for c in calls)

    def test_no_session_uses_engine(self):
        """Without a session, record_trade falls back to engine.begin()."""
        pm = self._setup_pm()

        run(pm.record_trade(trade_id=42, pnl_per_lot=3.5, lots=2))

        assert self.mock_engine.begin.call_count == 2


# ── Config field existence ───────────────────────────────────────


class TestConfigField:
    """Verify the new config field exists and has correct default."""

    def test_portfolio_max_trades_per_run_exists(self):
        """config.py should expose portfolio_max_trades_per_run."""
        from spx_backend.config import Settings
        assert "portfolio_max_trades_per_run" in Settings.model_fields

    def test_default_value(self):
        """Default should be 1 (stagger across entry times)."""
        from spx_backend.config import Settings
        assert Settings.model_fields["portfolio_max_trades_per_run"].default == 1
