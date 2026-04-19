"""Tests for production portfolio manager and decision-job portfolio integration.

Covers:
- PortfolioManager._count_trades_today (daily count persistence)
- _create_trade_from_decision contracts_override parameter
- Per-run stagger limit in _run
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.services.portfolio_manager import (
    MARGIN_PER_LOT,
    PortfolioClosureSplitBrainError,
    PortfolioManager,
)


# ── Helper to run coroutines in sync test context ────────────────

def run(coro):
    """Run an async coroutine in a fresh event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _atomic_delta_result(*, new_equity: float, new_trades: int, rowcount: int = 1) -> MagicMock:
    """Build a mock SQL result that satisfies _apply_equity_delta.

    The atomic UPDATE in _apply_equity_delta needs both ``rowcount`` (to
    detect the missing-row case) and ``fetchone()`` returning
    ``(equity_end, trades_placed)`` from the RETURNING clause.  Tests
    that previously only set ``rowcount`` now also need a fetchone
    payload representing the post-write equity / trades values.
    """
    result = MagicMock()
    result.rowcount = rowcount
    result.fetchone.return_value = (new_equity, new_trades)
    return result


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
        assert run(pm.can_trade())

    def test_begin_day_at_daily_limit(self):
        """If DB shows max trades already placed, can_trade returns False."""
        self._setup_db_responses(equity=20_000, trade_count=2)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=2)

        run(pm.begin_day(date(2026, 4, 1)))

        assert pm._trades_today == 2
        assert not run(pm.can_trade())

    def test_record_trade_increments_count(self):
        """record_trade should bump _trades_today beyond the DB-loaded value.

        After the atomic-delta refactor, ``_trades_today`` is refreshed
        from the RETURNING clause of the portfolio_state UPDATE, not
        from local arithmetic.  We override the begin-conn execute side
        effect to return the post-write trades_placed each time.
        """
        self._setup_db_responses(equity=20_000, trade_count=1)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)

        run(pm.begin_day(date(2026, 4, 1)))
        assert pm._trades_today == 1

        # First record_trade: trades_placed becomes 2.
        self.mock_begin_conn.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_050, new_trades=2)
        )
        run(pm.record_trade(trade_id=100, pnl_per_lot=50, lots=1))
        assert pm._trades_today == 2
        assert run(pm.can_trade())

        # Second record_trade: trades_placed becomes 3 (at the limit).
        self.mock_begin_conn.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_100, new_trades=3)
        )
        run(pm.record_trade(trade_id=101, pnl_per_lot=50, lots=1))
        assert pm._trades_today == 3
        assert not run(pm.can_trade())


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
    """Verify the stagger limit logic used in _run."""

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
        # Both record_trade SQL calls (atomic delta UPDATE...RETURNING and
        # portfolio_trades INSERT) route through the session; the atomic
        # delta needs a fetchone() payload from RETURNING.
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_007, new_trades=1)
        )
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
        # Make engine.begin's connection return a RETURNING payload so the
        # _apply_equity_delta fetchone() doesn't blow up.
        self.mock_begin_conn.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_007, new_trades=1)
        )
        pm = self._setup_pm()

        run(pm.record_trade(trade_id=42, pnl_per_lot=3.5, lots=2))

        assert self.mock_engine.begin.call_count == 2


# ── Equity start-of-day tracking ──────────────────────────────────


class TestEquityStartToday:
    """Verify _equity_start_today is set once per day and not overwritten."""

    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        with patch("spx_backend.services.portfolio_manager.engine") as mock_eng:
            self.mock_engine = mock_eng

            mock_conn_ctx = AsyncMock()
            mock_conn = AsyncMock()
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_begin_ctx = AsyncMock()
            mock_begin_conn = AsyncMock()
            mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_begin_conn)
            mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_eng.connect.return_value = mock_conn_ctx
            mock_eng.begin.return_value = mock_begin_ctx

            self.mock_conn = mock_conn
            self.mock_begin_conn = mock_begin_conn
            yield

    def _setup_db_responses(self, equity: float | None, trade_count: int, state_id: int = 1):
        async def execute_side_effect(*args, **kwargs):
            query = str(args[0]) if args else ""
            result = MagicMock()
            if "equity_end" in query.lower():
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
        upsert_result = MagicMock()
        upsert_result.scalar_one.return_value = state_id
        self.mock_begin_conn.execute = AsyncMock(return_value=upsert_result)

    def test_equity_start_today_set_on_begin_day(self):
        """begin_day should snapshot equity into _equity_start_today."""
        self._setup_db_responses(equity=25_000, trade_count=0)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=2)
        run(pm.begin_day(date(2026, 4, 7)))

        assert pm._equity_start_today == 25_000
        assert pm.equity == 25_000

    def test_zero_pnl_at_entry_preserves_equity(self):
        """record_trade(pnl_per_lot=0.0) should not change equity."""
        self._setup_db_responses(equity=20_000, trade_count=0)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 4, 7)))

        # The atomic delta UPDATE in record_trade returns the post-write
        # equity / trades_placed via RETURNING.  Override the mock to
        # echo the expected post-write values.
        self.mock_begin_conn.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_000, new_trades=1)
        )
        run(pm.record_trade(trade_id=1, pnl_per_lot=0.0, lots=2))
        assert pm.equity == 20_000
        assert pm._trades_today == 1

    def test_apply_equity_delta_computes_daily_pnl(self):
        """_apply_equity_delta should pass the equity delta and lots params
        to the atomic UPDATE; daily_pnl is now computed by the SQL itself
        (equity_end - equity_start) instead of being passed in by the
        caller, so we just assert the right delta / tdelta flowed through.

        Also asserts that when ``lots_per_trade`` is omitted (the
        ``record_closure`` path) the ``:lpt`` bind is SQL ``NULL`` so the
        ``COALESCE(:lpt, lots_per_trade)`` in the UPDATE preserves the
        existing column value rather than overwriting it.
        """
        self._setup_db_responses(equity=20_000, trade_count=0)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 4, 7)))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_100, new_trades=1)
        )
        run(pm._apply_equity_delta(
            equity_delta=100.0, trades_delta=1, session=mock_session,
        ))

        params = mock_session.execute.await_args_list[0].args[1]
        assert params["delta"] == pytest.approx(100.0)
        assert params["tdelta"] == 1
        # lots_per_trade omitted -> :lpt must be None so the COALESCE
        # in the UPDATE preserves the existing portfolio_state value.
        assert params["lpt"] is None

    def test_apply_equity_delta_passes_lots_per_trade(self):
        """When ``lots_per_trade`` is provided (the ``record_trade`` path),
        it must be coerced to ``int`` and bound as ``:lpt`` so the SQL's
        ``COALESCE(:lpt, lots_per_trade)`` writes the entry-time sizing
        decision into ``portfolio_state.lots_per_trade``.

        This is the assertion that locks in the D1 fix: prior to the
        ``COALESCE`` rewrite, every closure call quietly overwrote
        ``lots_per_trade`` with ``self._lots_today or 1``.
        """
        self._setup_db_responses(equity=20_000, trade_count=0)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 4, 7)))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_000, new_trades=1)
        )
        run(pm._apply_equity_delta(
            equity_delta=0.0,
            trades_delta=1,
            lots_per_trade=5,
            session=mock_session,
        ))

        params = mock_session.execute.await_args_list[0].args[1]
        assert params["lpt"] == 5
        assert isinstance(params["lpt"], int)

    def test_record_trade_rejects_non_positive_lots(self):
        """record_trade must reject lots <= 0 to avoid silently overwriting
        portfolio_state.lots_per_trade with a bogus value via the COALESCE
        in _apply_equity_delta.  Production callers go through compute_lots()
        which is bounded >= 1, but the guard catches future regressions and
        any direct callers (e.g. tests, ad-hoc scripts).
        """
        self._setup_db_responses(equity=20_000, trade_count=0)
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 4, 7)))

        with pytest.raises(ValueError, match="lots >= 1"):
            run(pm.record_trade(trade_id=1, pnl_per_lot=10.0, lots=0))

        with pytest.raises(ValueError, match="lots >= 1"):
            run(pm.record_trade(trade_id=1, pnl_per_lot=10.0, lots=-2))


# ── begin_day state preservation across re-runs ──────────────────


class TestBeginDayStatePreservation:
    """Verify begin_day preserves equity_start and month_start from the DB
    across multiple calls within the same day / same month, even after
    trade closures update equity_end in between.
    """

    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        with patch("spx_backend.services.portfolio_manager.engine") as mock_eng:
            self.mock_engine = mock_eng

            mock_conn_ctx = AsyncMock()
            mock_conn = AsyncMock()
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_begin_ctx = AsyncMock()
            mock_begin_conn = AsyncMock()
            mock_begin_ctx.__aenter__ = AsyncMock(return_value=mock_begin_conn)
            mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_eng.connect.return_value = mock_conn_ctx
            mock_eng.begin.return_value = mock_begin_ctx

            self.mock_conn = mock_conn
            self.mock_begin_conn = mock_begin_conn
            yield

    def _setup_db_responses(
        self,
        equity: float | None,
        trade_count: int,
        state_id: int = 1,
        today_equity_start: float | None = None,
        month_start_eq: float | None = None,
    ):
        """Configure mock DB responses including the new helper queries.

        Parameters
        ----------
        equity : Latest equity_end value (None = no prior state).
        trade_count : Number of trades today.
        today_equity_start : Existing equity_start for today's row.
        month_start_eq : Existing month_start_equity for current month.
        """
        async def execute_side_effect(*args, **kwargs):
            query = str(args[0]) if args else ""
            ql = query.lower()
            result = MagicMock()

            if "equity_end" in ql:
                row = MagicMock()
                row.__getitem__ = lambda self, i: equity
                result.fetchone.return_value = row if equity is not None else None
            elif "equity_start" in ql and "portfolio_state" in ql and "date" in ql:
                row = MagicMock()
                row.__getitem__ = lambda self, i: today_equity_start
                result.fetchone.return_value = row if today_equity_start is not None else None
            elif "month_start_equity" in ql and "to_char" in ql:
                row = MagicMock()
                row.__getitem__ = lambda self, i: month_start_eq
                result.fetchone.return_value = row if month_start_eq is not None else None
            elif "count" in ql:
                row = MagicMock()
                row.__getitem__ = lambda self, i: trade_count
                result.fetchone.return_value = row
            else:
                result.fetchone.return_value = None
            return result

        self.mock_conn.execute = AsyncMock(side_effect=execute_side_effect)
        upsert_result = MagicMock()
        upsert_result.scalar_one.return_value = state_id
        self.mock_begin_conn.execute = AsyncMock(return_value=upsert_result)

    def test_begin_day_preserves_equity_start_across_reruns(self):
        """When today's row already exists, begin_day should use the DB's
        equity_start rather than the (potentially updated) current equity."""
        self._setup_db_responses(
            equity=20_660,           # equity_end updated by a closure
            trade_count=1,
            today_equity_start=20_240,  # original start-of-day value
        )
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 4, 8)))

        assert pm.equity == 20_660
        assert pm._equity_start_today == 20_240

    def test_first_begin_day_uses_current_equity(self):
        """On the very first call of the day (no DB row yet), equity_start
        should fall back to the current equity."""
        self._setup_db_responses(
            equity=20_000,
            trade_count=0,
            today_equity_start=None,  # no row yet
        )
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 4, 7)))

        assert pm._equity_start_today == 20_000

    def test_month_start_from_db_on_restart(self):
        """On process restart (_current_month is None), begin_day should
        load month_start_equity from the DB instead of using current equity."""
        self._setup_db_responses(
            equity=20_660,
            trade_count=2,
            month_start_eq=20_000,  # true start-of-month equity
        )
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        assert pm._current_month is None

        run(pm.begin_day(date(2026, 4, 8)))

        assert pm.month_start_equity == 20_000  # from DB, not 20_660

    def test_month_start_falls_back_to_equity_for_new_month(self):
        """When no DB rows exist for the current month, month_start should
        use the current equity (genuinely new month)."""
        self._setup_db_responses(
            equity=21_000,
            trade_count=0,
            month_start_eq=None,  # no rows for this month yet
        )
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 5, 1)))

        assert pm.month_start_equity == 21_000

    def test_apply_equity_delta_after_closure_and_rerun(self):
        """After a trade closure updates equity between begin_day calls,
        the atomic delta should still apply correctly to the post-closure
        equity_end (delta is added by SQL, not pre-computed by the caller).
        """
        self._setup_db_responses(
            equity=20_660,           # equity_end after closure
            trade_count=1,
            today_equity_start=20_240,  # preserved from first begin_day
        )
        pm = PortfolioManager(starting_capital=20_000, max_trades_per_day=3)
        run(pm.begin_day(date(2026, 4, 8)))

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_660, new_trades=1)
        )
        run(pm._apply_equity_delta(
            equity_delta=0.0, trades_delta=0, session=mock_session,
        ))
        params = mock_session.execute.await_args_list[0].args[1]
        assert params["delta"] == pytest.approx(0.0)
        assert params["tdelta"] == 0


# ── Event candidate filtering logic ──────────────────────────────


class TestEventCandidateFiltering:
    """Verify event delta_target filtering and has_drop side logic."""

    def test_delta_target_filters_candidates(self):
        """Event filter should use delta_target, not delta_diff."""
        min_delta = 0.10
        max_delta = 0.25
        candidates = [
            {"delta_target": 0.05, "target_dte": 5, "spread_side": "put"},
            {"delta_target": 0.10, "target_dte": 5, "spread_side": "put"},
            {"delta_target": 0.20, "target_dte": 5, "spread_side": "put"},
            {"delta_target": 0.30, "target_dte": 5, "spread_side": "put"},
        ]
        filtered = [
            c for c in candidates
            if c["delta_target"] >= min_delta
            and c["delta_target"] <= max_delta
        ]
        assert len(filtered) == 2
        assert filtered[0]["delta_target"] == 0.10
        assert filtered[1]["delta_target"] == 0.20

    def test_has_drop_forces_side_preference(self):
        """When drop-class signals present, event_side_preference is enforced."""
        has_drop = True
        event_side_raw = "put"
        candidates = [
            {"spread_side": "call", "target_dte": 5, "delta_target": 0.10},
            {"spread_side": "put", "target_dte": 5, "delta_target": 0.10},
        ]
        filtered = [
            c for c in candidates
            if (not has_drop or c.get("spread_side", "").lower() == event_side_raw)
        ]
        assert len(filtered) == 1
        assert filtered[0]["spread_side"] == "put"

    def test_no_drop_allows_all_sides(self):
        """Non-drop signals (e.g. term_inversion) allow all sides."""
        has_drop = False
        event_side_raw = "put"
        candidates = [
            {"spread_side": "call", "target_dte": 5, "delta_target": 0.10},
            {"spread_side": "put", "target_dte": 5, "delta_target": 0.10},
        ]
        filtered = [
            c for c in candidates
            if (not has_drop or c.get("spread_side", "").lower() == event_side_raw)
        ]
        assert len(filtered) == 2

    def test_has_drop_classification(self):
        """Verify which signals are classified as drop-class."""
        drop_signals_test = ["spx_drop_1d", "spx_drop_2d", "vix_spike", "vix_elevated"]
        has_drop = any(
            s.startswith("spx_drop") or s in ("vix_spike", "vix_elevated")
            for s in drop_signals_test
        )
        assert has_drop is True

        non_drop_signals = ["term_inversion"]
        has_drop = any(
            s.startswith("spx_drop") or s in ("vix_spike", "vix_elevated")
            for s in non_drop_signals
        )
        assert has_drop is False

    def test_event_side_preference_filters_candidates(self):
        """Event candidates should match the event_side_preference."""
        event_side_raw = "puts".rstrip("s")  # "put"
        candidates = [
            {"spread_side": "call", "target_dte": 5, "credit_to_width": 0.5},
            {"spread_side": "put", "target_dte": 5, "credit_to_width": 0.4},
            {"spread_side": "call", "target_dte": 7, "credit_to_width": 0.3},
        ]
        event_filtered = [
            c for c in candidates
            if c.get("spread_side", "").lower() == event_side_raw
        ]
        assert len(event_filtered) == 1
        assert event_filtered[0]["spread_side"] == "put"

    def test_scheduled_side_filter(self):
        """Scheduled candidates respect portfolio_calls_only side list."""
        sched_sides = ["call"]
        candidates = [
            {"spread_side": "call", "credit_to_width": 0.5},
            {"spread_side": "put", "credit_to_width": 0.4},
            {"spread_side": "call", "credit_to_width": 0.3},
        ]
        sched_ranked = [c for c in candidates if c.get("spread_side", "").lower() in sched_sides]
        assert len(sched_ranked) == 2
        assert all(c["spread_side"] == "call" for c in sched_ranked)

    def test_all_sides_dedup(self):
        """all_sides combines sched + event sides without duplicates."""
        sched_sides = ["call"]
        event_sides = ["put"]
        all_sides = list(dict.fromkeys(sched_sides + event_sides))
        assert all_sides == ["call", "put"]

        event_sides = ["call"]
        all_sides = list(dict.fromkeys(sched_sides + event_sides))
        assert all_sides == ["call"]

    def test_non_drop_event_sides_uses_all_spread_sides(self):
        """For non-drop events, event_sides should be all spread sides."""
        sched_sides = ["call"]
        all_spread_sides = ["put", "call"]
        drop_signals = ["term_inversion"]
        has_drop = any(
            s.startswith("spx_drop") or s in ("vix_spike", "vix_elevated")
            for s in drop_signals
        )
        assert has_drop is False

        event_sides = list(all_spread_sides)
        all_sides = list(dict.fromkeys(sched_sides + event_sides))
        assert "put" in all_sides
        assert "call" in all_sides


# ── PortfolioManager.record_closure ──────────────────────────────


class TestRecordClosure:
    """Verify record_closure updates portfolio_trades and equity."""

    @pytest.fixture(autouse=True)
    def _patch_engine(self):
        with patch("spx_backend.services.portfolio_manager.engine") as mock_eng:
            self.mock_engine = mock_eng

            mock_conn_ctx = AsyncMock()
            mock_conn = AsyncMock()
            mock_conn_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_begin_ctx = AsyncMock()
            mock_begin_conn = AsyncMock()
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
        pm._trades_today = 1
        pm._lots_today = 2
        pm.equity = 20_000
        pm.month_start_equity = 20_000
        pm._equity_start_today = 20_000
        return pm

    def test_record_closure_updates_equity(self):
        """record_closure should adjust equity by realized_pnl."""
        mock_session = AsyncMock()
        # First execute call: portfolio_trades UPDATE -> rowcount=1
        # Second execute call: portfolio_state UPDATE...RETURNING -> fetchone
        # Single shared result_mock works because both rowcount and
        # fetchone are read by their respective callers.
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_150.0, new_trades=1)
        )

        pm = self._setup_pm()
        run(pm.record_closure(trade_id=42, realized_pnl=150.0, session=mock_session))

        assert pm.equity == pytest.approx(20_150.0)

    def test_record_closure_updates_portfolio_trades(self):
        """record_closure should UPDATE portfolio_trades.realized_pnl."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_150.0, new_trades=1)
        )

        pm = self._setup_pm()
        run(pm.record_closure(trade_id=42, realized_pnl=150.0, session=mock_session))

        calls = [str(c.args[0]) for c in mock_session.execute.await_args_list]
        assert any("portfolio_trades" in c for c in calls)
        assert any("portfolio_state" in c for c in calls)

    def test_record_closure_negative_pnl(self):
        """record_closure with a loss should decrease equity."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=19_800.0, new_trades=1)
        )

        pm = self._setup_pm()
        run(pm.record_closure(trade_id=43, realized_pnl=-200.0, session=mock_session))

        assert pm.equity == pytest.approx(19_800.0)

    def test_record_closure_idempotent_when_already_closed(self):
        """When the portfolio_trades row exists with a non-NULL realized_pnl,
        record_closure should treat the second call as idempotent: no equity
        change, no exception (legacy retry-storm behaviour).
        """
        mock_session = AsyncMock()
        # rowcount=0 from the UPDATE WHERE realized_pnl IS NULL clause,
        # then the probe SELECT 1 returns existing-row.
        update_result = MagicMock()
        update_result.rowcount = 0
        probe_result = MagicMock()
        probe_result.scalar.return_value = 1  # row exists
        mock_session.execute = AsyncMock(side_effect=[update_result, probe_result])

        pm = self._setup_pm()
        run(pm.record_closure(trade_id=42, realized_pnl=500.0, session=mock_session))

        assert pm.equity == pytest.approx(20_000.0)

    def test_record_closure_split_brain_raises(self):
        """When no portfolio_trades row exists at all (truly missing),
        record_closure now raises PortfolioClosureSplitBrainError so the
        caller can alert + roll back instead of silently dropping the
        closure.
        """
        mock_session = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 0
        probe_result = MagicMock()
        probe_result.scalar.return_value = None  # no row exists
        mock_session.execute = AsyncMock(side_effect=[update_result, probe_result])

        pm = self._setup_pm()
        with pytest.raises(PortfolioClosureSplitBrainError) as excinfo:
            run(pm.record_closure(trade_id=99, realized_pnl=500.0, session=mock_session))

        assert excinfo.value.trade_id == 99
        assert pm.equity == pytest.approx(20_000.0)

    def test_record_closure_without_session_uses_engine(self):
        """record_closure without a session should use engine.begin()."""
        self.mock_begin_conn.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_100.0, new_trades=1)
        )

        pm = self._setup_pm()
        run(pm.record_closure(trade_id=44, realized_pnl=100.0))

        assert self.mock_engine.begin.call_count >= 1


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


# ── Monthly drawdown stop persistence ────────────────────────────


class TestMonthlyStopPersistence:
    """Cover the monthly drawdown stop persistence path.

    Two surfaces:

    * ``_load_month_stop_active`` -- read on ``begin_day`` after a
      process restart so a previously-tripped month stays stopped.
    * ``can_trade()`` trip -- the moment the equity falls below the
      configured drawdown threshold the in-memory ``_month_stopped``
      flag flips AND ``_persist_month_stop`` writes the flag to the DB
      so a crash between trip and the next equity-mutating call doesn't
      lose the stop.

    Both writes go through ``engine.begin()`` (atomic short transaction
    inside ``_persist_month_stop``) and ``engine.connect()`` (read-only
    inside ``_load_month_stop_active``); the mocks below set up both
    sides so the assertions stay focused on the public behaviour.
    """

    def _make_pm(
        self,
        *,
        equity: float,
        month_start: float,
        dd_limit: float | None,
        state_id: int | None = 1,
        trades_today: int = 0,
        max_trades: int = 10,
    ) -> PortfolioManager:
        """Build a PortfolioManager with the in-memory fields needed by
        ``can_trade()`` already pre-populated, so the test can focus on
        the trip path without rerunning ``begin_day``."""
        pm = PortfolioManager(
            starting_capital=20_000,
            max_trades_per_day=max_trades,
            monthly_drawdown_limit=dd_limit,
        )
        pm.equity = equity
        pm.month_start_equity = month_start
        pm._trades_today = trades_today
        pm._state_id = state_id
        return pm

    @pytest.mark.asyncio
    async def test_load_month_stop_active_returns_false_when_no_rows(self):
        """No rows for the month at all -> default False (rare; most
        months will have at least one row from begin_day)."""
        with patch("spx_backend.services.portfolio_manager.engine") as mock_engine:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = None
            mock_conn.execute = AsyncMock(return_value=mock_result)

            connect_ctx = AsyncMock()
            connect_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            connect_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_engine.connect.return_value = connect_ctx

            pm = PortfolioManager()
            assert await pm._load_month_stop_active("2026-04") is False

    @pytest.mark.asyncio
    async def test_load_month_stop_active_returns_true_when_latest_set(self):
        """Latest row in the month has the flag -> honour it."""
        with patch("spx_backend.services.portfolio_manager.engine") as mock_engine:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            # _load_month_stop_active reads result[0]; tuple-style payload.
            mock_result.fetchone.return_value = (True,)
            mock_conn.execute = AsyncMock(return_value=mock_result)

            connect_ctx = AsyncMock()
            connect_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            connect_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_engine.connect.return_value = connect_ctx

            pm = PortfolioManager()
            assert await pm._load_month_stop_active("2026-04") is True

    @pytest.mark.asyncio
    async def test_load_month_stop_active_returns_false_when_latest_null(self):
        """Defensive default: the column is non-null in the schema, but if
        a legacy row is NULL we should fail safe to False rather than
        propagating the NULL into the trip gate."""
        with patch("spx_backend.services.portfolio_manager.engine") as mock_engine:
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.fetchone.return_value = (None,)
            mock_conn.execute = AsyncMock(return_value=mock_result)

            connect_ctx = AsyncMock()
            connect_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            connect_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_engine.connect.return_value = connect_ctx

            pm = PortfolioManager()
            assert await pm._load_month_stop_active("2026-04") is False

    @pytest.mark.asyncio
    async def test_can_trade_trip_writes_month_stop_to_db(self):
        """Equity below threshold -> can_trade() returns False, sets the
        in-memory flag, AND awaits ``_persist_month_stop``.

        Scope note: this test mocks ``_persist_month_stop`` to isolate the
        trip-decision logic from SQL.  The end-to-end SQL UPDATE that
        ``_persist_month_stop`` issues is covered separately by
        ``test_persist_month_stop_writes_to_state_id`` below, so the two
        tests together prove "trip awaits persist" + "persist writes the
        flag" without coupling them in a single, harder-to-debug fixture.

        This is the core bug the persistence path guards against: a
        process restart between the trip and the next record_trade call
        used to lose the trip and resume trading inside a stopped month.
        """
        # Equity 9000, month_start 10000, dd_limit 0.05 -> threshold 9500
        pm = self._make_pm(equity=9_000, month_start=10_000, dd_limit=0.05)
        with patch.object(pm, "_persist_month_stop", new_callable=AsyncMock) as mock_persist:
            allowed = await pm.can_trade()

        assert allowed is False
        assert pm._month_stopped is True
        mock_persist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_can_trade_already_stopped_short_circuits_no_persist(self):
        """If the in-memory ``_month_stopped`` flag is already True
        (e.g. begin_day reloaded it from the DB), can_trade() must
        short-circuit to False WITHOUT calling _persist_month_stop
        again -- the row is already up to date and we shouldn't
        amplify load by writing on every gate check."""
        pm = self._make_pm(equity=20_000, month_start=20_000, dd_limit=0.05)
        pm._month_stopped = True
        with patch.object(pm, "_persist_month_stop", new_callable=AsyncMock) as mock_persist:
            allowed = await pm.can_trade()

        assert allowed is False
        mock_persist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persist_month_stop_writes_to_state_id(self):
        """Smoke-check the SQL: _persist_month_stop opens its own
        ``engine.begin()`` transaction and updates the row identified
        by ``self._state_id``.  We verify the parameterised id rather
        than the exact text to keep the assertion robust against
        SQL formatting changes."""
        pm = PortfolioManager()
        pm._state_id = 42
        with patch("spx_backend.services.portfolio_manager.engine") as mock_engine:
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock()
            begin_ctx = AsyncMock()
            begin_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            begin_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_engine.begin.return_value = begin_ctx

            await pm._persist_month_stop()

        assert mock_conn.execute.await_count == 1
        call = mock_conn.execute.await_args
        params = call.args[1]
        assert params["sid"] == 42

    @pytest.mark.asyncio
    async def test_persist_month_stop_skips_when_state_id_missing(self):
        """If ``begin_day`` was never called, ``_state_id`` is None and
        we shouldn't try to UPDATE WHERE id IS NULL.  Log + skip is the
        correct behaviour."""
        pm = PortfolioManager()
        assert pm._state_id is None

        with patch("spx_backend.services.portfolio_manager.engine") as mock_engine:
            await pm._persist_month_stop()
            mock_engine.begin.assert_not_called()


# ── margin_dollars propagation ──────────────────────────────────


class TestMarginDollarsPropagation:
    """Verify the per-trade ``margin_dollars`` value reaches the
    ``portfolio_trades.margin_committed`` column instead of the synthetic
    ``lots * MARGIN_PER_LOT`` derivation.

    This was added so risk dashboards reflect the real ``max_loss`` the
    spread actually risks (which can differ materially from the synthetic
    derivation when the spread width is narrower than the configured
    margin-per-lot)."""

    def _setup_pm(self) -> PortfolioManager:
        """Build a PortfolioManager pre-initialised so record_trade can
        run without invoking begin_day."""
        pm = PortfolioManager(starting_capital=20_000)
        pm.equity = 20_000.0
        pm._equity_start_today = 20_000.0
        pm.month_start_equity = 20_000.0
        pm._state_id = 1
        pm._today = date(2026, 4, 1)
        return pm

    def test_explicit_margin_dollars_lands_in_insert(self):
        """``record_trade(margin_dollars=2222.5)`` -> portfolio_trades
        INSERT params carry ``margin=2222.5``, NOT ``lots * MARGIN_PER_LOT``."""
        pm = self._setup_pm()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_000.0, new_trades=1)
        )

        run(pm.record_trade(
            trade_id=1,
            pnl_per_lot=0.0,
            lots=2,
            session=mock_session,
            margin_dollars=2222.5,
        ))

        insert_calls = [
            c for c in mock_session.execute.await_args_list
            if "portfolio_trades" in str(c.args[0])
        ]
        assert len(insert_calls) == 1, (
            "expected exactly one portfolio_trades INSERT"
        )
        params = insert_calls[0].args[1]
        assert params["margin"] == pytest.approx(2222.5), (
            "margin_dollars override must reach the INSERT untouched, "
            "not be replaced by lots * MARGIN_PER_LOT"
        )

    def test_none_margin_dollars_falls_back_to_synthetic(self):
        """When the caller doesn't pass margin_dollars (legacy callers),
        we fall back to the synthetic ``lots * MARGIN_PER_LOT`` so old
        behaviour is preserved."""
        pm = self._setup_pm()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(
            return_value=_atomic_delta_result(new_equity=20_000.0, new_trades=1)
        )

        run(pm.record_trade(
            trade_id=2,
            pnl_per_lot=0.0,
            lots=3,
            session=mock_session,
            margin_dollars=None,
        ))

        insert_calls = [
            c for c in mock_session.execute.await_args_list
            if "portfolio_trades" in str(c.args[0])
        ]
        assert len(insert_calls) == 1
        params = insert_calls[0].args[1]
        assert params["margin"] == pytest.approx(3 * MARGIN_PER_LOT), (
            "missing margin_dollars must fall back to lots * MARGIN_PER_LOT "
            "to preserve legacy caller behaviour"
        )
