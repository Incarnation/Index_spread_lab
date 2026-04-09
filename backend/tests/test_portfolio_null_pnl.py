"""Test that record_trade inserts realized_pnl=NULL (not 0.0).

The B3 regression occurred because record_trade previously inserted
``realized_pnl = 0.0``, which made record_closure's idempotency guard
(``WHERE realized_pnl IS NULL``) unable to find the row to update.
This test verifies the fix is in place.
"""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spx_backend.services.portfolio_manager import PortfolioManager


def run(coro):
    """Run an async coroutine in the current event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestRecordTradeNullPnl:
    """Verify record_trade passes realized_pnl=None to the SQL INSERT."""

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
        pm._today = date(2026, 4, 8)
        pm._state_id = 1
        pm._trades_today = 0
        pm._lots_today = 2
        pm.equity = 20_000
        pm.month_start_equity = 20_000
        pm._equity_start_today = 20_000
        return pm

    def test_record_trade_inserts_null_realized_pnl_via_session(self):
        """When called with a session, the portfolio_trades INSERT must use
        realized_pnl=NULL so record_closure can match it later."""
        mock_session = AsyncMock()
        pm = self._setup_pm()

        run(pm.record_trade(
            trade_id=42, pnl_per_lot=0.0, lots=2,
            source="scheduled", session=mock_session,
        ))

        insert_calls = [
            c for c in mock_session.execute.await_args_list
            if "portfolio_trades" in str(c.args[0])
        ]
        assert len(insert_calls) == 1

        params = insert_calls[0].args[1]
        assert params["pnl"] is None, (
            "record_trade must INSERT realized_pnl=NULL, not 0.0"
        )

    def test_record_trade_inserts_null_realized_pnl_via_engine(self):
        """Without a session, the engine path must also use realized_pnl=NULL."""
        pm = self._setup_pm()

        run(pm.record_trade(trade_id=43, pnl_per_lot=0.0, lots=1))

        insert_calls = [
            c for c in self.mock_begin_conn.execute.await_args_list
            if "portfolio_trades" in str(c.args[0])
        ]
        assert len(insert_calls) == 1

        params = insert_calls[0].args[1]
        assert params["pnl"] is None, (
            "record_trade must INSERT realized_pnl=NULL, not 0.0"
        )

    def test_record_closure_matches_null_pnl(self):
        """record_closure's SQL uses WHERE realized_pnl IS NULL, confirming
        it pairs with the NULL inserted by record_trade."""
        mock_session = AsyncMock()
        result_mock = MagicMock()
        result_mock.rowcount = 1
        mock_session.execute = AsyncMock(return_value=result_mock)

        pm = self._setup_pm()
        pm._trades_today = 1

        run(pm.record_closure(trade_id=42, realized_pnl=150.0, session=mock_session))

        closure_sql = str(mock_session.execute.await_args_list[0].args[0])
        assert "realized_pnl IS NULL" in closure_sql
        assert pm.equity == pytest.approx(20_150.0)
