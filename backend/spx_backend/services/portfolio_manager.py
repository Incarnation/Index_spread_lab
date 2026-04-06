"""Production portfolio manager for capital-budgeted trading.

Tracks equity, enforces risk limits, computes lot sizes, and persists
daily state to the ``portfolio_state`` / ``portfolio_trades`` tables.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.database.connection import engine

MARGIN_PER_LOT = 1000  # 10-pt wide SPX spread, $100/pt multiplier


class PortfolioManager:
    """Manages portfolio state across the trading day.

    Reads initial equity from the database on first call of the day,
    enforces monthly drawdown stop-loss and daily trade limits,
    and persists state after each trade.

    Parameters
    ----------
    starting_capital : Override starting capital (default: from config).
    max_trades_per_day : Override daily trade cap (default: from config).
    monthly_drawdown_limit : Override monthly stop (default: from config).
        Set to 0 or None to disable.
    lot_per_equity : Gradual lot scaling factor (default: from config).
    """

    def __init__(
        self,
        starting_capital: float | None = None,
        max_trades_per_day: int | None = None,
        monthly_drawdown_limit: float | None = None,
        lot_per_equity: float | None = None,
    ) -> None:
        self.starting_capital = starting_capital or settings.portfolio_starting_capital
        self.max_trades = max_trades_per_day or settings.portfolio_max_trades_per_day
        raw_limit = monthly_drawdown_limit if monthly_drawdown_limit is not None else settings.portfolio_monthly_drawdown_limit
        self.monthly_dd_limit: float | None = raw_limit if raw_limit and raw_limit > 0 else None
        self.lot_per_equity = lot_per_equity or settings.portfolio_lot_per_equity
        self.max_risk_pct = settings.portfolio_max_equity_risk_pct

        self.equity: float = self.starting_capital
        self.month_start_equity: float = self.starting_capital
        self._equity_start_today: float = self.starting_capital
        self._current_month: str | None = None
        self._month_stopped = False
        self._trades_today = 0
        self._lots_today: int | None = None
        self._today: date | None = None
        self._state_id: int | None = None

    async def begin_day(self, today: date) -> None:
        """Initialize daily state.

        Loads the most recent ``portfolio_state`` row to get current equity
        and restores the trade count for *today* so the daily limit is
        enforced correctly across multiple decision runs.

        Parameters
        ----------
        today : The current trading date.
        """
        self._today = today
        self._lots_today = None

        prev_equity = await self._load_latest_equity()
        if prev_equity is not None:
            self.equity = prev_equity

        self._equity_start_today = self.equity
        self._trades_today = await self._count_trades_today(today)

        month_key = today.strftime("%Y-%m")
        if month_key != self._current_month:
            self._current_month = month_key
            self.month_start_equity = self.equity
            self._month_stopped = False

        self._state_id = await self._upsert_day_state(
            equity_start=self._equity_start_today,
            month_start_equity=self.month_start_equity,
        )
        logger.info(
            "portfolio_day_start date={} equity={:.0f} month_start={:.0f} lots={} trades_today={}",
            today, self.equity, self.month_start_equity, self.compute_lots(),
            self._trades_today,
        )

    def can_trade(self) -> bool:
        """Return True if another trade is permitted right now."""
        if self._month_stopped:
            return False
        if self.equity < MARGIN_PER_LOT:
            return False
        if self.monthly_dd_limit is not None:
            if self.equity < self.month_start_equity * (1 - self.monthly_dd_limit):
                self._month_stopped = True
                logger.warning(
                    "portfolio_monthly_stop equity={:.0f} threshold={:.0f}",
                    self.equity, self.month_start_equity * (1 - self.monthly_dd_limit),
                )
                return False
        if self._trades_today >= self.max_trades:
            return False
        return True

    @property
    def is_month_stopped(self) -> bool:
        return self._month_stopped

    def compute_lots(self) -> int:
        """Calculate lot count for the next trade.

        Uses gradual scaling (1 lot per ``lot_per_equity`` dollars) capped by
        per-trade risk limit.

        Returns
        -------
        int : Number of lots (always >= 1 while capital permits).
        """
        if self._lots_today is not None:
            return self._lots_today
        raw = max(1, int(self.equity / self.lot_per_equity))
        max_by_risk = max(1, int(self.equity * self.max_risk_pct / MARGIN_PER_LOT))
        self._lots_today = min(raw, max_by_risk)
        return self._lots_today

    async def record_trade(
        self,
        trade_id: int,
        pnl_per_lot: float,
        lots: int,
        source: str = "scheduled",
        event_signal: str | None = None,
        session: AsyncSession | None = None,
    ) -> float:
        """Record a new trade entry and persist to database.

        At entry time the caller should pass ``pnl_per_lot=0.0`` because no
        PnL is realised until the trade closes.  Actual realised PnL is
        tracked by ``trade_pnl_job`` on the ``trades`` table.

        Parameters
        ----------
        trade_id : Foreign key into the ``trades`` table.
        pnl_per_lot : PnL per lot to book (normally ``0.0`` at entry).
        lots : Number of lots filled.
        source : ``'scheduled'`` or ``'event'``.
        event_signal : Name of the triggering event signal (if source is 'event').
        session : If provided, execute SQL within this existing transaction
            instead of opening a new connection.  Required when the caller has
            an uncommitted ``trades`` row in the same session (avoids cross-txn
            FK violations under READ COMMITTED isolation).

        Returns
        -------
        float : Total PnL booked for this entry.
        """
        equity_before = self.equity
        total_pnl = pnl_per_lot * lots
        self.equity += total_pnl
        self._trades_today += 1

        await self._insert_portfolio_trade(
            session=session,
            trade_id=trade_id,
            source=source,
            event_signal=event_signal,
            lots=lots,
            margin=lots * MARGIN_PER_LOT,
            pnl=total_pnl,
            equity_before=equity_before,
            equity_after=self.equity,
        )
        await self._update_day_state(
            equity_end=self.equity, trades=self._trades_today, session=session,
        )

        logger.info(
            "portfolio_trade source={} pnl={:.0f} lots={} equity={:.0f}->={:.0f}",
            source, total_pnl, lots, equity_before, self.equity,
        )
        return total_pnl

    # ── DB helpers ────────────────────────────────────────────────

    async def _count_trades_today(self, today: date) -> int:
        """Count trades already placed today from ``portfolio_trades``.

        Joins against ``portfolio_state`` to filter by date so the daily
        trade cap is enforced correctly across separate decision runs.
        """
        async with engine.connect() as conn:
            row = await conn.execute(text(
                "SELECT COUNT(*) FROM portfolio_trades pt "
                "JOIN portfolio_state ps ON pt.portfolio_state_id = ps.id "
                "WHERE ps.date = :today"
            ), {"today": today})
            result = row.fetchone()
            return int(result[0]) if result else 0

    async def _load_latest_equity(self) -> float | None:
        """Read the most recent end-of-day equity from ``portfolio_state``."""
        async with engine.connect() as conn:
            row = await conn.execute(text(
                "SELECT equity_end FROM portfolio_state "
                "WHERE equity_end IS NOT NULL "
                "ORDER BY date DESC LIMIT 1"
            ))
            result = row.fetchone()
            return float(result[0]) if result else None

    async def _upsert_day_state(
        self,
        equity_start: float,
        month_start_equity: float,
    ) -> int:
        """Create or update today's portfolio_state row and return its id.

        On INSERT (first run of the day) sets ``equity_start``; on conflict
        (subsequent runs) only refreshes ``month_start_equity`` so the
        original start-of-day equity is preserved.
        """
        async with engine.begin() as conn:
            row = await conn.execute(text(
                "INSERT INTO portfolio_state (date, equity_start, month_start_equity) "
                "VALUES (:d, :es, :ms) "
                "ON CONFLICT (date) DO UPDATE SET month_start_equity = :ms "
                "RETURNING id"
            ), {"d": self._today, "es": equity_start, "ms": month_start_equity})
            return row.scalar_one()

    async def _update_day_state(
        self,
        equity_end: float,
        trades: int,
        event_signals: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Update today's portfolio_state with end-of-day values.

        Parameters
        ----------
        session : Optional existing session; when provided the UPDATE runs
            inside that transaction instead of opening a standalone one.
        """
        params = {
            "ee": equity_end,
            "tp": trades,
            "dpnl": equity_end - self._equity_start_today,
            "ms": self._month_stopped,
            "lpt": self._lots_today or 1,
            "ev": json.dumps(event_signals) if event_signals else None,
            "sid": self._state_id,
        }
        stmt = text(
            "UPDATE portfolio_state SET "
            "  equity_end = :ee, trades_placed = :tp, daily_pnl = :dpnl, "
            "  monthly_stop_active = :ms, lots_per_trade = :lpt, "
            "  event_signals = :ev "
            "WHERE id = :sid"
        )
        if session is not None:
            await session.execute(stmt, params)
        else:
            async with engine.begin() as conn:
                await conn.execute(stmt, params)

    async def _insert_portfolio_trade(
        self,
        session: AsyncSession | None = None,
        **kw: Any,
    ) -> None:
        """Insert a row into ``portfolio_trades``.

        Parameters
        ----------
        session : Optional existing session; when provided the INSERT runs
            inside that transaction so an uncommitted ``trades`` row from the
            same session is visible to the FK check.
        """
        params = {
            "trade_id": kw["trade_id"], "sid": self._state_id,
            "source": kw["source"], "event_signal": kw.get("event_signal"),
            "lots": kw["lots"], "margin": kw["margin"],
            "pnl": kw["pnl"], "equity_before": kw["equity_before"],
            "equity_after": kw["equity_after"],
        }
        stmt = text(
            "INSERT INTO portfolio_trades "
            "(trade_id, portfolio_state_id, trade_source, event_signal, "
            " lots, margin_committed, realized_pnl, equity_before, equity_after) "
            "VALUES (:trade_id, :sid, :source, :event_signal, "
            " :lots, :margin, :pnl, :equity_before, :equity_after)"
        )
        if session is not None:
            await session.execute(stmt, params)
        else:
            async with engine.begin() as conn:
                await conn.execute(stmt, params)
