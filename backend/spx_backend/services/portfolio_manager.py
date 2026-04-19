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


class PortfolioClosureSplitBrainError(RuntimeError):
    """Raised when ``record_closure`` cannot find the matching row.

    Indicates the system is in a split-brain state: ``trades`` table
    knows about a closure for a portfolio-managed trade but the
    accompanying ``portfolio_trades`` row is missing (or already had
    a non-NULL ``realized_pnl``).  The trade_pnl_job catches this
    exception, sends an operator alert, and rolls back the closure
    so the next pass can retry.

    Carrying the offending ``trade_id`` lets the alert handler key
    its cooldown on a single trade without spamming the inbox for
    every concurrent closure attempt.
    """

    def __init__(self, trade_id: int, message: str) -> None:
        super().__init__(message)
        self.trade_id = trade_id


MARGIN_PER_LOT = 1000  # keep in sync with scripts/_constants.py


def _margin_per_lot() -> float:
    """Derive margin per lot from configured spread width and multiplier.

    Returns ``spread_width_points * contract_multiplier`` (e.g. 10 * 100 = 1000).
    Falls back to the legacy ``MARGIN_PER_LOT`` constant if either setting is
    unavailable or yields a non-positive value.
    """
    try:
        val = float(settings.decision_spread_width_points) * float(settings.trade_pnl_contract_multiplier)
        return val if val > 0 else float(MARGIN_PER_LOT)
    except Exception:
        return float(MARGIN_PER_LOT)


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

        Preserves the original start-of-day equity and month-start equity
        across multiple ``begin_day`` calls within the same day (e.g. after
        ``trade_pnl_job`` closes a trade between decision runs) by reading
        the stored values from the DB rather than recomputing them from the
        latest (potentially updated) equity.

        Parameters
        ----------
        today : The current trading date.
        """
        self._today = today
        self._lots_today = None

        prev_equity = await self._load_latest_equity()
        if prev_equity is not None:
            self.equity = prev_equity

        # Use the DB's equity_start if today's row already exists so that
        # intra-day closures don't reset the daily PnL baseline.
        db_start = await self._load_today_equity_start(today)
        self._equity_start_today = db_start if db_start is not None else self.equity

        self._trades_today = await self._count_trades_today(today)

        month_key = today.strftime("%Y-%m")
        if month_key != self._current_month:
            self._current_month = month_key
            # Recover month-start from the DB so a process restart mid-month
            # doesn't reset the drawdown baseline to the current equity.
            db_month = await self._load_month_start_equity(month_key)
            self.month_start_equity = db_month if db_month is not None else self.equity
            # Recover monthly_stop_active too, so a process restart after a
            # drawdown trip doesn't accidentally allow new trades for the
            # rest of the month.  We read the LATEST row in the current
            # month: if any prior run flipped the flag, we honour it.
            self._month_stopped = await self._load_month_stop_active(month_key)

        self._state_id = await self._upsert_day_state(
            equity_start=self._equity_start_today,
            month_start_equity=self.month_start_equity,
        )
        logger.info(
            "portfolio_day_start date={} equity={:.0f} month_start={:.0f} "
            "lots={} trades_today={} month_stopped={}",
            today, self.equity, self.month_start_equity, self.compute_lots(),
            self._trades_today, self._month_stopped,
        )

    async def can_trade(self) -> bool:
        """Return True if another trade is permitted right now.

        Async so that when the monthly drawdown limit trips for the
        first time we can immediately persist ``monthly_stop_active=True``
        to ``portfolio_state``.  Without that write, a process restart
        between the trip and the next ``record_trade`` / ``record_closure``
        call would lose the trip and the system would resume trading
        within a stopped month.
        """
        if self._month_stopped:
            return False
        if self.equity < _margin_per_lot():
            return False
        if self.monthly_dd_limit is not None:
            threshold = self.month_start_equity * (1 - self.monthly_dd_limit)
            if self.equity < threshold:
                self._month_stopped = True
                logger.warning(
                    "portfolio_monthly_stop equity={:.0f} threshold={:.0f}",
                    self.equity, threshold,
                )
                # Persist immediately so a crash/restart between this trip
                # and the next equity-mutating call doesn't lose the stop.
                await self._persist_month_stop()
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
        max_by_risk = max(1, int(self.equity * self.max_risk_pct / _margin_per_lot()))
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
        margin_dollars: float | None = None,
    ) -> float:
        """Record a new trade entry and persist to database.

        At entry time the caller should pass ``pnl_per_lot=0.0`` because no
        PnL is realised until the trade closes.  The ``portfolio_trades`` row
        is inserted with ``realized_pnl = NULL`` so that ``record_closure``
        can later identify it via ``WHERE realized_pnl IS NULL``.

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
        margin_dollars : Per-trade total margin actually committed.  When
            provided, persisted as ``portfolio_trades.margin_committed`` so
            risk dashboards reflect the *actual* per-trade ``max_loss``
            instead of the synthetic ``lots * MARGIN_PER_LOT`` derivation.
            When ``None`` we fall back to the synthetic value for backward
            compatibility with callers that don't yet pass it.

        Returns
        -------
        float : Total PnL booked for this entry.
        """
        if lots <= 0:
            # Defence-in-depth: production callers go through compute_lots()
            # which is bounded >= 1, but a future caller passing lots=0 (or
            # negative) would silently overwrite portfolio_state.lots_per_trade
            # via the COALESCE in _apply_equity_delta.  Reject explicitly so
            # the bug surfaces at the call site rather than as quiet state
            # corruption discovered hours/days later.
            raise ValueError(f"record_trade requires lots >= 1; got lots={lots}")

        total_pnl = pnl_per_lot * lots
        margin_committed = (
            float(margin_dollars)
            if margin_dollars is not None
            else lots * _margin_per_lot()
        )
        equity_before = self.equity

        # Atomic delta first so the UPDATE's RETURNING gives us the
        # post-write equity (and trades_placed) value to use as
        # equity_after on the portfolio_trades row.  Pass lots so the
        # day's sizing decision is recorded once at entry and isn't
        # later clobbered by closures (record_closure deliberately omits
        # lots_per_trade so the COALESCE preserves the entry value).
        new_equity, _ = await self._apply_equity_delta(
            equity_delta=total_pnl,
            trades_delta=1,
            lots_per_trade=lots,
            session=session,
        )

        await self._insert_portfolio_trade(
            session=session,
            trade_id=trade_id,
            source=source,
            event_signal=event_signal,
            lots=lots,
            margin=margin_committed,
            pnl=None,
            equity_before=equity_before,
            equity_after=new_equity,
        )

        logger.info(
            "portfolio_trade source={} pnl={:.0f} lots={} margin={:.0f} equity={:.0f}->={:.0f}",
            source, total_pnl, lots, margin_committed, equity_before, new_equity,
        )
        return total_pnl

    async def record_closure(
        self,
        trade_id: int,
        realized_pnl: float,
        session: AsyncSession | None = None,
    ) -> None:
        """Record a trade closure and update portfolio equity.

        Called by ``trade_pnl_job`` after closing a trade via TP, SL, or
        expiration.  Updates the ``portfolio_trades.realized_pnl`` column
        and adjusts ``portfolio_state.equity_end`` so that lot sizing and
        drawdown checks reflect actual realized outcomes.

        Raises
        ------
        PortfolioClosureSplitBrainError
            When no matching ``portfolio_trades`` row exists for the given
            ``trade_id`` and the portfolio system is enabled.  This is a
            data-integrity failure: the ``trades`` table is about to record
            a closure for a portfolio-managed trade but the corresponding
            ``portfolio_trades`` row is missing.  The caller is expected to
            catch the exception, alert an operator, and roll back.

        Parameters
        ----------
        trade_id : The trade that was closed.
        realized_pnl : Realized dollar PnL from the closure.
        session : Optional session for transactional consistency with the
            caller's close-trade writes.
        """
        # Idempotency: only update rows that have never been closed
        # (realized_pnl IS NULL).  Previous version also matched
        # ``realized_pnl = 0`` which caused replayed closures with a zero
        # PnL to succeed again, re-running ``_update_day_state``.
        stmt = text(
            "UPDATE portfolio_trades SET realized_pnl = :pnl "
            "WHERE trade_id = :tid AND realized_pnl IS NULL"
        )
        params = {"pnl": realized_pnl, "tid": trade_id}

        if session is not None:
            result = await session.execute(stmt, params)
        else:
            async with engine.begin() as conn:
                result = await conn.execute(stmt, params)

        if result.rowcount == 0:
            # Probe whether ANY portfolio_trades row exists for this trade
            # (regardless of realized_pnl).  Two paths:
            #   * Row exists but already has realized_pnl: idempotent retry
            #     of an already-closed trade -> silently no-op (legacy
            #     behaviour preserved; this is normal in retry storms).
            #   * Row truly missing: split-brain.  Raise so the caller
            #     can alert + rollback.
            existed = await self._portfolio_trade_row_exists(
                trade_id=trade_id, session=session,
            )
            if existed:
                logger.info(
                    "portfolio_closure_idempotent: trade_id={} already had "
                    "realized_pnl set; no-op",
                    trade_id,
                )
                return
            # Truly missing row + portfolio is enabled = split-brain.
            raise PortfolioClosureSplitBrainError(
                trade_id=trade_id,
                message=(
                    f"portfolio_trades row missing for trade_id={trade_id}; "
                    "trades table closure must be rolled back"
                ),
            )

        equity_before = self.equity
        new_equity, _ = await self._apply_equity_delta(
            equity_delta=realized_pnl,
            trades_delta=0,
            session=session,
        )

        logger.info(
            "portfolio_closure trade_id={} pnl={:.2f} equity={:.2f}->{:.2f}",
            trade_id, realized_pnl, equity_before, new_equity,
        )

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

    async def _load_today_equity_start(self, today: date) -> float | None:
        """Read today's equity_start from ``portfolio_state`` if the row exists.

        Returns the stored value so that multiple ``begin_day`` calls in the
        same day preserve the original start-of-day baseline even when
        ``equity_end`` has been updated by trade closures in between.
        """
        async with engine.connect() as conn:
            row = await conn.execute(text(
                "SELECT equity_start FROM portfolio_state WHERE date = :d"
            ), {"d": today})
            result = row.fetchone()
            return float(result[0]) if result else None

    async def _load_month_start_equity(self, month_key: str) -> float | None:
        """Read month_start_equity from the earliest row in the given month.

        On a process restart ``_current_month`` is ``None``, so ``begin_day``
        always enters the month-change branch.  Without this DB lookup it
        would overwrite ``month_start_equity`` with the latest equity,
        breaking the monthly drawdown baseline.
        """
        async with engine.connect() as conn:
            row = await conn.execute(text(
                "SELECT month_start_equity FROM portfolio_state "
                "WHERE to_char(date, 'YYYY-MM') = :mk "
                "AND month_start_equity IS NOT NULL "
                "ORDER BY date ASC LIMIT 1"
            ), {"mk": month_key})
            result = row.fetchone()
            return float(result[0]) if result else None

    async def _load_month_stop_active(self, month_key: str) -> bool:
        """Read the latest ``monthly_stop_active`` flag in the given month.

        On process restart we need to know whether a prior run tripped the
        monthly drawdown stop.  We read the LATEST row of the month rather
        than the earliest because the trip can happen any day; once tripped
        the flag stays True for the remainder of the month, so the latest
        row's value is authoritative.

        Returns False when no rows exist for the month or when the column
        is NULL (defensive: schema default is False so this should be rare).
        """
        async with engine.connect() as conn:
            row = await conn.execute(text(
                "SELECT monthly_stop_active FROM portfolio_state "
                "WHERE to_char(date, 'YYYY-MM') = :mk "
                "ORDER BY date DESC LIMIT 1"
            ), {"mk": month_key})
            result = row.fetchone()
            if result is None or result[0] is None:
                return False
            return bool(result[0])

    async def _persist_month_stop(self) -> None:
        """Persist ``monthly_stop_active=True`` to today's portfolio_state row.

        Called from ``can_trade()`` the moment the drawdown trip is detected
        so the flag is durable across restarts even if no further write
        (record_trade / record_closure) happens before the process dies.

        Always opens its own short-lived transaction (no ``session`` kwarg)
        because the trip can fire from any caller context, including ones
        that haven't opened a session.
        """
        if self._state_id is None:
            logger.warning(
                "_persist_month_stop: skipped (state_id not initialized; "
                "begin_day was likely not called)"
            )
            return
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE portfolio_state SET monthly_stop_active = TRUE "
                    "WHERE id = :sid"
                ),
                {"sid": self._state_id},
            )

    async def _upsert_day_state(
        self,
        equity_start: float,
        month_start_equity: float,
    ) -> int:
        """Create or update today's portfolio_state row and return its id.

        On INSERT (first run of the day) sets ``equity_start`` and defaults
        ``equity_end`` to the same value so that no-trade days don't leave
        ``equity_end`` as NULL.  On conflict (subsequent runs) only refreshes
        ``month_start_equity`` so the original start-of-day equity is
        preserved.
        """
        async with engine.begin() as conn:
            row = await conn.execute(text(
                "INSERT INTO portfolio_state (date, equity_start, equity_end, month_start_equity) "
                "VALUES (:d, :es, :es, :ms) "
                "ON CONFLICT (date) DO UPDATE SET month_start_equity = :ms "
                "RETURNING id"
            ), {"d": self._today, "es": equity_start, "ms": month_start_equity})
            return row.scalar_one()

    async def _apply_equity_delta(
        self,
        *,
        equity_delta: float,
        trades_delta: int,
        lots_per_trade: int | None = None,
        event_signals: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> tuple[float, int]:
        """Atomically apply equity / trades deltas and return refreshed state.

        Replaces the previous absolute-overwrite ``_update_day_state``.
        Two concurrent trades booked in different sessions used to race:
        each read ``self.equity``, added their PnL, and overwrote
        ``equity_end`` -- the second write wiped the first's PnL.

        Now we let Postgres do the addition with a single SQL statement
        keyed on the row id, then refresh the in-memory cache with the
        ``RETURNING`` row.  ``daily_pnl`` is recomputed from the new
        equity_end so it never drifts from ``equity_end - equity_start``.

        Parameters
        ----------
        equity_delta:
            Signed dollar change to apply to ``equity_end``.  ``0`` is
            allowed (e.g. zero-PnL closure) and produces a no-op equity
            change but still refreshes the in-memory cache.
        trades_delta:
            Signed integer change to apply to ``trades_placed``.
            ``record_trade`` passes ``1``; ``record_closure`` passes ``0``.
        lots_per_trade:
            Optional per-day lot-sizing snapshot to merge into the row
            via ``COALESCE(:lpt, lots_per_trade)``.  ``record_trade``
            passes the freshly-computed ``lots`` so the entry path
            durably records the day's sizing decision; ``record_closure``
            omits it so the closure path doesn't clobber the original
            entry value with a stale fallback (the previous implementation
            derived ``lpt`` from ``self._lots_today`` which is ``None`` on
            the closure path's fresh PortfolioManager instance, causing
            every closure to overwrite ``lots_per_trade`` with ``1``).
        event_signals:
            Optional list of event-signal names to merge into the JSONB
            column.  ``None`` preserves the existing column value rather
            than overwriting with NULL.
        session:
            Optional existing async session; when provided the UPDATE runs
            inside that transaction so the caller can roll back the trade
            insert and the equity update together.

        Returns
        -------
        tuple[float, int]
            The post-update ``(equity_end, trades_placed)`` as read from
            ``RETURNING``.  These values are also written into
            ``self.equity`` and ``self._trades_today`` so the in-memory
            cache stays consistent for the next ``can_trade`` call.

        Notes
        -----
        ``monthly_stop_active`` is intentionally NOT in the ``RETURNING``
        clause.  The flag is written eagerly by ``_persist_month_stop``
        the moment ``can_trade`` trips the drawdown gate, and reloaded by
        ``begin_day`` via ``_load_month_stop_active`` whenever the
        ``PortfolioManager`` enters a new calendar month -- which in
        production includes every freshly-constructed instance, since
        ``decision_job._run`` builds a new
        ``ProdPortfolioManager`` per run and ``_current_month`` starts as
        ``None``.  Refreshing the flag from every equity-delta would
        couple two unrelated write paths and require additional retry
        logic if the column went NULL on a legacy row, with no
        observable benefit.
        """
        if self._state_id is None:
            raise RuntimeError(
                "_apply_equity_delta called before begin_day initialized "
                "self._state_id"
            )
        params: dict[str, Any] = {
            "delta": float(equity_delta),
            "tdelta": int(trades_delta),
            "lpt": int(lots_per_trade) if lots_per_trade is not None else None,
            "ev": json.dumps(event_signals) if event_signals else None,
            "sid": self._state_id,
        }
        # COALESCE for both equity_end and trades_placed because the row
        # was inserted with equity_end = equity_start (non-null) but the
        # trades_placed default could be NULL on legacy rows.  Using
        # COALESCE keeps the increment safe regardless.
        # lots_per_trade and event_signals both use COALESCE(:param,
        # column) so passing None preserves the current value instead of
        # nulling it out -- closures must not overwrite the entry-time
        # sizing or the entry-time signal list.
        stmt = text(
            "UPDATE portfolio_state SET "
            "  equity_end = COALESCE(equity_end, equity_start) + :delta, "
            "  trades_placed = COALESCE(trades_placed, 0) + :tdelta, "
            "  daily_pnl = COALESCE(equity_end, equity_start) + :delta - equity_start, "
            "  lots_per_trade = COALESCE(:lpt, lots_per_trade), "
            "  event_signals = COALESCE(:ev::jsonb, event_signals) "
            "WHERE id = :sid "
            "RETURNING equity_end, trades_placed"
        )
        if session is not None:
            result = await session.execute(stmt, params)
        else:
            async with engine.begin() as conn:
                result = await conn.execute(stmt, params)
        row = result.fetchone()
        if row is None:
            raise RuntimeError(
                f"_apply_equity_delta: no portfolio_state row for "
                f"id={self._state_id} (race or manual delete?)"
            )
        new_equity = float(row[0])
        new_trades = int(row[1])
        # Refresh in-memory cache so subsequent can_trade() / compute_lots
        # decisions see the durable post-write state.
        self.equity = new_equity
        self._trades_today = new_trades
        return new_equity, new_trades

    async def _portfolio_trade_row_exists(
        self,
        *,
        trade_id: int,
        session: AsyncSession | None,
    ) -> bool:
        """Probe whether ANY portfolio_trades row exists for ``trade_id``.

        Used by ``record_closure`` to distinguish the two reasons a 0-rowcount
        UPDATE can happen:

        * Row exists with non-NULL realized_pnl -> idempotent retry, ok.
        * Row truly missing -> split-brain, raise.

        Runs in the caller's session when provided so it sees uncommitted
        writes from the same transaction.
        """
        stmt = text(
            "SELECT 1 FROM portfolio_trades WHERE trade_id = :tid LIMIT 1"
        )
        params = {"tid": trade_id}
        if session is not None:
            result = await session.execute(stmt, params)
        else:
            async with engine.connect() as conn:
                result = await conn.execute(stmt, params)
        return result.scalar() is not None

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
