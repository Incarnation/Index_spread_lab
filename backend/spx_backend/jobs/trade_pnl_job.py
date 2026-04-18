from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth
from spx_backend.services.alerts import send_alert
from spx_backend.services.portfolio_manager import (
    PortfolioClosureSplitBrainError,
    PortfolioManager,
)
from spx_backend.utils.options import resolve_option_right
from spx_backend.utils.pricing import mid_price

if TYPE_CHECKING:
    from spx_backend.services.sms_notifier import SmsNotifier


def derive_stop_loss_target(
    *,
    existing_target: float | None,
    basis: str,
    pct: float,
    max_profit: float | None,
    max_loss: float | None,
) -> float | None:
    """Compute a stop-loss dollar threshold from trade attributes.

    If *existing_target* is already set, it is returned as-is.  Otherwise
    the threshold is derived from *max_loss* (when ``basis == "max_loss"``
    and the value is present) or falls back to *max_profit*.

    Returns ``None`` when neither *max_profit* nor *max_loss* is available.
    """
    if existing_target is not None:
        return existing_target
    if basis == "max_loss" and max_loss is not None:
        return float(max_loss) * pct
    if max_profit is not None:
        return float(max_profit) * pct
    return None


@dataclass
class TradePnlJob:
    """Mark-to-market open trades and close by TP/SL/expiry."""

    clock_cache: MarketClockCache | None = None
    notifier: SmsNotifier | None = None
    _pm: PortfolioManager | None = field(default=None, init=False, repr=False)

    async def _market_open(self, now_et: datetime) -> bool:
        """Check market-open state from clock cache with RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    async def _latest_spread_mark(
        self,
        *,
        session,
        short_symbol: str,
        long_symbol: str,
        now_utc: datetime,
    ) -> dict | None:
        """Load latest joint mark where both short/long symbols are present."""
        row = await session.execute(
            text(
                """
                SELECT cs.snapshot_id, cs.ts, s.bid AS short_bid, s.ask AS short_ask, l.bid AS long_bid, l.ask AS long_ask
                FROM chain_snapshots cs
                JOIN option_chain_rows s ON s.snapshot_id = cs.snapshot_id AND s.option_symbol = :short_symbol
                JOIN option_chain_rows l ON l.snapshot_id = cs.snapshot_id AND l.option_symbol = :long_symbol
                WHERE cs.ts <= :now_ts
                ORDER BY cs.ts DESC
                LIMIT 1
                """
            ),
            {"short_symbol": short_symbol, "long_symbol": long_symbol, "now_ts": now_utc},
        )
        result = row.fetchone()
        if not result:
            return None
        return {
            "snapshot_id": result.snapshot_id,
            "ts": result.ts,
            "short_bid": result.short_bid,
            "short_ask": result.short_ask,
            "long_bid": result.long_bid,
            "long_ask": result.long_ask,
        }

    async def _latest_spot(self, session, *, now_utc: datetime) -> float | None:
        """Return the latest underlying quote price for intrinsic settlement."""
        row = await session.execute(
            text(
                """
                SELECT last
                FROM underlying_quotes
                WHERE symbol = :symbol AND ts <= :now_ts
                ORDER BY ts DESC
                LIMIT 1
                """
            ),
            {"symbol": settings.snapshot_underlying, "now_ts": now_utc},
        )
        result = row.fetchone()
        return float(result.last) if result and result.last is not None else None

    async def _trade_legs(self, session, trade_id: int) -> tuple[dict, dict] | None:
        """Return short and long leg rows for a trade."""
        rows = await session.execute(
            text(
                """
                SELECT leg_index, option_symbol, side, qty, entry_price, strike
                FROM trade_legs
                WHERE trade_id = :trade_id
                ORDER BY leg_index ASC
                """
            ),
            {"trade_id": trade_id},
        )
        short_leg = None
        long_leg = None
        for row in rows.fetchall():
            side = (row.side or "").upper()
            leg = {
                "leg_index": row.leg_index,
                "option_symbol": row.option_symbol,
                "side": side,
                "qty": row.qty,
                "entry_price": row.entry_price,
                "strike": float(row.strike) if row.strike is not None else None,
            }
            if side in {"STO", "SHORT"} and short_leg is None:
                short_leg = leg
            elif side in {"BTO", "LONG"} and long_leg is None:
                long_leg = leg
        if not short_leg or not long_leg:
            return None
        return short_leg, long_leg

    async def _bulk_trade_legs(self, session, trade_ids: list[int]) -> dict[int, tuple[dict, dict]]:
        """Load legs for multiple trades in a single query.

        Parameters
        ----------
        session:
            Async database session.
        trade_ids:
            Trade IDs to load legs for.

        Returns
        -------
        dict[int, tuple[dict, dict]]
            Mapping from trade_id to (short_leg, long_leg).  Trades with
            missing or incomplete legs are omitted from the result.
        """
        if not trade_ids:
            return {}
        rows = await session.execute(
            text(
                """
                SELECT trade_id, leg_index, option_symbol, side, qty, entry_price, strike
                FROM trade_legs
                WHERE trade_id = ANY(:trade_ids)
                ORDER BY trade_id, leg_index ASC
                """
            ),
            {"trade_ids": trade_ids},
        )
        legs_by_trade: dict[int, dict[str, dict | None]] = {}
        for row in rows.fetchall():
            tid = row.trade_id
            side = (row.side or "").upper()
            leg = {
                "leg_index": row.leg_index,
                "option_symbol": row.option_symbol,
                "side": side,
                "qty": row.qty,
                "entry_price": row.entry_price,
                "strike": float(row.strike) if row.strike is not None else None,
            }
            slot = legs_by_trade.setdefault(tid, {"short": None, "long": None})
            if side in {"STO", "SHORT"} and slot["short"] is None:
                slot["short"] = leg
            elif side in {"BTO", "LONG"} and slot["long"] is None:
                slot["long"] = leg

        result: dict[int, tuple[dict, dict]] = {}
        for tid, slot in legs_by_trade.items():
            if slot["short"] is not None and slot["long"] is not None:
                result[tid] = (slot["short"], slot["long"])
        return result

    async def _close_trade(
        self,
        *,
        session,
        trade_id: int,
        exit_time: datetime,
        pnl: float,
        exit_cost: float,
        exit_reason: str,
        short_leg: dict | None = None,
        long_leg: dict | None = None,
        short_exit_price: float | None = None,
        long_exit_price: float | None = None,
    ) -> None:
        """Persist a trade closure: update status, set realized PnL, record leg exit prices.

        Parameters
        ----------
        session:
            Async database session.
        trade_id:
            ID of the trade to close.
        exit_time:
            Timestamp of the exit event.
        pnl:
            Realized PnL in dollars.
        exit_cost:
            Per-unit exit cost of the spread.
        exit_reason:
            Machine-readable close reason (TAKE_PROFIT_<pct>, STOP_LOSS, EXPIRED).
        short_leg / long_leg:
            Leg dicts; when provided the corresponding exit_price is written.
        short_exit_price / long_exit_price:
            Mid prices for the exit mark on each leg.
        """
        await session.execute(
            text(
                """
                UPDATE trades
                SET status = 'CLOSED',
                    exit_time = :exit_time,
                    current_pnl = :current_pnl,
                    realized_pnl = :realized_pnl,
                    current_exit_cost = :current_exit_cost,
                    exit_reason = :exit_reason
                WHERE trade_id = :trade_id
                """
            ),
            {
                "trade_id": trade_id,
                "exit_time": exit_time,
                "current_pnl": pnl,
                "realized_pnl": pnl,
                "current_exit_cost": exit_cost,
                "exit_reason": exit_reason,
            },
        )
        if short_leg is not None and short_exit_price is not None:
            await session.execute(
                text(
                    "UPDATE trade_legs SET exit_price = :exit_price "
                    "WHERE trade_id = :trade_id AND option_symbol = :option_symbol"
                ),
                {"trade_id": trade_id, "option_symbol": short_leg["option_symbol"], "exit_price": short_exit_price},
            )
        if long_leg is not None and long_exit_price is not None:
            await session.execute(
                text(
                    "UPDATE trade_legs SET exit_price = :exit_price "
                    "WHERE trade_id = :trade_id AND option_symbol = :option_symbol"
                ),
                {"trade_id": trade_id, "option_symbol": long_leg["option_symbol"], "exit_price": long_exit_price},
            )

    async def _get_portfolio_manager(self, now_et: datetime) -> PortfolioManager:
        """Lazy-init a PortfolioManager for recording closures.

        Loads current equity from DB via ``begin_day`` so that the equity
        adjustment is based on the latest persisted state.
        """
        if self._pm is None:
            self._pm = PortfolioManager()
            await self._pm.begin_day(now_et.date())
        return self._pm

    async def _close_with_portfolio(
        self,
        *,
        pm: PortfolioManager,
        trade_id: int,
        pnl: float,
        session,
    ) -> None:
        """Call ``pm.record_closure`` with split-brain alert + rollback semantics.

        ``record_closure`` raises :class:`PortfolioClosureSplitBrainError`
        when the portfolio is enabled but the corresponding
        ``portfolio_trades`` row is missing.  In that case we:

        1. Send a SendGrid operator alert keyed on the trade_id (so the
           same offending trade doesn't spam the inbox on every retry
           attempt -- the cooldown is per-trade).
        2. Re-raise so the surrounding ``SessionLocal`` context exits
           without committing.  The earlier ``_close_trade`` UPDATE in
           the same session is rolled back, leaving the trade OPEN so
           the next ``trade_pnl_job`` run can retry once the operator
           has reconciled the portfolio_trades row.
        """
        try:
            await pm.record_closure(trade_id, pnl, session=session)
        except PortfolioClosureSplitBrainError as exc:
            # Fire alert before re-raising.  send_alert swallows its
            # own SendGrid errors so the alert delivery can never
            # mask the underlying split-brain.
            await send_alert(
                subject=(
                    f"[IndexSpreadLab] Portfolio closure split-brain "
                    f"trade_id={trade_id}"
                ),
                body_html=(
                    f"<p><strong>Portfolio closure split-brain detected.</strong></p>"
                    f"<p>trade_id={trade_id}, attempted realized_pnl={pnl:.2f}.</p>"
                    f"<p>The trades-table closure has been rolled back; "
                    f"the next trade_pnl_job run will retry once the "
                    f"portfolio_trades row is reconciled manually.</p>"
                    f"<p>Detail: {exc}</p>"
                ),
                cooldown_key=f"split_brain:trade_id={trade_id}",
                cooldown_minutes=60,
            )
            raise

    @staticmethod
    def _intrinsic_exit_cost(
        option_right: str | None,
        short_strike: float | None,
        long_strike: float | None,
        spot: float | None,
    ) -> float | None:
        """Compute intrinsic-value exit cost at expiration from spot and strikes.

        ``option_right`` must be one of ``"put"`` / ``"call"`` -- typically
        produced by :func:`spx_backend.utils.options.resolve_option_right`
        from the trade's ``trade_legs.option_right`` (with strategy_type
        substring fallback for legacy rows).

        For a put credit spread (short put above long put):
          - spot >= short_strike -> both expire worthless -> exit_cost = 0
          - spot <= long_strike  -> max loss -> exit_cost = short - long
          - otherwise            -> partial intrinsic on short, long worthless

        For a call credit spread (short call below long call):
          - spot <= short_strike -> both expire worthless -> exit_cost = 0
          - spot >= long_strike  -> max loss -> exit_cost = -(long - short)
          - otherwise            -> partial intrinsic on short, long worthless

        Returns exit_cost in points (same sign convention as mark-based:
        ``exit_cost = short_value - long_value``), or ``None`` when
        inputs are insufficient (missing spot/strikes/right).
        """
        if spot is None or short_strike is None or long_strike is None:
            return None
        if option_right == "put":
            short_intrinsic = max(short_strike - spot, 0.0)
            long_intrinsic = max(long_strike - spot, 0.0)
            return short_intrinsic - long_intrinsic
        if option_right == "call":
            short_intrinsic = max(spot - short_strike, 0.0)
            long_intrinsic = max(spot - long_strike, 0.0)
            return short_intrinsic - long_intrinsic
        return None

    async def _close_expired_trades(
        self,
        *,
        session,
        trades: list,
        all_legs: dict,
        now_et: datetime,
        now_utc: datetime,
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Close all expired trades and update portfolio equity.

        Uses three fallback tiers for exit valuation:
        1. Last snapshot mark (mid-to-mid) if available.
        2. Intrinsic-value settlement from latest spot price and leg strikes.
        3. Worst-case assumption of full max loss.

        Returns (expired_closed, total_closed_delta, sms_close_infos).
        """
        expired_closed = 0
        closed = 0
        sms_close_infos: list[dict[str, Any]] = []

        spot: float | None = None
        spot_loaded = False

        for trade in trades:
            entry_credit = float(trade.entry_credit or 0.0)
            contracts = int(trade.contracts or settings.decision_contracts or 1)
            contract_multiplier = int(trade.contract_multiplier or settings.trade_pnl_contract_multiplier)

            legs = all_legs.get(trade.trade_id)
            pnl: float
            exit_cost: float
            short_leg_ref: dict | None = None
            long_leg_ref: dict | None = None
            short_exit: float | None = None
            long_exit: float | None = None

            if legs is not None:
                short_leg_ref, long_leg_ref = legs
                mark = await self._latest_spread_mark(
                    session=session,
                    short_symbol=short_leg_ref["option_symbol"],
                    long_symbol=long_leg_ref["option_symbol"],
                    now_utc=now_utc,
                )
                short_mid = mid_price(mark["short_bid"], mark["short_ask"]) if mark else None
                long_mid = mid_price(mark["long_bid"], mark["long_ask"]) if mark else None
                if short_mid is not None and long_mid is not None:
                    exit_cost = short_mid - long_mid
                    pnl = (entry_credit - exit_cost) * contracts * contract_multiplier
                    short_exit = short_mid
                    long_exit = long_mid
                else:
                    # Tier 2: intrinsic settlement from spot and leg strikes.
                    # Resolve put/call from trade_legs.option_right first,
                    # falling back to strategy_type substring for legacy
                    # rows.  Avoids the previous bug where strategy_type
                    # values without "put"/"call" tokens (e.g.
                    # "credit_spread") silently returned None and dropped
                    # to tier 3 even with valid leg strikes.
                    if not spot_loaded:
                        spot = await self._latest_spot(session, now_utc=now_utc)
                        spot_loaded = True
                    option_right = resolve_option_right(
                        getattr(trade, "strategy_type", None),
                        short_leg_ref,
                        long_leg_ref,
                    )
                    intrinsic = self._intrinsic_exit_cost(
                        option_right=option_right,
                        short_strike=short_leg_ref.get("strike"),
                        long_strike=long_leg_ref.get("strike"),
                        spot=spot,
                    )
                    if intrinsic is not None:
                        exit_cost = intrinsic
                        pnl = (entry_credit - exit_cost) * contracts * contract_multiplier
                    else:
                        # Tier 3: refuse to default to zero PnL when we
                        # can't even derive max_loss.  Raising rolls back
                        # the surrounding session so the next run can
                        # retry with fresher spot/mark data instead of
                        # silently booking the trade as flat.
                        if trade.max_loss is None:
                            raise RuntimeError(
                                f"trade_pnl_job: cannot settle expired trade_id={trade.trade_id}; "
                                "no mark, no spot/intrinsic, and trade.max_loss is NULL"
                            )
                        pnl = -abs(float(trade.max_loss))
                        exit_cost = entry_credit - (pnl / max(contracts * contract_multiplier, 1))
            else:
                # No legs at all: same tier-3 hard-fail rule applies.
                if trade.max_loss is None:
                    raise RuntimeError(
                        f"trade_pnl_job: cannot settle expired trade_id={trade.trade_id}; "
                        "no legs available and trade.max_loss is NULL"
                    )
                pnl = -abs(float(trade.max_loss))
                exit_cost = entry_credit - (pnl / max(contracts * contract_multiplier, 1))

            await self._close_trade(
                session=session,
                trade_id=trade.trade_id,
                exit_time=now_utc,
                pnl=pnl,
                exit_cost=exit_cost,
                exit_reason="EXPIRED",
                short_leg=short_leg_ref,
                long_leg=long_leg_ref,
                short_exit_price=short_exit,
                long_exit_price=long_exit,
            )

            if settings.portfolio_enabled:
                pm = await self._get_portfolio_manager(now_et)
                await self._close_with_portfolio(
                    pm=pm, trade_id=trade.trade_id, pnl=pnl, session=session,
                )

            expired_closed += 1
            closed += 1
            sms_close_infos.append({
                "trade_id": trade.trade_id,
                "strategy_type": getattr(trade, "strategy_type", None),
                "exit_reason": "EXPIRED",
                "entry_credit": entry_credit,
                "exit_cost": exit_cost,
                "realized_pnl": pnl,
                "contracts": contracts,
                "source": getattr(trade, "trade_source", ""),
            })
            logger.info(
                "trade_pnl_job: trade_id={} closed=EXPIRED expiration={} pnl={:.2f}",
                trade.trade_id, trade.expiration, pnl,
            )

        return expired_closed, closed, sms_close_infos

    async def run_once(self, *, force: bool = False) -> dict:
        """Run one live mark-to-market cycle for all open trades.

        Handles three close triggers in priority order:
        1. **Expiration** -- always processed regardless of RTH so that
           trades expiring on Friday are closed over the weekend.
        2. **Take-profit** / **Stop-loss** -- evaluated against the latest
           spread mark from chain snapshot data (RTH-only unless forced).
        3. Otherwise the trade mark is updated with the current unrealized PnL.

        On each closure, ``PortfolioManager.record_closure`` is called to
        flow realized PnL back to ``portfolio_state.equity_end``.
        """
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        logger.info("trade_pnl_job: start force={} now_et={}", force, now_et.isoformat())

        self._pm = None

        if not settings.trade_pnl_enabled:
            return {"skipped": True, "reason": "trade_pnl_disabled", "now_et": now_et.isoformat()}

        updated = 0
        closed = 0
        expired_closed = 0
        marks_written = 0
        skipped_no_legs = 0
        skipped_no_mark = 0
        skipped_stale = 0

        async with SessionLocal() as session:
            open_rows = await session.execute(
                text(
                    """
                    SELECT trade_id, entry_credit, contracts, contract_multiplier, expiration,
                           take_profit_target, stop_loss_target, max_profit, max_loss,
                           strategy_type, trade_source
                    FROM trades
                    WHERE status = 'OPEN'
                    ORDER BY entry_time ASC
                    """
                )
            )
            trades_list = open_rows.fetchall()

            trade_ids = [t.trade_id for t in trades_list]
            all_legs = await self._bulk_trade_legs(session, trade_ids)

            # Phase 1: process expirations (runs even outside RTH).
            # Close on the expiration day itself once the market has closed
            # (16:00 ET) rather than waiting until the next calendar day.
            market_close_hour = 16
            past_close_today = now_et.hour >= market_close_hour

            def _is_expired(t) -> bool:
                if t.expiration is None:
                    return False
                if now_et.date() > t.expiration:
                    return True
                return now_et.date() == t.expiration and past_close_today

            expired_trades = [t for t in trades_list if _is_expired(t)]
            active_trades = [t for t in trades_list if not _is_expired(t)]

            sms_close_infos: list[dict[str, Any]] = []
            if expired_trades:
                expired_closed, closed, expired_sms = await self._close_expired_trades(
                    session=session,
                    trades=expired_trades,
                    all_legs=all_legs,
                    now_et=now_et,
                    now_utc=now_utc,
                )
                sms_close_infos.extend(expired_sms)

            # Phase 2: RTH gate for mark-to-market and TP/SL
            if (not force) and (not settings.trade_pnl_allow_outside_rth):
                if not await self._market_open(now_et):
                    await session.commit()
                    if self.notifier and sms_close_infos:
                        for ci in sms_close_infos:
                            await self.notifier.notify_trade_closed(ci)
                    logger.info(
                        "trade_pnl_job: market_closed expired_closed={} closed={}",
                        expired_closed, closed,
                    )
                    return {
                        "skipped": expired_closed == 0,
                        "reason": "market_closed" if expired_closed == 0 else None,
                        "now_et": now_et.isoformat(),
                        "updated": 0,
                        "closed": closed,
                        "expired_closed": expired_closed,
                        "marks_written": 0,
                        "skipped_no_legs": 0,
                        "skipped_no_mark": 0,
                        "skipped_stale": 0,
                    }

            # Phase 3: MTM + TP/SL for non-expired trades
            for trade in active_trades:
                entry_credit = float(trade.entry_credit or 0.0)
                contracts = int(trade.contracts or settings.decision_contracts or 1)
                contract_multiplier = int(trade.contract_multiplier or settings.trade_pnl_contract_multiplier)

                legs = all_legs.get(trade.trade_id)
                if legs is None:
                    skipped_no_legs += 1
                    continue
                short_leg, long_leg = legs

                mark = await self._latest_spread_mark(
                    session=session,
                    short_symbol=short_leg["option_symbol"],
                    long_symbol=long_leg["option_symbol"],
                    now_utc=now_utc,
                )
                if mark is None:
                    skipped_no_mark += 1
                    continue

                mark_ts = mark["ts"]
                age_minutes = (now_utc - mark_ts).total_seconds() / 60.0
                if (not force) and age_minutes > settings.trade_pnl_mark_max_age_minutes:
                    skipped_stale += 1
                    continue

                short_mid = mid_price(mark["short_bid"], mark["short_ask"])
                long_mid = mid_price(mark["long_bid"], mark["long_ask"])
                if short_mid is None or long_mid is None:
                    skipped_no_mark += 1
                    continue

                exit_cost = short_mid - long_mid
                pnl = (entry_credit - exit_cost) * contracts * contract_multiplier

                take_profit_target = trade.take_profit_target
                if take_profit_target is None and trade.max_profit is not None:
                    take_profit_target = float(trade.max_profit) * settings.trade_pnl_take_profit_pct
                stop_loss_target = None
                if settings.trade_pnl_stop_loss_enabled:
                    stop_loss_target = derive_stop_loss_target(
                        existing_target=trade.stop_loss_target,
                        basis=settings.trade_pnl_stop_loss_basis,
                        pct=settings.trade_pnl_stop_loss_pct,
                        max_profit=trade.max_profit,
                        max_loss=trade.max_loss,
                    )

                close_reason = None
                if take_profit_target is not None and pnl >= float(take_profit_target):
                    tp_pct = int(settings.trade_pnl_take_profit_pct * 100)
                    close_reason = f"TAKE_PROFIT_{tp_pct}"
                elif stop_loss_target is not None and pnl <= -abs(float(stop_loss_target)):
                    close_reason = "STOP_LOSS"

                await session.execute(
                    text(
                        """
                        UPDATE trades
                        SET current_pnl = :current_pnl,
                            current_exit_cost = :current_exit_cost,
                            last_mark_ts = :last_mark_ts,
                            last_snapshot_id = :last_snapshot_id
                        WHERE trade_id = :trade_id
                        """
                    ),
                    {
                        "trade_id": trade.trade_id,
                        "current_pnl": pnl,
                        "current_exit_cost": exit_cost,
                        "last_mark_ts": mark_ts,
                        "last_snapshot_id": mark["snapshot_id"],
                    },
                )
                updated += 1

                await session.execute(
                    text(
                        """
                        INSERT INTO trade_marks (
                          trade_id, snapshot_id, ts, short_mid, long_mid, exit_cost, pnl, status
                        )
                        VALUES (
                          :trade_id, :snapshot_id, :ts, :short_mid, :long_mid, :exit_cost, :pnl, :status
                        )
                        ON CONFLICT (trade_id, ts) DO UPDATE SET
                          snapshot_id = EXCLUDED.snapshot_id,
                          short_mid = EXCLUDED.short_mid,
                          long_mid = EXCLUDED.long_mid,
                          exit_cost = EXCLUDED.exit_cost,
                          pnl = EXCLUDED.pnl,
                          status = EXCLUDED.status
                        """
                    ),
                    {
                        "trade_id": trade.trade_id,
                        "snapshot_id": mark["snapshot_id"],
                        "ts": mark_ts,
                        "short_mid": short_mid,
                        "long_mid": long_mid,
                        "exit_cost": exit_cost,
                        "pnl": pnl,
                        "status": "CLOSED" if close_reason else "OPEN",
                    },
                )
                marks_written += 1

                if close_reason:
                    await self._close_trade(
                        session=session,
                        trade_id=trade.trade_id,
                        exit_time=mark_ts,
                        pnl=pnl,
                        exit_cost=exit_cost,
                        exit_reason=close_reason,
                        short_leg=short_leg,
                        long_leg=long_leg,
                        short_exit_price=short_mid,
                        long_exit_price=long_mid,
                    )

                    if settings.portfolio_enabled:
                        pm = await self._get_portfolio_manager(now_et)
                        await self._close_with_portfolio(
                            pm=pm, trade_id=trade.trade_id, pnl=pnl, session=session,
                        )

                    sms_close_infos.append({
                        "trade_id": trade.trade_id,
                        "strategy_type": getattr(trade, "strategy_type", None),
                        "exit_reason": close_reason,
                        "entry_credit": entry_credit,
                        "exit_cost": exit_cost,
                        "realized_pnl": pnl,
                        "contracts": contracts,
                        "source": getattr(trade, "trade_source", ""),
                    })
                    closed += 1

            await session.commit()

        if self.notifier and sms_close_infos:
            for ci in sms_close_infos:
                await self.notifier.notify_trade_closed(ci)

        logger.info(
            "trade_pnl_job: updated={} closed={} expired_closed={} marks_written={} "
            "skipped_no_legs={} skipped_no_mark={} skipped_stale={}",
            updated, closed, expired_closed, marks_written,
            skipped_no_legs, skipped_no_mark, skipped_stale,
        )
        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "updated": updated,
            "closed": closed,
            "expired_closed": expired_closed,
            "marks_written": marks_written,
            "skipped_no_legs": skipped_no_legs,
            "skipped_no_mark": skipped_no_mark,
            "skipped_stale": skipped_stale,
        }


def build_trade_pnl_job(
    clock_cache: MarketClockCache | None = None,
    notifier: SmsNotifier | None = None,
) -> TradePnlJob:
    """Factory helper for TradePnlJob.

    Parameters
    ----------
    clock_cache:
        Shared market-clock cache for RTH checks.
    notifier:
        Optional SMS notifier; when provided, trade-closed messages are sent
        after the DB commit for each closed trade.
    """
    return TradePnlJob(clock_cache=clock_cache, notifier=notifier)
