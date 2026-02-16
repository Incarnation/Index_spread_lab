from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth


def _mid(bid: float | None, ask: float | None) -> float | None:
    """Return mid quote when both bid and ask are available."""
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


@dataclass(frozen=True)
class TradePnlJob:
    """Mark-to-market open trades and close by TP/SL/expiry."""

    clock_cache: MarketClockCache | None = None

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

    async def _trade_legs(self, session, trade_id: int) -> tuple[dict, dict] | None:
        """Return short and long leg rows for a trade."""
        rows = await session.execute(
            text(
                """
                SELECT leg_index, option_symbol, side, qty, entry_price
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
            }
            if side in {"STO", "SHORT"} and short_leg is None:
                short_leg = leg
            elif side in {"BTO", "LONG"} and long_leg is None:
                long_leg = leg
        if not short_leg or not long_leg:
            return None
        return short_leg, long_leg

    async def run_once(self, *, force: bool = False) -> dict:
        """Run one live mark-to-market cycle for all open trades."""
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        logger.info("trade_pnl_job: start force={} now_et={}", force, now_et.isoformat())

        if not settings.trade_pnl_enabled:
            return {"skipped": True, "reason": "trade_pnl_disabled", "now_et": now_et.isoformat()}

        if (not force) and (not settings.trade_pnl_allow_outside_rth):
            if not await self._market_open(now_et):
                return {"skipped": True, "reason": "market_closed", "now_et": now_et.isoformat(), "updated": 0}

        updated = 0
        closed = 0
        marks_written = 0
        skipped_no_legs = 0
        skipped_no_mark = 0
        skipped_stale = 0

        async with SessionLocal() as session:
            open_rows = await session.execute(
                text(
                    """
                    SELECT trade_id, entry_credit, contracts, contract_multiplier, expiration,
                           take_profit_target, stop_loss_target, max_profit
                    FROM trades
                    WHERE status = 'OPEN'
                    ORDER BY entry_time ASC
                    """
                )
            )

            for trade in open_rows.fetchall():
                legs = await self._trade_legs(session, trade.trade_id)
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

                short_mid = _mid(mark["short_bid"], mark["short_ask"])
                long_mid = _mid(mark["long_bid"], mark["long_ask"])
                if short_mid is None or long_mid is None:
                    skipped_no_mark += 1
                    continue

                entry_credit = float(trade.entry_credit or 0.0)
                contracts = int(trade.contracts or settings.decision_contracts or 1)
                contract_multiplier = int(trade.contract_multiplier or settings.trade_pnl_contract_multiplier)
                exit_cost = short_mid - long_mid
                pnl = (entry_credit - exit_cost) * contracts * contract_multiplier

                take_profit_target = trade.take_profit_target
                if take_profit_target is None and trade.max_profit is not None:
                    take_profit_target = float(trade.max_profit) * settings.trade_pnl_take_profit_pct
                stop_loss_target = trade.stop_loss_target
                if stop_loss_target is None and trade.max_profit is not None:
                    stop_loss_target = float(trade.max_profit) * settings.trade_pnl_stop_loss_pct

                close_reason = None
                if take_profit_target is not None and pnl >= float(take_profit_target):
                    close_reason = "TAKE_PROFIT_50"
                elif stop_loss_target is not None and pnl <= -abs(float(stop_loss_target)):
                    close_reason = "STOP_LOSS"
                elif trade.expiration is not None and mark_ts.date() > trade.expiration:
                    close_reason = "EXPIRED"

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
                            "trade_id": trade.trade_id,
                            "exit_time": mark_ts,
                            "current_pnl": pnl,
                            "realized_pnl": pnl,
                            "current_exit_cost": exit_cost,
                            "exit_reason": close_reason,
                        },
                    )
                    await session.execute(
                        text(
                            """
                            UPDATE trade_legs
                            SET exit_price = :exit_price
                            WHERE trade_id = :trade_id
                              AND option_symbol = :option_symbol
                            """
                        ),
                        {
                            "trade_id": trade.trade_id,
                            "option_symbol": short_leg["option_symbol"],
                            "exit_price": short_mid,
                        },
                    )
                    await session.execute(
                        text(
                            """
                            UPDATE trade_legs
                            SET exit_price = :exit_price
                            WHERE trade_id = :trade_id
                              AND option_symbol = :option_symbol
                            """
                        ),
                        {
                            "trade_id": trade.trade_id,
                            "option_symbol": long_leg["option_symbol"],
                            "exit_price": long_mid,
                        },
                    )
                    closed += 1

            await session.commit()

        logger.info(
            "trade_pnl_job: updated={} closed={} marks_written={} skipped_no_legs={} skipped_no_mark={} skipped_stale={}",
            updated,
            closed,
            marks_written,
            skipped_no_legs,
            skipped_no_mark,
            skipped_stale,
        )
        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "updated": updated,
            "closed": closed,
            "marks_written": marks_written,
            "skipped_no_legs": skipped_no_legs,
            "skipped_no_mark": skipped_no_mark,
            "skipped_stale": skipped_stale,
        }


def build_trade_pnl_job(clock_cache: MarketClockCache | None = None) -> TradePnlJob:
    """Factory helper for TradePnlJob."""
    return TradePnlJob(clock_cache=clock_cache)
