from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth


@dataclass(frozen=True)
class GexJob:
    """Compute and persist GEX aggregates for new snapshots."""
    clock_cache: MarketClockCache | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        """Check if market is open using cache or RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    async def run_once(self, *, force: bool = False) -> dict:
        """Compute GEX for snapshots that are missing results.

        Parameters
        ----------
        force:
            When true, bypasses regular-trading-hours gating and computes GEX
            regardless of market status.

        Returns
        -------
        dict
            Job result payload containing computed count, skipped/failure
            counters, and failure details for snapshots that could not be
            processed.
        """
        if not settings.gex_enabled:
            return {"skipped": True, "reason": "gex_disabled"}

        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        if (not force) and (not settings.gex_allow_outside_rth):
            if not await self._market_open(now_et):
                logger.info("gex_job: market closed; skipping (now_et={})", now_et.isoformat())
                return {
                    "skipped": True,
                    "reason": "market_closed",
                    "computed_snapshots": 0,
                    "skipped_snapshots": 0,
                    "failed_snapshots": [],
                    "now_et": now_et.isoformat(),
                }
        computed = 0
        skipped_snapshots = 0
        failed_snapshots: list[dict] = []

        async with SessionLocal() as session:
            try:
                # Find snapshots that don't have a gex snapshot yet.
                rows = await session.execute(
                    text(
                        """
                        SELECT cs.snapshot_id, cs.ts, cs.underlying, cs.target_dte
                        FROM chain_snapshots cs
                        LEFT JOIN gex_snapshots gs ON gs.snapshot_id = cs.snapshot_id
                        WHERE gs.snapshot_id IS NULL
                          AND cs.source = 'TRADIER'
                          AND EXISTS (
                            SELECT 1
                            FROM option_chain_rows ocr
                            WHERE ocr.snapshot_id = cs.snapshot_id
                          )
                        ORDER BY cs.snapshot_id DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": settings.gex_snapshot_batch_limit},
                )
                candidates = rows.fetchall()
            except Exception as exc:
                await session.rollback()
                logger.exception("gex_job: candidate_query_failed error={}", exc)
                return {"skipped": True, "reason": "candidate_query_failed", "computed_snapshots": 0, "now_et": now_et.isoformat()}

            if not candidates:
                return {"skipped": True, "reason": "no_pending_snapshots"}

            for snapshot_id, ts, underlying, snapshot_target_dte in candidates:
                try:
                    # Isolate each snapshot in a savepoint to keep processing
                    # the rest of the batch after one malformed snapshot fails.
                    async with session.begin_nested():
                        spot = await self._get_spot_price(session, ts, underlying)
                        if spot is None:
                            skipped_snapshots += 1
                            logger.info("gex_job: no spot price for snapshot_id={}; skipping snapshot", snapshot_id)
                            continue

                        opt_rows = await session.execute(
                            text(
                                """
                                SELECT strike, option_right, open_interest, gamma, contract_size, expiration
                                FROM option_chain_rows
                                WHERE snapshot_id = :snapshot_id
                                """
                            ),
                            {"snapshot_id": snapshot_id},
                        )
                        opts = opt_rows.fetchall()
                        if not opts:
                            skipped_snapshots += 1
                            logger.info("gex_job: no option rows for snapshot_id={}; skipping snapshot", snapshot_id)
                            continue

                        # Filter by DTE window and select top-N strikes near spot.
                        filtered = []
                        resolved_snapshot_dte = int(snapshot_target_dte) if snapshot_target_dte is not None else None
                        for strike, right, oi, gamma, contract_size, expiration in opts:
                            if strike is None or expiration is None:
                                continue
                            dte = resolved_snapshot_dte
                            if dte is None:
                                dte = (expiration - ts.date()).days
                            if dte < 0 or dte > settings.gex_max_dte_days:
                                continue
                            filtered.append((strike, right, oi, gamma, contract_size, expiration, dte))

                        if not filtered:
                            skipped_snapshots += 1
                            logger.info(
                                "gex_job: no options within dte_limit={} for snapshot_id={}; skipping snapshot",
                                settings.gex_max_dte_days,
                                snapshot_id,
                            )
                            continue

                        unique_strikes = sorted({float(r[0]) for r in filtered}, key=lambda s: abs(s - float(spot)))
                        selected_strikes = set(unique_strikes[: settings.gex_strike_limit])

                        per_strike = {}
                        per_expiry_strike = {}
                        gex_calls = 0.0
                        gex_puts = 0.0
                        gex_abs = 0.0

                        for strike, right, oi, gamma, contract_size, expiration, dte in filtered:
                            if float(strike) not in selected_strikes:
                                continue
                            if strike is None or oi is None or gamma is None:
                                continue
                            multiplier = contract_size or settings.gex_contract_multiplier
                            sign = 1.0
                            if settings.gex_puts_negative and (right == "P"):
                                sign = -1.0
                            gex_val = sign * float(gamma) * float(oi) * float(multiplier) * (float(spot) ** 2)

                            # Totals by option side
                            if right == "P":
                                gex_puts += gex_val
                            else:
                                gex_calls += gex_val
                            gex_abs += abs(gex_val)

                            # Aggregate by strike
                            ps = per_strike.setdefault(strike, {"gex_calls": 0.0, "gex_puts": 0.0, "oi": 0})
                            if right == "P":
                                ps["gex_puts"] += gex_val
                            else:
                                ps["gex_calls"] += gex_val
                            ps["oi"] += int(oi)

                            # Aggregate by expiry+strike
                            if settings.gex_store_by_expiry and expiration is not None:
                                key = (expiration, strike)
                                pe = per_expiry_strike.setdefault(
                                    key,
                                    {"gex_calls": 0.0, "gex_puts": 0.0, "oi": 0, "dte_days": dte},
                                )
                                if right == "P":
                                    pe["gex_puts"] += gex_val
                                else:
                                    pe["gex_calls"] += gex_val
                                pe["oi"] += int(oi)

                        # Compute zero gamma level from cumulative net gex by strike
                        zero_gamma = self._zero_gamma_level(per_strike)
                        gex_net = gex_calls + gex_puts

                        method = f"oi_gamma_spot_top{settings.gex_strike_limit}_dte{settings.gex_max_dte_days}"

                        await session.execute(
                            text(
                                """
                                INSERT INTO gex_snapshots (
                                  snapshot_id, ts, underlying, source, spot_price, gex_net, gex_calls, gex_puts, gex_abs, zero_gamma_level, method
                                )
                                VALUES (
                                  :snapshot_id, :ts, :underlying, :source, :spot_price, :gex_net, :gex_calls, :gex_puts, :gex_abs, :zero_gamma, :method
                                )
                                ON CONFLICT (snapshot_id) DO NOTHING
                                """
                            ),
                            {
                                "snapshot_id": snapshot_id,
                                "ts": ts,
                                "underlying": underlying,
                                "source": "TRADIER",
                                "spot_price": spot,
                                "gex_net": gex_net,
                                "gex_calls": gex_calls,
                                "gex_puts": gex_puts,
                                "gex_abs": gex_abs,
                                "zero_gamma": zero_gamma,
                                "method": method,
                            },
                        )

                        # Insert gex_by_strike
                        for strike, data in per_strike.items():
                            await session.execute(
                                text(
                                    """
                                    INSERT INTO gex_by_strike (snapshot_id, strike, gex_net, gex_calls, gex_puts, oi_total, method)
                                    VALUES (:snapshot_id, :strike, :gex_net, :gex_calls, :gex_puts, :oi_total, :method)
                                    ON CONFLICT (snapshot_id, strike) DO NOTHING
                                    """
                                ),
                                {
                                    "snapshot_id": snapshot_id,
                                    "strike": strike,
                                    "gex_net": data["gex_calls"] + data["gex_puts"],
                                    "gex_calls": data["gex_calls"],
                                    "gex_puts": data["gex_puts"],
                                    "oi_total": data["oi"],
                                    "method": method,
                                },
                            )

                        # Insert gex_by_expiry_strike
                        if settings.gex_store_by_expiry:
                            for (expiration, strike), data in per_expiry_strike.items():
                                dte_days = data.get("dte_days")
                                await session.execute(
                                    text(
                                        """
                                        INSERT INTO gex_by_expiry_strike (
                                          snapshot_id, expiration, dte_days, strike, gex_net, gex_calls, gex_puts, oi_total, method
                                        )
                                        VALUES (
                                          :snapshot_id, :expiration, :dte_days, :strike, :gex_net, :gex_calls, :gex_puts, :oi_total, :method
                                        )
                                        ON CONFLICT (snapshot_id, expiration, strike) DO UPDATE SET
                                          dte_days = EXCLUDED.dte_days,
                                          gex_net = EXCLUDED.gex_net,
                                          gex_calls = EXCLUDED.gex_calls,
                                          gex_puts = EXCLUDED.gex_puts,
                                          oi_total = EXCLUDED.oi_total,
                                          method = EXCLUDED.method
                                        """
                                    ),
                                    {
                                        "snapshot_id": snapshot_id,
                                        "expiration": expiration,
                                        "dte_days": dte_days,
                                        "strike": strike,
                                        "gex_net": data["gex_calls"] + data["gex_puts"],
                                        "gex_calls": data["gex_calls"],
                                        "gex_puts": data["gex_puts"],
                                        "oi_total": data["oi"],
                                        "method": method,
                                    },
                                )

                        # Update context_snapshots with gex fields.
                        await session.execute(
                            text(
                                """
                                INSERT INTO context_snapshots (ts, gex_net, zero_gamma_level, notes_json)
                                VALUES (:ts, :gex_net, :zero_gamma_level, CAST(:notes AS jsonb))
                                ON CONFLICT (ts) DO UPDATE SET
                                  gex_net = EXCLUDED.gex_net,
                                  zero_gamma_level = EXCLUDED.zero_gamma_level,
                                  notes_json = EXCLUDED.notes_json
                                """
                            ),
                            {
                                "ts": ts,
                                "gex_net": gex_net,
                                "zero_gamma_level": zero_gamma,
                                "notes": "{}",
                            },
                        )
                        computed += 1
                except Exception as exc:
                    failed_snapshots.append(
                        {
                            "snapshot_id": int(snapshot_id),
                            "underlying": str(underlying),
                            "error": str(exc),
                        }
                    )
                    logger.exception("gex_job: snapshot_processing_failed snapshot_id={} error={}", snapshot_id, exc)
                    continue

            try:
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.exception(
                    "gex_job: commit_failed computed_snapshots={} failed_snapshots={} error={}",
                    computed,
                    len(failed_snapshots),
                    exc,
                )
                return {
                    "skipped": True,
                    "reason": "db_commit_failed",
                    "computed_snapshots": computed,
                    "skipped_snapshots": skipped_snapshots,
                    "failed_snapshots": failed_snapshots,
                    "now_et": now_et.isoformat(),
                }

        logger.info(
            "gex_job: computed_snapshots={} skipped_snapshots={} failed_snapshots={}",
            computed,
            skipped_snapshots,
            len(failed_snapshots),
        )
        return {
            "skipped": False,
            "computed_snapshots": computed,
            "skipped_snapshots": skipped_snapshots,
            "failed_snapshots": failed_snapshots,
            "now_et": now_et.isoformat(),
        }

    async def _get_spot_price(self, session, ts, underlying: str) -> float | None:
        """Fetch latest spot price at or before ts with staleness guard."""
        # Find the most recent quote <= snapshot time.
        row = await session.execute(
            text(
                """
                SELECT last, ts
                FROM underlying_quotes
                WHERE symbol = :symbol AND ts <= :ts
                ORDER BY ts DESC
                LIMIT 1
                """
            ),
            {"symbol": underlying, "ts": ts},
        )
        result = row.fetchone()
        if not result:
            return None
        last, quote_ts = result
        if last is None:
            return None
        # Enforce staleness guard
        age = (ts - quote_ts).total_seconds()
        if age > settings.gex_spot_max_age_seconds:
            return None
        return float(last)

    def _zero_gamma_level(self, per_strike: dict) -> float | None:
        """Estimate zero gamma level from cumulative net GEX by strike."""
        if not per_strike:
            return None
        strikes = sorted(per_strike.keys())
        cum = 0.0
        prev_cum = 0.0
        prev_strike = None
        for strike in strikes:
            prev_cum = cum
            cum += per_strike[strike]["gex_calls"] + per_strike[strike]["gex_puts"]
            if prev_strike is None:
                prev_strike = strike
                continue
            if prev_cum == 0.0:
                return float(prev_strike)
            if (prev_cum < 0 and cum > 0) or (prev_cum > 0 and cum < 0):
                # Linear interpolation between strikes
                denom = abs(prev_cum) + abs(cum)
                if denom == 0:
                    return float(strike)
                w = abs(prev_cum) / denom
                return float(prev_strike + (strike - prev_strike) * w)
            prev_strike = strike
        return None
