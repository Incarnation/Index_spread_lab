from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.db import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth


def _mid(bid: float | None, ask: float | None) -> float | None:
    """Return mid price when both bid/ask are present."""
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


@dataclass(frozen=True)
class DecisionJob:
    """Rule-based decision runner for credit spreads."""
    clock_cache: MarketClockCache | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        """Check if the market is open using cache or RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    def _is_entry_time(self, now_et: datetime) -> bool:
        """Return True if now matches configured entry times."""
        allowed = settings.decision_entry_times_list()
        return (now_et.hour, now_et.minute) in allowed

    async def run_once(self, *, force: bool = False) -> dict:
        """Run one decision cycle and write TRADE/SKIP."""
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        entry_slot = now_et.hour
        logger.info("decision_job: start force={} now_et={}", force, now_et.isoformat())

        if (not force) and (not self._is_entry_time(now_et)):
            return {"skipped": True, "reason": "not_entry_time", "now_et": now_et.isoformat()}

        if (not force) and (not settings.decision_allow_outside_rth):
            if not await self._market_open(now_et):
                return {"skipped": True, "reason": "market_closed", "now_et": now_et.isoformat()}

        decision_dtes = settings.decision_dte_targets_list()
        delta_targets = settings.decision_delta_targets_list()
        if not decision_dtes or not delta_targets:
            return {"skipped": True, "reason": "missing_targets", "now_et": now_et.isoformat()}

        async with SessionLocal() as session:
            if await self._max_open_trades(session) >= settings.decision_max_open_trades:
                await self._insert_decision(
                    session=session,
                    now_et=now_et,
                    entry_slot=entry_slot,
                    target_dte=decision_dtes[0],
                    delta_target=delta_targets[0],
                    decision="SKIP",
                    reason="max_open_trades",
                )
                await session.commit()
                return {"skipped": True, "reason": "max_open_trades", "now_et": now_et.isoformat()}

            if await self._trades_today(session, now_et) >= settings.decision_max_trades_per_day:
                await self._insert_decision(
                    session=session,
                    now_et=now_et,
                    entry_slot=entry_slot,
                    target_dte=decision_dtes[0],
                    delta_target=delta_targets[0],
                    decision="SKIP",
                    reason="max_trades_per_day",
                )
                await session.commit()
                return {"skipped": True, "reason": "max_trades_per_day", "now_et": now_et.isoformat()}

            spot = await self._get_spot_price(session, now_et, settings.snapshot_underlying)
            context = await self._get_latest_context(session, now_et)

            candidates: list[dict] = []
            for target_dte in decision_dtes:
                snapshot = await self._get_latest_snapshot_for_dte(session, now_et, target_dte, force=force)
                if snapshot is None:
                    continue
                options = await self._get_option_rows(session, snapshot["snapshot_id"])
                if not options:
                    continue
                for delta_target in delta_targets:
                    candidate = self._build_candidate(
                        options=options,
                        target_dte=snapshot["target_dte"],
                        delta_target=delta_target,
                        spread_side=settings.decision_spread_side,
                        width_points=settings.decision_spread_width_points,
                        snapshot_id=snapshot["snapshot_id"],
                        expiration=snapshot["expiration"],
                        spot=spot,
                        context=context,
                    )
                    if candidate:
                        candidates.append(candidate)

            if not candidates:
                await self._insert_decision(
                    session=session,
                    now_et=now_et,
                    entry_slot=entry_slot,
                    target_dte=decision_dtes[0],
                    delta_target=delta_targets[0],
                    decision="SKIP",
                    reason="no_candidates",
                )
                await session.commit()
                return {"skipped": True, "reason": "no_candidates", "now_et": now_et.isoformat()}

            # Pick best candidate by score, then closest delta.
            candidates.sort(key=lambda c: (-c["score"], c["delta_diff"]))
            chosen = candidates[0]

            candidate_ref = await self._find_candidate_reference(session, now_et, chosen)
            decision_id = await self._insert_decision(
                session=session,
                now_et=now_et,
                entry_slot=entry_slot,
                target_dte=chosen["target_dte"],
                delta_target=chosen["delta_target"],
                decision="TRADE",
                reason=None,
                chain_snapshot_id=chosen["snapshot_id"],
                score=chosen["score"],
                chosen_legs_json=chosen["chosen_legs_json"],
                strategy_params_json=chosen["strategy_params_json"],
                candidate_id=candidate_ref.get("candidate_id"),
                feature_snapshot_id=candidate_ref.get("feature_snapshot_id"),
                decision_source="rules",
            )
            trade_id = await self._create_trade_from_decision(
                session=session,
                decision_id=decision_id,
                now_et=now_et,
                chosen=chosen,
                candidate_ref=candidate_ref,
            )
            await session.commit()

        logger.info("decision_job: decision=TRADE target_dte={} delta_target={}", chosen["target_dte"], chosen["delta_target"])
        return {
            "skipped": False,
            "decision": "TRADE",
            "now_et": now_et.isoformat(),
            "decision_id": decision_id,
            "trade_id": trade_id,
            "chosen": chosen,
        }

    async def _get_latest_snapshot_for_dte(
        self,
        session,
        now_et: datetime,
        target_dte: int,
        *,
        force: bool = False,
    ) -> dict | None:
        """Select the most recent snapshot within DTE tolerance and age window."""
        tol = settings.decision_dte_tolerance_days
        min_dte = target_dte - tol
        max_dte = target_dte + tol
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        min_ts = now_utc - timedelta(minutes=settings.decision_snapshot_max_age_minutes)
        freshness_sql = "" if force else "AND ts >= :min_ts"
        row = await session.execute(
            text(
                f"""
                SELECT snapshot_id, ts, expiration, target_dte
                FROM chain_snapshots
                WHERE underlying = :underlying
                  AND ts <= :now_ts
                  {freshness_sql}
                  AND target_dte BETWEEN :min_dte AND :max_dte
                ORDER BY ABS(target_dte - :target_dte) ASC, ts DESC, snapshot_id DESC
                LIMIT 1
                """
            ),
            {
                "underlying": settings.snapshot_underlying,
                "min_dte": min_dte,
                "max_dte": max_dte,
                "target_dte": target_dte,
                "now_ts": now_utc,
                "min_ts": min_ts,
            },
        )
        result = row.fetchone()
        used_fallback = False
        if (not result) and settings.snapshot_range_fallback_enabled:
            # In sandbox/range-fallback mode we may not have near-term DTE snapshots.
            row = await session.execute(
                text(
                    f"""
                    SELECT snapshot_id, ts, expiration, target_dte
                    FROM chain_snapshots
                    WHERE underlying = :underlying
                      AND ts <= :now_ts
                      {freshness_sql}
                    ORDER BY ABS(target_dte - :target_dte) ASC, ts DESC, snapshot_id DESC
                    LIMIT 1
                    """
                ),
                {
                    "underlying": settings.snapshot_underlying,
                    "target_dte": target_dte,
                    "now_ts": now_utc,
                    "min_ts": min_ts,
                },
            )
            result = row.fetchone()
            used_fallback = result is not None
        if not result:
            return None
        snapshot_id, ts, expiration, actual_dte = result
        if used_fallback:
            logger.warning(
                "decision_job: no snapshot in tolerance for target_dte={}; using closest snapshot target_dte={}",
                target_dte,
                actual_dte,
            )
        return {"snapshot_id": snapshot_id, "ts": ts, "expiration": expiration, "target_dte": actual_dte}

    async def _get_option_rows(self, session, snapshot_id: int) -> list[dict]:
        """Load and sanitize option rows for a given snapshot."""
        right = "P" if settings.decision_spread_side.lower() == "put" else "C"
        rows = await session.execute(
            text(
                """
                SELECT option_symbol, strike, bid, ask, delta
                FROM option_chain_rows
                WHERE snapshot_id = :snapshot_id AND option_right = :right AND strike IS NOT NULL
                """
            ),
            {"snapshot_id": snapshot_id, "right": right},
        )
        out: list[dict] = []
        for r in rows.fetchall():
            bid = r.bid
            ask = r.ask
            delta = r.delta
            strike = r.strike
            if bid is None or ask is None or delta is None or strike is None:
                continue
            if ask <= 0 or bid < 0:
                continue
            out.append(
                {
                    "symbol": r.option_symbol,
                    "strike": float(strike),
                    "bid": float(bid),
                    "ask": float(ask),
                    "delta": float(delta),
                }
            )
        return out

    def _build_candidate(
        self,
        *,
        options: list[dict],
        target_dte: int,
        delta_target: float,
        spread_side: str,
        width_points: float,
        snapshot_id: int,
        expiration,
        spot: float | None,
        context: dict | None,
    ) -> dict | None:
        """Build a single vertical spread candidate from option rows."""
        if not options:
            return None

        def delta_diff(opt: dict) -> float:
            return abs(abs(opt["delta"]) - delta_target)

        short = min(options, key=delta_diff)
        desired_long = short["strike"] - width_points if spread_side.lower() == "put" else short["strike"] + width_points
        long_candidates = sorted(
            [o for o in options if o["strike"] != short["strike"]],
            key=lambda o: abs(o["strike"] - desired_long),
        )
        if not long_candidates:
            return None
        long = long_candidates[0]

        short_mid = _mid(short["bid"], short["ask"])
        long_mid = _mid(long["bid"], long["ask"])
        if short_mid is None or long_mid is None:
            return None

        credit = short_mid - long_mid
        if credit <= 0:
            return None

        context_score, context_flags = self._context_score(context, spread_side, spot)
        diff = delta_diff(short)
        score = credit - diff + context_score
        chosen_legs_json = {
            "underlying": settings.snapshot_underlying,
            "expiration": str(expiration),
            "target_dte": target_dte,
            "delta_target": delta_target,
            "spread_side": spread_side.lower(),
            "width_points": width_points,
            "spot": spot,
            "credit": credit,
            "context": context or {},
            "context_score": context_score,
            "context_flags": context_flags,
            "score_components": {"credit": credit, "delta_diff": diff, "context_score": context_score},
            "short": {
                "symbol": short["symbol"],
                "strike": short["strike"],
                "delta": short["delta"],
                "bid": short["bid"],
                "ask": short["ask"],
                "mid": short_mid,
                "qty": settings.decision_contracts,
                "side": "STO" if spread_side.lower() == "put" or spread_side.lower() == "call" else "STO",
            },
            "long": {
                "symbol": long["symbol"],
                "strike": long["strike"],
                "delta": long["delta"],
                "bid": long["bid"],
                "ask": long["ask"],
                "mid": long_mid,
                "qty": settings.decision_contracts,
                "side": "BTO" if spread_side.lower() == "put" or spread_side.lower() == "call" else "BTO",
            },
        }
        strategy_params_json = {
            "spread_side": spread_side.lower(),
            "width_points": width_points,
            "delta_target": delta_target,
            "contracts": settings.decision_contracts,
        }
        return {
            "target_dte": target_dte,
            "delta_target": delta_target,
            "snapshot_id": snapshot_id,
            "expiration": expiration,
            "credit": credit,
            "delta_diff": diff,
            "score": score,
            "context_score": context_score,
            "chosen_legs_json": chosen_legs_json,
            "strategy_params_json": strategy_params_json,
        }

    async def _insert_decision(
        self,
        *,
        session,
        now_et: datetime,
        entry_slot: int,
        target_dte: int,
        delta_target: float,
        decision: str,
        reason: str | None,
        chain_snapshot_id: int | None = None,
        score: float | None = None,
        chosen_legs_json: dict | None = None,
        strategy_params_json: dict | None = None,
        candidate_id: int | None = None,
        feature_snapshot_id: int | None = None,
        decision_source: str = "rules",
    ) -> int:
        """Persist a decision row to trade_decisions."""
        result = await session.execute(
            text(
                """
                INSERT INTO trade_decisions (
                  ts, target_dte, entry_slot, delta_target,
                  chosen_legs_json, strategy_params_json, ruleset_version,
                  score, decision, reason, chain_snapshot_id, feature_snapshot_id, candidate_id, decision_source
                )
                VALUES (
                  :ts, :target_dte, :entry_slot, :delta_target,
                  CAST(:chosen_legs_json AS jsonb), CAST(:strategy_params_json AS jsonb), :ruleset_version,
                  :score, :decision, :reason, :chain_snapshot_id, :feature_snapshot_id, :candidate_id, :decision_source
                )
                RETURNING decision_id
                """
            ),
            {
                "ts": now_et.astimezone(ZoneInfo("UTC")),
                "target_dte": target_dte,
                "entry_slot": entry_slot,
                "delta_target": delta_target,
                "chosen_legs_json": None if chosen_legs_json is None else json_dumps(chosen_legs_json),
                "strategy_params_json": None if strategy_params_json is None else json_dumps(strategy_params_json),
                "ruleset_version": settings.decision_ruleset_version,
                "score": score,
                "decision": decision,
                "reason": reason,
                "chain_snapshot_id": chain_snapshot_id,
                "feature_snapshot_id": feature_snapshot_id,
                "candidate_id": candidate_id,
                "decision_source": decision_source,
            },
        )
        return int(result.scalar_one())

    async def _find_candidate_reference(self, session, now_et: datetime, chosen: dict) -> dict:
        """Best-effort lookup to link trade_decision to trade_candidates rows."""
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        window_start = now_utc - timedelta(minutes=5)
        window_end = now_utc + timedelta(minutes=5)
        chosen_legs = chosen.get("chosen_legs_json") or {}
        short_symbol = ((chosen_legs.get("short") or {}).get("symbol")) if isinstance(chosen_legs, dict) else None
        long_symbol = ((chosen_legs.get("long") or {}).get("symbol")) if isinstance(chosen_legs, dict) else None
        if not short_symbol or not long_symbol:
            return {}
        row = await session.execute(
            text(
                """
                SELECT candidate_id, feature_snapshot_id
                FROM trade_candidates
                WHERE snapshot_id = :snapshot_id
                  AND ts >= :window_start
                  AND ts <= :window_end
                  AND (candidate_json->'legs'->'short'->>'symbol') = :short_symbol
                  AND (candidate_json->'legs'->'long'->>'symbol') = :long_symbol
                ORDER BY ts DESC, candidate_id DESC
                LIMIT 1
                """
            ),
            {
                "snapshot_id": chosen["snapshot_id"],
                "window_start": window_start,
                "window_end": window_end,
                "short_symbol": short_symbol,
                "long_symbol": long_symbol,
            },
        )
        result = row.fetchone()
        if not result:
            return {}
        return {"candidate_id": result.candidate_id, "feature_snapshot_id": result.feature_snapshot_id}

    async def _create_trade_from_decision(
        self,
        *,
        session,
        decision_id: int,
        now_et: datetime,
        chosen: dict,
        candidate_ref: dict,
    ) -> int:
        """Insert one OPEN trade and its legs from the chosen decision candidate."""
        chosen_legs = chosen.get("chosen_legs_json") or {}
        short_leg = chosen_legs.get("short") or {}
        long_leg = chosen_legs.get("long") or {}

        contracts = int(settings.decision_contracts or 1)
        multiplier = int(settings.trade_pnl_contract_multiplier)
        width_points = float(settings.decision_spread_width_points)
        entry_credit = float(chosen.get("credit") or 0.0)
        max_profit = max(entry_credit, 0.0) * contracts * multiplier
        max_loss = max(width_points - entry_credit, 0.0) * contracts * multiplier
        take_profit_target = max_profit * settings.trade_pnl_take_profit_pct
        stop_loss_target = max_profit * settings.trade_pnl_stop_loss_pct

        expiration_raw = chosen.get("expiration")
        expiration: date | None = None
        if isinstance(expiration_raw, date):
            expiration = expiration_raw
        elif isinstance(expiration_raw, str):
            try:
                expiration = date.fromisoformat(expiration_raw)
            except ValueError:
                expiration = None

        option_right = "P" if settings.decision_spread_side.lower() == "put" else "C"
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        trade_insert = await session.execute(
            text(
                """
                INSERT INTO trades (
                  decision_id, candidate_id, feature_snapshot_id, entry_snapshot_id,
                  trade_source, strategy_type, status, underlying, entry_time,
                  target_dte, expiration, contracts, contract_multiplier, spread_width_points,
                  entry_credit, max_profit, max_loss, take_profit_target, stop_loss_target,
                  current_exit_cost, current_pnl, realized_pnl
                )
                VALUES (
                  :decision_id, :candidate_id, :feature_snapshot_id, :entry_snapshot_id,
                  :trade_source, :strategy_type, :status, :underlying, :entry_time,
                  :target_dte, :expiration, :contracts, :contract_multiplier, :spread_width_points,
                  :entry_credit, :max_profit, :max_loss, :take_profit_target, :stop_loss_target,
                  :current_exit_cost, :current_pnl, :realized_pnl
                )
                RETURNING trade_id
                """
            ),
            {
                "decision_id": decision_id,
                "candidate_id": candidate_ref.get("candidate_id"),
                "feature_snapshot_id": candidate_ref.get("feature_snapshot_id"),
                "entry_snapshot_id": chosen.get("snapshot_id"),
                "trade_source": "paper",
                "strategy_type": f"credit_vertical_{settings.decision_spread_side.lower()}",
                "status": "OPEN",
                "underlying": settings.snapshot_underlying,
                "entry_time": now_utc,
                "target_dte": chosen.get("target_dte"),
                "expiration": expiration,
                "contracts": contracts,
                "contract_multiplier": multiplier,
                "spread_width_points": width_points,
                "entry_credit": entry_credit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "take_profit_target": take_profit_target,
                "stop_loss_target": stop_loss_target,
                "current_exit_cost": None,
                "current_pnl": 0.0,
                "realized_pnl": None,
            },
        )
        trade_id = int(trade_insert.scalar_one())

        await session.execute(
            text(
                """
                INSERT INTO trade_legs (
                  trade_id, leg_index, option_symbol, side, qty, entry_price, strike, expiration, option_right
                )
                VALUES (
                  :trade_id, :leg_index, :option_symbol, :side, :qty, :entry_price, :strike, :expiration, :option_right
                )
                """
            ),
            {
                "trade_id": trade_id,
                "leg_index": 1,
                "option_symbol": short_leg.get("symbol"),
                "side": "STO",
                "qty": contracts,
                "entry_price": short_leg.get("mid"),
                "strike": short_leg.get("strike"),
                "expiration": expiration,
                "option_right": option_right,
            },
        )
        await session.execute(
            text(
                """
                INSERT INTO trade_legs (
                  trade_id, leg_index, option_symbol, side, qty, entry_price, strike, expiration, option_right
                )
                VALUES (
                  :trade_id, :leg_index, :option_symbol, :side, :qty, :entry_price, :strike, :expiration, :option_right
                )
                """
            ),
            {
                "trade_id": trade_id,
                "leg_index": 2,
                "option_symbol": long_leg.get("symbol"),
                "side": "BTO",
                "qty": contracts,
                "entry_price": long_leg.get("mid"),
                "strike": long_leg.get("strike"),
                "expiration": expiration,
                "option_right": option_right,
            },
        )
        return trade_id

    async def _max_open_trades(self, session) -> int:
        """Count OPEN trades to enforce risk limits."""
        row = await session.execute(
            text(
                """
                SELECT COUNT(*) AS cnt
                FROM trades
                WHERE status = 'OPEN'
                """
            )
        )
        result = row.fetchone()
        return int(result.cnt if result else 0)

    async def _trades_today(self, session, now_et: datetime) -> int:
        """Count TRADE decisions for the current day."""
        day_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = day_start + timedelta(days=1)
        row = await session.execute(
            text(
                """
                SELECT COUNT(*) AS cnt
                FROM trade_decisions
                WHERE decision = 'TRADE'
                  AND ts >= :start_ts
                  AND ts < :end_ts
                """
            ),
            {
                "start_ts": day_start.astimezone(ZoneInfo("UTC")),
                "end_ts": next_day.astimezone(ZoneInfo("UTC")),
            },
        )
        result = row.fetchone()
        return int(result.cnt if result else 0)

    async def _get_spot_price(self, session, ts: datetime, underlying: str) -> float | None:
        """Fetch latest spot price at or before ts."""
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
            {"symbol": underlying, "ts": ts.astimezone(ZoneInfo("UTC"))},
        )
        result = row.fetchone()
        if not result:
            return None
        last, _ = result
        if last is None:
            return None
        return float(last)

    async def _get_latest_context(self, session, now_et: datetime) -> dict | None:
        """Fetch the latest context snapshot at or before now."""
        row = await session.execute(
            text(
                """
                SELECT ts, spx_price, spy_price, vix, vix9d, term_structure, gex_net, zero_gamma_level
                FROM context_snapshots
                WHERE ts <= :ts
                ORDER BY ts DESC
                LIMIT 1
                """
            ),
            {"ts": now_et.astimezone(ZoneInfo("UTC"))},
        )
        result = row.fetchone()
        if not result:
            return None
        return {
            "ts": result.ts.isoformat(),
            "spx_price": result.spx_price,
            "spy_price": result.spy_price,
            "vix": result.vix,
            "vix9d": result.vix9d,
            "term_structure": result.term_structure,
            "gex_net": result.gex_net,
            "zero_gamma_level": result.zero_gamma_level,
        }

    def _context_score(self, context: dict | None, spread_side: str, spot: float | None) -> tuple[float, list[str]]:
        """Compute a small score adjustment based on GEX and macro context."""
        if not context:
            return 0.0, []
        flags: list[str] = []
        score = 0.0
        gex_net = context.get("gex_net")
        zero_gamma = context.get("zero_gamma_level")
        vix = context.get("vix")
        term_structure = context.get("term_structure")
        side = spread_side.lower()

        if gex_net is not None:
            if side == "put":
                if gex_net >= 0:
                    score += 0.2
                    flags.append("gex_support")
                else:
                    score -= 0.2
                    flags.append("gex_headwind")
            else:
                if gex_net <= 0:
                    score += 0.2
                    flags.append("gex_support")
                else:
                    score -= 0.2
                    flags.append("gex_headwind")

        if spot is not None and zero_gamma is not None:
            if side == "put":
                if spot >= zero_gamma:
                    score += 0.1
                    flags.append("spot_above_zero_gamma")
                else:
                    score -= 0.1
                    flags.append("spot_below_zero_gamma")
            else:
                if spot <= zero_gamma:
                    score += 0.1
                    flags.append("spot_below_zero_gamma")
                else:
                    score -= 0.1
                    flags.append("spot_above_zero_gamma")

        if term_structure is not None and term_structure > 1.05:
            score -= 0.1
            flags.append("term_structure_inverted")

        if vix is not None and vix >= 25:
            score -= 0.1
            flags.append("vix_high")

        return score, flags


def json_dumps(payload: dict) -> str:
    """Serialize payload to JSON for JSONB inserts."""
    import json

    return json.dumps(payload, default=str)


def build_decision_job(clock_cache: MarketClockCache | None = None) -> DecisionJob:
    """Factory helper for DecisionJob."""
    return DecisionJob(clock_cache=clock_cache)
