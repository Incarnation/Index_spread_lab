from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth
from spx_backend.services.portfolio_manager import PortfolioManager as ProdPortfolioManager
from spx_backend.services.event_signals import EventSignalDetector as ProdEventSignalDetector
from spx_backend.utils.pricing import mid_price

if TYPE_CHECKING:
    from spx_backend.services.sms_notifier import SmsNotifier


@dataclass(frozen=True)
class CreatedTrade:
    """Return payload from :meth:`DecisionJob._create_trade_from_decision`.

    Hoisted to a typed object so the portfolio path can pass the
    per-trade ``max_loss`` straight into
    ``PortfolioManager.record_trade(margin_dollars=...)`` without
    re-deriving it.  ``trade_id`` was previously the sole return value
    when there were two execution paths; since the legacy non-portfolio
    branch was deleted in Track A, the only reason to keep this as a
    dataclass (rather than collapsing it back to ``int``) is to keep
    ``max_loss`` / ``spread_side`` / ``entry_credit`` / ``contracts``
    available without an extra DB read.
    """

    trade_id: int
    # Naming intentional: matches ``trades.max_loss`` schema column and
    # the existing ``trade.max_loss`` accessor in ``trade_pnl_job``.
    # Renaming to ``max_loss_dollars`` would create a vocabulary split
    # for no behavioural benefit.
    max_loss: float
    spread_side: str
    entry_credit: float
    contracts: int


@dataclass(frozen=True)
class DecisionJob:
    """Rule-based decision runner for credit spreads."""
    clock_cache: MarketClockCache | None = None
    notifier: SmsNotifier | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        """Check if the market is open using cache or RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    def _is_entry_time(self, now_et: datetime) -> bool:
        """Return True if now matches configured entry times."""
        allowed = settings.decision_entry_times_list()
        return (now_et.hour, now_et.minute) in allowed

    @staticmethod
    def _build_sms_trade_info(chosen: dict, *, contracts: int, source: str,
                              event_signals: Any = None) -> dict[str, Any]:
        """Assemble the dict expected by ``SmsNotifier.notify_trade_opened``.

        Pulls leg details, credit, width, and computed max-loss from the
        candidate data that is already in scope at trade-creation time.
        """
        legs = chosen.get("chosen_legs_json") or {}
        short = legs.get("short") or {}
        long = legs.get("long") or {}
        spread_side = str(legs.get("spread_side") or settings.decision_spread_side).lower()
        width = float(legs.get("width_points") or settings.decision_spread_width_points)
        credit = float(chosen.get("credit") or 0.0)
        multiplier = int(settings.trade_pnl_contract_multiplier)
        max_loss = max(width - credit, 0.0) * contracts * multiplier
        return {
            "spread_side": spread_side,
            "target_dte": chosen.get("target_dte"),
            "expiration": str(chosen.get("expiration", "")),
            "short": {
                "symbol": short.get("symbol", ""),
                "strike": short.get("strike"),
                "entry_price": short.get("mid"),
                "delta": short.get("delta"),
            },
            "long": {
                "symbol": long.get("symbol", ""),
                "strike": long.get("strike"),
                "entry_price": long.get("mid"),
            },
            "credit": credit,
            "width_points": width,
            "contracts": contracts,
            "max_loss": max_loss,
            "source": source,
            "event_signals": event_signals,
        }

    @staticmethod
    async def _is_opex_day(session, today: date) -> bool:
        """Check if *today* is a monthly OPEX day (3rd Friday) by querying economic_events.

        Returns True when a row with ``event_type = 'OPEX'`` exists for the
        given date.  The ``economic_events`` table is populated by
        ``eod_events_job`` using ``generate_economic_calendar.generate_rows()``.
        """
        result = await session.execute(
            text("SELECT 1 FROM economic_events WHERE date = :d AND event_type = 'OPEX'"),
            {"d": today},
        )
        return result.first() is not None

    def _build_run_result(
        self,
        *,
        now_et: datetime,
        skipped: bool,
        reason: str | None,
        decisions_created: list[dict] | None = None,
        trades_created: list[int] | None = None,
        selection_meta: dict | None = None,
    ) -> dict:
        """Build a normalized decision-run payload for UI/API consumers.

        Parameters
        ----------
        now_et:
            Current ET timestamp for consistent response metadata.
        skipped:
            Whether the run produced no trades.
        reason:
            Skip reason when skipped is true; otherwise optional context.
        decisions_created:
            Optional decision/trade summaries persisted during this run.
        trades_created:
            Optional list of newly created trade IDs.
        selection_meta:
            Optional ranking/dedupe/clipping diagnostics.

        Returns
        -------
        dict
            Multi-trade-first payload with counts, IDs, and diagnostics.
        """
        decisions = decisions_created or []
        trades = trades_created or []
        return {
            "skipped": skipped,
            "reason": reason,
            "now_et": now_et.isoformat(),
            "decisions_created_count": len(decisions),
            "trades_created_count": len(trades),
            "decisions_created": decisions,
            "trades_created": trades,
            "selection_meta": selection_meta,
        }

    async def _run_portfolio_managed(self, *, now_et: datetime, force: bool) -> dict:
        """Portfolio-managed decision path using capital budgeting.

        Replaces the legacy ML-gated scoring with credit_to_width ranking,
        enforces portfolio risk limits, and processes event-driven signals.

        Parameters
        ----------
        now_et : Current ET timestamp.
        force : Bypass entry-time and RTH checks.

        Returns
        -------
        dict with the standard run-result payload.
        """
        entry_slot = now_et.hour
        pm = ProdPortfolioManager()
        event_det = ProdEventSignalDetector()

        await pm.begin_day(now_et.date())

        # Hard day-level filter: skip all entries on OPEX days (3rd Friday).
        if settings.decision_avoid_opex:
            async with SessionLocal() as session:
                if await self._is_opex_day(session, now_et.date()):
                    logger.info("decision_job: skipping OPEX day {}", now_et.date())
                    return self._build_run_result(now_et=now_et, skipped=True, reason="opex_day")

        if not await pm.can_trade():
            skip_reason = f"portfolio_{pm.status_label() if hasattr(pm, 'status_label') else 'limit'}"
            dte_list = settings.decision_dte_targets_list()
            delta_list = settings.decision_delta_targets_list()
            async with SessionLocal() as session:
                await self._insert_decision(
                    session=session, now_et=now_et, entry_slot=entry_slot,
                    target_dte=dte_list[0] if dte_list else 0,
                    delta_target=delta_list[0] if delta_list else 0.10,
                    decision="SKIP", reason=skip_reason,
                    decision_source="portfolio_managed",
                )
                await session.commit()
            return self._build_run_result(
                now_et=now_et, skipped=True, reason=skip_reason,
                selection_meta={"equity": pm.equity, "month_stopped": pm.is_month_stopped},
            )

        signals = await event_det.detect(now_et.date())
        skip_scheduled = settings.event_rally_avoidance and "rally" in signals

        sched_sides = ["call"] if settings.portfolio_calls_only else settings.decision_spread_sides_list()
        decision_dtes = settings.decision_dte_targets_list()

        # Mirror the rules-path gate: bail early on misconfigured targets
        # instead of running an empty candidate loop and logging SKIP with
        # ``target_dte=0``.
        has_delta_targets = any(
            settings.decision_delta_targets_for_side(s) for s in sched_sides
        ) if sched_sides else bool(settings.decision_delta_targets_list())
        if not decision_dtes or not has_delta_targets:
            return self._build_run_result(now_et=now_et, skipped=True, reason="missing_targets")

        drop_signals = [s for s in signals if s != "rally"]
        event_side_raw = settings.event_side_preference.rstrip("s")
        # has_drop drives "use event_side_preference for the spread side"
        # vs "use the configured spread sides".  We mirror the backtester
        # here: only outright drop signals (spx_drop*) and VIX signals
        # (vix_spike, vix_elevated) qualify.  term_inversion was previously
        # included in live but excluded by the backtester, causing live
        # event trades on inversion-only days to bias toward the wrong
        # side relative to the historical study.
        has_drop = any(
            s.startswith("spx_drop") or s in ("vix_spike", "vix_elevated")
            for s in drop_signals
        )

        if settings.event_enabled and drop_signals:
            if has_drop:
                event_sides = [event_side_raw]
            else:
                event_sides = list(settings.decision_spread_sides_list())
        else:
            event_sides = []

        all_sides = list(dict.fromkeys(sched_sides + event_sides))

        async with SessionLocal() as session:
            spot = await self._get_spot_price(session, now_et, settings.snapshot_underlying)
            context = await self._get_latest_context(session, now_et)

            candidates: list[dict] = []
            for spread_side in all_sides:
                side_delta_targets = settings.decision_delta_targets_for_side(spread_side)
                for target_dte in decision_dtes:
                    snapshot = await self._get_latest_snapshot_for_dte(session, now_et, target_dte)
                    if snapshot is None:
                        continue
                    options = await self._get_option_rows(session, snapshot["snapshot_id"], spread_side=spread_side)
                    if not options:
                        continue
                    for delta_target in side_delta_targets:
                        candidate = self._build_candidate(
                            options=options, target_dte=snapshot["target_dte"],
                            delta_target=delta_target, spread_side=spread_side,
                            width_points=settings.decision_spread_width_points,
                            snapshot_id=snapshot["snapshot_id"],
                            expiration=snapshot["expiration"],
                            spot=spot, context=context,
                        )
                        if candidate:
                            candidate["spread_side"] = spread_side.lower()
                            candidates.append(candidate)

            if not candidates:
                fallback_delta = settings.decision_delta_targets_list()
                await self._insert_decision(
                    session=session, now_et=now_et, entry_slot=entry_slot,
                    target_dte=decision_dtes[0] if decision_dtes else 0,
                    delta_target=fallback_delta[0] if fallback_delta else 0.10,
                    decision="SKIP", reason="no_candidates",
                    decision_source="portfolio_managed",
                )
                await session.commit()
                return self._build_run_result(now_et=now_et, skipped=True, reason="no_candidates")

            # Rank by credit_to_width using actual strike spacing from each candidate
            for c in candidates:
                legs = c.get("chosen_legs_json") or {}
                w = float(legs.get("width_points") or settings.decision_spread_width_points)
                c["credit_to_width"] = c["credit"] / w if w > 0 else 0

            ranked = sorted(candidates, key=lambda c: -c["credit_to_width"])
            lots = pm.compute_lots()

            # Event trades first on signal days
            event_trades_placed = 0
            decisions_created: list[dict] = []
            trades_created: list[int] = []
            sms_trade_infos: list[dict] = []

            run_limit = settings.portfolio_max_trades_per_run
            # Shared dedupe across event + scheduled loops: a high-rank
            # candidate placed by the event loop must not be placed again
            # as a "scheduled" trade in the same run.  Previously
            # seen_keys was scoped to the scheduled loop, which let the
            # same spread duplicate when both loops fired in one run.
            seen_keys: set[tuple] = set()

            if settings.event_enabled and drop_signals:
                event_cap = min(settings.event_max_trades, run_limit)
                event_candidates = [
                    c for c in ranked
                    if (not has_drop or c.get("spread_side", "").lower() == event_side_raw)
                    and c["target_dte"] >= settings.event_min_dte
                    and c["target_dte"] <= settings.event_max_dte
                    and c["delta_target"] >= settings.event_min_delta
                    and c["delta_target"] <= settings.event_max_delta
                ]
                for c in event_candidates[:event_cap]:
                    if not await pm.can_trade():
                        break
                    key = self._candidate_dedupe_key(c)
                    if key in seen_keys:
                        continue
                    decision_id = await self._insert_decision(
                        session=session, now_et=now_et, entry_slot=entry_slot,
                        target_dte=c["target_dte"], delta_target=c["delta_target"],
                        decision="TRADE", reason=None,
                        chain_snapshot_id=c["snapshot_id"], score=c["credit_to_width"],
                        chosen_legs_json=c["chosen_legs_json"],
                        strategy_params_json=c["strategy_params_json"],
                        decision_source="portfolio_event",
                    )
                    created = await self._create_trade_from_decision(
                        session=session, decision_id=decision_id,
                        now_et=now_et, chosen=c,
                        contracts_override=lots,
                    )
                    await pm.record_trade(
                        created.trade_id, 0.0, lots,
                        source="event",
                        event_signal=",".join(drop_signals),
                        session=session,
                        margin_dollars=created.max_loss,
                    )
                    decisions_created.append({
                        "decision_id": int(decision_id),
                        "trade_id": int(created.trade_id),
                        "target_dte": c["target_dte"], "delta_target": c["delta_target"],
                        "spread_side": (c.get("chosen_legs_json") or {}).get("spread_side", ""),
                        "score": c["credit_to_width"], "decision_source": "portfolio_event",
                    })
                    trades_created.append(int(created.trade_id))
                    sms_trade_infos.append(self._build_sms_trade_info(
                        c, contracts=lots, source="portfolio_event",
                        event_signals=drop_signals,
                    ))
                    event_trades_placed += 1
                    # Mark this candidate seen so the scheduled loop below
                    # does not pick the same legs again.
                    seen_keys.add(key)

            # Scheduled trades (capped by per-run stagger limit)
            if not skip_scheduled and not settings.event_only_mode:
                sched_limit = min(run_limit, settings.portfolio_max_trades_per_day)
                if settings.event_enabled and settings.event_budget_mode == "shared":
                    sched_limit = max(0, sched_limit - event_trades_placed)
                sched_ranked = [c for c in ranked if c.get("spread_side", "").lower() in sched_sides]
                sched_candidates = sched_ranked[:sched_limit * 2]
                placed = 0
                for c in sched_candidates:
                    if placed >= sched_limit:
                        break
                    if not await pm.can_trade():
                        break
                    key = self._candidate_dedupe_key(c)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    decision_id = await self._insert_decision(
                        session=session, now_et=now_et, entry_slot=entry_slot,
                        target_dte=c["target_dte"], delta_target=c["delta_target"],
                        decision="TRADE", reason=None,
                        chain_snapshot_id=c["snapshot_id"], score=c["credit_to_width"],
                        chosen_legs_json=c["chosen_legs_json"],
                        strategy_params_json=c["strategy_params_json"],
                        decision_source="portfolio_scheduled",
                    )
                    created = await self._create_trade_from_decision(
                        session=session, decision_id=decision_id,
                        now_et=now_et, chosen=c,
                        contracts_override=lots,
                    )
                    await pm.record_trade(
                        created.trade_id, 0.0, lots, source="scheduled",
                        session=session,
                        margin_dollars=created.max_loss,
                    )
                    decisions_created.append({
                        "decision_id": int(decision_id),
                        "trade_id": int(created.trade_id),
                        "target_dte": c["target_dte"], "delta_target": c["delta_target"],
                        "spread_side": (c.get("chosen_legs_json") or {}).get("spread_side", ""),
                        "score": c["credit_to_width"], "decision_source": "portfolio_scheduled",
                    })
                    trades_created.append(int(created.trade_id))
                    sms_trade_infos.append(self._build_sms_trade_info(
                        c, contracts=lots, source="portfolio_scheduled",
                    ))
                    placed += 1

            if not trades_created:
                fallback_delta = settings.decision_delta_targets_list()
                skip_decision_id = await self._insert_decision(
                    session=session, now_et=now_et, entry_slot=entry_slot,
                    target_dte=decision_dtes[0] if decision_dtes else 0,
                    delta_target=fallback_delta[0] if fallback_delta else 0.10,
                    decision="SKIP", reason="no_trades_after_filters",
                    decision_source="portfolio_managed",
                    strategy_params_json={
                        "candidates_total": len(candidates),
                        "event_signals": signals,
                    },
                )
                decisions_created.append({
                    "decision_id": int(skip_decision_id),
                    "decision": "SKIP",
                    "reason": "no_trades_after_filters",
                    "decision_source": "portfolio_managed",
                })

            await session.commit()

        if self.notifier and sms_trade_infos:
            for ti in sms_trade_infos:
                await self.notifier.notify_trade_opened(ti)

        return self._build_run_result(
            now_et=now_et,
            skipped=len(trades_created) == 0,
            reason=None if trades_created else "no_trades_after_filters",
            decisions_created=decisions_created,
            trades_created=trades_created,
            selection_meta={
                "mode": "portfolio_managed",
                "equity": pm.equity,
                "lots": lots,
                "event_signals": signals,
                "event_trades": event_trades_placed,
                "candidates_total": len(candidates),
            },
        )

    async def run_once(self, *, force: bool = False) -> dict:
        """Run one decision cycle and persist top-N trades.

        Parameters
        ----------
        force:
            When true, bypasses entry-time and regular-hours checks.

        Returns
        -------
        dict
            Multi-trade run payload with created decision/trade IDs, skip
            reason (when applicable), and selection diagnostics.

        Notes
        -----
        Live trading is portfolio-managed by ``credit_to_width`` ranking
        (see :meth:`_run_portfolio_managed`).  The legacy hybrid ML/rules
        path was removed when the online ML pipeline was decommissioned;
        future ML re-entry plugs into ``_run_portfolio_managed`` rather
        than restoring a parallel branch.
        """
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        logger.info("decision_job: start force={} now_et={}", force, now_et.isoformat())

        if (not force) and (not self._is_entry_time(now_et)):
            return self._build_run_result(now_et=now_et, skipped=True, reason="not_entry_time")

        if (not force) and (not settings.decision_allow_outside_rth):
            if not await self._market_open(now_et):
                return self._build_run_result(now_et=now_et, skipped=True, reason="market_closed")

        return await self._run_portfolio_managed(now_et=now_et, force=force)

    async def _get_latest_snapshot_for_dte(
        self,
        session,
        now_et: datetime,
        target_dte: int,
    ) -> dict | None:
        """Select the most recent snapshot within DTE tolerance and age window.

        ``force`` mode in ``run_once`` bypasses entry-time and RTH guards
        but MUST NOT bypass snapshot freshness: trading on a 2-day-old
        chain because an admin manually triggered a run is exactly the
        kind of mistake we want to fail closed.  The freshness clause is
        therefore always applied here.
        """
        tol = settings.decision_dte_tolerance_days
        min_dte = target_dte - tol
        max_dte = target_dte + tol
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        min_ts = now_utc - timedelta(minutes=settings.decision_snapshot_max_age_minutes)
        # Always-on freshness guard.  Previously this was conditionally
        # disabled when the caller passed force=True, which silently
        # allowed admin-triggered runs to use stale chain data.
        freshness_sql = "AND ts >= :min_ts"
        row = await session.execute(
            text(
                f"""
                SELECT snapshot_id, ts, expiration, target_dte
                FROM chain_snapshots
                WHERE underlying = :underlying
                  AND source = 'TRADIER'
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
                      AND source = 'TRADIER'
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

    async def _get_option_rows(self, session, snapshot_id: int, *, spread_side: str) -> list[dict]:
        """Load and sanitize option rows for a given snapshot."""
        side = spread_side.lower().strip()
        if side not in {"put", "call"}:
            return []
        right = "P" if side == "put" else "C"
        rows = await session.execute(
            text(
                """
                SELECT option_symbol, strike, bid, ask, delta, mid_iv
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
            entry: dict = {
                "symbol": r.option_symbol,
                "strike": float(strike),
                "bid": float(bid),
                "ask": float(ask),
                "delta": float(delta),
            }
            if r.mid_iv is not None:
                entry["iv"] = float(r.mid_iv)
            out.append(entry)
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
            """Measure distance between option absolute delta and target delta."""
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

        # Vertical integrity: put spreads must have short above long;
        # call spreads must have short below long.
        side_lower = spread_side.lower()
        if side_lower == "put" and long["strike"] >= short["strike"]:
            return None
        if side_lower == "call" and long["strike"] <= short["strike"]:
            return None

        short_mid = mid_price(short["bid"], short["ask"])
        long_mid = mid_price(long["bid"], long["ask"])
        if short_mid is None or long_mid is None:
            return None

        credit = short_mid - long_mid
        if credit <= 0:
            return None

        actual_width = abs(short["strike"] - long["strike"])

        width_deviation = abs(actual_width - width_points)
        width_deviation_flag = width_deviation > 1.0
        if width_deviation_flag:
            logger.warning(
                "width_deviation: requested={} actual={} deviation={:.1f} side={} short={} long={}",
                width_points, actual_width, width_deviation, spread_side,
                short["strike"], long["strike"],
            )

        context_score, context_flags = self._context_score(context, spread_side, spot)
        diff = delta_diff(short)
        score = credit - diff + context_score
        chosen_legs_json = {
            "underlying": settings.snapshot_underlying,
            "expiration": str(expiration),
            "target_dte": target_dte,
            "delta_target": delta_target,
            "spread_side": side_lower,
            "width_points": actual_width,
            "requested_width_points": width_points,
            "width_deviation_flag": width_deviation_flag,
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
                "iv": short.get("iv"),
                "bid": short["bid"],
                "ask": short["ask"],
                "mid": short_mid,
                "qty": settings.decision_contracts,
                "side": "STO",
            },
            "long": {
                "symbol": long["symbol"],
                "strike": long["strike"],
                "delta": long["delta"],
                "iv": long.get("iv"),
                "bid": long["bid"],
                "ask": long["ask"],
                "mid": long_mid,
                "qty": settings.decision_contracts,
                "side": "BTO",
            },
        }
        strategy_params_json = {
            "spread_side": side_lower,
            "width_points": actual_width,
            "requested_width_points": width_points,
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

    def _candidate_dedupe_key(self, candidate: dict) -> tuple[str, str, str, str]:
        """Return a stable identity key used to dedupe repeated spread legs.

        The key captures spread side, expiration, short symbol, and long symbol
        so one exact leg pair is only executed once per run.
        """
        chosen_legs = candidate.get("chosen_legs_json") or {}
        short_symbol = str(((chosen_legs.get("short") or {}).get("symbol")) or "")
        long_symbol = str(((chosen_legs.get("long") or {}).get("symbol")) or "")
        spread_side = str(chosen_legs.get("spread_side") or "")
        expiration = str(candidate.get("expiration") or chosen_legs.get("expiration") or "")
        return (spread_side, expiration, short_symbol, long_symbol)

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
        decision_source: str = "portfolio_managed",
    ) -> int:
        """Persist a decision row to ``trade_decisions``.

        After the online ML pipeline was decommissioned (Track A) the only
        live producer of decision rows is :meth:`_run_portfolio_managed`.
        Migration ``015_decommission_online_ml_schema.sql`` then dropped
        the always-NULL legacy columns (``model_score``, ``expected_value``,
        ``prediction_id``, ``feature_snapshot_id``, ``candidate_id``,
        ``policy_version``, ``risk_gate_json``, ``experiment_tag``).
        ``model_version_id`` is preserved on the schema for offline ML
        re-entry but is not populated by the rules-based path.

        The legacy ``decision_source="rules"`` default has been removed in
        favour of ``"portfolio_managed"`` so an accidental call with no
        ``decision_source`` annotates correctly in audit logs.
        """
        result = await session.execute(
            text(
                """
                INSERT INTO trade_decisions (
                  ts, target_dte, entry_slot, delta_target,
                  chosen_legs_json, strategy_params_json, ruleset_version,
                  score, decision, reason, chain_snapshot_id, decision_source
                )
                VALUES (
                  :ts, :target_dte, :entry_slot, :delta_target,
                  CAST(:chosen_legs_json AS jsonb), CAST(:strategy_params_json AS jsonb), :ruleset_version,
                  :score, :decision, :reason, :chain_snapshot_id, :decision_source
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
                "decision_source": decision_source,
            },
        )
        return int(result.scalar_one())

    async def _create_trade_from_decision(
        self,
        *,
        session,
        decision_id: int,
        now_et: datetime,
        chosen: dict,
        contracts_override: int | None = None,
    ) -> CreatedTrade:
        """Insert one OPEN trade and its legs from the chosen decision candidate.

        Returns a :class:`CreatedTrade` so the portfolio path can pass the
        per-trade ``max_loss`` straight into ``record_trade(margin_dollars=...)``
        instead of re-deriving it.

        Parameters
        ----------
        contracts_override : When set (e.g. from ``PortfolioManager.compute_lots()``),
            overrides the static ``settings.decision_contracts`` for lot sizing.

        Notes
        -----
        ``trades.candidate_id`` and ``trades.feature_snapshot_id`` were
        dropped by migration 015 (Track A.7) along with their referenced
        tables.  ``trades.model_version_id`` is preserved for offline ML
        re-entry but is not populated by this rules-based path.
        """
        chosen_legs = chosen.get("chosen_legs_json") or {}
        short_leg = chosen_legs.get("short") or {}
        long_leg = chosen_legs.get("long") or {}
        spread_side = str(chosen_legs.get("spread_side") or settings.decision_spread_side).lower()
        if spread_side not in {"put", "call"}:
            spread_side = settings.decision_spread_side.lower()

        contracts = contracts_override if contracts_override is not None else int(settings.decision_contracts or 1)
        multiplier = int(settings.trade_pnl_contract_multiplier)
        width_points = float(chosen_legs.get("width_points") or settings.decision_spread_width_points)
        entry_credit = float(chosen.get("credit") or 0.0)
        max_profit = max(entry_credit, 0.0) * contracts * multiplier
        max_loss = max(width_points - entry_credit, 0.0) * contracts * multiplier
        # max_loss == 0 means the spread sells for full width (no risk),
        # which only happens with stale/dead quotes that slipped past
        # mid_price's strict gate.  Refuse to insert such a trade rather
        # than silently book a zero-margin position that the portfolio
        # manager (and downstream PnL math) will treat as risk-free.
        if max_loss <= 0.0:
            raise RuntimeError(
                f"_create_trade_from_decision: refusing trade with non-positive "
                f"max_loss={max_loss} (width={width_points}, credit={entry_credit}, "
                f"contracts={contracts}, multiplier={multiplier})"
            )
        take_profit_target = max_profit * settings.trade_pnl_take_profit_pct
        sl_basis_value = (
            max_loss if settings.trade_pnl_stop_loss_basis == "max_loss"
            else max_profit
        )
        stop_loss_target = (
            sl_basis_value * settings.trade_pnl_stop_loss_pct
            if settings.trade_pnl_stop_loss_enabled
            else None
        )

        expiration_raw = chosen.get("expiration")
        expiration: date | None = None
        if isinstance(expiration_raw, date):
            expiration = expiration_raw
        elif isinstance(expiration_raw, str):
            try:
                expiration = date.fromisoformat(expiration_raw)
            except ValueError:
                expiration = None

        option_right = "P" if spread_side == "put" else "C"
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        trade_insert = await session.execute(
            text(
                """
                INSERT INTO trades (
                  decision_id, entry_snapshot_id,
                  trade_source, strategy_type, status, underlying, entry_time,
                  target_dte, expiration, contracts, contract_multiplier, spread_width_points,
                  entry_credit, max_profit, max_loss, take_profit_target, stop_loss_target,
                  current_exit_cost, current_pnl, realized_pnl
                )
                VALUES (
                  :decision_id, :entry_snapshot_id,
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
                "entry_snapshot_id": chosen.get("snapshot_id"),
                "trade_source": "paper",
                "strategy_type": f"credit_vertical_{spread_side}",
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
        return CreatedTrade(
            trade_id=trade_id,
            max_loss=float(max_loss),
            spread_side=spread_side,
            entry_credit=float(entry_credit),
            contracts=int(contracts),
        )

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
        """Fetch the latest context snapshot at or before now for SPX."""
        row = await session.execute(
            text(
                """
                SELECT ts, spx_price, spy_price, vix, vix9d, term_structure,
                       vvix, skew, gex_net, zero_gamma_level,
                       gex_net_tradier, zero_gamma_level_tradier,
                       gex_net_cboe, zero_gamma_level_cboe
                FROM context_snapshots
                WHERE ts <= :ts AND underlying = :underlying
                ORDER BY ts DESC
                LIMIT 1
                """
            ),
            {"ts": now_et.astimezone(ZoneInfo("UTC")), "underlying": "SPX"},
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
            "vvix": result.vvix,
            "skew": result.skew,
            "gex_net": result.gex_net,
            "zero_gamma_level": result.zero_gamma_level,
            "gex_net_tradier": result.gex_net_tradier,
            "zero_gamma_level_tradier": result.zero_gamma_level_tradier,
            "gex_net_cboe": result.gex_net_cboe,
            "zero_gamma_level_cboe": result.zero_gamma_level_cboe,
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


def build_decision_job(
    clock_cache: MarketClockCache | None = None,
    notifier: SmsNotifier | None = None,
) -> DecisionJob:
    """Factory helper for DecisionJob.

    Parameters
    ----------
    clock_cache:
        Shared market-clock cache for RTH checks.
    notifier:
        Optional SMS notifier; when provided, trade-opened messages are sent
        after the DB commit for each placed trade.
    """
    return DecisionJob(clock_cache=clock_cache, notifier=notifier)
