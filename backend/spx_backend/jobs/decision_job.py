from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth
from spx_backend.services.portfolio_manager import PortfolioManager as ProdPortfolioManager
from spx_backend.services.event_signals import EventSignalDetector as ProdEventSignalDetector


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

        if not pm.can_trade():
            return self._build_run_result(
                now_et=now_et, skipped=True,
                reason=f"portfolio_{pm.status_label() if hasattr(pm, 'status_label') else 'limit'}",
                selection_meta={"equity": pm.equity, "month_stopped": pm.is_month_stopped},
            )

        signals = await event_det.detect(now_et.date())
        skip_scheduled = settings.event_rally_avoidance and "rally" in signals

        spread_sides = ["call"] if settings.portfolio_calls_only else settings.decision_spread_sides_list()
        decision_dtes = settings.decision_dte_targets_list()
        delta_targets = settings.decision_delta_targets_list()

        async with SessionLocal() as session:
            spot = await self._get_spot_price(session, now_et, settings.snapshot_underlying)
            context = await self._get_latest_context(session, now_et)

            candidates: list[dict] = []
            for spread_side in spread_sides:
                for target_dte in decision_dtes:
                    snapshot = await self._get_latest_snapshot_for_dte(session, now_et, target_dte, force=force)
                    if snapshot is None:
                        continue
                    options = await self._get_option_rows(session, snapshot["snapshot_id"], spread_side=spread_side)
                    if not options:
                        continue
                    for delta_target in delta_targets:
                        candidate = self._build_candidate(
                            options=options, target_dte=snapshot["target_dte"],
                            delta_target=delta_target, spread_side=spread_side,
                            width_points=settings.decision_spread_width_points,
                            snapshot_id=snapshot["snapshot_id"],
                            expiration=snapshot["expiration"],
                            spot=spot, context=context,
                        )
                        if candidate:
                            candidates.append(candidate)

            if not candidates:
                return self._build_run_result(now_et=now_et, skipped=True, reason="no_candidates")

            # Rank by credit_to_width (credit / spread width)
            width = settings.decision_spread_width_points
            for c in candidates:
                c["credit_to_width"] = c["credit"] / width if width > 0 else 0

            ranked = sorted(candidates, key=lambda c: -c["credit_to_width"])
            lots = pm.compute_lots()

            # Event trades first on signal days
            event_trades_placed = 0
            drop_signals = [s for s in signals if s != "rally"]
            decisions_created: list[dict] = []
            trades_created: list[int] = []

            run_limit = settings.portfolio_max_trades_per_run

            if settings.event_enabled and drop_signals:
                event_cap = min(settings.event_max_trades, run_limit)
                event_candidates = [
                    c for c in ranked
                    if c["target_dte"] >= settings.event_min_dte
                    and c["target_dte"] <= settings.event_max_dte
                    and abs(c.get("delta_diff", 0)) <= settings.event_max_delta
                ]
                for c in event_candidates[:event_cap]:
                    if not pm.can_trade():
                        break
                    decision_id = await self._insert_decision(
                        session=session, now_et=now_et, entry_slot=entry_slot,
                        target_dte=c["target_dte"], delta_target=c["delta_target"],
                        decision="TRADE", reason=None,
                        chain_snapshot_id=c["snapshot_id"], score=c["credit_to_width"],
                        chosen_legs_json=c["chosen_legs_json"],
                        strategy_params_json=c["strategy_params_json"],
                        decision_source="portfolio_event",
                    )
                    trade_id = await self._create_trade_from_decision(
                        session=session, decision_id=decision_id,
                        now_et=now_et, chosen=c, candidate_ref={},
                        contracts_override=lots,
                    )
                    await pm.record_trade(trade_id, c["credit"], lots,
                                          source="event",
                                          event_signal=",".join(drop_signals))
                    decisions_created.append({
                        "decision_id": int(decision_id), "trade_id": int(trade_id),
                        "target_dte": c["target_dte"], "delta_target": c["delta_target"],
                        "spread_side": (c.get("chosen_legs_json") or {}).get("spread_side", ""),
                        "score": c["credit_to_width"], "decision_source": "portfolio_event",
                    })
                    trades_created.append(int(trade_id))
                    event_trades_placed += 1

            # Scheduled trades (capped by per-run stagger limit)
            if not skip_scheduled:
                sched_limit = min(run_limit, settings.portfolio_max_trades_per_day)
                if settings.event_enabled and settings.event_budget_mode == "shared":
                    sched_limit = max(0, sched_limit - event_trades_placed)
                sched_candidates = ranked[:sched_limit * 2]  # over-fetch for dedup
                seen_keys: set[tuple] = set()
                placed = 0
                for c in sched_candidates:
                    if placed >= sched_limit or not pm.can_trade():
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
                    trade_id = await self._create_trade_from_decision(
                        session=session, decision_id=decision_id,
                        now_et=now_et, chosen=c, candidate_ref={},
                        contracts_override=lots,
                    )
                    await pm.record_trade(trade_id, c["credit"], lots, source="scheduled")
                    decisions_created.append({
                        "decision_id": int(decision_id), "trade_id": int(trade_id),
                        "target_dte": c["target_dte"], "delta_target": c["delta_target"],
                        "spread_side": (c.get("chosen_legs_json") or {}).get("spread_side", ""),
                        "score": c["credit_to_width"], "decision_source": "portfolio_scheduled",
                    })
                    trades_created.append(int(trade_id))
                    placed += 1

            await session.commit()

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
        """
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        entry_slot = now_et.hour
        logger.info("decision_job: start force={} now_et={}", force, now_et.isoformat())

        if (not force) and (not self._is_entry_time(now_et)):
            return self._build_run_result(now_et=now_et, skipped=True, reason="not_entry_time")

        if (not force) and (not settings.decision_allow_outside_rth):
            if not await self._market_open(now_et):
                return self._build_run_result(now_et=now_et, skipped=True, reason="market_closed")

        if settings.portfolio_enabled:
            return await self._run_portfolio_managed(now_et=now_et, force=force)

        decision_dtes = settings.decision_dte_targets_list()
        delta_targets = settings.decision_delta_targets_list()
        spread_sides = settings.decision_spread_sides_list()
        if not decision_dtes or not delta_targets:
            return self._build_run_result(now_et=now_et, skipped=True, reason="missing_targets")
        if not spread_sides:
            return self._build_run_result(now_et=now_et, skipped=True, reason="missing_spread_sides")

        async with SessionLocal() as session:
            open_trades_now = await self._max_open_trades(session)
            if open_trades_now >= settings.decision_max_open_trades:
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
                return self._build_run_result(
                    now_et=now_et,
                    skipped=True,
                    reason="max_open_trades",
                    selection_meta={
                        "max_trades_per_run": max(1, int(settings.decision_max_trades_per_run)),
                        "day_remaining_before": None,
                        "open_remaining_before": 0,
                    },
                )

            trades_today_now = await self._trades_today(session, now_et)
            if trades_today_now >= settings.decision_max_trades_per_day:
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
                return self._build_run_result(
                    now_et=now_et,
                    skipped=True,
                    reason="max_trades_per_day",
                    selection_meta={
                        "max_trades_per_run": max(1, int(settings.decision_max_trades_per_run)),
                        "day_remaining_before": 0,
                        "open_remaining_before": max(0, settings.decision_max_open_trades - open_trades_now),
                    },
                )

            side_limits: dict[str, str] = {}
            eligible_spread_sides: list[str] = []
            for spread_side in spread_sides:
                if settings.decision_max_open_trades_per_side > 0:
                    open_for_side = await self._max_open_trades_by_side(session, spread_side)
                    if open_for_side >= settings.decision_max_open_trades_per_side:
                        side_limits[spread_side] = "max_open_trades_per_side"
                        continue
                if settings.decision_max_trades_per_side_per_day > 0:
                    trades_for_side = await self._trades_today_by_side(session, now_et, spread_side)
                    if trades_for_side >= settings.decision_max_trades_per_side_per_day:
                        side_limits[spread_side] = "max_trades_per_side_per_day"
                        continue
                eligible_spread_sides.append(spread_side)

            if not eligible_spread_sides:
                await self._insert_decision(
                    session=session,
                    now_et=now_et,
                    entry_slot=entry_slot,
                    target_dte=decision_dtes[0],
                    delta_target=delta_targets[0],
                    decision="SKIP",
                    reason="side_limits_reached",
                    strategy_params_json={
                        "requested_spread_sides": spread_sides,
                        "side_limits": side_limits,
                    },
                )
                await session.commit()
                return self._build_run_result(
                    now_et=now_et,
                    skipped=True,
                    reason="side_limits_reached",
                    selection_meta={
                        "requested_spread_sides": spread_sides,
                        "side_limits": side_limits,
                    },
                )

            spot = await self._get_spot_price(session, now_et, settings.snapshot_underlying)
            context = await self._get_latest_context(session, now_et)

            candidates: list[dict] = []
            for spread_side in eligible_spread_sides:
                for target_dte in decision_dtes:
                    snapshot = await self._get_latest_snapshot_for_dte(session, now_et, target_dte, force=force)
                    if snapshot is None:
                        continue
                    options = await self._get_option_rows(session, snapshot["snapshot_id"], spread_side=spread_side)
                    if not options:
                        continue
                    for delta_target in delta_targets:
                        candidate = self._build_candidate(
                            options=options,
                            target_dte=snapshot["target_dte"],
                            delta_target=delta_target,
                            spread_side=spread_side,
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
                    strategy_params_json={
                        "spread_sides_considered": eligible_spread_sides,
                        "side_limits": side_limits,
                    },
                )
                await session.commit()
                return self._build_run_result(
                    now_et=now_et,
                    skipped=True,
                    reason="no_candidates",
                    selection_meta={
                        "candidates_total": 0,
                        "candidates_ranked": 0,
                        "candidates_after_dedupe": 0,
                        "duplicates_removed": 0,
                        "max_trades_per_run": max(1, int(settings.decision_max_trades_per_run)),
                        "day_remaining_before": max(0, settings.decision_max_trades_per_day - trades_today_now),
                        "open_remaining_before": max(0, settings.decision_max_open_trades - open_trades_now),
                        "selected_count": 0,
                        "clipped_by": "no_candidates",
                    },
                )

            ranked = await self._rank_candidates_with_policy(session=session, now_et=now_et, candidates=candidates)
            ranked_deduped, duplicates_removed = self._dedupe_ranked_selections(ranked)
            if not ranked_deduped:
                await self._insert_decision(
                    session=session,
                    now_et=now_et,
                    entry_slot=entry_slot,
                    target_dte=decision_dtes[0],
                    delta_target=delta_targets[0],
                    decision="SKIP",
                    reason="no_candidates",
                    strategy_params_json={
                        "spread_sides_considered": eligible_spread_sides,
                        "side_limits": side_limits,
                    },
                )
                await session.commit()
                return self._build_run_result(
                    now_et=now_et,
                    skipped=True,
                    reason="no_candidates",
                    selection_meta={
                        "candidates_total": len(candidates),
                        "candidates_ranked": len(ranked),
                        "candidates_after_dedupe": 0,
                        "duplicates_removed": duplicates_removed,
                        "max_trades_per_run": max(1, int(settings.decision_max_trades_per_run)),
                        "day_remaining_before": max(0, settings.decision_max_trades_per_day - trades_today_now),
                        "open_remaining_before": max(0, settings.decision_max_open_trades - open_trades_now),
                        "selected_count": 0,
                        "clipped_by": "all_duplicates",
                    },
                )

            max_per_run = max(1, int(settings.decision_max_trades_per_run))
            day_remaining_before = max(0, settings.decision_max_trades_per_day - trades_today_now)
            open_remaining_before = max(0, settings.decision_max_open_trades - open_trades_now)
            capacity_limit = min(max_per_run, day_remaining_before, open_remaining_before)

            if capacity_limit <= 0:
                clipped_by = "day_cap" if day_remaining_before <= 0 else "open_cap"
                await self._insert_decision(
                    session=session,
                    now_et=now_et,
                    entry_slot=entry_slot,
                    target_dte=decision_dtes[0],
                    delta_target=delta_targets[0],
                    decision="SKIP",
                    reason="capacity_reached",
                    strategy_params_json={
                        "day_remaining_before": day_remaining_before,
                        "open_remaining_before": open_remaining_before,
                        "max_trades_per_run": max_per_run,
                    },
                )
                await session.commit()
                return self._build_run_result(
                    now_et=now_et,
                    skipped=True,
                    reason="capacity_reached",
                    selection_meta={
                        "candidates_total": len(candidates),
                        "candidates_ranked": len(ranked),
                        "candidates_after_dedupe": len(ranked_deduped),
                        "duplicates_removed": duplicates_removed,
                        "max_trades_per_run": max_per_run,
                        "day_remaining_before": day_remaining_before,
                        "open_remaining_before": open_remaining_before,
                        "selected_count": 0,
                        "clipped_by": clipped_by,
                    },
                )

            selected: list[dict[str, Any]] = []
            selected_by_side: dict[str, int] = {}
            open_by_side: dict[str, int] = {}
            trades_today_by_side: dict[str, int] = {}
            skipped_by_side_limits: dict[str, str] = {}

            # Walk ranked candidates in order and keep selecting until the
            # clipped capacity limit is reached.
            for selection in ranked_deduped:
                if len(selected) >= capacity_limit:
                    break
                chosen = selection["chosen"]
                legs = chosen.get("chosen_legs_json") or {}
                spread_side = str(legs.get("spread_side") or "").lower()
                if spread_side not in {"put", "call"}:
                    spread_side = settings.decision_spread_side.lower()

                if settings.decision_max_open_trades_per_side > 0:
                    if spread_side not in open_by_side:
                        open_by_side[spread_side] = await self._max_open_trades_by_side(session, spread_side)
                    side_open_after = open_by_side[spread_side] + selected_by_side.get(spread_side, 0)
                    if side_open_after >= settings.decision_max_open_trades_per_side:
                        skipped_by_side_limits[spread_side] = "max_open_trades_per_side"
                        continue

                if settings.decision_max_trades_per_side_per_day > 0:
                    if spread_side not in trades_today_by_side:
                        trades_today_by_side[spread_side] = await self._trades_today_by_side(session, now_et, spread_side)
                    side_day_after = trades_today_by_side[spread_side] + selected_by_side.get(spread_side, 0)
                    if side_day_after >= settings.decision_max_trades_per_side_per_day:
                        skipped_by_side_limits[spread_side] = "max_trades_per_side_per_day"
                        continue

                selected.append(selection)
                selected_by_side[spread_side] = selected_by_side.get(spread_side, 0) + 1

            if not selected:
                await self._insert_decision(
                    session=session,
                    now_et=now_et,
                    entry_slot=entry_slot,
                    target_dte=decision_dtes[0],
                    delta_target=delta_targets[0],
                    decision="SKIP",
                    reason="side_limits_reached",
                    strategy_params_json={
                        "requested_spread_sides": spread_sides,
                        "side_limits": side_limits | skipped_by_side_limits,
                    },
                )
                await session.commit()
                return self._build_run_result(
                    now_et=now_et,
                    skipped=True,
                    reason="side_limits_reached",
                    selection_meta={
                        "candidates_total": len(candidates),
                        "candidates_ranked": len(ranked),
                        "candidates_after_dedupe": len(ranked_deduped),
                        "duplicates_removed": duplicates_removed,
                        "max_trades_per_run": max_per_run,
                        "day_remaining_before": day_remaining_before,
                        "open_remaining_before": open_remaining_before,
                        "selected_count": 0,
                        "clipped_by": "side_cap",
                    },
                )

            decisions_created: list[dict] = []
            trades_created: list[int] = []
            for selection in selected:
                chosen = selection["chosen"]
                candidate_ref = selection.get("candidate_ref") or {}
                model_prediction = selection.get("model_prediction")
                decision_source = str(selection.get("decision_source") or "rules")
                model_score = model_prediction.get("score_raw") if isinstance(model_prediction, dict) else None
                expected_value = model_prediction.get("expected_value") if isinstance(model_prediction, dict) else None
                prediction_id = model_prediction.get("prediction_id") if isinstance(model_prediction, dict) else None
                model_version_id = model_prediction.get("model_version_id") if isinstance(model_prediction, dict) else None

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
                    decision_source=decision_source,
                    model_score=(float(model_score) if model_score is not None else None),
                    expected_value=(float(expected_value) if expected_value is not None else None),
                    prediction_id=(int(prediction_id) if prediction_id is not None else None),
                    model_version_id=(int(model_version_id) if model_version_id is not None else None),
                )
                trade_id = await self._create_trade_from_decision(
                    session=session,
                    decision_id=decision_id,
                    now_et=now_et,
                    chosen=chosen,
                    candidate_ref=candidate_ref,
                    model_version_id=(int(model_version_id) if model_version_id is not None else None),
                )
                spread_side = str((chosen.get("chosen_legs_json") or {}).get("spread_side") or "").lower()
                decisions_created.append(
                    {
                        "decision_id": int(decision_id),
                        "trade_id": int(trade_id),
                        "target_dte": int(chosen["target_dte"]),
                        "delta_target": float(chosen["delta_target"]),
                        "spread_side": spread_side if spread_side in {"put", "call"} else settings.decision_spread_side.lower(),
                        "score": float(chosen["score"]) if chosen.get("score") is not None else None,
                        "decision_source": decision_source,
                    }
                )
                trades_created.append(int(trade_id))
            await session.commit()

        clipped_by = None
        if len(selected) < max_per_run:
            if len(selected) == open_remaining_before:
                clipped_by = "open_cap"
            elif len(selected) == day_remaining_before:
                clipped_by = "day_cap"
            elif len(selected) < capacity_limit:
                clipped_by = "side_cap"
            else:
                clipped_by = "insufficient_candidates"

        selection_meta = {
            "candidates_total": len(candidates),
            "candidates_ranked": len(ranked),
            "candidates_after_dedupe": len(ranked_deduped),
            "duplicates_removed": duplicates_removed,
            "max_trades_per_run": max_per_run,
            "day_remaining_before": day_remaining_before,
            "open_remaining_before": open_remaining_before,
            "selected_count": len(selected),
            "clipped_by": clipped_by,
        }
        logger.info(
            "decision_job: created_trades={} ranked={} deduped={} clipped_by={}",
            len(trades_created),
            len(ranked),
            len(ranked_deduped),
            clipped_by,
        )
        return self._build_run_result(
            now_et=now_et,
            skipped=False,
            reason=None,
            decisions_created=decisions_created,
            trades_created=trades_created,
            selection_meta=selection_meta,
        )

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

    async def _get_hybrid_model_version(self, session) -> dict[str, Any] | None:
        """Load model version eligible for hybrid selection."""
        if settings.decision_hybrid_require_active_model:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT model_version_id, model_name, version, rollout_status, is_active
                        FROM model_versions
                        WHERE model_name = :model_name
                          AND is_active = true
                          AND rollout_status = 'active'
                        ORDER BY created_at DESC, model_version_id DESC
                        LIMIT 1
                        """
                    ),
                    {"model_name": settings.decision_hybrid_model_name},
                )
            ).fetchone()
        else:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT model_version_id, model_name, version, rollout_status, is_active
                        FROM model_versions
                        WHERE model_name = :model_name
                          AND rollout_status IN ('canary', 'active')
                        ORDER BY created_at DESC, model_version_id DESC
                        LIMIT 1
                        """
                    ),
                    {"model_name": settings.decision_hybrid_model_name},
                )
            ).fetchone()
        if row is None:
            return None
        return {
            "model_version_id": int(row.model_version_id),
            "model_name": str(row.model_name),
            "version": str(row.version),
            "rollout_status": str(row.rollout_status),
            "is_active": bool(row.is_active),
        }

    async def _get_candidate_prediction(self, session, *, candidate_id: int, model_version_id: int) -> dict[str, Any] | None:
        """Load latest shadow/hybrid prediction for one candidate."""
        row = (
            await session.execute(
                text(
                    """
                    SELECT prediction_id, score_raw, probability_win, expected_value, rank_in_snapshot
                    FROM model_predictions
                    WHERE candidate_id = :candidate_id
                      AND model_version_id = :model_version_id
                    ORDER BY created_at DESC, prediction_id DESC
                    LIMIT 1
                    """
                ),
                {"candidate_id": candidate_id, "model_version_id": model_version_id},
            )
        ).fetchone()
        if row is None:
            return None
        return {
            "prediction_id": int(row.prediction_id),
            "score_raw": float(row.score_raw),
            "probability_win": (float(row.probability_win) if row.probability_win is not None else None),
            "expected_value": (float(row.expected_value) if row.expected_value is not None else None),
            "rank_in_snapshot": (int(row.rank_in_snapshot) if row.rank_in_snapshot is not None else None),
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

    def _dedupe_ranked_selections(self, ranked: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Drop duplicate leg-pair selections while preserving ranking order."""
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[dict[str, Any]] = []
        duplicates_removed = 0
        for selection in ranked:
            key = self._candidate_dedupe_key(selection["chosen"])
            if key in seen:
                duplicates_removed += 1
                continue
            seen.add(key)
            deduped.append(selection)
        return deduped, duplicates_removed

    async def _rank_candidates_with_policy(self, *, session, now_et: datetime, candidates: list[dict]) -> list[dict[str, Any]]:
        """Rank candidates using rules first, then hybrid-model overrides.

        Behavior
        --------
        - Rules ranking: score desc, delta_diff asc.
        - Hybrid ranking (when enabled): model score desc, then rules score.
        - Fallback: append non-model-qualified rules candidates after hybrid
          picks so runs can still fill top-N when model predictions are sparse.
        """
        candidates_sorted = sorted(candidates, key=lambda c: (-c["score"], c["delta_diff"]))
        rules_ranked: list[dict[str, Any]] = []
        for candidate in candidates_sorted:
            candidate_ref = await self._find_candidate_reference(session, now_et, candidate)
            rules_ranked.append(
                {
                    "chosen": candidate,
                    "candidate_ref": candidate_ref,
                    "decision_source": "rules",
                    "model_prediction": None,
                }
            )
        if not settings.decision_hybrid_enabled:
            return rules_ranked

        model_version = await self._get_hybrid_model_version(session)
        if model_version is None:
            return rules_ranked

        ranked_by_model: list[tuple[float, float, dict[str, Any]]] = []
        for rules_entry in rules_ranked:
            candidate_ref = rules_entry.get("candidate_ref") or {}
            candidate_id = candidate_ref.get("candidate_id")
            if candidate_id is None:
                continue
            pred = await self._get_candidate_prediction(
                session,
                candidate_id=int(candidate_id),
                model_version_id=int(model_version["model_version_id"]),
            )
            if pred is None:
                continue

            prob = pred.get("probability_win")
            ev = pred.get("expected_value")
            if prob is None or ev is None:
                continue
            if float(prob) < settings.decision_hybrid_min_probability:
                continue
            if float(ev) < settings.decision_hybrid_min_expected_pnl:
                continue
            ranked_by_model.append(
                (
                    float(pred["score_raw"]),
                    float(rules_entry["chosen"]["score"]),
                    {
                        **rules_entry,
                        "decision_source": "hybrid_model",
                        "model_prediction": {
                            **pred,
                            "model_version_id": int(model_version["model_version_id"]),
                            "model_name": model_version["model_name"],
                            "model_version": model_version["version"],
                        },
                    },
                )
            )

        if not ranked_by_model:
            return rules_ranked

        ranked_by_model.sort(key=lambda item: (-item[0], -item[1]))
        hybrid_ranked = [item[2] for item in ranked_by_model]

        # Keep hybrid-qualified picks first, then append remaining rules-ranked
        # candidates so top-N can still fill when hybrid coverage is incomplete.
        hybrid_keys = {self._candidate_dedupe_key(entry["chosen"]) for entry in hybrid_ranked}
        for rules_entry in rules_ranked:
            key = self._candidate_dedupe_key(rules_entry["chosen"])
            if key in hybrid_keys:
                continue
            hybrid_ranked.append(rules_entry)
        return hybrid_ranked

    async def _select_candidate_with_policy(self, *, session, now_et: datetime, candidates: list[dict]) -> dict[str, Any]:
        """Select a single best candidate (compatibility helper for tests).

        The production run path uses `_rank_candidates_with_policy` for top-N
        execution. This helper preserves legacy behavior by returning the first
        ranked candidate.
        """
        ranked = await self._rank_candidates_with_policy(session=session, now_et=now_et, candidates=candidates)
        if not ranked:
            chosen_rules = sorted(candidates, key=lambda c: (-c["score"], c["delta_diff"]))[0]
            rules_ref = await self._find_candidate_reference(session, now_et, chosen_rules)
            return {
                "chosen": chosen_rules,
                "candidate_ref": rules_ref,
                "decision_source": "rules",
                "model_prediction": None,
            }
        return ranked[0]

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
        model_score: float | None = None,
        expected_value: float | None = None,
        prediction_id: int | None = None,
        model_version_id: int | None = None,
        decision_source: str = "rules",
    ) -> int:
        """Persist a decision row to trade_decisions."""
        result = await session.execute(
            text(
                """
                INSERT INTO trade_decisions (
                  ts, target_dte, entry_slot, delta_target,
                  chosen_legs_json, strategy_params_json, ruleset_version,
                  score, model_score, expected_value,
                  decision, reason, chain_snapshot_id, feature_snapshot_id, candidate_id, prediction_id, model_version_id, decision_source
                )
                VALUES (
                  :ts, :target_dte, :entry_slot, :delta_target,
                  CAST(:chosen_legs_json AS jsonb), CAST(:strategy_params_json AS jsonb), :ruleset_version,
                  :score, :model_score, :expected_value,
                  :decision, :reason, :chain_snapshot_id, :feature_snapshot_id, :candidate_id, :prediction_id, :model_version_id, :decision_source
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
                "model_score": model_score,
                "expected_value": expected_value,
                "decision": decision,
                "reason": reason,
                "chain_snapshot_id": chain_snapshot_id,
                "feature_snapshot_id": feature_snapshot_id,
                "candidate_id": candidate_id,
                "prediction_id": prediction_id,
                "model_version_id": model_version_id,
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
        model_version_id: int | None = None,
        contracts_override: int | None = None,
    ) -> int:
        """Insert one OPEN trade and its legs from the chosen decision candidate.

        Parameters
        ----------
        contracts_override : When set (e.g. from PortfolioManager.compute_lots()),
            overrides the static ``settings.decision_contracts`` for lot sizing.
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
        take_profit_target = max_profit * settings.trade_pnl_take_profit_pct
        stop_loss_target = (
            max_profit * settings.trade_pnl_stop_loss_pct
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
                  decision_id, candidate_id, feature_snapshot_id, entry_snapshot_id,
                  trade_source, strategy_type, status, underlying, model_version_id, entry_time,
                  target_dte, expiration, contracts, contract_multiplier, spread_width_points,
                  entry_credit, max_profit, max_loss, take_profit_target, stop_loss_target,
                  current_exit_cost, current_pnl, realized_pnl
                )
                VALUES (
                  :decision_id, :candidate_id, :feature_snapshot_id, :entry_snapshot_id,
                  :trade_source, :strategy_type, :status, :underlying, :model_version_id, :entry_time,
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
                "strategy_type": f"credit_vertical_{spread_side}",
                "status": "OPEN",
                "underlying": settings.snapshot_underlying,
                "model_version_id": model_version_id,
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

    async def _max_open_trades_by_side(self, session, spread_side: str) -> int:
        """Count OPEN trades for one spread side (put/call)."""
        side = spread_side.lower().strip()
        if side not in {"put", "call"}:
            return 0
        row = await session.execute(
            text(
                """
                SELECT COUNT(*) AS cnt
                FROM trades
                WHERE status = 'OPEN'
                  AND strategy_type LIKE :strategy_type_pattern
                """
            ),
            {"strategy_type_pattern": f"%_{side}"},
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

    async def _trades_today_by_side(self, session, now_et: datetime, spread_side: str) -> int:
        """Count trades entered today for one spread side (put/call)."""
        side = spread_side.lower().strip()
        if side not in {"put", "call"}:
            return 0
        day_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = day_start + timedelta(days=1)
        row = await session.execute(
            text(
                """
                SELECT COUNT(*) AS cnt
                FROM trades
                WHERE entry_time >= :start_ts
                  AND entry_time < :end_ts
                  AND strategy_type LIKE :strategy_type_pattern
                """
            ),
            {
                "start_ts": day_start.astimezone(ZoneInfo("UTC")),
                "end_ts": next_day.astimezone(ZoneInfo("UTC")),
                "strategy_type_pattern": f"%_{side}",
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
                SELECT ts, spx_price, spy_price, vix, vix9d, term_structure,
                       vvix, skew, gex_net, zero_gamma_level
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
            "vvix": result.vvix,
            "skew": result.skew,
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
