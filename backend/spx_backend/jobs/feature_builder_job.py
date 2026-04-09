from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import statistics
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.jobs.decision_job import DecisionJob
from spx_backend.market_clock import MarketClockCache, is_rth


def stable_json_hash(payload: dict) -> str:
    """Return stable SHA-256 hash for deterministic JSON payloads."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_candidate_hash_payload(candidate_json: dict) -> dict:
    """Build canonical fields used for candidate hashing."""
    legs = candidate_json.get("legs") or {}
    short_leg = legs.get("short") or {}
    long_leg = legs.get("long") or {}
    return {
        "underlying": candidate_json.get("underlying"),
        "snapshot_id": candidate_json.get("snapshot_id"),
        "expiration": candidate_json.get("expiration"),
        "target_dte": candidate_json.get("target_dte"),
        "delta_target": candidate_json.get("delta_target"),
        "spread_side": candidate_json.get("spread_side"),
        "width_points": candidate_json.get("width_points"),
        "contracts": candidate_json.get("contracts"),
        "short_symbol": short_leg.get("symbol"),
        "short_strike": short_leg.get("strike"),
        "long_symbol": long_leg.get("symbol"),
        "long_strike": long_leg.get("strike"),
    }


def build_candidate_hash(candidate_json: dict) -> str:
    """Return deterministic candidate hash from canonical candidate fields."""
    return stable_json_hash(build_candidate_hash_payload(candidate_json))


def _to_json(value: dict | None) -> str | None:
    """Serialize dict payloads for JSONB SQL inserts."""
    if value is None:
        return None
    return json.dumps(value, default=str)


def _iso_to_dt(value: str | None) -> datetime | None:
    """Parse ISO datetime when available."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _to_float(value: object) -> float | None:
    """Convert an arbitrary object to float when possible."""
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def classify_vix_regime(vix: float | None) -> str:
    """Classify VIX level into low/normal/high regimes for model features."""
    if vix is None:
        return "unknown"
    if vix < 15.0:
        return "low"
    if vix <= 25.0:
        return "normal"
    return "high"


def classify_term_structure_regime(term_structure: float | None) -> str:
    """Classify VIX term-structure into contango/flat/backwardation regimes."""
    if term_structure is None:
        return "unknown"
    if term_structure < 0.95:
        return "contango"
    if term_structure <= 1.05:
        return "flat"
    return "backwardation"


def classify_spy_spx_ratio_regime(spy_spx_ratio: float | None) -> str:
    """Classify SPY/SPX ratio into discount/parity/premium regimes."""
    if spy_spx_ratio is None:
        return "unknown"
    if spy_spx_ratio < 0.099:
        return "discount"
    if spy_spx_ratio <= 0.101:
        return "parity"
    return "premium"


@dataclass(frozen=True)
class FeatureBuilderJob:
    """Generate feature snapshots and trade candidates at decision times."""

    clock_cache: MarketClockCache | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        """Check market state using cached clock, fallback to simple RTH."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    def _is_entry_time(self, now_et: datetime) -> bool:
        """Return True if current ET hour/minute is configured for entries."""
        return (now_et.hour, now_et.minute) in settings.decision_entry_times_list()

    async def _nearest_cboe_batch_ts(
        self,
        *,
        session,
        target_ts: datetime,
        underlying: str,
    ) -> datetime | None:
        """Return nearest CBOE GEX batch timestamp for the given underlying.

        Parameters
        ----------
        session:
            Open SQLAlchemy async session.
        target_ts:
            Timestamp we want to align with (typically the Tradier snapshot ts).
        underlying:
            Underlying symbol (for example ``SPX``).

        Returns
        -------
        datetime | None
            UTC timestamp of the nearest CBOE batch when available.
        """
        target_utc = target_ts if target_ts.tzinfo is not None else target_ts.replace(tzinfo=ZoneInfo("UTC"))
        row = await session.execute(
            text(
                """
                SELECT ts
                FROM gex_snapshots
                WHERE underlying = :underlying
                  AND source = 'CBOE'
                ORDER BY ABS(EXTRACT(EPOCH FROM (ts - :target_ts))) ASC, snapshot_id DESC
                LIMIT 1
                """
            ),
            {"underlying": underlying, "target_ts": target_utc.astimezone(ZoneInfo("UTC"))},
        )
        result = row.fetchone()
        if result is None:
            return None
        return result.ts

    async def _build_cboe_expiration_context(
        self,
        *,
        session,
        target_ts: datetime,
        underlying: str,
        expiration: date,
        spot: float | None,
    ) -> dict | None:
        """Build CBOE-derived expiry context for one candidate expiration.

        Parameters
        ----------
        session:
            Open SQLAlchemy async session.
        target_ts:
            Timestamp used to find the nearest CBOE batch.
        underlying:
            Underlying symbol (for example ``SPX``).
        expiration:
            Candidate expiration date.
        spot:
            Current spot used for wall-distance normalization.

        Returns
        -------
        dict | None
            Expiration-level CBOE context with wall levels and distance ratios.
        """
        batch_ts = await self._nearest_cboe_batch_ts(session=session, target_ts=target_ts, underlying=underlying)
        if batch_ts is None:
            return None
        rows = (
            await session.execute(
                text(
                    """
                    SELECT gbes.strike, gbes.gex_net, gbes.gex_calls, gbes.gex_puts
                    FROM gex_by_expiry_strike gbes
                    JOIN gex_snapshots gs ON gs.snapshot_id = gbes.snapshot_id
                    WHERE gs.ts = :batch_ts
                      AND gs.underlying = :underlying
                      AND gs.source = 'CBOE'
                      AND gbes.expiration = :expiration
                    ORDER BY gbes.strike ASC
                    """
                ),
                {
                    "batch_ts": batch_ts,
                    "underlying": underlying,
                    "expiration": expiration,
                },
            )
        ).fetchall()
        if not rows:
            return None

        expiry_gex_net = 0.0
        expiry_gex_calls = 0.0
        expiry_gex_puts = 0.0
        gamma_wall_strike: float | None = None
        gamma_wall_abs = -1.0
        call_wall_strike: float | None = None
        call_wall_value = float("-inf")
        put_wall_strike: float | None = None
        put_wall_value = float("inf")
        for row in rows:
            strike = _to_float(row.strike)
            gex_net = _to_float(row.gex_net) or 0.0
            gex_calls = _to_float(row.gex_calls) or 0.0
            gex_puts = _to_float(row.gex_puts) or 0.0
            expiry_gex_net += gex_net
            expiry_gex_calls += gex_calls
            expiry_gex_puts += gex_puts
            if strike is None:
                continue
            if abs(gex_net) > gamma_wall_abs:
                gamma_wall_abs = abs(gex_net)
                gamma_wall_strike = strike
            if gex_calls > call_wall_value:
                call_wall_value = gex_calls
                call_wall_strike = strike
            if gex_puts < put_wall_value:
                put_wall_value = gex_puts
                put_wall_strike = strike

        distance_ratio_denominator = float(spot) if spot is not None and float(spot) > 0 else None
        gamma_wall_distance_ratio = (
            None
            if distance_ratio_denominator is None or gamma_wall_strike is None
            else abs(float(spot) - gamma_wall_strike) / distance_ratio_denominator
        )
        call_wall_distance_ratio = (
            None
            if distance_ratio_denominator is None or call_wall_strike is None
            else abs(float(spot) - call_wall_strike) / distance_ratio_denominator
        )
        put_wall_distance_ratio = (
            None
            if distance_ratio_denominator is None or put_wall_strike is None
            else abs(float(spot) - put_wall_strike) / distance_ratio_denominator
        )

        return {
            "source": "CBOE",
            "batch_ts_utc": batch_ts.astimezone(ZoneInfo("UTC")).isoformat(),
            "expiration": expiration.isoformat(),
            "expiry_gex_net": expiry_gex_net,
            "expiry_gex_calls": expiry_gex_calls,
            "expiry_gex_puts": expiry_gex_puts,
            "gamma_wall_strike": gamma_wall_strike,
            "call_wall_strike": call_wall_strike,
            "put_wall_strike": put_wall_strike,
            "gamma_wall_distance_ratio": gamma_wall_distance_ratio,
            "call_wall_distance_ratio": call_wall_distance_ratio,
            "put_wall_distance_ratio": put_wall_distance_ratio,
        }

    def _build_feature_payload(
        self,
        *,
        now_et: datetime,
        snapshot: dict,
        options: list[dict],
        spot: float | None,
        context: dict | None,
        spread_side: str,
        cboe_context: dict | None,
    ) -> dict:
        """Build compact, reproducible market-state feature payload."""
        spreads = [float(o["ask"]) - float(o["bid"]) for o in options]
        mids = [(float(o["ask"]) + float(o["bid"])) / 2.0 for o in options]
        deltas = [abs(float(o["delta"])) for o in options]
        strikes = [float(o["strike"]) for o in options]

        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        snapshot_ts = snapshot["ts"]
        context_ts = _iso_to_dt(context.get("ts")) if context else None

        snapshot_age_sec = (now_utc - snapshot_ts).total_seconds() if snapshot_ts else None
        context_age_sec = (now_utc - context_ts).total_seconds() if context_ts else None
        vix = _to_float(None if context is None else context.get("vix"))
        vix9d = _to_float(None if context is None else context.get("vix9d"))
        term_structure = _to_float(None if context is None else context.get("term_structure"))
        vvix = _to_float(None if context is None else context.get("vvix"))
        skew = _to_float(None if context is None else context.get("skew"))
        spy_price = _to_float(None if context is None else context.get("spy_price"))
        spx_price = _to_float(None if context is None else context.get("spx_price"))
        if spx_price is None:
            spx_price = _to_float(spot)
        spy_spx_ratio = (spy_price / spx_price) if spy_price is not None and spx_price is not None and spx_price > 0 else None

        return {
            "schema_version": settings.feature_schema_version,
            "asof_ts_utc": now_utc.isoformat(),
            "underlying": settings.snapshot_underlying,
            "target_dte": snapshot["target_dte"],
            "entry_slot": now_et.hour,
            "snapshot_id": snapshot["snapshot_id"],
            "expiration": str(snapshot["expiration"]),
            "spot": {"spx_last": spot},
            "vol_context": {
                "spx_price": spx_price,
                "spy_price": spy_price,
                "spy_spx_ratio": spy_spx_ratio,
                "spy_spx_ratio_regime": classify_spy_spx_ratio_regime(spy_spx_ratio),
                "vix": vix,
                "vix9d": vix9d,
                "term_structure": term_structure,
                "vvix": vvix,
                "skew": skew,
                "vix_regime": classify_vix_regime(vix),
                "term_structure_regime": classify_term_structure_regime(term_structure),
            },
            "gex_context": {
                "gex_net": None if context is None else context.get("gex_net"),
                "zero_gamma_level": None if context is None else context.get("zero_gamma_level"),
                "gex_net_tradier": None if context is None else context.get("gex_net_tradier"),
                "zero_gamma_level_tradier": None if context is None else context.get("zero_gamma_level_tradier"),
                "gex_net_cboe": None if context is None else context.get("gex_net_cboe"),
                "zero_gamma_level_cboe": None if context is None else context.get("zero_gamma_level_cboe"),
            },
            "cboe_context": cboe_context,
            "chain_quality": {
                "rows_total": len(options),
                "strike_min": min(strikes) if strikes else None,
                "strike_max": max(strikes) if strikes else None,
                "median_spread": statistics.median(spreads) if spreads else None,
                "median_mid": statistics.median(mids) if mids else None,
                "median_abs_delta": statistics.median(deltas) if deltas else None,
            },
            "staleness": {
                "snapshot_age_sec": snapshot_age_sec,
                "context_age_sec": context_age_sec,
            },
            "decision_config": {
                "spread_side": spread_side.lower(),
                "spread_sides_enabled": settings.decision_spread_sides_list(),
                "width_points": settings.decision_spread_width_points,
                "contracts": settings.decision_contracts,
                "delta_targets": settings.decision_delta_targets_for_side(spread_side),
            },
        }

    def _build_candidate_json(self, candidate: dict, *, cboe_context: dict | None) -> dict:
        """Normalize candidate payload to a stable JSON document."""
        chosen = candidate["chosen_legs_json"]
        short_leg = chosen.get("short") or {}
        long_leg = chosen.get("long") or {}
        context = chosen.get("context") if isinstance(chosen.get("context"), dict) else {}
        vix = _to_float(context.get("vix")) if isinstance(context, dict) else None
        term_structure = _to_float(context.get("term_structure")) if isinstance(context, dict) else None
        spy_price = _to_float(context.get("spy_price")) if isinstance(context, dict) else None
        spx_price = _to_float(context.get("spx_price")) if isinstance(context, dict) else None
        if spx_price is None:
            spx_price = _to_float(chosen.get("spot"))
        spy_spx_ratio = (spy_price / spx_price) if spy_price is not None and spx_price is not None and spx_price > 0 else None
        spread_side = str(chosen.get("spread_side") or settings.decision_spread_side).lower()
        width_points = float(chosen.get("width_points") or settings.decision_spread_width_points)
        contracts = int(short_leg.get("qty") or settings.decision_contracts)
        vix9d = _to_float(context.get("vix9d")) if isinstance(context, dict) else None
        vvix = _to_float(context.get("vvix")) if isinstance(context, dict) else None
        skew = _to_float(context.get("skew")) if isinstance(context, dict) else None

        cal_flags = candidate.get("_calendar_flags") or {}

        entry_credit = candidate["credit"]
        credit_to_width = (entry_credit / width_points) if width_points > 0 else None

        return {
            "schema_version": settings.candidate_schema_version,
            "underlying": settings.snapshot_underlying,
            "snapshot_id": candidate["snapshot_id"],
            "expiration": str(candidate["expiration"]),
            "target_dte": candidate["target_dte"],
            "delta_target": candidate["delta_target"],
            "spread_side": spread_side,
            "width_points": width_points,
            "contracts": contracts,
            "entry_credit": entry_credit,
            "credit_to_width": credit_to_width,
            "score": candidate["score"],
            "delta_diff": candidate["delta_diff"],
            "context_score": candidate.get("context_score"),
            "spot": spx_price,
            "spy_price": spy_price,
            "spx_price": spx_price,
            "vix": vix,
            "vix9d": vix9d,
            "term_structure": term_structure,
            "vvix": vvix,
            "skew": skew,
            "is_opex_day": cal_flags.get("is_opex", False),
            "is_fomc_day": cal_flags.get("is_fomc", False),
            "is_triple_witching": cal_flags.get("is_triple_witching", False),
            "is_cpi_day": cal_flags.get("is_cpi", False),
            "is_nfp_day": cal_flags.get("is_nfp", False),
            "spy_spx_ratio": spy_spx_ratio,
            "spy_spx_ratio_regime": classify_spy_spx_ratio_regime(spy_spx_ratio),
            "vix_regime": classify_vix_regime(vix),
            "term_structure_regime": classify_term_structure_regime(term_structure),
            "legs": {"short": short_leg, "long": long_leg},
            "context": context,
            "context_flags": chosen.get("context_flags"),
            "cboe_context": cboe_context,
        }

    async def _get_calendar_flags(self, session, today: date) -> dict:
        """Query economic_events for today's calendar flags.

        Returns a dict with boolean keys ``is_opex``, ``is_fomc``,
        ``is_triple_witching``, ``is_cpi``, and ``is_nfp``.  Defaults to
        all-False when no events are found for the given date.
        """
        rows = await session.execute(
            text(
                "SELECT event_type, is_triple_witching "
                "FROM economic_events WHERE date = :d"
            ),
            {"d": today},
        )
        flags = {
            "is_opex": False, "is_fomc": False, "is_triple_witching": False,
            "is_cpi": False, "is_nfp": False,
        }
        for r in rows.fetchall():
            evt = str(r.event_type).upper()
            if evt == "OPEX":
                flags["is_opex"] = True
            elif evt == "FOMC":
                flags["is_fomc"] = True
            elif evt == "CPI":
                flags["is_cpi"] = True
            elif evt == "NFP":
                flags["is_nfp"] = True
            if r.is_triple_witching:
                flags["is_triple_witching"] = True
        return flags

    async def run_once(self, *, force: bool = False) -> dict:
        """Generate features and candidates for each target DTE."""
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        entry_slot = now_et.hour
        logger.info("feature_builder_job: start force={} now_et={}", force, now_et.isoformat())

        if not settings.feature_builder_enabled:
            return {"skipped": True, "reason": "feature_builder_disabled", "now_et": now_et.isoformat()}

        if (not force) and (not self._is_entry_time(now_et)):
            return {"skipped": True, "reason": "not_entry_time", "now_et": now_et.isoformat()}

        if (not force) and (not settings.feature_builder_allow_outside_rth):
            if not await self._market_open(now_et):
                return {"skipped": True, "reason": "market_closed", "now_et": now_et.isoformat()}

        target_dtes = settings.decision_dte_targets_list()
        spread_sides = settings.decision_spread_sides_list()
        has_delta_targets = any(
            settings.decision_delta_targets_for_side(s) for s in spread_sides
        ) if spread_sides else bool(settings.decision_delta_targets_list())
        if not target_dtes or not has_delta_targets:
            return {"skipped": True, "reason": "missing_targets", "now_et": now_et.isoformat()}
        if not spread_sides:
            return {"skipped": True, "reason": "missing_spread_sides", "now_et": now_et.isoformat()}

        helper = DecisionJob(clock_cache=self.clock_cache)
        features_inserted = 0
        candidates_inserted = 0
        built: list[dict] = []
        cboe_context_cache: dict[tuple[int, str], dict | None] = {}

        async with SessionLocal() as session:
            spot = await helper._get_spot_price(session, now_et, settings.snapshot_underlying)
            context = await helper._get_latest_context(session, now_et)
            context_ts = _iso_to_dt(context.get("ts")) if context else None

            # Economic calendar flags for today
            calendar_flags = await self._get_calendar_flags(session, now_et.date())

            for spread_side in spread_sides:
                for target_dte in target_dtes:
                    snapshot = await helper._get_latest_snapshot_for_dte(session, now_et, target_dte, force=force)
                    if snapshot is None:
                        continue
                    options = await helper._get_option_rows(session, snapshot["snapshot_id"], spread_side=spread_side)
                    if not options:
                        continue

                    cache_key = (int(snapshot["snapshot_id"]), str(snapshot["expiration"]))
                    if cache_key not in cboe_context_cache:
                        # Cache by snapshot/expiration to avoid re-querying the same
                        # CBOE batch context for put/call loops in one run.
                        expiration_value = snapshot["expiration"]
                        if isinstance(expiration_value, date):
                            cboe_context_cache[cache_key] = await self._build_cboe_expiration_context(
                                session=session,
                                target_ts=snapshot["ts"],
                                underlying=settings.snapshot_underlying,
                                expiration=expiration_value,
                                spot=spot,
                            )
                        else:
                            cboe_context_cache[cache_key] = None
                    cboe_context = cboe_context_cache[cache_key]

                    feature_payload = self._build_feature_payload(
                        now_et=now_et,
                        snapshot=snapshot,
                        options=options,
                        spot=spot,
                        context=context,
                        spread_side=spread_side,
                        cboe_context=cboe_context,
                    )
                    feature_hash = stable_json_hash(feature_payload)
                    feature_insert = await session.execute(
                        text(
                            """
                            INSERT INTO feature_snapshots (
                              ts, snapshot_id, context_ts, underlying, target_dte, entry_slot,
                              strategy_version_id, data_source, feature_schema_version, feature_hash,
                              source_job, source_run_id, features_json, label_status, label_horizon
                            )
                            VALUES (
                              :ts, :snapshot_id, :context_ts, :underlying, :target_dte, :entry_slot,
                              :strategy_version_id, :data_source, :feature_schema_version, :feature_hash,
                              :source_job, :source_run_id, CAST(:features_json AS jsonb), :label_status, :label_horizon
                            )
                            RETURNING feature_snapshot_id
                            """
                        ),
                        {
                            "ts": now_et.astimezone(ZoneInfo("UTC")),
                            "snapshot_id": snapshot["snapshot_id"],
                            "context_ts": context_ts,
                            "underlying": settings.snapshot_underlying,
                            "target_dte": snapshot["target_dte"],
                            "entry_slot": entry_slot,
                            "strategy_version_id": None,
                            "data_source": "live",
                            "feature_schema_version": settings.feature_schema_version,
                            "feature_hash": feature_hash,
                            "source_job": "feature_builder_job",
                            "source_run_id": None,
                            "features_json": _to_json(feature_payload),
                            "label_status": "pending",
                            "label_horizon": "to_expiration",
                        },
                    )
                    feature_snapshot_id = feature_insert.scalar_one()
                    features_inserted += 1

                    candidates: list[dict] = []
                    side_delta_targets = settings.decision_delta_targets_for_side(spread_side)
                    for delta_target in side_delta_targets:
                        candidate = helper._build_candidate(
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

                    candidates.sort(key=lambda c: (-c["score"], c["delta_diff"]))
                    for rank, candidate in enumerate(candidates, start=1):
                        candidate["_calendar_flags"] = calendar_flags
                        candidate_json = self._build_candidate_json(candidate, cboe_context=cboe_context)
                        candidate_hash = build_candidate_hash(candidate_json)
                        entry_credit = float(candidate["credit"])
                        width_points = float(candidate_json["width_points"])
                        max_loss = max(width_points - entry_credit, 0.0)
                        credit_to_width = (entry_credit / width_points) if width_points > 0 else None
                        constraints_json = {
                            "credit_positive": entry_credit > 0,
                            "width_points": width_points,
                            "delta_target": candidate["delta_target"],
                            "spread_side": spread_side,
                        }

                        insert_candidate = await session.execute(
                            text(
                                """
                                INSERT INTO trade_candidates (
                                  ts, feature_snapshot_id, snapshot_id, strategy_version_id,
                                  candidate_hash, candidate_schema_version, candidate_rank,
                                  entry_credit, max_loss, credit_to_width, candidate_json, constraints_json,
                                  label_status, label_horizon
                                )
                                VALUES (
                                  :ts, :feature_snapshot_id, :snapshot_id, :strategy_version_id,
                                  :candidate_hash, :candidate_schema_version, :candidate_rank,
                                  :entry_credit, :max_loss, :credit_to_width, CAST(:candidate_json AS jsonb), CAST(:constraints_json AS jsonb),
                                  :label_status, :label_horizon
                                )
                                ON CONFLICT (feature_snapshot_id, candidate_hash) DO NOTHING
                                RETURNING candidate_id
                                """
                            ),
                            {
                                "ts": now_et.astimezone(ZoneInfo("UTC")),
                                "feature_snapshot_id": feature_snapshot_id,
                                "snapshot_id": candidate["snapshot_id"],
                                "strategy_version_id": None,
                                "candidate_hash": candidate_hash,
                                "candidate_schema_version": settings.candidate_schema_version,
                                "candidate_rank": rank,
                                "entry_credit": entry_credit,
                                "max_loss": max_loss,
                                "credit_to_width": credit_to_width,
                                "candidate_json": _to_json(candidate_json),
                                "constraints_json": _to_json(constraints_json),
                                "label_status": "pending",
                                "label_horizon": "to_expiration",
                            },
                        )
                        candidate_id = insert_candidate.scalar_one_or_none()
                        if candidate_id is not None:
                            candidates_inserted += 1

                    built.append(
                        {
                            "spread_side": spread_side,
                            "target_dte": snapshot["target_dte"],
                            "snapshot_id": snapshot["snapshot_id"],
                            "feature_snapshot_id": feature_snapshot_id,
                            "candidate_count": len(candidates),
                        }
                    )

            await session.commit()

        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "features_inserted": features_inserted,
            "candidates_inserted": candidates_inserted,
            "items": built,
        }


def build_feature_builder_job(clock_cache: MarketClockCache | None = None) -> FeatureBuilderJob:
    """Factory helper for FeatureBuilderJob."""
    return FeatureBuilderJob(clock_cache=clock_cache)
