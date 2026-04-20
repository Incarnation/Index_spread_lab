from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.dte import trading_dte_lookup
from spx_backend.ingestion.mzdata_client import MzDataClient, get_mzdata_client
# Refactor #1 (audit): pure parsers were extracted to
# ``spx_backend.ingestion.parsers``. The underscored aliases below
# preserve every existing internal call-site / test import unchanged
# (bytecode-equivalent thin wrappers); new code should import the
# public names directly from ``ingestion.parsers``.
from spx_backend.ingestion.parsers import (
    CboeExposureItem,
    normalize_cboe_exposure_items as _normalize_exposure_items,
    parse_iso_date as _parse_iso_date,
    parse_payload_timestamp as _parse_payload_timestamp,
    series_float as _series_float,
    series_int as _series_int,
    to_float as _to_float,
    to_float_list as _to_float_list,
    to_int as _to_int,
    to_int_list as _to_int_list,
)
from spx_backend.jobs._chain_snapshot_dao import (
    PAYLOAD_KIND_GEX_ANCHOR,
    get_or_insert_anchor,
)
from spx_backend.market_clock import MarketClockCache, is_rth
from spx_backend.services import alerts
from spx_backend.services.gex_math import apply_vendor_units


def _checksum_payload(payload: object) -> str:
    """Compute a deterministic checksum for one JSON-like payload.

    Kept in this module (not promoted to ``ingestion.parsers``)
    because the canonical payload shape is owned by ``cboe_gex_job``;
    other parsers should not assume the same field ordering.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CboeGexJob:
    """Ingest precomputed CBOE-mode GEX data and persist source-tagged batches."""

    mzdata: MzDataClient
    clock_cache: MarketClockCache | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        """Check market state using cached clock, fallback to simple RTH logic."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    def _build_trading_slot_dte_lookup(
        self,
        *,
        exposure_items: list[CboeExposureItem],
        as_of_trading_date: date,
    ) -> dict[date, int]:
        """Build one expiration->trading-slot DTE map for a CBOE batch.

        Parameters
        ----------
        exposure_items:
            One normalized MZData payload expressed as per-expiration rows.
        as_of_trading_date:
            Trading date derived from batch timestamp in market timezone.

        Returns
        -------
        dict[date, int]
            Mapping from expiration date to trading-slot DTE where 0DTE is
            same-day expiration and subsequent expirations increment by one.
        """
        expirations = sorted({item.expiration for item in exposure_items})
        return trading_dte_lookup(expirations, as_of_trading_date)

    def _resolve_target_dte(
        self,
        *,
        item: CboeExposureItem,
        trading_slot_dte_by_expiration: dict[date, int],
    ) -> int | None:
        """Resolve one CBOE expiration into Tradier-style trading-slot DTE.

        Parameters
        ----------
        item:
            One normalized CBOE expiration payload.
        trading_slot_dte_by_expiration:
            Batch-level expiration->trading-slot mapping.

        Returns
        -------
        int | None
            Trading-slot DTE for the expiration, or ``None`` when the
            expiration is not eligible in the current trading-date window.
        """
        return trading_slot_dte_by_expiration.get(item.expiration)

    def _zero_gamma_level(self, per_strike: dict[float, dict[str, float]]) -> float | None:
        """Estimate zero-gamma strike from cumulative net GEX crossing.

        Uses the ``gex_net`` key (MZData precomputed net gamma) when available
        so the crossing is consistent with the aggregate gex_net stored in
        context_snapshots.  Falls back to ``gex_calls + gex_puts`` for Tradier
        or any source that lacks a dedicated net field.
        """
        if not per_strike:
            return None
        strikes = sorted(per_strike.keys())
        cumulative = 0.0
        prev_cumulative = 0.0
        prev_strike: float | None = None
        for strike in strikes:
            prev_cumulative = cumulative
            sdata = per_strike[strike]
            if "gex_net" in sdata:
                cumulative += float(sdata["gex_net"])
            else:
                cumulative += float(sdata["gex_calls"]) + float(sdata["gex_puts"])
            if prev_strike is None:
                prev_strike = strike
                continue
            if prev_cumulative == 0.0:
                return float(prev_strike)
            crossed = (prev_cumulative < 0 and cumulative > 0) or (prev_cumulative > 0 and cumulative < 0)
            if crossed:
                denom = abs(prev_cumulative) + abs(cumulative)
                if denom == 0:
                    return float(strike)
                weight = abs(prev_cumulative) / denom
                return float(prev_strike + (strike - prev_strike) * weight)
            prev_strike = strike
        return None

    def _empty_underlying_result(self, *, now_et: datetime, underlying: str) -> dict[str, Any]:
        """Build a default per-underlying result payload with zeroed counters."""
        return {
            "underlying": underlying,
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "batch_ts_utc": None,
            "inserted_snapshots": 0,
            "reused_snapshots": 0,
            "gex_snapshots_upserted": 0,
            "gex_by_strike_upserted": 0,
            "gex_by_expiry_strike_upserted": 0,
            "skipped_items": 0,
            "skipped_items_by_reason": {},
            "failed_items": [],
        }

    async def _run_once_for_underlying(
        self,
        *,
        session: Any,
        underlying: str,
        now_et: datetime,
    ) -> dict[str, Any]:
        """Fetch and persist one underlying's CBOE exposure payload."""
        result = self._empty_underlying_result(now_et=now_et, underlying=underlying)
        # Defense-in-depth: VIX dealer-gamma snapshots were dropped in
        # audit Wave 1 (finding H3) because mzdata coverage was 0% and
        # the symbol's options aren't dealer-hedged the same way as SPX
        # / SPY. We still need VIX *index* quotes (handled in
        # quote_job), but never an exposure snapshot. Guard here so a
        # config-only re-introduction can't write VIX rows.
        if underlying.strip().upper() == "VIX":
            logger.info(
                "cboe_gex_job: skipping VIX exposure write (audit Wave 1 H3); "
                "remove 'VIX' from CBOE_GEX_UNDERLYINGS to silence this log"
            )
            result["skipped"] = True
            result["reason"] = "vix_excluded"
            return result
        try:
            payload = await self.mzdata.get_live_option_exposure(underlying)
        except Exception as exc:
            logger.exception("cboe_gex_job: exposure_fetch_failed underlying={} error={}", underlying, exc)
            result["skipped"] = True
            result["reason"] = "exposure_fetch_failed"
            result["failed_items"] = [{"underlying": underlying, "reason": "exposure_fetch_failed", "error": str(exc)}]
            return result

        spot_price = _to_float(payload.get("spotPrice"))
        batch_ts_utc = _parse_payload_timestamp(payload.get("timestamp"), fallback=now_et.astimezone(ZoneInfo("UTC")))
        result["batch_ts_utc"] = batch_ts_utc.isoformat()

        # M10 (audit): distinguish "HTTP 200 + empty data" (vendor schema
        # drift OR vendor outage) from "HTTP 200 + well-formed data" so
        # the operator gets paged on the former. Without this branch
        # both shapes silently no-op via the ``no_exposure_items`` skip.
        raw_data = payload.get("data")
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            result["skipped"] = True
            result["reason"] = "vendor_empty_data" if isinstance(raw_data, list) else "vendor_schema_drift"
            try:
                await alerts.send_alert(
                    subject=f"[IndexSpreadLab] cboe_gex_job vendor anomaly -- {underlying}",
                    body_html=(
                        f"<p>mzdata returned HTTP 200 with anomalous payload for "
                        f"<b>{underlying}</b>.</p>"
                        f"<ul>"
                        f"<li>shape: {result['reason']}</li>"
                        f"<li>data type: {type(raw_data).__name__}</li>"
                        f"<li>data length: "
                        f"{len(raw_data) if isinstance(raw_data, list) else 'n/a'}</li>"
                        f"<li>batch_ts_utc: {batch_ts_utc.isoformat()}</li>"
                        f"<li>now_et: {now_et.isoformat()}</li>"
                        f"</ul>"
                        f"<p>Check mzdata's API contract; this previously surfaced "
                        f"as a silent no-op snapshot.</p>"
                    ),
                    cooldown_key=f"cboe_vendor_anomaly:{underlying}:{result['reason']}",
                    cooldown_minutes=int(getattr(settings, "cboe_vendor_alert_cooldown_minutes", 60)),
                )
            except Exception as alert_exc:
                logger.warning(
                    "cboe_gex_job: vendor_alert_failed underlying={} error={}",
                    underlying, alert_exc,
                )
            return result

        exposure_items = _normalize_exposure_items(payload)
        if not exposure_items:
            # Reach here only if the data list is non-empty but every item
            # failed normalization (e.g. malformed expirations). Different
            # signal from M10's empty-data shape; keep the legacy reason.
            result["skipped"] = True
            result["reason"] = "no_exposure_items"
            return result

        as_of_trading_date = batch_ts_utc.astimezone(ZoneInfo(settings.tz)).date()
        trading_slot_dte_by_expiration = self._build_trading_slot_dte_lookup(
            exposure_items=exposure_items,
            as_of_trading_date=as_of_trading_date,
        )
        max_allowed_dte = max(int(settings.gex_max_dte_days), 0)

        inserted_snapshots = 0
        reused_snapshots = 0
        gex_snapshots_upserted = 0
        strike_rows_upserted = 0
        expiry_strike_rows_upserted = 0
        skipped_items_by_reason: dict[str, int] = {}
        failed_items: list[dict[str, Any]] = []

        agg_per_strike: dict[float, dict[str, float]] = {}

        for item in exposure_items:
            try:
                # Build CBOE DTE labels with the same trading-slot indexing used
                # by Tradier snapshots so cross-source filters remain aligned.
                target_dte = self._resolve_target_dte(
                    item=item,
                    trading_slot_dte_by_expiration=trading_slot_dte_by_expiration,
                )
                if target_dte is None:
                    skipped_items_by_reason["missing_trading_slot_dte"] = (
                        skipped_items_by_reason.get("missing_trading_slot_dte", 0) + 1
                    )
                    logger.info(
                        "cboe_gex_job: skipping underlying={} expiration={} reason=missing_trading_slot_dte",
                        underlying,
                        item.expiration.isoformat(),
                    )
                    continue
                if target_dte < 0 or target_dte > max_allowed_dte:
                    skipped_items_by_reason["dte_out_of_range"] = skipped_items_by_reason.get("dte_out_of_range", 0) + 1
                    logger.info(
                        "cboe_gex_job: skipping underlying={} expiration={} reason=dte_out_of_range target_dte={} max_allowed_dte={}",
                        underlying,
                        item.expiration.isoformat(),
                        target_dte,
                        max_allowed_dte,
                    )
                    continue
                async with session.begin_nested():
                    snapshot_payload = {
                        "source": "CBOE",
                        "underlying": underlying,
                        "timestamp": batch_ts_utc.isoformat(),
                        "spotPrice": spot_price,
                        "item": item.raw_item,
                    }
                    checksum = _checksum_payload(snapshot_payload)
                    # Refactor #2 + M4 + M1 (audit): centralized DAO replaces
                    # the prior SELECT-then-INSERT race window. payload_kind
                    # is mandatory and set to ``gex_anchor`` so retention /
                    # backfill can distinguish CBOE FK-anchor rows (no
                    # option_chain_rows) from snapshot_job's full chains.
                    snapshot_id, was_inserted = await get_or_insert_anchor(
                        session=session,
                        ts=batch_ts_utc,
                        underlying=underlying,
                        source="CBOE",
                        target_dte=target_dte,
                        expiration=item.expiration,
                        checksum=checksum,
                        payload_kind=PAYLOAD_KIND_GEX_ANCHOR,
                    )
                    if was_inserted:
                        inserted_snapshots += 1
                    else:
                        reused_snapshots += 1

                    per_strike: dict[float, dict[str, float]] = {}
                    gex_calls_total = 0.0
                    gex_puts_total = 0.0
                    gex_abs_total = 0.0

                    # mzdata publishes per-strike exposures in dollars-per-
                    # share-per-1%-move (omits the 100-share contract
                    # multiplier). apply_vendor_units(...) scales each
                    # series into the SqueezeMetrics convention so this
                    # writer's gex_net is directly comparable to the
                    # TRADIER writer (audit Wave 1, finding C1).
                    spot_for_units = float(spot_price) if spot_price is not None else 0.0
                    raw_net_gamma_series = item.net_gamma if isinstance(item.net_gamma, list) else []
                    for index, strike_value in enumerate(item.strikes):
                        if strike_value is None:
                            continue
                        strike = float(strike_value)
                        call_gamma = apply_vendor_units(
                            _series_float(item.call_abs_gamma, index, default=0.0),
                            spot=spot_for_units,
                        )
                        put_gamma = -abs(
                            apply_vendor_units(
                                _series_float(item.put_abs_gamma, index, default=0.0),
                                spot=spot_for_units,
                            )
                        )
                        # Use the vendor's net_gamma when present; otherwise
                        # fall back to the call+put sum already in canonical
                        # units. We avoid double-scaling by only piping raw
                        # vendor values through apply_vendor_units.
                        if 0 <= index < len(raw_net_gamma_series) and raw_net_gamma_series[index] is not None:
                            net_gamma = apply_vendor_units(
                                _series_float(raw_net_gamma_series, index, default=0.0),
                                spot=spot_for_units,
                            )
                        else:
                            net_gamma = call_gamma + put_gamma
                        call_oi = _series_int(item.call_open_interest, index, default=0)
                        put_oi = _series_int(item.put_open_interest, index, default=0)
                        oi_total = max(call_oi + put_oi, 0)

                        bucket = per_strike.setdefault(
                            strike,
                            {"gex_calls": 0.0, "gex_puts": 0.0, "gex_net": 0.0, "oi_total": 0.0},
                        )
                        bucket["gex_calls"] += call_gamma
                        bucket["gex_puts"] += put_gamma
                        bucket["gex_net"] += net_gamma
                        bucket["oi_total"] += float(oi_total)
                        gex_calls_total += call_gamma
                        gex_puts_total += put_gamma
                        gex_abs_total += abs(call_gamma) + abs(put_gamma)

                    if not per_strike:
                        failed_items.append(
                            {
                                "underlying": underlying,
                                "expiration": item.expiration.isoformat(),
                                "reason": "no_valid_strikes",
                            }
                        )
                        continue

                    for strike, sdata in per_strike.items():
                        agg = agg_per_strike.setdefault(
                            strike,
                            {"gex_calls": 0.0, "gex_puts": 0.0, "gex_net": 0.0},
                        )
                        agg["gex_calls"] += sdata["gex_calls"]
                        agg["gex_puts"] += sdata["gex_puts"]
                        agg["gex_net"] += sdata["gex_net"]

                    method = "cboe_mz_precomputed"
                    zero_gamma_level = self._zero_gamma_level(per_strike)
                    gex_net_total = sum(strike_data["gex_net"] for strike_data in per_strike.values())

                    await session.execute(
                        # M6 (audit) ON CONFLICT policy:
                        # ``DO UPDATE`` is intentional and asymmetric
                        # with gex_job's ``DO NOTHING``.
                        # Rationale: mzdata occasionally republishes
                        # an exposure with revised numbers for the same
                        # ``(ts, underlying, expiration)`` -- e.g. when
                        # a late-arriving OPRA strike is added to the
                        # vendor's book. We want the latest revision to
                        # win in place rather than stale-stamp a snapshot
                        # the consumer already cached. The ``snapshot_id``
                        # is shared via _chain_snapshot_dao.get_or_insert_anchor
                        # so a vendor revision lands on the same row
                        # without creating a duplicate anchor.
                        text(
                            """
                            INSERT INTO gex_snapshots (
                              snapshot_id, ts, underlying, source, spot_price, gex_net, gex_calls, gex_puts, gex_abs, zero_gamma_level, method
                            )
                            VALUES (
                              :snapshot_id, :ts, :underlying, :source, :spot_price, :gex_net, :gex_calls, :gex_puts, :gex_abs, :zero_gamma_level, :method
                            )
                            ON CONFLICT (snapshot_id) DO UPDATE SET
                              ts = EXCLUDED.ts,
                              underlying = EXCLUDED.underlying,
                              source = EXCLUDED.source,
                              spot_price = EXCLUDED.spot_price,
                              gex_net = EXCLUDED.gex_net,
                              gex_calls = EXCLUDED.gex_calls,
                              gex_puts = EXCLUDED.gex_puts,
                              gex_abs = EXCLUDED.gex_abs,
                              zero_gamma_level = EXCLUDED.zero_gamma_level,
                              method = EXCLUDED.method
                            """
                        ),
                        {
                            "snapshot_id": snapshot_id,
                            "ts": batch_ts_utc,
                            "underlying": underlying,
                            "source": "CBOE",
                            "spot_price": spot_price,
                            "gex_net": gex_net_total,
                            "gex_calls": gex_calls_total,
                            "gex_puts": gex_puts_total,
                            "gex_abs": gex_abs_total,
                            "zero_gamma_level": zero_gamma_level,
                            "method": method,
                        },
                    )
                    gex_snapshots_upserted += 1

                    # L9 (audit) + paired with M3: bind every strike row
                    # for this expiration up front, then issue one
                    # ``executemany`` per child table. A 600-strike chain
                    # used to fire 1200 round trips (one per
                    # gex_by_strike + one per gex_by_expiry_strike); now
                    # it fires 2 (regardless of strike count). The
                    # ON CONFLICT semantics are unchanged.
                    by_strike_rows: list[dict] = []
                    by_expiry_strike_rows: list[dict] = []
                    for strike, strike_data in per_strike.items():
                        by_strike_rows.append(
                            {
                                "snapshot_id": snapshot_id,
                                "strike": strike,
                                "gex_net": strike_data["gex_net"],
                                "gex_calls": strike_data["gex_calls"],
                                "gex_puts": strike_data["gex_puts"],
                                "oi_total": int(strike_data["oi_total"]),
                                "method": method,
                            }
                        )
                        by_expiry_strike_rows.append(
                            {
                                "snapshot_id": snapshot_id,
                                "expiration": item.expiration,
                                "dte_days": target_dte,
                                "strike": strike,
                                "gex_net": strike_data["gex_net"],
                                "gex_calls": strike_data["gex_calls"],
                                "gex_puts": strike_data["gex_puts"],
                                "oi_total": int(strike_data["oi_total"]),
                                "method": method,
                            }
                        )

                    if by_strike_rows:
                        await session.execute(
                            text(
                                """
                                INSERT INTO gex_by_strike (snapshot_id, strike, gex_net, gex_calls, gex_puts, oi_total, method)
                                VALUES (:snapshot_id, :strike, :gex_net, :gex_calls, :gex_puts, :oi_total, :method)
                                ON CONFLICT (snapshot_id, strike) DO UPDATE SET
                                  gex_net = EXCLUDED.gex_net,
                                  gex_calls = EXCLUDED.gex_calls,
                                  gex_puts = EXCLUDED.gex_puts,
                                  oi_total = EXCLUDED.oi_total,
                                  method = EXCLUDED.method
                                """
                            ),
                            by_strike_rows,
                        )
                        strike_rows_upserted += len(by_strike_rows)

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
                            by_expiry_strike_rows,
                        )
                        expiry_strike_rows_upserted += len(by_expiry_strike_rows)
            except Exception as exc:
                failed_items.append(
                    {
                        "underlying": underlying,
                        "expiration": item.expiration.isoformat(),
                        "error": str(exc),
                    }
                )
                logger.exception(
                    "cboe_gex_job: process_item_failed underlying={} expiration={} error={}",
                    underlying,
                    item.expiration.isoformat(),
                    exc,
                )
                continue

        if agg_per_strike and gex_snapshots_upserted > 0:
            agg_gex_net = sum(s["gex_net"] for s in agg_per_strike.values())
            agg_zero_gamma = self._zero_gamma_level(agg_per_strike)
            try:
                # M2 (audit): gex_net is a GENERATED column (migration 020)
                # derived from COALESCE(gex_net_cboe, gex_net_tradier);
                # writing it directly raises PG 428C9. We populate
                # gex_net_cboe and zero_gamma_level_cboe; gex_net is
                # auto-derived. zero_gamma_level (the legacy aggregate
                # column) keeps the "CBOE wins" precedence at app level
                # for now -- migration 020 only promoted gex_net.
                # H6 (audit): inline spot subqueries use COALESCE(vendor_ts, ts)
                # so as-of lookups prefer vendor observation time.
                await session.execute(
                    text(
                        """
                        INSERT INTO context_snapshots
                            (ts, underlying, gex_net_cboe, zero_gamma_level_cboe,
                             zero_gamma_level,
                             spx_price, spy_price, vix, vix9d, term_structure,
                             vvix, skew, notes_json)
                        VALUES (
                            :ts, :underlying, :gex_net, :zero_gamma_level,
                            :zero_gamma_level,
                            (SELECT last FROM underlying_quotes WHERE symbol='SPX' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1),
                            (SELECT last FROM underlying_quotes WHERE symbol='SPY' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1),
                            (SELECT last FROM underlying_quotes WHERE symbol='VIX' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1),
                            (SELECT last FROM underlying_quotes WHERE symbol='VIX9D' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1),
                            (SELECT CASE WHEN v.last > 0 THEN v9.last / v.last END
                             FROM (SELECT last FROM underlying_quotes WHERE symbol='VIX' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1) v,
                                  (SELECT last FROM underlying_quotes WHERE symbol='VIX9D' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1) v9),
                            (SELECT last FROM underlying_quotes WHERE symbol='VVIX' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1),
                            (SELECT last FROM underlying_quotes WHERE symbol='SKEW' AND COALESCE(vendor_ts, ts) <= :ts ORDER BY COALESCE(vendor_ts, ts) DESC LIMIT 1),
                            CAST(:notes AS jsonb)
                        )
                        ON CONFLICT (ts, underlying) DO UPDATE SET
                          gex_net_cboe = EXCLUDED.gex_net_cboe,
                          zero_gamma_level_cboe = EXCLUDED.zero_gamma_level_cboe,
                          zero_gamma_level = EXCLUDED.zero_gamma_level_cboe,
                          spx_price = COALESCE(context_snapshots.spx_price, EXCLUDED.spx_price),
                          spy_price = COALESCE(context_snapshots.spy_price, EXCLUDED.spy_price),
                          vix = COALESCE(context_snapshots.vix, EXCLUDED.vix),
                          vix9d = COALESCE(context_snapshots.vix9d, EXCLUDED.vix9d),
                          term_structure = COALESCE(context_snapshots.term_structure, EXCLUDED.term_structure),
                          vvix = COALESCE(context_snapshots.vvix, EXCLUDED.vvix),
                          skew = COALESCE(context_snapshots.skew, EXCLUDED.skew)
                        """
                    ),
                    {
                        "ts": batch_ts_utc,
                        "underlying": underlying,
                        "gex_net": agg_gex_net,
                        "zero_gamma_level": agg_zero_gamma,
                        "notes": "{}",
                    },
                )
                logger.info(
                    "cboe_gex_job: context_snapshots upserted underlying={} gex_net={:.2f} zero_gamma={}",
                    underlying,
                    agg_gex_net,
                    agg_zero_gamma,
                )
            except Exception as exc:
                logger.warning(
                    "cboe_gex_job: context_snapshots upsert failed underlying={} error={}",
                    underlying,
                    exc,
                )

        result.update(
            {
                "inserted_snapshots": inserted_snapshots,
                "reused_snapshots": reused_snapshots,
                "gex_snapshots_upserted": gex_snapshots_upserted,
                "gex_by_strike_upserted": strike_rows_upserted,
                "gex_by_expiry_strike_upserted": expiry_strike_rows_upserted,
                "skipped_items": sum(skipped_items_by_reason.values()),
                "skipped_items_by_reason": skipped_items_by_reason,
                "failed_items": failed_items,
            }
        )
        return result

    def _aggregate_underlying_results(
        self,
        *,
        now_et: datetime,
        requested_underlyings: list[str],
        underlying_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate per-underlying counters into a single run response."""
        inserted_snapshots = sum(int(item.get("inserted_snapshots", 0) or 0) for item in underlying_results)
        reused_snapshots = sum(int(item.get("reused_snapshots", 0) or 0) for item in underlying_results)
        gex_snapshots_upserted = sum(int(item.get("gex_snapshots_upserted", 0) or 0) for item in underlying_results)
        strike_rows_upserted = sum(int(item.get("gex_by_strike_upserted", 0) or 0) for item in underlying_results)
        expiry_strike_rows_upserted = sum(
            int(item.get("gex_by_expiry_strike_upserted", 0) or 0) for item in underlying_results
        )
        failed_items: list[dict[str, Any]] = []
        skipped_items_by_reason: dict[str, int] = {}
        for item in underlying_results:
            for reason, count in dict(item.get("skipped_items_by_reason", {})).items():
                skipped_items_by_reason[reason] = skipped_items_by_reason.get(reason, 0) + int(count)
            failed_items.extend(list(item.get("failed_items", [])))

        processed_underlyings = [
            str(item.get("underlying", "")).upper() for item in underlying_results if not bool(item.get("skipped"))
        ]
        skipped_underlyings = [
            {"underlying": str(item.get("underlying", "")).upper(), "reason": item.get("reason")}
            for item in underlying_results
            if bool(item.get("skipped"))
        ]
        all_skipped = len(processed_underlyings) == 0
        if all_skipped and len(underlying_results) == 1:
            run_reason = underlying_results[0].get("reason")
        elif all_skipped:
            run_reason = "all_underlyings_skipped"
        else:
            run_reason = None
        first_result = underlying_results[0] if len(underlying_results) == 1 else None
        return {
            "skipped": all_skipped,
            "reason": run_reason,
            "now_et": now_et.isoformat(),
            # Keep single-symbol keys for backwards compatibility where possible.
            "underlying": first_result.get("underlying") if first_result else None,
            "batch_ts_utc": first_result.get("batch_ts_utc") if first_result else None,
            "underlyings": requested_underlyings,
            "processed_underlyings": processed_underlyings,
            "skipped_underlyings": skipped_underlyings,
            "underlying_results": underlying_results,
            "inserted_snapshots": inserted_snapshots,
            "reused_snapshots": reused_snapshots,
            "gex_snapshots_upserted": gex_snapshots_upserted,
            "gex_by_strike_upserted": strike_rows_upserted,
            "gex_by_expiry_strike_upserted": expiry_strike_rows_upserted,
            "skipped_items": sum(skipped_items_by_reason.values()),
            "skipped_items_by_reason": skipped_items_by_reason,
            "failed_items": failed_items,
        }

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        """Fetch MZData exposure payloads and persist CBOE rows by symbol.

        The CBOE pipeline intentionally aligns DTE labeling to Tradier trading
        slot semantics and only stores expirations within the configured
        ``0..gex_max_dte_days`` window for every configured underlying.
        """
        if not settings.cboe_gex_enabled:
            return {"skipped": True, "reason": "cboe_gex_disabled"}

        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        if (not force) and (not settings.cboe_gex_allow_outside_rth):
            if not await self._market_open(now_et):
                logger.info("cboe_gex_job: market closed; skipping (now_et={})", now_et.isoformat())
                return {"skipped": True, "reason": "market_closed", "now_et": now_et.isoformat()}

        underlyings = settings.cboe_gex_underlyings_list()
        if not underlyings:
            return {"skipped": True, "reason": "missing_underlyings", "now_et": now_et.isoformat()}

        underlying_results: list[dict[str, Any]] = []
        async with SessionLocal() as session:
            for underlying in underlyings:
                underlying_results.append(
                    await self._run_once_for_underlying(session=session, underlying=underlying, now_et=now_et)
                )
            await session.commit()

        aggregated_result = self._aggregate_underlying_results(
            now_et=now_et,
            requested_underlyings=underlyings,
            underlying_results=underlying_results,
        )
        logger.info(
            "cboe_gex_job: underlyings={} processed_underlyings={} skipped_underlyings={} inserted_snapshots={} reused_snapshots={} gex_snapshots={} strikes={} expiry_strikes={} skipped_items={} skipped_items_by_reason={} failed_items={}",
            aggregated_result["underlyings"],
            aggregated_result["processed_underlyings"],
            aggregated_result["skipped_underlyings"],
            aggregated_result["inserted_snapshots"],
            aggregated_result["reused_snapshots"],
            aggregated_result["gex_snapshots_upserted"],
            aggregated_result["gex_by_strike_upserted"],
            aggregated_result["gex_by_expiry_strike_upserted"],
            aggregated_result["skipped_items"],
            aggregated_result["skipped_items_by_reason"],
            len(aggregated_result["failed_items"]),
        )
        return aggregated_result


def build_cboe_gex_job(
    *,
    clock_cache: MarketClockCache | None = None,
    mzdata_client: MzDataClient | None = None,
) -> CboeGexJob:
    """Factory helper for the CBOE-mode precomputed GEX ingestion job."""
    return CboeGexJob(mzdata=mzdata_client or get_mzdata_client(), clock_cache=clock_cache)
