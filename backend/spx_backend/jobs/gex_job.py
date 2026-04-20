from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth
from spx_backend.services.gex_math import compute_gex_per_strike

# Audit Wave 1 finding H4: 25% of TRADIER snapshots historically wrote
# zero_gamma_level=NULL because the strike-count window
# (settings.gex_strike_limit, default 150) is too narrow for chains where
# the dealer-flip lives outside the densest strike cluster. The aggregate
# gex_net columns are intentionally still windowed (performance), but the
# zero-gamma walker now uses a percent-of-spot window so it can resolve
# crossings in wider chains. The two constants are module-level so future
# tuning is one edit.
_ZERO_GAMMA_WINDOW_PCT_PRIMARY: Final[float] = 0.20
_ZERO_GAMMA_WINDOW_PCT_FALLBACK: Final[float] = 0.35


@dataclass(frozen=True)
class GexJob:
    """Compute and persist GEX aggregates from Tradier option chain snapshots.

    Per-strike values are produced by
    :func:`spx_backend.services.gex_math.compute_gex_per_strike` so the
    SqueezeMetrics convention (``OI * gamma * multiplier * S^2 * 0.01``
    with calls signed ``+`` and puts signed ``-``) lives in exactly one
    place across both writers (TRADIER here and CBOE in ``cboe_gex_job``).

    Per-snapshot magnitudes for SPX live in the canonical $1B-$100B
    range; the cross-source parity check vs CBOE is documented in
    ``backend/spx_backend/jobs/INGEST_AUDIT.md`` (finding C1).
    """
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
                        # M11 (audit): chain_snapshots older than the
                        # configured threshold are skipped with a loud warning.
                        # Force=True bypasses the gate for explicit backfills.
                        # ``ts`` is already UTC-aware from the SELECT; convert
                        # now_et to UTC for an apples-to-apples diff.
                        max_age_seconds = int(getattr(settings, "gex_chain_max_age_seconds", 0) or 0)
                        if not force and max_age_seconds > 0:
                            now_utc = now_et.astimezone(ZoneInfo("UTC"))
                            chain_age_seconds = (now_utc - ts).total_seconds()
                            if chain_age_seconds > max_age_seconds:
                                skipped_snapshots += 1
                                logger.warning(
                                    "gex_job: chain_too_old snapshot_id={} underlying={} "
                                    "age_seconds={:.0f} threshold_seconds={} -- skipping (use force=True to override)",
                                    snapshot_id,
                                    underlying,
                                    chain_age_seconds,
                                    max_age_seconds,
                                )
                                continue
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
                            # Route through the canonical SqueezeMetrics
                            # formula in services/gex_math.py so any future
                            # convention change is one edit. The canonical
                            # formula always signs puts negative; the legacy
                            # `gex_puts_negative` setting is enforced by
                            # treating right == 'P' as positive when False
                            # to preserve the original opt-out behavior.
                            if (not settings.gex_puts_negative) and right == "P":
                                effective_right = "C"  # opt-out flips the sign
                            else:
                                effective_right = right
                            gex_val = compute_gex_per_strike(
                                oi=int(oi),
                                gamma_per_share=float(gamma),
                                spot=float(spot),
                                right=effective_right,
                                contract_multiplier=int(multiplier),
                            )

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

                        # Build a separate per-strike map for the zero-gamma
                        # walker (audit Wave 1 H4). The aggregate
                        # gex_net/gex_calls/gex_puts/gex_abs above are
                        # intentionally windowed by gex_strike_limit (150 by
                        # default) for write performance, but the zero-gamma
                        # walker needs to see strikes far enough out to find
                        # sign changes that occur outside that strike-count
                        # window. The widest possible window
                        # (_ZERO_GAMMA_WINDOW_PCT_FALLBACK) is materialised
                        # once here; _zero_gamma_level itself filters down to
                        # the primary window first and falls back to the
                        # wider window only if no crossing is found.
                        zg_per_strike: dict[float, float] = {}
                        zg_lo = float(spot) * (1.0 - _ZERO_GAMMA_WINDOW_PCT_FALLBACK)
                        zg_hi = float(spot) * (1.0 + _ZERO_GAMMA_WINDOW_PCT_FALLBACK)
                        for strike, right, oi, gamma, contract_size, expiration, dte in filtered:
                            if strike is None or oi is None or gamma is None:
                                continue
                            s_val = float(strike)
                            if s_val < zg_lo or s_val > zg_hi:
                                continue
                            multiplier_zg = contract_size or settings.gex_contract_multiplier
                            # Mirror the gex_puts_negative opt-out so the
                            # walker uses the same sign convention as the
                            # aggregate; otherwise zero-gamma would be
                            # computed under a different convention than
                            # the gex_net it ships alongside.
                            if (not settings.gex_puts_negative) and right == "P":
                                effective_right_zg = "C"
                            else:
                                effective_right_zg = right
                            gex_val_zg = compute_gex_per_strike(
                                oi=int(oi),
                                gamma_per_share=float(gamma),
                                spot=float(spot),
                                right=effective_right_zg,
                                contract_multiplier=int(multiplier_zg),
                            )
                            zg_per_strike[s_val] = zg_per_strike.get(s_val, 0.0) + gex_val_zg

                        zero_gamma = self._zero_gamma_level(
                            zg_per_strike,
                            spot=float(spot),
                            snapshot_id=snapshot_id,
                            underlying=underlying,
                        )
                        gex_net = gex_calls + gex_puts

                        method = f"oi_gamma_spot_top{settings.gex_strike_limit}_dte{settings.gex_max_dte_days}"

                        # M6 (audit) ON CONFLICT policy:
                        # ``DO NOTHING`` is intentional and asymmetric
                        # with cboe_gex_job's ``DO UPDATE``.
                        # Rationale: a Tradier-derived snapshot is a
                        # *function of an immutable* chain_snapshots
                        # row (the chain text was already frozen by
                        # snapshot_job + chain_snapshot_dao). Re-running
                        # gex_job for the same ``snapshot_id`` should
                        # therefore produce the same numbers (modulo
                        # bug-fixes, which we always backfill via a
                        # migration, not a silent UPDATE). If you ever
                        # need to rewrite Tradier GEX in place, do it
                        # via an explicit migration so the change is
                        # audit-loggable -- not via a quiet
                        # ``DO UPDATE`` on the hot path.
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

                        # Update context_snapshots with gex fields, carrying
                        # forward quote values from underlying_quotes when
                        # inserting a new row so spx_price/vix/etc. aren't NULL.
                        #
                        # M2 (audit): gex_net is now a GENERATED column
                        # (migration 020) computed from
                        # COALESCE(gex_net_cboe, gex_net_tradier); writing it
                        # directly would raise PG error 428C9. Same for the
                        # ON CONFLICT path -- only the source-tagged columns
                        # are touched here.
                        # H6 (audit): the inline spot-quote subqueries use
                        # COALESCE(vendor_ts, ts) so as-of lookups prefer the
                        # vendor's observation timestamp when available;
                        # idx_underlying_quotes_symbol_vendor_or_ts (migration
                        # 019) supports the new ORDER BY shape.
                        await session.execute(
                            text(
                                """
                                INSERT INTO context_snapshots
                                    (ts, underlying, gex_net_tradier, zero_gamma_level_tradier,
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
                                  gex_net_tradier = EXCLUDED.gex_net_tradier,
                                  zero_gamma_level_tradier = EXCLUDED.zero_gamma_level_tradier,
                                  zero_gamma_level = COALESCE(context_snapshots.zero_gamma_level_cboe, EXCLUDED.zero_gamma_level_tradier),
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
                                "ts": ts,
                                "underlying": underlying,
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
        """Fetch latest spot price at or before ts with staleness guard.

        H6 (audit): switched the WHERE / ORDER BY to ``COALESCE(vendor_ts, ts)``
        so as-of lookups prefer the vendor's observation timestamp when
        present; the staleness guard then measures snapshot-to-vendor delay
        instead of snapshot-to-ingest delay (more accurate).
        ``idx_underlying_quotes_symbol_vendor_or_ts`` (migration 019)
        supports the new index-only scan.
        """
        row = await session.execute(
            text(
                """
                SELECT last, COALESCE(vendor_ts, ts) AS effective_ts
                FROM underlying_quotes
                WHERE symbol = :symbol
                  AND COALESCE(vendor_ts, ts) <= :ts
                ORDER BY COALESCE(vendor_ts, ts) DESC
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
        age = (ts - quote_ts).total_seconds()
        if age > settings.gex_spot_max_age_seconds:
            return None
        return float(last)

    def _zero_gamma_level(
        self,
        per_strike: Mapping[float, float],
        *,
        spot: float,
        snapshot_id: int | None = None,
        underlying: str | None = None,
    ) -> float | None:
        """Estimate the zero-gamma (dealer-flip) level by walking cumulative net GEX.

        Args:
            per_strike: Pre-built map of ``{strike: signed_net_gex}`` covering
                strikes within ``±_ZERO_GAMMA_WINDOW_PCT_FALLBACK`` of spot.
                Per-strike values are already signed via
                :func:`compute_gex_per_strike` so this method only sums.
            spot: Underlying spot price; defines the percent-of-spot window.
            snapshot_id: For structured-log correlation when no crossing exists.
            underlying: For structured-log correlation when no crossing exists.

        Returns:
            The interpolated dollar-strike at which cumulative net GEX crosses
            zero, preferring the narrower ``±_ZERO_GAMMA_WINDOW_PCT_PRIMARY``
            window. Falls back to ``±_ZERO_GAMMA_WINDOW_PCT_FALLBACK`` if the
            primary window is monotone. Returns ``None`` and emits a structured
            ``zero_gamma_chain_monotone`` warning if no crossing exists in
            either window — distinguishing a genuinely sign-stable chain from
            a search-window-too-narrow case (audit Wave 1 H4).
        """
        if not per_strike:
            return None

        # Try primary window first; fall back to wider window if monotone.
        for window_pct, window_label in (
            (_ZERO_GAMMA_WINDOW_PCT_PRIMARY, "primary"),
            (_ZERO_GAMMA_WINDOW_PCT_FALLBACK, "fallback"),
        ):
            lo = float(spot) * (1.0 - window_pct)
            hi = float(spot) * (1.0 + window_pct)
            windowed = sorted(s for s in per_strike if lo <= s <= hi)
            if len(windowed) < 2:
                # Need at least two strikes to detect a crossing.
                continue
            cum = 0.0
            prev_cum = 0.0
            prev_strike: float | None = None
            for strike in windowed:
                prev_cum = cum
                cum += float(per_strike[strike])
                if prev_strike is None:
                    prev_strike = strike
                    continue
                # Exact zero on the previous step is a crossing at prev_strike.
                if prev_cum == 0.0:
                    if window_label == "fallback":
                        logger.info(
                            "gex_job: zero_gamma_resolved_in_fallback_window "
                            "snapshot_id={} underlying={} window_pct={} strike={}",
                            snapshot_id,
                            underlying,
                            window_pct,
                            float(prev_strike),
                        )
                    return float(prev_strike)
                if (prev_cum < 0 and cum > 0) or (prev_cum > 0 and cum < 0):
                    denom = abs(prev_cum) + abs(cum)
                    if denom == 0:
                        return float(strike)
                    w = abs(prev_cum) / denom
                    interpolated = float(prev_strike + (strike - prev_strike) * w)
                    if window_label == "fallback":
                        logger.info(
                            "gex_job: zero_gamma_resolved_in_fallback_window "
                            "snapshot_id={} underlying={} window_pct={} strike={}",
                            snapshot_id,
                            underlying,
                            window_pct,
                            interpolated,
                        )
                    return interpolated
                prev_strike = strike

        # Both windows monotone -> persist NULL and log the discriminator.
        observed_min = min(per_strike.keys())
        observed_max = max(per_strike.keys())
        cum_total = sum(per_strike.values())
        logger.warning(
            "gex_job: zero_gamma_chain_monotone snapshot_id={} underlying={} "
            "spot={} observed_min={} observed_max={} cum_net={}",
            snapshot_id,
            underlying,
            float(spot),
            observed_min,
            observed_max,
            cum_total,
        )
        return None
