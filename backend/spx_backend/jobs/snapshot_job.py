from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.dte import (
    choose_expiration_for_trading_dte,
    closest_expiration_for_trading_dte,
    trading_dte_lookup,
)
from spx_backend.ingestion.tradier_client import TradierClient, get_tradier_client
from spx_backend.jobs._chain_snapshot_dao import (
    PAYLOAD_KIND_OPTIONS_CHAIN,
    get_or_insert_anchor,
)
from spx_backend.market_clock import MarketClockCache, is_rth
from spx_backend.services import alerts
from spx_backend.services.option_row_sanitizer import (
    normalize_option_right as _sanitizer_normalize_option_right,
    sanitize_chain_options,
)


_OPTION_ROW_INSERT_PAGE_SIZE = 200  # M3 (audit) executemany page size


def _checksum(payload: object) -> str:
    """Compute a stable checksum for the snapshot payload."""
    b = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _parse_expirations(resp: dict) -> list[date]:
    """Parse Tradier expirations response into sorted dates."""
    # Tradier often returns { "expirations": { "date": ["2026-02-06", ...] } }
    dates = resp.get("expirations", {}).get("date", [])
    out: list[date] = []
    for d in dates:
        try:
            out.append(date.fromisoformat(d))
        except Exception:
            continue
    return sorted(out)


def _parse_chain_options(chain: dict) -> list[dict]:
    """Normalize Tradier chain payload into a validated option-row list.

    Parameters
    ----------
    chain:
        One Tradier option-chain payload as returned by ``get_option_chain``.
        Some edge responses use ``{"options": None}`` or malformed shapes
        rather than a nested dict/list, especially for sparse expirations.

    Returns
    -------
    list[dict]
        A list of option-row dictionaries safe for downstream processing.
        Non-dict payload fragments are discarded so snapshot ingestion can
        continue without raising during startup warmup or scheduler runs.
    """
    options_container = chain.get("options")
    if not isinstance(options_container, dict):
        return []
    options = options_container.get("option")
    if options is None:
        return []
    if isinstance(options, list):
        return [opt for opt in options if isinstance(opt, dict)]
    if isinstance(options, dict):
        return [options]
    return []


def _normalize_option_right(opt: dict) -> str | None:
    """Normalize option right to 'C' or 'P'.

    Refactor #4 (audit): the canonical implementation now lives in
    ``services.option_row_sanitizer.normalize_option_right`` and is
    re-exported here as a thin wrapper so existing imports of
    ``snapshot_job._normalize_option_right`` (notably the unit tests)
    keep working bytecode-identically.
    """
    return _sanitizer_normalize_option_right(opt)


def _to_float(value: object) -> float | None:
    """Convert value to float when possible."""
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_date(value: object) -> date | None:
    """Convert ISO date string/object to date when possible."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except Exception:
            return None
    return None


def _select_strikes_near_spot(options: list[dict], spot: float, each_side: int) -> set[float]:
    """Return strike set around spot with N strikes on each side."""
    strikes: list[float] = []
    for opt in options:
        strike_val = _to_float(opt.get("strike"))
        if strike_val is not None:
            strikes.append(strike_val)
    if not strikes:
        return set()
    strikes_sorted = sorted(set(strikes))
    idx = bisect.bisect_left(strikes_sorted, spot)
    below = strikes_sorted[max(0, idx - each_side) : idx]
    above = strikes_sorted[idx : idx + each_side]
    return set(below + above)


@dataclass(frozen=True)
class SnapshotJobConfig:
    """Configuration values for one snapshot ingestion stream."""

    job_name: str
    underlying: str
    dte_mode: str
    dte_targets: list[int]
    dte_min_days: int
    dte_max_days: int
    range_fallback_enabled: bool
    range_fallback_count: int
    dte_tolerance_days: int
    strikes_each_side: int
    allow_outside_rth: bool


def _default_snapshot_job_config() -> SnapshotJobConfig:
    """Build the default SPX snapshot configuration from global settings."""
    return SnapshotJobConfig(
        job_name="snapshot_job",
        underlying=settings.snapshot_underlying,
        dte_mode=settings.snapshot_dte_mode,
        dte_targets=settings.dte_targets_list(),
        dte_min_days=settings.snapshot_dte_min_days,
        dte_max_days=settings.snapshot_dte_max_days,
        range_fallback_enabled=settings.snapshot_range_fallback_enabled,
        range_fallback_count=settings.snapshot_range_fallback_count,
        dte_tolerance_days=settings.snapshot_dte_tolerance_days,
        strikes_each_side=settings.snapshot_strikes_each_side,
        allow_outside_rth=settings.allow_snapshot_outside_rth,
    )


def _default_spy_snapshot_job_config() -> SnapshotJobConfig:
    """Build the default SPY snapshot configuration from global settings."""
    return SnapshotJobConfig(
        job_name="snapshot_job_spy",
        underlying=settings.spy_snapshot_underlying,
        dte_mode=settings.spy_snapshot_dte_mode,
        dte_targets=settings.spy_snapshot_dte_targets_list(),
        dte_min_days=settings.spy_snapshot_dte_min_days,
        dte_max_days=settings.spy_snapshot_dte_max_days,
        range_fallback_enabled=settings.spy_snapshot_range_fallback_enabled,
        range_fallback_count=settings.spy_snapshot_range_fallback_count,
        dte_tolerance_days=settings.spy_snapshot_dte_tolerance_days,
        strikes_each_side=settings.spy_snapshot_strikes_each_side,
        allow_outside_rth=settings.spy_allow_snapshot_outside_rth,
    )


@dataclass(frozen=True)
class SnapshotJob:
    """Periodic Tradier chain snapshot job."""

    tradier: TradierClient
    clock_cache: MarketClockCache | None = None
    config: SnapshotJobConfig = field(default_factory=_default_snapshot_job_config)

    async def _market_open(self, now_et: datetime) -> bool:
        """Check if market is open using cache or RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    async def run_once(self, *, force: bool = False) -> dict:
        """Run one snapshot cycle and persist chain snapshots/rows.

        Parameters
        ----------
        force:
            When true, bypasses regular-trading-hours gating and allows relaxed
            target-DTE fallback behavior.

        Returns
        -------
        dict
            Job result payload with skip reason, inserted chain metadata,
            cumulative option-row count, and per-expiration failures.
        """
        tz = ZoneInfo(settings.tz)
        utc = ZoneInfo("UTC")
        now_et = datetime.now(tz=tz)
        logger.info("{}: start force={} now_et={}", self.config.job_name, force, now_et.isoformat())

        if (not force) and (not self.config.allow_outside_rth):
            if not await self._market_open(now_et):
                logger.info("{}: market closed; skipping (now_et={})", self.config.job_name, now_et.isoformat())
                return {"skipped": True, "reason": "market_closed", "now_et": now_et.isoformat(), "inserted": []}

        underlying = self.config.underlying
        dte_mode = self.config.dte_mode.strip().lower()

        try:
            exp_resp = await self.tradier.get_option_expirations(underlying)
            expirations = _parse_expirations(exp_resp)
        except Exception as exc:
            logger.exception("{}: expirations_fetch_failed underlying={} error={}", self.config.job_name, underlying, exc)
            return {
                "skipped": True,
                "reason": "expirations_fetch_failed",
                "now_et": now_et.isoformat(),
                "inserted": [],
                "chain_rows_inserted": 0,
                "fallback_used": False,
                "failed_items": [],
            }
        as_of = now_et.date()
        exp_to_trading_dte = trading_dte_lookup(expirations, as_of)

        inserted: list[dict] = []
        failed_items: list[dict] = []
        chain_rows_inserted = 0
        async with SessionLocal() as session:
            if not expirations:
                logger.warning("{}: no expirations returned for {}", self.config.job_name, underlying)
                return {
                    "skipped": True,
                    "reason": "no_expirations",
                    "now_et": now_et.isoformat(),
                    "inserted": [],
                    "chain_rows_inserted": 0,
                    "fallback_used": False,
                    "failed_items": [],
                }

            try:
                spot = await self._get_spot_price(session, now_et, underlying)
            except Exception as exc:
                await session.rollback()
                logger.exception("{}: spot_lookup_failed underlying={} error={}", self.config.job_name, underlying, exc)
                return {
                    "skipped": True,
                    "reason": "spot_lookup_failed",
                    "now_et": now_et.isoformat(),
                    "inserted": [],
                    "chain_rows_inserted": 0,
                    "fallback_used": False,
                    "failed_items": [],
                }
            if spot is None:
                logger.warning("{}: no spot price for {}; storing full chains", self.config.job_name, underlying)

            selected: list[tuple[int, date]] = []
            fallback_used = False
            seen_exp: set[date] = set()
            if dte_mode == "range":
                min_dte = min(self.config.dte_min_days, self.config.dte_max_days)
                max_dte = max(self.config.dte_min_days, self.config.dte_max_days)
                for exp, dte in exp_to_trading_dte.items():
                    if min_dte <= dte <= max_dte:
                        selected.append((dte, exp))
                if not selected:
                    if self.config.range_fallback_enabled:
                        fallback_count = max(1, self.config.range_fallback_count)
                        center = (min_dte + max_dte) / 2.0
                        ranked = sorted(exp_to_trading_dte.items(), key=lambda item: abs(item[1] - center))
                        fallback_items = ranked[:fallback_count]
                        selected = [(dte, exp) for exp, dte in fallback_items]
                        fallback_used = True
                        logger.warning(
                            "{}: no expirations in trading-dte range {}-{}; fallback enabled, selected {} closest expirations",
                            self.config.job_name,
                            min_dte,
                            max_dte,
                            len(selected),
                        )
                    else:
                        logger.warning("{}: no expirations in trading-dte range {}-{}", self.config.job_name, min_dte, max_dte)
                        return {
                            "skipped": True,
                            "reason": "no_expirations_in_range",
                            "now_et": now_et.isoformat(),
                            "inserted": [],
                            "chain_rows_inserted": 0,
                            "fallback_used": False,
                            "failed_items": [],
                        }
            else:
                dte_targets = self.config.dte_targets
                for target_dte in dte_targets:
                    exp = choose_expiration_for_trading_dte(
                        expirations,
                        target_dte=target_dte,
                        as_of=as_of,
                        tolerance=self.config.dte_tolerance_days,
                    )
                    if exp is None:
                        if force:
                            exp = closest_expiration_for_trading_dte(expirations, target_dte=target_dte, as_of=as_of)
                            if exp is None:
                                logger.warning("{}: no expirations available to fallback", self.config.job_name)
                                continue
                            logger.warning(
                                "{}: no expiration within tolerance for trading target_dte={}; using closest exp={} (force mode)",
                                self.config.job_name,
                                target_dte,
                                exp.isoformat(),
                            )
                        else:
                            logger.warning(
                                "{}: no expiration found for trading target_dte={} ({} expirations)",
                                self.config.job_name,
                                target_dte,
                                len(expirations),
                            )
                            continue
                    if exp in seen_exp:
                        continue
                    seen_exp.add(exp)
                    selected.append((exp_to_trading_dte.get(exp, target_dte), exp))

            for target_dte, exp in selected:
                try:
                    chain = await self.tradier.get_option_chain(underlying=underlying, expiration=exp.isoformat(), greeks=True)
                except Exception as exc:
                    failed_items.append(
                        {
                            "target_dte": target_dte,
                            "expiration": exp.isoformat(),
                            "stage": "fetch_chain",
                            "error": str(exc),
                        }
                    )
                    logger.exception(
                        "{}: chain_fetch_failed target_dte={} expiration={} error={}",
                        self.config.job_name,
                        target_dte,
                        exp.isoformat(),
                        exc,
                    )
                    continue

                options = _parse_chain_options(chain)
                if not options:
                    # Empty provider payloads are not useful snapshots and would
                    # otherwise consume downstream GEX batch capacity.
                    failed_items.append(
                        {
                            "target_dte": target_dte,
                            "expiration": exp.isoformat(),
                            "stage": "empty_chain",
                            "error": "no_option_rows",
                        }
                    )
                    logger.warning(
                        "{}: empty_chain_payload target_dte={} expiration={} underlying={}; skipping snapshot insert",
                        self.config.job_name,
                        target_dte,
                        exp.isoformat(),
                        underlying,
                    )
                    continue

                chk = _checksum(chain)
                item_chain_rows_inserted = 0
                try:
                    # Use per-expiration savepoints so one bad chain payload or DB
                    # write does not roll back successful snapshots from this run.
                    async with session.begin_nested():
                        # Refactor #2 + M4 + M1 (audit): centralized DAO
                        # writes the (ts, underlying, source, expiration)
                        # anchor with ON CONFLICT DO NOTHING and stamps
                        # payload_kind='options_chain' (full chain follows
                        # in option_chain_rows below).
                        snapshot_id, _was_inserted = await get_or_insert_anchor(
                            session=session,
                            ts=now_et.astimezone(utc),
                            underlying=underlying,
                            source="TRADIER",
                            target_dte=target_dte,
                            expiration=exp,
                            checksum=chk,
                            payload_kind=PAYLOAD_KIND_OPTIONS_CHAIN,
                        )

                        # Extract per-option rows with open_interest + greeks if present.
                        selected_strikes: set[float] | None = None
                        if spot is not None and self.config.strikes_each_side > 0:
                            selected_strikes = _select_strikes_near_spot(
                                options,
                                float(spot),
                                self.config.strikes_each_side,
                            )
                            if selected_strikes:
                                logger.info(
                                    "{}: filtering to {} strikes around spot={} (exp={})",
                                    self.config.job_name,
                                    len(selected_strikes),
                                    spot,
                                    exp.isoformat(),
                                )
                            else:
                                # H2 (audit): _select_strikes_near_spot returned an empty set
                                # (sparse chain, malformed strikes, etc.) despite a non-None
                                # spot. Treating ``set()`` as "filter to nothing" silently
                                # produces an empty-but-FK-valid chain_snapshot (Q23 evidence
                                # showed 222 such rows). Convert to None so we fall through
                                # to the full-chain ingest with a loud warning.
                                logger.warning(
                                    "{}: empty_strike_filter underlying={} exp={} spot={}; "
                                    "ingesting full chain (H2 fallback)",
                                    self.config.job_name,
                                    underlying,
                                    exp.isoformat(),
                                    spot,
                                )
                                selected_strikes = None
                        # Refactor #4 + M18 + M3 (audit): centralised
                        # sanitiser filters non-dict / missing-symbol /
                        # null-strike / null-option_right rows up front
                        # and emits one structured log per snapshot.
                        # Insert is then issued via executemany in
                        # _OPTION_ROW_INSERT_PAGE_SIZE-row pages so a
                        # 600-row chain fires 3 round trips instead of
                        # 600 (M3).
                        sanitized = sanitize_chain_options(
                            options,
                            snapshot_id=snapshot_id,
                            underlying=underlying,
                            fallback_expiration=exp,
                            selected_strikes=selected_strikes,
                            job_name=self.config.job_name,
                        )
                        bound_rows = sanitized.rows
                        if bound_rows:
                            insert_stmt = text(
                                """
                                INSERT INTO option_chain_rows (
                                  snapshot_id, option_symbol, underlying, expiration, strike, option_right,
                                  bid, ask, last, volume, open_interest, contract_size,
                                  delta, gamma, theta, vega, rho,
                                  bid_iv, mid_iv, ask_iv, greeks_updated_at,
                                  raw_json
                                )
                                VALUES (
                                  :snapshot_id, :option_symbol, :underlying, :expiration, :strike, :option_right,
                                  :bid, :ask, :last, :volume, :open_interest, :contract_size,
                                  :delta, :gamma, :theta, :vega, :rho,
                                  :bid_iv, :mid_iv, :ask_iv, :greeks_updated_at,
                                  CAST(:raw_json AS jsonb)
                                )
                                """
                            )
                            for page_start in range(0, len(bound_rows), _OPTION_ROW_INSERT_PAGE_SIZE):
                                page = bound_rows[page_start : page_start + _OPTION_ROW_INSERT_PAGE_SIZE]
                                await session.execute(insert_stmt, page)
                            item_chain_rows_inserted += len(bound_rows)
                    chain_rows_inserted += item_chain_rows_inserted
                    inserted.append(
                        {
                            "target_dte": target_dte,
                            "expiration": exp.isoformat(),
                            "actual_dte_days": (exp - now_et.date()).days,
                            "actual_trading_dte": exp_to_trading_dte.get(exp),
                            "checksum": chk,
                            "fallback_used": fallback_used,
                        }
                    )
                except Exception as exc:
                    failed_items.append(
                        {
                            "target_dte": target_dte,
                            "expiration": exp.isoformat(),
                            "stage": "persist_chain",
                            "error": str(exc),
                        }
                    )
                    logger.exception(
                        "{}: persist_chain_failed target_dte={} expiration={} error={}",
                        self.config.job_name,
                        target_dte,
                        exp.isoformat(),
                        exc,
                    )
                    continue

            try:
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.exception(
                    "{}: commit_failed inserted_chains={} failed_items={} error={}",
                    self.config.job_name,
                    len(inserted),
                    len(failed_items),
                    exc,
                )
                return {
                    "skipped": True,
                    "reason": "db_commit_failed",
                    "now_et": now_et.isoformat(),
                    "inserted": inserted,
                    "chain_rows_inserted": chain_rows_inserted,
                    "fallback_used": fallback_used,
                    "failed_items": failed_items,
                }
        # H1 (audit): per-expiration savepoints commit independently inside one
        # outer transaction. A failure on expiration N does NOT roll back the
        # already-savepoint-committed expirations 1..N-1, and decision_job
        # cannot see "this run was incomplete" from the snapshot table alone.
        # Emit a structured log + best-effort SendGrid alert (cooldown-gated)
        # so the operator notices a partial run quickly.
        expected_expirations = len(selected)
        inserted_count = len(inserted)
        if expected_expirations > 0 and inserted_count < expected_expirations:
            failed_stages = sorted({str(fi.get("stage", "?")) for fi in failed_items})
            logger.warning(
                "{}: partial_snapshot_run underlying={} expected={} inserted={} "
                "failed={} stages={}",
                self.config.job_name,
                underlying,
                expected_expirations,
                inserted_count,
                len(failed_items),
                failed_stages,
            )
            try:
                await alerts.send_alert(
                    subject=f"[IndexSpreadLab] snapshot_job partial run -- {underlying}",
                    body_html=(
                        f"<p><b>{self.config.job_name}</b> partial-batch run.</p>"
                        f"<ul>"
                        f"<li>underlying: {underlying}</li>"
                        f"<li>expected expirations: {expected_expirations}</li>"
                        f"<li>inserted expirations: {inserted_count}</li>"
                        f"<li>failed expirations: {len(failed_items)}</li>"
                        f"<li>failed stages: {', '.join(failed_stages) or 'none'}</li>"
                        f"<li>now_et: {now_et.isoformat()}</li>"
                        f"</ul>"
                        f"<p>See logs for per-expiration failure detail.</p>"
                    ),
                    cooldown_key=f"snapshot_partial:{self.config.job_name}:{underlying}",
                    cooldown_minutes=int(
                        getattr(settings, "snapshot_partial_alert_cooldown_minutes", 30)
                    ),
                )
            except Exception as alert_exc:
                # Never let the alert path bring down the snapshot job.
                logger.warning(
                    "{}: partial_snapshot_alert_failed underlying={} error={}",
                    self.config.job_name,
                    underlying,
                    alert_exc,
                )
        logger.info(
            "{}: inserted_chains={} chain_rows={} expected_chains={} failed_items={}",
            self.config.job_name,
            len(inserted),
            chain_rows_inserted,
            expected_expirations,
            len(failed_items),
        )
        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "inserted": inserted,
            "chain_rows_inserted": chain_rows_inserted,
            "expected_expirations": expected_expirations,
            "inserted_count": inserted_count,
            "fallback_used": fallback_used,
            "failed_items": failed_items,
        }

    async def _get_spot_price(self, session, ts: datetime, underlying: str) -> float | None:
        """Fetch latest spot price at or before ts.

        H6 (audit): switched to ``COALESCE(vendor_ts, ts)`` so as-of lookups
        prefer the vendor's observation timestamp (Tradier ``trade_date``)
        when present. Backward-compatible: historical rows where
        vendor_ts IS NULL fall back to ingest ts. The functional index
        ``idx_underlying_quotes_symbol_vendor_or_ts`` (migration 019)
        supports the ORDER BY without a sequential scan.
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
            {"symbol": underlying, "ts": ts.astimezone(ZoneInfo("UTC"))},
        )
        result = row.fetchone()
        if not result:
            return None
        last, _ = result
        if last is None:
            return None
        return float(last)


def build_snapshot_job(
    clock_cache: MarketClockCache | None = None,
    tradier: TradierClient | None = None,
    config: SnapshotJobConfig | None = None,
) -> SnapshotJob:
    """Factory helper for SPX snapshot ingestion job."""
    client = tradier or get_tradier_client()
    snapshot_config = config or _default_snapshot_job_config()
    return SnapshotJob(tradier=client, clock_cache=clock_cache, config=snapshot_config)


def build_spy_snapshot_job(clock_cache: MarketClockCache | None = None, tradier: TradierClient | None = None) -> SnapshotJob:
    """Factory helper for SPY snapshot ingestion job."""
    client = tradier or get_tradier_client()
    return SnapshotJob(tradier=client, clock_cache=clock_cache, config=_default_spy_snapshot_job_config())

