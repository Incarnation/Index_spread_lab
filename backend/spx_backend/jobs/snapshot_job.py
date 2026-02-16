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
from spx_backend.market_clock import MarketClockCache, is_rth


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
    """Normalize Tradier chain options payload to a list."""
    options = chain.get("options", {}).get("option")
    if options is None:
        return []
    if isinstance(options, list):
        return options
    return [options]


def _normalize_option_right(opt: dict) -> str | None:
    """Normalize option right to 'C' or 'P'."""
    val = opt.get("option_type") or opt.get("put_call") or opt.get("right")
    if isinstance(val, str):
        v = val.strip().upper()
        if v in {"CALL", "C"}:
            return "C"
        if v in {"PUT", "P"}:
            return "P"
    return None


def _to_int(value: object) -> int | None:
    """Convert value to int when possible."""
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


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


def _default_vix_snapshot_job_config() -> SnapshotJobConfig:
    """Build the default VIX snapshot configuration from global settings."""
    return SnapshotJobConfig(
        job_name="snapshot_job_vix",
        underlying=settings.vix_snapshot_underlying,
        dte_mode=settings.vix_snapshot_dte_mode,
        dte_targets=settings.vix_snapshot_dte_targets_list(),
        dte_min_days=settings.vix_snapshot_dte_min_days,
        dte_max_days=settings.vix_snapshot_dte_max_days,
        range_fallback_enabled=settings.vix_snapshot_range_fallback_enabled,
        range_fallback_count=settings.vix_snapshot_range_fallback_count,
        dte_tolerance_days=settings.vix_snapshot_dte_tolerance_days,
        strikes_each_side=settings.vix_snapshot_strikes_each_side,
        allow_outside_rth=settings.vix_allow_snapshot_outside_rth,
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
        """Run one snapshot cycle and store chains."""
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        logger.info("{}: start force={} now_et={}", self.config.job_name, force, now_et.isoformat())

        if (not force) and (not self.config.allow_outside_rth):
            if not await self._market_open(now_et):
                logger.info("{}: market closed; skipping (now_et={})", self.config.job_name, now_et.isoformat())
                return {"skipped": True, "reason": "market_closed", "now_et": now_et.isoformat(), "inserted": []}

        underlying = self.config.underlying
        dte_mode = self.config.dte_mode.strip().lower()

        exp_resp = await self.tradier.get_option_expirations(underlying)
        expirations = _parse_expirations(exp_resp)
        as_of = now_et.date()
        exp_to_trading_dte = trading_dte_lookup(expirations, as_of)

        inserted: list[dict] = []
        chain_rows_inserted = 0
        async with SessionLocal() as session:
            if not expirations:
                logger.warning("{}: no expirations returned for {}", self.config.job_name, underlying)
                await session.commit()
                return {
                    "skipped": True,
                    "reason": "no_expirations",
                    "now_et": now_et.isoformat(),
                    "inserted": [],
                    "chain_rows_inserted": 0,
                    "fallback_used": False,
                }

            spot = await self._get_spot_price(session, now_et, underlying)
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
                        await session.commit()
                        return {
                            "skipped": True,
                            "reason": "no_expirations_in_range",
                            "now_et": now_et.isoformat(),
                            "inserted": [],
                            "chain_rows_inserted": 0,
                            "fallback_used": False,
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

                chain = await self.tradier.get_option_chain(underlying=underlying, expiration=exp.isoformat(), greeks=True)
                chk = _checksum(chain)

                result = await session.execute(
                    text(
                        """
                        INSERT INTO chain_snapshots (ts, underlying, target_dte, expiration, payload_json, checksum)
                        VALUES (:ts, :underlying, :target_dte, :expiration, CAST(:payload AS jsonb), :checksum)
                        RETURNING snapshot_id
                        """
                    ),
                    {
                        "ts": now_et.astimezone(ZoneInfo("UTC")),
                        "underlying": underlying,
                        "target_dte": target_dte,
                        "expiration": exp,
                        "payload": json.dumps(chain, default=str),
                        "checksum": chk,
                    },
                )
                snapshot_id = result.scalar_one()

                # Extract per-option rows with open_interest + greeks if present.
                options = _parse_chain_options(chain)
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
                for opt in options:
                    symbol = opt.get("symbol")
                    if not symbol:
                        continue
                    greeks = opt.get("greeks") or {}
                    strike_val = _to_float(opt.get("strike"))
                    if selected_strikes is not None:
                        if strike_val is None or strike_val not in selected_strikes:
                            continue
                    await session.execute(
                        text(
                            """
                            INSERT INTO option_chain_rows (
                              snapshot_id, option_symbol, underlying, expiration, strike, option_right,
                              bid, ask, last, volume, open_interest,
                              delta, gamma, theta, vega, rho,
                              bid_iv, mid_iv, ask_iv, greeks_updated_at,
                              raw_json
                            )
                            VALUES (
                              :snapshot_id, :option_symbol, :underlying, :expiration, :strike, :option_right,
                              :bid, :ask, :last, :volume, :open_interest,
                              :delta, :gamma, :theta, :vega, :rho,
                              :bid_iv, :mid_iv, :ask_iv, :greeks_updated_at,
                              CAST(:raw_json AS jsonb)
                            )
                            """
                        ),
                        {
                            "snapshot_id": snapshot_id,
                            "option_symbol": symbol,
                            "underlying": underlying,
                            "expiration": (_to_date(opt.get("expiration_date")) or exp),
                            "strike": strike_val,
                            "option_right": _normalize_option_right(opt),
                            "bid": _to_float(opt.get("bid")),
                            "ask": _to_float(opt.get("ask")),
                            "last": _to_float(opt.get("last")),
                            "volume": _to_int(opt.get("volume")),
                            "open_interest": _to_int(opt.get("open_interest")),
                            "contract_size": _to_int(opt.get("contract_size")),
                            "delta": _to_float(greeks.get("delta")),
                            "gamma": _to_float(greeks.get("gamma")),
                            "theta": _to_float(greeks.get("theta")),
                            "vega": _to_float(greeks.get("vega")),
                            "rho": _to_float(greeks.get("rho")),
                            "bid_iv": _to_float(greeks.get("bid_iv")),
                            "mid_iv": _to_float(greeks.get("mid_iv")),
                            "ask_iv": _to_float(greeks.get("ask_iv")),
                            "greeks_updated_at": greeks.get("updated_at"),
                            "raw_json": json.dumps(opt, default=str),
                        },
                    )
                    chain_rows_inserted += 1
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

            await session.commit()
        logger.info("{}: inserted_chains={} chain_rows={}", self.config.job_name, len(inserted), chain_rows_inserted)
        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "inserted": inserted,
            "chain_rows_inserted": chain_rows_inserted,
            "fallback_used": fallback_used,
        }

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


def build_snapshot_job(
    clock_cache: MarketClockCache | None = None,
    tradier: TradierClient | None = None,
    config: SnapshotJobConfig | None = None,
) -> SnapshotJob:
    """Factory helper for SPX snapshot ingestion job."""
    client = tradier or get_tradier_client()
    snapshot_config = config or _default_snapshot_job_config()
    return SnapshotJob(tradier=client, clock_cache=clock_cache, config=snapshot_config)


def build_vix_snapshot_job(clock_cache: MarketClockCache | None = None, tradier: TradierClient | None = None) -> SnapshotJob:
    """Factory helper for VIX snapshot ingestion job."""
    client = tradier or get_tradier_client()
    return SnapshotJob(tradier=client, clock_cache=clock_cache, config=_default_vix_snapshot_job_config())

