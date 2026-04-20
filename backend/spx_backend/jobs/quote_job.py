from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.ingestion.tradier_client import TradierClient, get_tradier_client
from spx_backend.market_clock import MarketClockCache, is_rth


def _parse_quotes(resp: dict) -> list[dict]:
    """Normalize Tradier quotes payload to a list."""
    quotes = resp.get("quotes", {}).get("quote")
    if quotes is None:
        return []
    if isinstance(quotes, list):
        return quotes
    return [quotes]


def _quotes_by_symbol(quotes: list[dict]) -> dict[str, dict]:
    """Index quote objects by symbol."""
    out: dict[str, dict] = {}
    for q in quotes:
        sym = q.get("symbol")
        if sym:
            out[sym] = q
    return out


def _parse_tradier_epoch_ms(value: object) -> datetime | None:
    """Convert one Tradier ms-since-epoch field into a tz-aware UTC datetime.

    Tradier publishes ``trade_date``, ``bid_date``, and ``ask_date`` as integer
    milliseconds since the Unix epoch. Returns ``None`` for missing, non-numeric,
    or non-positive values so callers can write the column as NULL without
    extra branching. Used by H6 (audit) to populate ``underlying_quotes.vendor_ts``
    from ``trade_date`` so downstream as-of joins can move off ingest-time.
    """
    if value is None:
        return None
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=ZoneInfo("UTC"))
    except (OverflowError, OSError, ValueError):
        return None


class _FetchFailureCounter:
    """Module-level counter for consecutive get_quotes failures.

    Used by M13 (audit) to escalate a sustained outage past the per-tick
    "swallow into log" path so APScheduler's job-failure email fires.
    A small mutable holder rather than a global int so dataclass(frozen=True)
    callers can still mutate it without ``global`` declarations.
    """

    def __init__(self) -> None:
        self.consecutive: int = 0


_fetch_failures = _FetchFailureCounter()


def reset_fetch_failure_counter() -> None:
    """Test-only helper to clear the in-process counter between scenarios."""
    _fetch_failures.consecutive = 0


@dataclass(frozen=True)
class QuoteJob:
    """Periodic underlying quote capture job."""
    tradier: TradierClient
    clock_cache: MarketClockCache | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        """Check if market is open using cache or RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    async def run_once(self, *, force: bool = False) -> dict:
        """Run one quote capture cycle and store quote/context records.

        Parameters
        ----------
        force:
            When true, bypasses regular-trading-hours gating.

        Returns
        -------
        dict
            Job result payload containing skip reason (if any), run timestamp,
            and insert/failure counters for quote rows.
        """
        tz = ZoneInfo(settings.tz)
        utc = ZoneInfo("UTC")
        now_et = datetime.now(tz=tz)
        logger.info("quote_job: start force={} now_et={}", force, now_et.isoformat())

        if (not force) and (not settings.allow_quotes_outside_rth):
            if not await self._market_open(now_et):
                logger.info("quote_job: market closed; skipping (now_et={})", now_et.isoformat())
                return {"skipped": True, "reason": "market_closed", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        quote_symbols = settings.quote_symbols_list()
        if not quote_symbols:
            return {"skipped": True, "reason": "no_symbols", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        # M13 (audit): swallow transient fetch failures up to a threshold so
        # one-off Tradier 5xx blips don't spam APScheduler email, but escalate
        # past the threshold so a sustained outage raises into the scheduler's
        # job-failure path. The counter resets to zero on the first successful
        # fetch.
        try:
            quote_resp = await self.tradier.get_quotes(quote_symbols)
            quotes = _parse_quotes(quote_resp)
        except Exception as exc:
            _fetch_failures.consecutive += 1
            threshold = int(getattr(settings, "quote_consecutive_failure_threshold", 3))
            logger.exception(
                "quote_job: failed to fetch quotes consecutive_failures={}/{} symbols={} error={}",
                _fetch_failures.consecutive,
                threshold,
                quote_symbols,
                exc,
            )
            if _fetch_failures.consecutive >= threshold:
                # Re-raise so APScheduler emails the operator. Reset the
                # counter on raise so the next run starts a fresh window
                # rather than instantly re-escalating on a flapping outage.
                _fetch_failures.consecutive = 0
                raise
            return {"skipped": True, "reason": "quote_fetch_failed", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        # Successful fetch: clear the consecutive-failure window.
        _fetch_failures.consecutive = 0

        if not quotes:
            return {"skipped": True, "reason": "no_quotes", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        by_symbol = _quotes_by_symbol(quotes)
        quotes_inserted = 0
        quotes_failed = 0
        quotes_dedup_skipped = 0

        async with SessionLocal() as session:
            try:
                # Isolate per-symbol insert failures with savepoints so one bad
                # record does not abort the entire ingestion run.
                for q in quotes:
                    try:
                        async with session.begin_nested():
                            # H6 (audit): persist Tradier vendor timestamp from
                            # ``trade_date`` (ms epoch) into the new vendor_ts
                            # column. Falls through to NULL when Tradier omits
                            # it; downstream consumers use COALESCE(vendor_ts, ts)
                            # so historical rows keep working.
                            vendor_ts = _parse_tradier_epoch_ms(q.get("trade_date"))
                            # H5 (audit): ON CONFLICT against the
                            # uq_underlying_quotes_symbol_ts_minute expression
                            # index makes retries idempotent at minute
                            # granularity. ``ts`` here is ingest time
                            # (now_et.astimezone(utc)) which the dedup window
                            # is keyed on; vendor_ts can NULL through.
                            # NOTE: PG forbids STABLE functions in unique
                            # indexes, so the index uses date_bin (IMMUTABLE)
                            # rather than date_trunc. The ON CONFLICT
                            # expression must match the index expression
                            # EXACTLY for the planner to recognize it.
                            insert_result = await session.execute(
                                text(
                                    """
                                    INSERT INTO underlying_quotes (
                                      ts, vendor_ts, symbol, last, bid, ask, open, high, low, close,
                                      volume, change, change_percent, prevclose, source, raw_json
                                    )
                                    VALUES (
                                      :ts, :vendor_ts, :symbol, :last, :bid, :ask, :open, :high, :low, :close,
                                      :volume, :change, :change_percent, :prevclose, :source, CAST(:raw_json AS jsonb)
                                    )
                                    ON CONFLICT (symbol, (date_bin('1 minute', ts, TIMESTAMPTZ '2000-01-01 00:00:00+00'))) DO NOTHING
                                    RETURNING quote_id
                                    """
                                ),
                                {
                                    "ts": now_et.astimezone(utc),
                                    "vendor_ts": vendor_ts,
                                    "symbol": q.get("symbol"),
                                    "last": q.get("last"),
                                    "bid": q.get("bid"),
                                    "ask": q.get("ask"),
                                    "open": q.get("open"),
                                    "high": q.get("high"),
                                    "low": q.get("low"),
                                    "close": q.get("close"),
                                    "volume": q.get("volume"),
                                    "change": q.get("change"),
                                    "change_percent": q.get("change_percentage"),
                                    "prevclose": q.get("prevclose"),
                                    "source": "tradier",
                                    "raw_json": json.dumps(q, default=str),
                                },
                            )
                            inserted_row = insert_result.fetchone()
                            if inserted_row is None:
                                # Conflict path -- another writer (or this
                                # process on retry) already landed a row in
                                # the same (symbol, minute) bucket.
                                quotes_dedup_skipped += 1
                            else:
                                quotes_inserted += 1
                    except Exception as exc:
                        quotes_failed += 1
                        logger.exception(
                            "quote_job: quote_insert_failed symbol={} now_et={} error={}",
                            q.get("symbol"),
                            now_et.isoformat(),
                            exc,
                        )

                # Derive context snapshot fields from quotes.
                vix = by_symbol.get("VIX", {}).get("last")
                vix9d = by_symbol.get("VIX9D", {}).get("last")
                spx_price = by_symbol.get("SPX", {}).get("last")
                spy_price = by_symbol.get("SPY", {}).get("last")
                vvix = by_symbol.get("VVIX", {}).get("last")
                skew = by_symbol.get("SKEW", {}).get("last")
                term_structure = None
                if vix and vix9d and vix > 0:
                    term_structure = vix9d / vix

                # M2 (audit): the canonical context_snapshots.gex_net is now a
                # GENERATED column (migration 020) computed from
                # COALESCE(gex_net_cboe, gex_net_tradier), so quote_job no
                # longer touches gex_net / zero_gamma_level here. Keeping the
                # COALESCE on the source-tagged columns preserves the "first
                # writer wins" semantics for those.
                await session.execute(
                    text(
                        """
                        INSERT INTO context_snapshots
                          (ts, underlying, spx_price, spy_price, vix, vix9d, term_structure, vvix, skew, notes_json)
                        VALUES
                          (:ts, :underlying, :spx_price, :spy_price, :vix, :vix9d, :term_structure, :vvix, :skew, CAST(:notes AS jsonb))
                        ON CONFLICT (ts, underlying) DO UPDATE SET
                          spx_price = EXCLUDED.spx_price,
                          spy_price = EXCLUDED.spy_price,
                          vix = EXCLUDED.vix,
                          vix9d = EXCLUDED.vix9d,
                          term_structure = EXCLUDED.term_structure,
                          vvix = EXCLUDED.vvix,
                          skew = EXCLUDED.skew,
                          gex_net_tradier = COALESCE(context_snapshots.gex_net_tradier, EXCLUDED.gex_net_tradier),
                          zero_gamma_level_tradier = COALESCE(context_snapshots.zero_gamma_level_tradier, EXCLUDED.zero_gamma_level_tradier),
                          gex_net_cboe = COALESCE(context_snapshots.gex_net_cboe, EXCLUDED.gex_net_cboe),
                          zero_gamma_level_cboe = COALESCE(context_snapshots.zero_gamma_level_cboe, EXCLUDED.zero_gamma_level_cboe),
                          notes_json = EXCLUDED.notes_json
                        """
                    ),
                    {
                        "ts": now_et.astimezone(utc),
                        "underlying": "SPX",
                        "spx_price": spx_price,
                        "spy_price": spy_price,
                        "vix": vix,
                        "vix9d": vix9d,
                        "term_structure": term_structure,
                        "vvix": vvix,
                        "skew": skew,
                        "notes": json.dumps({"source": "tradier", "symbols": quote_symbols}),
                    },
                )
                await session.commit()
            except Exception as exc:
                await session.rollback()
                logger.exception("quote_job: db_write_failed now_et={} error={}", now_et.isoformat(), exc)
                return {
                    "skipped": True,
                    "reason": "db_write_failed",
                    "now_et": now_et.isoformat(),
                    "quotes_inserted": quotes_inserted,
                    "quotes_failed": quotes_failed,
                    "quotes_dedup_skipped": quotes_dedup_skipped,
                }

        if quotes_inserted == 0 and quotes_failed > 0 and quotes_dedup_skipped == 0:
            return {
                "skipped": True,
                "reason": "all_quote_inserts_failed",
                "now_et": now_et.isoformat(),
                "quotes_inserted": quotes_inserted,
                "quotes_failed": quotes_failed,
                "quotes_dedup_skipped": quotes_dedup_skipped,
            }
        logger.info(
            "quote_job: inserted_quotes={} failed_quotes={} dedup_skipped={}",
            quotes_inserted,
            quotes_failed,
            quotes_dedup_skipped,
        )
        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "quotes_inserted": quotes_inserted,
            "quotes_failed": quotes_failed,
            "quotes_dedup_skipped": quotes_dedup_skipped,
        }


def build_quote_job(clock_cache: MarketClockCache | None = None, tradier: TradierClient | None = None) -> QuoteJob:
    """Factory helper for QuoteJob."""
    client = tradier or get_tradier_client()
    return QuoteJob(tradier=client, clock_cache=clock_cache)
