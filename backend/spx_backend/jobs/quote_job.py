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

        try:
            quote_resp = await self.tradier.get_quotes(quote_symbols)
            quotes = _parse_quotes(quote_resp)
        except Exception as exc:
            logger.exception("quote_job: failed to fetch quotes symbols={} error={}", quote_symbols, exc)
            return {"skipped": True, "reason": "quote_fetch_failed", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        if not quotes:
            return {"skipped": True, "reason": "no_quotes", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        by_symbol = _quotes_by_symbol(quotes)
        quotes_inserted = 0
        quotes_failed = 0

        async with SessionLocal() as session:
            try:
                # Isolate per-symbol insert failures with savepoints so one bad
                # record does not abort the entire ingestion run.
                for q in quotes:
                    try:
                        async with session.begin_nested():
                            await session.execute(
                                text(
                                    """
                                    INSERT INTO underlying_quotes (
                                      ts, symbol, last, bid, ask, open, high, low, close,
                                      volume, change, change_percent, prevclose, source, raw_json
                                    )
                                    VALUES (
                                      :ts, :symbol, :last, :bid, :ask, :open, :high, :low, :close,
                                      :volume, :change, :change_percent, :prevclose, :source, CAST(:raw_json AS jsonb)
                                    )
                                    """
                                ),
                                {
                                    "ts": now_et.astimezone(utc),
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

                await session.execute(
                    text(
                        """
                        INSERT INTO context_snapshots
                          (ts, spx_price, spy_price, vix, vix9d, term_structure, vvix, skew, notes_json)
                        VALUES
                          (:ts, :spx_price, :spy_price, :vix, :vix9d, :term_structure, :vvix, :skew, CAST(:notes AS jsonb))
                        ON CONFLICT (ts) DO UPDATE SET
                          spx_price = EXCLUDED.spx_price,
                          spy_price = EXCLUDED.spy_price,
                          vix = EXCLUDED.vix,
                          vix9d = EXCLUDED.vix9d,
                          term_structure = EXCLUDED.term_structure,
                          vvix = EXCLUDED.vvix,
                          skew = EXCLUDED.skew,
                          notes_json = EXCLUDED.notes_json
                        """
                    ),
                    {
                        "ts": now_et.astimezone(utc),
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
                }

        if quotes_inserted == 0 and quotes_failed > 0:
            return {
                "skipped": True,
                "reason": "all_quote_inserts_failed",
                "now_et": now_et.isoformat(),
                "quotes_inserted": quotes_inserted,
                "quotes_failed": quotes_failed,
            }
        logger.info("quote_job: inserted_quotes={} failed_quotes={}", quotes_inserted, quotes_failed)
        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "quotes_inserted": quotes_inserted,
            "quotes_failed": quotes_failed,
        }


def build_quote_job(clock_cache: MarketClockCache | None = None, tradier: TradierClient | None = None) -> QuoteJob:
    """Factory helper for QuoteJob."""
    client = tradier or get_tradier_client()
    return QuoteJob(tradier=client, clock_cache=clock_cache)
