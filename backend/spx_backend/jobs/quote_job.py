from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.db import SessionLocal
from spx_backend.ingestion.tradier_client import TradierClient, get_tradier_client
from spx_backend.market_clock import MarketClockCache, is_rth


def _parse_quotes(resp: dict) -> list[dict]:
    quotes = resp.get("quotes", {}).get("quote")
    if quotes is None:
        return []
    if isinstance(quotes, list):
        return quotes
    return [quotes]


def _quotes_by_symbol(quotes: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for q in quotes:
        sym = q.get("symbol")
        if sym:
            out[sym] = q
    return out


@dataclass(frozen=True)
class QuoteJob:
    tradier: TradierClient
    clock_cache: MarketClockCache | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    async def run_once(self, *, force: bool = False) -> dict:
        tz = ZoneInfo(settings.tz)
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
        except Exception:
            logger.warning("quote_job: failed to fetch quotes for {}", quote_symbols)
            return {"skipped": True, "reason": "quote_fetch_failed", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        if not quotes:
            return {"skipped": True, "reason": "no_quotes", "now_et": now_et.isoformat(), "quotes_inserted": 0}

        by_symbol = _quotes_by_symbol(quotes)
        quotes_inserted = 0

        async with SessionLocal() as session:
            for q in quotes:
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
                        "ts": now_et.astimezone(ZoneInfo("UTC")),
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

            # Derive context snapshot fields from quotes.
            vix = by_symbol.get("VIX", {}).get("last")
            vix9d = by_symbol.get("VIX9D", {}).get("last")
            spx_price = by_symbol.get("SPX", {}).get("last")
            spy_price = by_symbol.get("SPY", {}).get("last")
            term_structure = None
            if vix and vix9d and vix > 0:
                term_structure = vix9d / vix

            await session.execute(
                text(
                    """
                    INSERT INTO context_snapshots (ts, spx_price, spy_price, vix, vix9d, term_structure, notes_json)
                    VALUES (:ts, :spx_price, :spy_price, :vix, :vix9d, :term_structure, CAST(:notes AS jsonb))
                    ON CONFLICT (ts) DO UPDATE SET
                      spx_price = EXCLUDED.spx_price,
                      spy_price = EXCLUDED.spy_price,
                      vix = EXCLUDED.vix,
                      vix9d = EXCLUDED.vix9d,
                      term_structure = EXCLUDED.term_structure,
                      notes_json = EXCLUDED.notes_json
                    """
                ),
                {
                    "ts": now_et.astimezone(ZoneInfo("UTC")),
                    "spx_price": spx_price,
                    "spy_price": spy_price,
                    "vix": vix,
                    "vix9d": vix9d,
                    "term_structure": term_structure,
                    "notes": json.dumps({"source": "tradier", "symbols": quote_symbols}),
                },
            )

            await session.commit()

        logger.info("quote_job: inserted_quotes={}", quotes_inserted)
        return {"skipped": False, "reason": None, "now_et": now_et.isoformat(), "quotes_inserted": quotes_inserted}


def build_quote_job(clock_cache: MarketClockCache | None = None, tradier: TradierClient | None = None) -> QuoteJob:
    client = tradier or get_tradier_client()
    return QuoteJob(tradier=client, clock_cache=clock_cache)
