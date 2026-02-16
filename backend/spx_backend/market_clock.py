from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import json

from loguru import logger

from sqlalchemy import text

from spx_backend.database import SessionLocal
from spx_backend.ingestion.tradier_client import TradierClient


def is_rth(now_et: datetime) -> bool:
    """Return True when timestamp falls inside regular US equity trading hours."""
    # MVP: Monday-Friday 09:30-16:00 ET.
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    return (t >= datetime.strptime("09:30", "%H:%M").time()) and (t <= datetime.strptime("16:00", "%H:%M").time())


@dataclass
class MarketClockCache:
    tradier: TradierClient
    ttl_seconds: int = 300
    _cached_at: datetime | None = None
    _cached_open: bool | None = None
    _cached_state: str | None = None

    async def _record(self, now_et: datetime, state: str | None, is_open: bool | None, raw: dict | None, error: str | None) -> None:
        """Persist one market-clock audit row for observability and troubleshooting."""
        try:
            async with SessionLocal() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO market_clock_audit (ts, state, is_open, source, raw_json, error)
                        VALUES (:ts, :state, :is_open, :source, CAST(:raw_json AS jsonb), :error)
                        """
                    ),
                    {
                        "ts": now_et.astimezone(ZoneInfo("UTC")),
                        "state": state,
                        "is_open": is_open,
                        "source": "tradier",
                        "raw_json": None if raw is None else json.dumps(raw, default=str),
                        "error": error,
                    },
                )
                await session.commit()
        except Exception:
            logger.warning("market_clock: failed to record audit row")

    async def is_open(self, now_et: datetime) -> bool:
        """Return exchange open/closed status with TTL cache and fallback behavior."""
        if self._cached_at and self._cached_open is not None:
            if (now_et - self._cached_at).total_seconds() < self.ttl_seconds:
                return self._cached_open

        try:
            clock = await self.tradier.get_market_clock()
            state = (clock.get("clock", {}) or {}).get("state")
            is_open = isinstance(state, str) and state.lower() == "open"
            self._cached_at = now_et
            self._cached_open = is_open
            self._cached_state = state if isinstance(state, str) else None
            await self._record(now_et, self._cached_state, is_open, clock, None)
            return is_open
        except Exception:
            logger.warning("market_clock: failed to fetch; using cached or RTH")
            await self._record(now_et, None, None, None, "fetch_failed")
            if self._cached_open is not None:
                return self._cached_open
            return is_rth(now_et)
