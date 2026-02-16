from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from spx_backend.config import settings


@dataclass(frozen=True)
class TradierClient:
    base_url: str
    token: str

    @property
    def _headers(self) -> dict[str, str]:
        """Build shared HTTP headers required for all Tradier requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_option_expirations(self, underlying: str) -> dict[str, Any]:
        """Fetch available expirations for an underlying symbol."""
        url = f"{self.base_url}/markets/options/expirations"
        async with httpx.AsyncClient(timeout=30) as client:
            # includeAllRoots=true is required for SPX weeklies/daily expirations.
            r = await client.get(
                url,
                params={"symbol": underlying, "includeAllRoots": "true", "strikes": "false"},
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_option_chain(self, underlying: str, expiration: str, greeks: bool = True) -> dict[str, Any]:
        """Fetch one option chain snapshot for the given symbol/expiration."""
        url = f"{self.base_url}/markets/options/chains"
        params = {"symbol": underlying, "expiration": expiration, "greeks": "true" if greeks else "false"}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url, params=params, headers=self._headers)
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_quotes(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch quote payload for one or more comma-joined symbols."""
        url = f"{self.base_url}/markets/quotes"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params={"symbols": ",".join(symbols)}, headers=self._headers)
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_market_clock(self) -> dict[str, Any]:
        """Fetch current market clock status from Tradier."""
        url = f"{self.base_url}/markets/clock"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            return r.json()


def get_tradier_client() -> TradierClient:
    """Construct a Tradier client from environment-backed application settings."""
    return TradierClient(base_url=settings.tradier_base_url, token=settings.tradier_access_token)

