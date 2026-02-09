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
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_option_expirations(self, underlying: str) -> dict[str, Any]:
        url = f"{self.base_url}/markets/options/expirations"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params={"symbol": underlying}, headers=self._headers)
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_option_chain(self, underlying: str, expiration: str, greeks: bool = True) -> dict[str, Any]:
        url = f"{self.base_url}/markets/options/chains"
        params = {"symbol": underlying, "expiration": expiration, "greeks": "true" if greeks else "false"}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(url, params=params, headers=self._headers)
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_quotes(self, symbols: list[str]) -> dict[str, Any]:
        url = f"{self.base_url}/markets/quotes"
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, params={"symbols": ",".join(symbols)}, headers=self._headers)
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
    async def get_market_clock(self) -> dict[str, Any]:
        url = f"{self.base_url}/markets/clock"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            return r.json()


def get_tradier_client() -> TradierClient:
    return TradierClient(base_url=settings.tradier_base_url, token=settings.tradier_access_token)

