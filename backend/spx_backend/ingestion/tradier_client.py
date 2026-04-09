from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from spx_backend.config import settings


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient HTTP errors (5xx, 429) and network failures."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


_RETRY_POLICY = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception(_is_retryable),
)


@dataclass(eq=False)
class TradierClient:
    """HTTP client for Tradier REST APIs with connection pooling.

    A single ``httpx.AsyncClient`` is reused across all requests so TCP
    connections and TLS sessions are kept alive, reducing latency on the
    high-frequency snapshot/quote/GEX polling paths.
    """

    base_url: str
    token: str
    _client: httpx.AsyncClient = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            },
            timeout=30,
        )

    @retry(**_RETRY_POLICY)
    async def get_option_expirations(self, underlying: str) -> dict[str, Any]:
        """Fetch available expirations for an underlying symbol."""
        r = await self._client.get(
            "/markets/options/expirations",
            params={"symbol": underlying, "includeAllRoots": "true", "strikes": "false"},
        )
        r.raise_for_status()
        return r.json()

    @retry(**_RETRY_POLICY)
    async def get_option_chain(self, underlying: str, expiration: str, greeks: bool = True) -> dict[str, Any]:
        """Fetch one option chain snapshot for the given symbol/expiration."""
        r = await self._client.get(
            "/markets/options/chains",
            params={"symbol": underlying, "expiration": expiration, "greeks": "true" if greeks else "false"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    @retry(**_RETRY_POLICY)
    async def get_quotes(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch quote payload for one or more comma-joined symbols."""
        r = await self._client.get(
            "/markets/quotes",
            params={"symbols": ",".join(symbols)},
        )
        r.raise_for_status()
        return r.json()

    @retry(**_RETRY_POLICY)
    async def get_market_clock(self) -> dict[str, Any]:
        """Fetch current market clock status from Tradier."""
        r = await self._client.get("/markets/clock", timeout=15)
        r.raise_for_status()
        return r.json()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()


def get_tradier_client() -> TradierClient:
    """Construct a Tradier client from environment-backed application settings."""
    return TradierClient(base_url=settings.tradier_base_url, token=settings.tradier_access_token)
