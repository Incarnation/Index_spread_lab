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
class MzDataClient:
    """HTTP client for precomputed options exposure payloads from MZData.

    A single ``httpx.AsyncClient`` is reused for connection pooling.
    """

    base_url: str
    _client: httpx.AsyncClient = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers={"Accept": "application/json"},
            timeout=45,
        )

    @retry(**_RETRY_POLICY)
    async def get_live_option_exposure(self, symbol: str) -> dict[str, Any]:
        """Fetch one live precomputed exposure payload for a symbol.

        Parameters
        ----------
        symbol:
            Underlying symbol expected by the MZData exposure endpoint.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response with top-level fields like ``data``,
            ``spotPrice``, and ``timestamp``.
        """
        normalized_symbol = symbol.strip().upper()
        response = await self._client.get(f"/api/options/{normalized_symbol}/exposure")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("mzdata exposure response must be a JSON object")
        return payload

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()


def get_mzdata_client() -> MzDataClient:
    """Construct an MZData client from environment-backed settings."""
    return MzDataClient(base_url=settings.mzdata_base_url)
