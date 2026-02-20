from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from spx_backend.config import settings


@dataclass(frozen=True)
class MzDataClient:
    """HTTP client for precomputed options exposure payloads from MZData."""

    base_url: str

    @property
    def _headers(self) -> dict[str, str]:
        """Build the shared request headers for MZData calls."""
        return {"Accept": "application/json"}

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=0.5, min=0.5, max=8))
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
        url = f"{self.base_url.rstrip('/')}/api/options/{normalized_symbol}/exposure"
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(url, headers=self._headers)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("mzdata exposure response must be a JSON object")
            return payload


def get_mzdata_client() -> MzDataClient:
    """Construct an MZData client from environment-backed settings."""
    return MzDataClient(base_url=settings.mzdata_base_url)
