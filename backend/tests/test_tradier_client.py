from __future__ import annotations

from typing import Any

import pytest

from spx_backend.ingestion.tradier_client import TradierClient


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"expirations": {"date": ["2026-02-13"]}}


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, params: dict[str, Any], headers: dict[str, str]) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        return _FakeResponse()


@pytest.mark.asyncio
async def test_get_option_expirations_requests_all_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: dict[str, Any] = {}

    def _factory(*args, **kwargs) -> _FakeAsyncClient:
        client = _FakeAsyncClient(*args, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("spx_backend.ingestion.tradier_client.httpx.AsyncClient", _factory)

    client = TradierClient(base_url="https://api.tradier.com/v1", token="token")
    payload = await client.get_option_expirations("SPX")

    assert payload["expirations"]["date"] == ["2026-02-13"]
    call = holder["client"].calls[0]
    assert call["url"].endswith("/markets/options/expirations")
    assert call["params"]["symbol"] == "SPX"
    assert call["params"]["includeAllRoots"] == "true"
    assert call["params"]["strikes"] == "false"
