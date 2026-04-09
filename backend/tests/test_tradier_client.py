from __future__ import annotations

from typing import Any

import pytest

from spx_backend.ingestion.tradier_client import TradierClient


class _FakeResponse:
    """Configurable fake HTTP response for tradier client tests."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self._payload = payload or {"expirations": {"date": ["2026-02-13"]}}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    """Captures HTTP calls and returns a configurable fake response."""

    def __init__(self, *args, response_payload: dict[str, Any] | None = None, **kwargs):
        self.calls: list[dict[str, Any]] = []
        self._response_payload = response_payload

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "headers": headers})
        return _FakeResponse(self._response_payload)


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


@pytest.mark.asyncio
async def test_get_option_chain_sends_correct_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_option_chain passes underlying, expiration, and greeks params."""
    chain_payload = {"options": {"option": [{"symbol": "SPXW_OPT"}]}}
    holder: dict[str, Any] = {}

    def _factory(*args, **kwargs) -> _FakeAsyncClient:
        client = _FakeAsyncClient(*args, response_payload=chain_payload, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("spx_backend.ingestion.tradier_client.httpx.AsyncClient", _factory)

    tc = TradierClient(base_url="https://api.tradier.com/v1", token="tok")
    payload = await tc.get_option_chain("SPX", "2026-04-17", greeks=True)

    assert payload == chain_payload
    call = holder["client"].calls[0]
    assert call["url"].endswith("/markets/options/chains")
    assert call["params"]["symbol"] == "SPX"
    assert call["params"]["expiration"] == "2026-04-17"
    assert call["params"]["greeks"] == "true"


@pytest.mark.asyncio
async def test_get_option_chain_greeks_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_option_chain passes greeks=false when requested."""
    holder: dict[str, Any] = {}

    def _factory(*args, **kwargs) -> _FakeAsyncClient:
        client = _FakeAsyncClient(*args, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("spx_backend.ingestion.tradier_client.httpx.AsyncClient", _factory)

    tc = TradierClient(base_url="https://api.tradier.com/v1", token="tok")
    await tc.get_option_chain("SPX", "2026-04-17", greeks=False)

    assert holder["client"].calls[0]["params"]["greeks"] == "false"


@pytest.mark.asyncio
async def test_get_quotes_joins_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_quotes comma-joins multiple symbols in the request."""
    quotes_payload = {"quotes": {"quote": [{"symbol": "SPX"}, {"symbol": "VIX"}]}}
    holder: dict[str, Any] = {}

    def _factory(*args, **kwargs) -> _FakeAsyncClient:
        client = _FakeAsyncClient(*args, response_payload=quotes_payload, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("spx_backend.ingestion.tradier_client.httpx.AsyncClient", _factory)

    tc = TradierClient(base_url="https://api.tradier.com/v1", token="tok")
    payload = await tc.get_quotes(["SPX", "VIX"])

    assert payload == quotes_payload
    call = holder["client"].calls[0]
    assert call["url"].endswith("/markets/quotes")
    assert call["params"]["symbols"] == "SPX,VIX"


@pytest.mark.asyncio
async def test_get_market_clock_sends_no_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_market_clock hits the clock endpoint without query params."""
    clock_payload = {"clock": {"state": "open", "timestamp": 1700000000}}
    holder: dict[str, Any] = {}

    def _factory(*args, **kwargs) -> _FakeAsyncClient:
        client = _FakeAsyncClient(*args, response_payload=clock_payload, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("spx_backend.ingestion.tradier_client.httpx.AsyncClient", _factory)

    tc = TradierClient(base_url="https://api.tradier.com/v1", token="tok")
    payload = await tc.get_market_clock()

    assert payload == clock_payload
    call = holder["client"].calls[0]
    assert call["url"].endswith("/markets/clock")


@pytest.mark.asyncio
async def test_auth_header_uses_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """All requests include the Bearer token in the Authorization header."""
    holder: dict[str, Any] = {}

    def _factory(*args, **kwargs) -> _FakeAsyncClient:
        client = _FakeAsyncClient(*args, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("spx_backend.ingestion.tradier_client.httpx.AsyncClient", _factory)

    tc = TradierClient(base_url="https://api.tradier.com/v1", token="secret-token")
    await tc.get_market_clock()

    headers = holder["client"].calls[0]["headers"]
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["Accept"] == "application/json"
