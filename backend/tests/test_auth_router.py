"""
Unit tests for auth router: login, register, and get_current_user behavior.

Uses a minimal FastAPI app with auth router and mocked or overridden dependencies
to test success/failure paths without a real DB in unit mode.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from passlib.context import CryptContext

from spx_backend.config import settings
from spx_backend.web.routers import auth

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


@pytest.fixture
def app_with_auth(monkeypatch) -> FastAPI:
    """FastAPI app with auth router only; JWT_SECRET set for token issuance."""
    monkeypatch.setattr(settings, "jwt_secret", "test-secret-for-unit-tests")
    monkeypatch.setattr(settings, "auth_register_enabled", True)
    app = FastAPI()
    app.include_router(auth.router)
    return app


@pytest.fixture
def client(app_with_auth: FastAPI) -> TestClient:
    return TestClient(app_with_auth)


@pytest.mark.e2e
def test_login_returns_401_when_no_user_in_db(client: TestClient, monkeypatch) -> None:
    """Login with unknown username returns 401."""
    from spx_backend.database import get_db_session

    async def _fake_session():
        session = AsyncMock()
        result = MagicMock()
        result.fetchone.return_value = None
        session.execute = AsyncMock(return_value=result)
        yield session

    app = client.app
    app.dependency_overrides[get_db_session] = _fake_session

    r = client.post(
        "/api/auth/login",
        json={"username": "nobody", "password": "anything"},
    )
    assert r.status_code == 401
    assert "Invalid" in (r.json().get("detail") or "")


@pytest.mark.e2e
def test_register_returns_409_when_username_taken(client: TestClient, monkeypatch) -> None:
    """Register with existing username returns 409."""
    from sqlalchemy.exc import IntegrityError

    from spx_backend.database import get_db_session

    async def _fake_session():
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=IntegrityError("", "", "", ""))
        session.commit = AsyncMock()
        yield session

    app = client.app
    app.dependency_overrides[get_db_session] = _fake_session

    r = client.post(
        "/api/auth/register",
        json={"username": "taken", "password": "password123"},
    )
    assert r.status_code == 409


@pytest.mark.e2e
def test_register_validation_rejects_short_password(client: TestClient) -> None:
    """Register with password shorter than 8 chars returns 422."""
    r = client.post(
        "/api/auth/register",
        json={"username": "user", "password": "short"},
    )
    assert r.status_code == 422


@pytest.mark.e2e
def test_me_returns_401_without_auth_header(client: TestClient) -> None:
    """GET /api/auth/me without Authorization returns 401."""
    r = client.get("/api/auth/me")
    assert r.status_code == 401


@pytest.mark.e2e
def test_me_returns_401_with_invalid_token(client: TestClient) -> None:
    """GET /api/auth/me with invalid JWT returns 401."""
    r = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalid.jwt.token"},
    )
    assert r.status_code == 401
