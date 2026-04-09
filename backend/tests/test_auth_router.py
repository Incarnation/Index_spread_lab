"""
Unit tests for auth router: login, register, and get_current_user behavior.

Uses a minimal FastAPI app with auth router and mocked or overridden dependencies
to test success/failure paths without a real DB in unit mode.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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


def test_bcrypt_truncation_short_password_unchanged() -> None:
    """Short password (<=72 bytes) is not modified by truncation."""
    short = "test_user"
    assert auth._truncate_password_for_bcrypt(short) == short
    assert len(auth._truncate_password_for_bcrypt(short).encode("utf-8")) <= 72


def test_bcrypt_truncation_long_password_truncated_to_72_bytes() -> None:
    """Password longer than 72 utf-8 bytes is truncated so bcrypt never raises."""
    # 80 ASCII chars = 80 bytes
    long_pass = "a" * 80
    truncated = auth._truncate_password_for_bcrypt(long_pass)
    assert len(truncated.encode("utf-8")) == 72
    assert truncated == "a" * 72


def test_hash_and_verify_long_password_roundtrip() -> None:
    """Hashing and verifying a password >72 bytes works; effective comparison is truncated."""
    long_pass = "password123" + "x" * 80
    hashed = auth._hash_password(long_pass)
    assert hashed.startswith("$2")
    assert auth._verify_password(long_pass, hashed) is True


def test_verify_truncation_consistency_same_first_72_bytes() -> None:
    """Two passwords that share the same first 72 bytes verify against the same hash."""
    prefix = "same_prefix_" + "b" * (72 - 12)  # 72 bytes total
    pass_a = prefix + "extra_a"
    pass_b = prefix + "extra_b"
    hashed = auth._hash_password(pass_a)
    assert auth._verify_password(pass_a, hashed) is True
    assert auth._verify_password(pass_b, hashed) is True
