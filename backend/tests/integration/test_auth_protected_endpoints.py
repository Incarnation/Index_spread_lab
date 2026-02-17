"""
Integration tests: auth login/register with real DB, and protected endpoints return 401 without JWT.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI
from passlib.context import CryptContext
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import get_db_session
from spx_backend.web.routers import admin, auth, public

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def _execute_sql_file(engine, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.exec_driver_sql(stmt)


@pytest.fixture
async def integration_client_with_auth(integration_db_session, database_url_test, monkeypatch):
    """Build FastAPI app with auth, public, admin and DB override; seed one user."""
    from sqlalchemy.ext.asyncio import create_async_engine

    monkeypatch.setattr(settings, "jwt_secret", "integration-test-jwt-secret")
    monkeypatch.setattr(settings, "auth_register_enabled", True)

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(public.router)
    app.include_router(admin.router)

    async def _override_db():
        yield integration_db_session

    app.dependency_overrides[get_db_session] = _override_db

    # Seed user for login test
    await integration_db_session.execute(
        text(
            "INSERT INTO users (username, password_hash) VALUES (:u, :h)"
        ),
        {"u": "testuser", "h": pwd_ctx.hash("testpass123")},
    )
    await integration_db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_protected_endpoint_returns_401_without_token(integration_client_with_auth: AsyncClient) -> None:
    """GET /api/chain-snapshots without Authorization returns 401."""
    r = await integration_client_with_auth.get("/api/chain-snapshots")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_health_remains_public(integration_client_with_auth: AsyncClient) -> None:
    """GET /health does not require auth."""
    r = await integration_client_with_auth.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_login_success_returns_token(integration_client_with_auth: AsyncClient) -> None:
    """POST /api/auth/login with valid credentials returns 200 and access_token."""
    r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data.get("token_type") == "bearer"
    assert data.get("user", {}).get("username") == "testuser"


@pytest.mark.asyncio
async def test_protected_endpoint_200_with_bearer_token(integration_client_with_auth: AsyncClient) -> None:
    """After login, GET /api/chain-snapshots with Bearer token returns 200."""
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]

    r = await integration_client_with_auth.get(
        "/api/chain-snapshots",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_register_success_then_login(integration_client_with_auth: AsyncClient) -> None:
    """Register new user then login works."""
    r = await integration_client_with_auth.post(
        "/api/auth/register",
        json={"username": "newuser", "password": "newpass123"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("user", {}).get("username") == "newuser"
    assert "access_token" in data

    r2 = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "newuser", "password": "newpass123"},
    )
    assert r2.status_code == 200
    assert r2.json().get("user", {}).get("username") == "newuser"
