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

    # Seed users: testuser (admin for auth-audit tests) and otheruser (non-admin for 403 test).
    await integration_db_session.execute(
        text(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (:u, :h, :admin)"
        ),
        {"u": "testuser", "h": pwd_ctx.hash("testpass123"), "admin": True},
    )
    await integration_db_session.execute(
        text(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (:u, :h, :admin)"
        ),
        {"u": "otheruser", "h": pwd_ctx.hash("otherpass123"), "admin": False},
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
async def test_performance_analytics_endpoint_returns_401_without_token(
    integration_client_with_auth: AsyncClient,
) -> None:
    """GET /api/performance-analytics without Authorization returns 401."""
    r = await integration_client_with_auth.get("/api/performance-analytics")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_health_remains_public(integration_client_with_auth: AsyncClient) -> None:
    """GET /health does not require auth."""
    r = await integration_client_with_auth.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_login_success_returns_token(integration_client_with_auth: AsyncClient) -> None:
    """POST /api/auth/login with valid credentials returns 200 and access_token with is_admin."""
    r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data.get("token_type") == "bearer"
    assert data.get("user", {}).get("username") == "testuser"
    assert data.get("user", {}).get("is_admin") is True


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


@pytest.mark.asyncio
async def test_login_success_records_audit_event(integration_client_with_auth: AsyncClient) -> None:
    """After successful login, GET /api/admin/auth-audit (as admin) returns at least one login_success event."""
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]
    r = await integration_client_with_auth.get(
        "/api/admin/auth-audit",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    data = r.json()
    events = data.get("events") or []
    assert any(e.get("event_type") == "login_success" for e in events)


@pytest.mark.asyncio
async def test_login_failure_records_audit_event(integration_client_with_auth: AsyncClient) -> None:
    """Failed login records login_failure; admin can see it via auth-audit."""
    await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "wrongpassword"},
    )
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]
    r = await integration_client_with_auth.get(
        "/api/admin/auth-audit",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    events = r.json().get("events") or []
    assert any(e.get("event_type") == "login_failure" for e in events)


@pytest.mark.asyncio
async def test_logout_records_audit_event(integration_client_with_auth: AsyncClient) -> None:
    """POST /api/auth/logout records logout event; admin can see it via auth-audit."""
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]
    logout_r = await integration_client_with_auth.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert logout_r.status_code == 204
    login_r2 = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    token2 = login_r2.json()["access_token"]
    r = await integration_client_with_auth.get(
        "/api/admin/auth-audit",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert r.status_code == 200
    events = r.json().get("events") or []
    assert any(e.get("event_type") == "logout" for e in events)


@pytest.mark.asyncio
async def test_auth_audit_returns_403_for_non_admin(integration_client_with_auth: AsyncClient) -> None:
    """GET /api/admin/auth-audit as non-admin user returns 403."""
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "otheruser", "password": "otherpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]
    r = await integration_client_with_auth.get(
        "/api/admin/auth-audit",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_run_snapshot_returns_403_for_non_admin(integration_client_with_auth: AsyncClient) -> None:
    """POST /api/admin/run-snapshot should reject non-admin users."""
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "otheruser", "password": "otherpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]
    r = await integration_client_with_auth.post(
        "/api/admin/run-snapshot",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_preflight_returns_200_for_non_admin(integration_client_with_auth: AsyncClient) -> None:
    """GET /api/admin/preflight should stay available to authenticated non-admin users."""
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "otheruser", "password": "otherpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]
    r = await integration_client_with_auth.get(
        "/api/admin/preflight",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_auth_audit_returns_200_for_admin(integration_client_with_auth: AsyncClient) -> None:
    """GET /api/admin/auth-audit as admin returns 200 with total and events list."""
    login_r = await integration_client_with_auth.post(
        "/api/auth/login",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert login_r.status_code == 200
    token = login_r.json()["access_token"]
    r = await integration_client_with_auth.get(
        "/api/admin/auth-audit",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "events" in data
    assert isinstance(data["events"], list)
