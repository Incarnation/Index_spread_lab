"""Tests for monitoring endpoints and job-failure alerting.

Covers:
- Deep ``/health`` endpoint (DB, scheduler, freshness)
- ``/api/pipeline-status`` endpoint (authenticated, sanitized freshness)
- ``_scheduler_event_listener`` and ``_send_job_failure_email`` alert logic
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spx_backend.config import settings
from spx_backend.web.routers import auth, public


# ── Fake DB helpers ────────────────────────────────────────────


class _FakeResult:
    """Minimal SQLAlchemy-like result wrapper."""

    def __init__(self, value: Any = None):
        self._value = value

    def scalar(self) -> Any:
        return self._value

    def fetchone(self) -> SimpleNamespace | None:
        if self._value is None:
            return None
        return SimpleNamespace(latest=self._value)

    def fetchall(self) -> list:
        return []


class _FakeSession:
    """Async context-manager session for monkeypatching SessionLocal."""

    def __init__(self, results: dict[str, Any] | None = None, *, fail: bool = False):
        self._results = results or {}
        self._fail = fail

    async def execute(self, stmt, params=None):
        if self._fail:
            raise ConnectionError("DB unreachable")
        sql = str(stmt)
        if "SELECT 1" in sql:
            return _FakeResult(1)
        for key, value in self._results.items():
            if key in sql:
                return _FakeResult(value)
        return _FakeResult(None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _session_factory(session: _FakeSession):
    """Return a callable matching SessionLocal() that yields a context manager."""
    @asynccontextmanager
    async def _factory():
        yield session
    return _factory


class _FakeScheduler:
    """Minimal scheduler stub with a ``running`` attribute."""
    def __init__(self, *, running: bool = True):
        self.running = running


# ── Test client builder ────────────────────────────────────────


def _build_client(monkeypatch, *, db_fail=False, scheduler_running=True, freshness=None):
    """Build a TestClient with mocked DB and scheduler.

    Parameters
    ----------
    db_fail : bool
        When True, all DB operations raise ConnectionError.
    scheduler_running : bool
        Value for the scheduler stub's ``running`` attribute.
    freshness : dict | None
        Maps table substrings to datetime values for MAX(ts) queries.
    """
    monkeypatch.setattr(settings, "jwt_secret", "test-monitoring-secret")

    fake_session = _FakeSession(results=freshness or {}, fail=db_fail)
    monkeypatch.setattr(public, "SessionLocal", _session_factory(fake_session))

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(public.router)

    app.state.scheduler = _FakeScheduler(running=scheduler_running)

    fake_db_session = _FakeSession(results=freshness or {}, fail=db_fail)

    async def _override_db():
        yield fake_db_session

    app.dependency_overrides[public.get_db_session] = _override_db

    return TestClient(app)


def _auth_headers(client: TestClient) -> dict[str, str]:
    """Return JWT auth headers using a fake user lookup.

    Patches the auth session to return a testuser so login succeeds.
    """
    from spx_backend.web.routers.auth import pwd_ctx

    class _AuthSession:
        async def execute(self, stmt, params=None):
            sql = str(stmt)
            if "password_hash" in sql:
                return _FakeResult(
                    SimpleNamespace(
                        id=1,
                        username="testuser",
                        password_hash=pwd_ctx.hash("testpass123"),
                    )
                )
            if "WHERE id" in sql:
                return _FakeResult(
                    SimpleNamespace(id=1, username="testuser", is_admin=False),
                )
            return _FakeResult(None)

        async def commit(self):
            pass

        def fetchone(self):
            return None

    class _AuthResult(_FakeResult):
        def fetchone(self):
            return self._value

    orig_execute = _AuthSession.execute

    async def _patched_execute(self, stmt, params=None):
        result = await orig_execute(self, stmt, params)
        if result._value is not None and isinstance(result._value, SimpleNamespace):
            return _AuthResult(result._value)
        return result

    _AuthSession.execute = _patched_execute

    auth_session = _AuthSession()

    async def _auth_db():
        yield auth_session

    client.app.dependency_overrides[auth.get_db_session] = _auth_db

    r = client.post("/api/auth/login", json={"username": "testuser", "password": "testpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ── /health tests ──────────────────────────────────────────────


class TestHealthEndpoint:
    """Test the deep /health endpoint."""

    def test_healthy_when_db_and_scheduler_ok(self, monkeypatch):
        fresh_ts = datetime.now(tz=timezone.utc)
        client = _build_client(
            monkeypatch,
            freshness={
                "underlying_quotes": fresh_ts,
                "chain_snapshots": fresh_ts,
                "trade_decisions": fresh_ts,
            },
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["checks"]["database"]["ok"] is True
        assert body["checks"]["scheduler"]["ok"] is True

    def test_unhealthy_when_db_down(self, monkeypatch):
        client = _build_client(monkeypatch, db_fail=True)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["checks"]["database"]["ok"] is False

    def test_degraded_when_scheduler_stopped(self, monkeypatch):
        client = _build_client(monkeypatch, scheduler_running=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["scheduler"]["ok"] is False

    def test_no_auth_required(self, monkeypatch):
        """Health endpoint should be accessible without JWT."""
        client = _build_client(monkeypatch)
        resp = client.get("/health")
        assert resp.status_code == 200


# ── /api/pipeline-status tests ─────────────────────────────────


class TestPipelineStatusEndpoint:
    """Test the authenticated pipeline-status endpoint."""

    def test_returns_freshness_for_authenticated_user(self, monkeypatch):
        fresh_ts = datetime.now(tz=timezone.utc)
        client = _build_client(
            monkeypatch,
            freshness={
                "underlying_quotes": fresh_ts,
                "chain_snapshots": fresh_ts,
                "gex_snapshots": fresh_ts,
                "trade_decisions": fresh_ts,
            },
        )
        headers = _auth_headers(client)
        resp = client.get("/api/pipeline-status", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert "freshness" in body
        assert "warnings" in body
        assert isinstance(body["warnings"], list)

    def test_401_without_auth(self, monkeypatch):
        client = _build_client(monkeypatch)
        resp = client.get("/api/pipeline-status")
        assert resp.status_code == 401


# ── Job failure alert tests ────────────────────────────────────


class TestJobFailureAlerts:
    """Test the scheduler event listener email alerts."""

    def setup_method(self):
        from spx_backend.web import app as app_module
        app_module._job_failure_last_alert.clear()

    def test_listener_calls_email_on_failure(self, monkeypatch):
        """Exception events should log error and call _send_job_failure_email with FAILURE."""
        from spx_backend.web import app as app_module
        from spx_backend.web.app import _scheduler_event_listener

        calls: list[dict] = []
        original_send = app_module._send_job_failure_email

        def _spy(*, job_id, kind, detail):
            calls.append({"job_id": job_id, "kind": kind, "detail": detail})

        monkeypatch.setattr(app_module, "_send_job_failure_email", _spy)

        event = SimpleNamespace(
            job_id="test_job",
            exception=RuntimeError("boom"),
            traceback="Traceback...",
        )
        _scheduler_event_listener(event)

        assert len(calls) == 1
        assert calls[0]["job_id"] == "test_job"
        assert calls[0]["kind"] == "FAILURE"
        assert "boom" in calls[0]["detail"]

    def test_listener_calls_email_on_misfire(self, monkeypatch):
        """Misfire events (no exception) should log warning and call _send_job_failure_email with MISFIRE."""
        from spx_backend.web import app as app_module
        from spx_backend.web.app import _scheduler_event_listener

        calls: list[dict] = []

        def _spy(*, job_id, kind, detail):
            calls.append({"job_id": job_id, "kind": kind, "detail": detail})

        monkeypatch.setattr(app_module, "_send_job_failure_email", _spy)

        event = SimpleNamespace(job_id="test_job", exception=None, traceback=None)
        _scheduler_event_listener(event)

        assert len(calls) == 1
        assert calls[0]["job_id"] == "test_job"
        assert calls[0]["kind"] == "MISFIRE"

    def test_email_sent_on_failure_when_configured(self, monkeypatch):
        """When SendGrid is configured, an email should be sent."""
        from spx_backend.web.app import _send_job_failure_email

        monkeypatch.setattr(settings, "job_failure_alert_enabled", True)
        monkeypatch.setattr(settings, "sendgrid_api_key", "SG.test")
        monkeypatch.setattr(settings, "email_alert_recipient", "user@test.com")
        monkeypatch.setattr(settings, "email_alert_sender", "alerts@test.com")

        mock_sg_class = MagicMock()
        mock_sg_instance = MagicMock()
        mock_sg_instance.send.return_value = SimpleNamespace(status_code=202)
        mock_sg_class.return_value = mock_sg_instance

        with patch.dict("sys.modules", {
            "sendgrid": MagicMock(SendGridAPIClient=mock_sg_class),
            "sendgrid.helpers.mail": MagicMock(),
        }):
            _send_job_failure_email(
                job_id="decision_job", kind="FAILURE", detail="Error: timeout",
            )

        mock_sg_instance.send.assert_called_once()

    def test_cooldown_prevents_spam(self, monkeypatch):
        """Second alert for the same job within cooldown should be suppressed."""
        from spx_backend.web.app import _send_job_failure_email, _job_failure_last_alert

        monkeypatch.setattr(settings, "job_failure_alert_enabled", True)
        monkeypatch.setattr(settings, "sendgrid_api_key", "SG.test")
        monkeypatch.setattr(settings, "email_alert_recipient", "user@test.com")
        monkeypatch.setattr(settings, "job_failure_alert_cooldown_minutes", 30)

        _job_failure_last_alert["decision_job"] = datetime.now(tz=timezone.utc)

        mock_sg_class = MagicMock()
        mock_sg_instance = MagicMock()
        mock_sg_class.return_value = mock_sg_instance

        with patch.dict("sys.modules", {
            "sendgrid": MagicMock(SendGridAPIClient=mock_sg_class),
            "sendgrid.helpers.mail": MagicMock(),
        }):
            _send_job_failure_email(
                job_id="decision_job", kind="FAILURE", detail="Error: timeout",
            )

        mock_sg_instance.send.assert_not_called()

    def test_disabled_skips_alert(self, monkeypatch):
        """When job_failure_alert_enabled is False, no email attempt."""
        from spx_backend.web.app import _send_job_failure_email

        monkeypatch.setattr(settings, "job_failure_alert_enabled", False)

        mock_sg_class = MagicMock()
        with patch.dict("sys.modules", {
            "sendgrid": MagicMock(SendGridAPIClient=mock_sg_class),
        }):
            _send_job_failure_email(
                job_id="quote_job", kind="MISFIRE", detail="Missed window",
            )

        mock_sg_class.assert_not_called()
