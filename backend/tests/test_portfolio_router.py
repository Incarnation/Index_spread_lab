"""
Unit tests for the portfolio router endpoints.

Uses a minimal FastAPI app with mocked DB sessions and overridden auth
to test the four portfolio API endpoints without a live database.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spx_backend.config import settings
from spx_backend.database import get_db_session
from spx_backend.web.routers import portfolio
from spx_backend.web.routers.auth import get_current_user


def _fake_user():
    """Override auth dependency to always return a valid user."""
    return {"username": "test"}


def _make_app(monkeypatch) -> FastAPI:
    """Create a minimal FastAPI app with portfolio router and no-auth override."""
    monkeypatch.setattr(settings, "portfolio_starting_capital", 20_000.0)
    monkeypatch.setattr(settings, "portfolio_max_trades_per_day", 2)
    monkeypatch.setattr(settings, "portfolio_max_trades_per_run", 1)
    monkeypatch.setattr(settings, "portfolio_monthly_drawdown_limit", 0.15)
    monkeypatch.setattr(settings, "portfolio_lot_per_equity", 10_000.0)
    monkeypatch.setattr(settings, "portfolio_max_equity_risk_pct", 0.05)
    monkeypatch.setattr(settings, "portfolio_max_margin_pct", 0.25)
    monkeypatch.setattr(settings, "portfolio_calls_only", True)
    monkeypatch.setattr(settings, "event_enabled", True)
    monkeypatch.setattr(settings, "event_budget_mode", "shared")
    monkeypatch.setattr(settings, "event_max_trades", 1)
    monkeypatch.setattr(settings, "event_spx_drop_threshold", -0.005)
    monkeypatch.setattr(settings, "event_spx_drop_2d_threshold", -0.01)
    monkeypatch.setattr(settings, "event_vix_spike_threshold", 0.15)
    monkeypatch.setattr(settings, "event_vix_elevated_threshold", 25.0)
    monkeypatch.setattr(settings, "event_term_inversion_threshold", 1.15)
    monkeypatch.setattr(settings, "event_side_preference", "puts")
    monkeypatch.setattr(settings, "event_min_dte", 3)
    monkeypatch.setattr(settings, "event_max_dte", 7)
    monkeypatch.setattr(settings, "event_min_delta", 0.10)
    monkeypatch.setattr(settings, "event_max_delta", 0.25)
    monkeypatch.setattr(settings, "event_rally_avoidance", True)
    monkeypatch.setattr(settings, "event_rally_threshold", 0.01)
    monkeypatch.setattr(settings, "decision_entry_times", "10:02,11:02,12:02")
    monkeypatch.setattr(settings, "decision_dte_targets", "0,3,5")
    monkeypatch.setattr(settings, "decision_delta_targets", "0.10,0.20")
    monkeypatch.setattr(settings, "decision_spread_width_points", 5)
    app = FastAPI()
    app.include_router(portfolio.router)
    app.dependency_overrides[get_current_user] = _fake_user
    return app


def _mock_session_with_state():
    """Return a mock async session that provides a portfolio_state row."""
    session = AsyncMock()

    state_row = SimpleNamespace(
        id=1, date=date(2026, 4, 5), equity_start=20_000.0, equity_end=20_500.0,
        month_start_equity=20_000.0, trades_placed=1, lots_per_trade=2,
        daily_pnl=500.0, monthly_stop_active=False, event_signals=None,
    )
    trade_count_row = SimpleNamespace()

    state_result = MagicMock()
    state_result.fetchone.return_value = state_row

    count_result = MagicMock()
    count_result.fetchone.return_value = (1,)

    session.execute = AsyncMock(side_effect=[state_result, count_result])
    return session


def _mock_session_empty():
    """Return a mock session that returns no rows."""
    session = AsyncMock()
    result = MagicMock()
    result.fetchone.return_value = None
    result.fetchall.return_value = []
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.fixture
def client_with_state(monkeypatch) -> TestClient:
    """Client where portfolio_state has a row."""
    app = _make_app(monkeypatch)

    async def _session():
        yield _mock_session_with_state()

    app.dependency_overrides[get_db_session] = _session
    return TestClient(app)


@pytest.fixture
def client_empty(monkeypatch) -> TestClient:
    """Client where portfolio tables are empty."""
    app = _make_app(monkeypatch)

    async def _session():
        yield _mock_session_empty()

    app.dependency_overrides[get_db_session] = _session
    return TestClient(app)


# ── GET /api/portfolio/config ──────────────────────────────────────────

class TestPortfolioConfig:
    """Tests for the /api/portfolio/config endpoint."""

    def test_returns_portfolio_section(self, client_empty: TestClient) -> None:
        """Config response includes portfolio, event, and decision sections."""
        r = client_empty.get("/api/portfolio/config")
        assert r.status_code == 200
        data = r.json()
        assert "portfolio" in data
        assert "event" in data
        assert "decision" in data

    def test_portfolio_fields(self, client_empty: TestClient) -> None:
        """Portfolio section contains expected config keys.

        ``enabled`` was dropped from the response when ``portfolio_enabled``
        was removed (the live decision job is always portfolio-managed).
        """
        data = client_empty.get("/api/portfolio/config").json()
        p = data["portfolio"]
        assert "enabled" not in p
        assert p["starting_capital"] == 20_000
        assert p["max_trades_per_day"] == 2
        assert p["max_trades_per_run"] == 1
        assert p["monthly_drawdown_limit"] == 0.15
        assert p["calls_only"] is True

    def test_event_fields(self, client_empty: TestClient) -> None:
        """Event section has expected threshold values."""
        data = client_empty.get("/api/portfolio/config").json()
        e = data["event"]
        assert e["enabled"] is True
        assert e["budget_mode"] == "shared"
        assert e["spx_drop_threshold"] == -0.005
        assert e["rally_avoidance"] is True

    def test_decision_fields(self, client_empty: TestClient) -> None:
        """Decision section has entry times, DTE targets, and delta targets."""
        data = client_empty.get("/api/portfolio/config").json()
        d = data["decision"]
        assert "10:02" in d["entry_times"]
        assert d["spread_width_points"] == 5


# ── GET /api/portfolio/status ──────────────────────────────────────────

class TestPortfolioStatus:
    """Tests for the /api/portfolio/status endpoint."""

    @patch("spx_backend.web.routers.portfolio.EventSignalDetector")
    def test_returns_status_with_data(self, mock_detector_cls, client_with_state: TestClient) -> None:
        """Status endpoint returns equity, lots, and trade counts when data exists."""
        mock_instance = MagicMock()
        mock_instance.detect = AsyncMock(return_value=["spx_drop_1d"])
        mock_detector_cls.return_value = mock_instance

        r = client_with_state.get("/api/portfolio/status")
        assert r.status_code == 200
        data = r.json()
        assert data["equity"] == 20_500.0
        assert data["lots_per_trade"] == 2
        assert data["trades_today"] == 1
        assert data["max_trades_per_day"] == 2
        assert data["max_trades_per_run"] == 1
        assert data["monthly_stop_active"] is False
        # ``portfolio_enabled`` was removed from the response when the
        # config flag was deleted; assert explicitly that no consumer
        # accidentally re-adds it.
        assert "portfolio_enabled" not in data
        assert "spx_drop_1d" in data["event_signals"]

    @patch("spx_backend.web.routers.portfolio.EventSignalDetector")
    def test_returns_defaults_when_empty(self, mock_detector_cls, client_empty: TestClient) -> None:
        """Status returns starting capital defaults when no portfolio_state row exists."""
        mock_instance = MagicMock()
        mock_instance.detect = AsyncMock(return_value=[])
        mock_detector_cls.return_value = mock_instance

        r = client_empty.get("/api/portfolio/status")
        assert r.status_code == 200
        data = r.json()
        assert data["equity"] == 20_000.0
        assert data["event_signals"] == []
        assert data["monthly_stop_active"] is False


# ── GET /api/portfolio/history ─────────────────────────────────────────

class TestPortfolioHistory:
    """Tests for the /api/portfolio/history endpoint."""

    def test_returns_empty_items(self, client_empty: TestClient) -> None:
        """History returns an empty items list when no data exists."""
        r = client_empty.get("/api/portfolio/history?days=30")
        assert r.status_code == 200
        data = r.json()
        assert data["days"] == 30
        assert data["items"] == []

    def test_history_with_rows(self, monkeypatch) -> None:
        """History returns formatted rows from portfolio_state."""
        app = _make_app(monkeypatch)
        session = AsyncMock()
        result = MagicMock()
        result.fetchall.return_value = [
            SimpleNamespace(
                date=date(2026, 4, 4), equity_start=20_000, equity_end=20_300,
                trades_placed=1, lots_per_trade=2, daily_pnl=300,
                monthly_stop_active=False, event_signals=None,
            ),
            SimpleNamespace(
                date=date(2026, 4, 5), equity_start=20_300, equity_end=20_500,
                trades_placed=2, lots_per_trade=2, daily_pnl=200,
                monthly_stop_active=False, event_signals=None,
            ),
        ]
        session.execute = AsyncMock(return_value=result)

        async def _session():
            yield session

        app.dependency_overrides[get_db_session] = _session
        client = TestClient(app)

        r = client.get("/api/portfolio/history?days=7")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 2
        assert items[0]["date"] == "2026-04-04"
        assert items[1]["equity_end"] == 20_500


# ── GET /api/portfolio/trades ──────────────────────────────────────────

class TestPortfolioTrades:
    """Tests for the /api/portfolio/trades endpoint."""

    def test_returns_empty_items(self, client_empty: TestClient) -> None:
        """Trades returns empty items when no data exists."""
        r = client_empty.get("/api/portfolio/trades")
        assert r.status_code == 200
        data = r.json()
        assert data["items"] == []
        assert data["limit"] == 100

    def test_source_filter_param(self, client_empty: TestClient) -> None:
        """Source filter param is echoed in response."""
        r = client_empty.get("/api/portfolio/trades?source=event")
        assert r.status_code == 200
        assert r.json()["source_filter"] == "event"

    def test_invalid_source_ignored(self, client_empty: TestClient) -> None:
        """Invalid source values are passed through but do not filter results."""
        r = client_empty.get("/api/portfolio/trades?source=invalid")
        assert r.status_code == 200
        assert r.json()["source_filter"] == "invalid"
        assert r.json()["items"] == []

    def test_custom_limit(self, client_empty: TestClient) -> None:
        """Custom limit is applied and echoed."""
        r = client_empty.get("/api/portfolio/trades?limit=50")
        assert r.status_code == 200
        assert r.json()["limit"] == 50
