from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from spx_backend.config import settings
from spx_backend.web.routers import admin, auth, public

pytestmark = pytest.mark.e2e


class _FakeExecResult:
    """Minimal SQLAlchemy-like result wrapper for router tests."""

    def __init__(self, rows: list[SimpleNamespace]):
        self._rows = rows

    def fetchall(self) -> list[SimpleNamespace]:
        return self._rows

    def fetchone(self) -> SimpleNamespace | None:
        return self._rows[0] if self._rows else None

    def scalar_one(self) -> Any:
        """Return one scalar-like value from the first stored row."""
        if not self._rows:
            raise AssertionError("Expected one scalar row but no rows were stored")
        first = self._rows[0]
        if isinstance(first, (int, float, str, bool)):
            return first
        if hasattr(first, "value"):
            return getattr(first, "value")
        if hasattr(first, "count"):
            return getattr(first, "count")
        raise AssertionError(f"Unable to read scalar from fake row type: {type(first)!r}")


class _RouterAwareSession:
    """Route-aware fake DB session serving deterministic endpoint fixtures."""

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.commits = 0

    async def execute(self, stmt, params=None):  # noqa: ANN001 - SQLAlchemy text object
        sql = str(stmt)
        query_params = params or {}
        self.calls.append((sql, query_params))

        # Auth: login and get_current_user lookups (e2e uses testuser / testpass123).
        if "FROM users" in sql:
            if "password_hash" in sql and "username" in sql:
                from spx_backend.web.routers.auth import pwd_ctx
                return _FakeExecResult(
                    [
                        SimpleNamespace(
                            id=1,
                            username="testuser",
                            password_hash=pwd_ctx.hash("testpass123"),
                        )
                    ]
                )
            if "id" in sql and "username" in sql and "WHERE id" in sql:
                return _FakeExecResult(
                    [SimpleNamespace(id=1, username="testuser", is_admin=True)]
                )

        if "SELECT COUNT(*) FROM auth_audit_log" in sql:
            return _FakeExecResult([1])

        if "FROM auth_audit_log" in sql and "ORDER BY occurred_at DESC" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        id=901,
                        event_type="login_success",
                        user_id=1,
                        username="testuser",
                        occurred_at=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        ip_address="73.83.187.53/32",
                        user_agent="unit-test-agent",
                        country="United States",
                        geo_json={"country": "United States", "regionName": "Washington"},
                        details=None,
                    )
                ]
            )

        if "FROM chain_snapshots" in sql and "checksum" in sql and "LIMIT :limit" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        snapshot_id=101,
                        ts=datetime(2026, 2, 1, 14, 30, tzinfo=timezone.utc),
                        underlying="SPX",
                        target_dte=3,
                        expiration=date(2026, 2, 5),
                        checksum="abc123",
                    ),
                    SimpleNamespace(
                        snapshot_id=102,
                        ts=datetime(2026, 2, 1, 14, 31, tzinfo=timezone.utc),
                        underlying="SPY",
                        target_dte=3,
                        expiration=date(2026, 2, 5),
                        checksum="spy123",
                    ),
                ]
            )

        if "FROM trade_decisions" in sql and "LIMIT :limit" in sql and "decision_source" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        decision_id=77,
                        ts=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        target_dte=3,
                        entry_slot=10,
                        delta_target=0.2,
                        decision="TRADE",
                        reason=None,
                        score=1.23,
                        chain_snapshot_id=101,
                        decision_source="rules",
                        ruleset_version="rules_v1",
                        chosen_legs_json={"short": {"symbol": "SPX_P_SHORT"}},
                        strategy_params_json={"spread_side": "put"},
                    )
                ]
            )

        if "FROM gex_snapshots" in sql and "ORDER BY ts DESC" in sql and "LIMIT :limit" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        snapshot_id=101,
                        ts=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        underlying="SPX",
                        spot_price=6020.0,
                        gex_net=1200.0,
                        gex_calls=1600.0,
                        gex_puts=-400.0,
                        gex_abs=2000.0,
                        zero_gamma_level=6000.0,
                        method="oi_gamma_spot",
                    )
                ]
            )

        if "SELECT DISTINCT gbes.dte_days" in sql:
            return _FakeExecResult([SimpleNamespace(dte_days=3), SimpleNamespace(dte_days=5)])

        if "SELECT DISTINCT gbes.expiration, gbes.dte_days" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(expiration=date(2026, 2, 5), dte_days=3),
                    SimpleNamespace(expiration=date(2026, 2, 7), dte_days=5),
                ]
            )

        if "FROM gex_by_expiry_strike" in sql and "GROUP BY gbes.strike" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(strike=6000.0, gex_net=100.0, gex_calls=150.0, gex_puts=-50.0),
                    SimpleNamespace(strike=6050.0, gex_net=80.0, gex_calls=130.0, gex_puts=-50.0),
                ]
            )

        if "FROM trades t" in sql and "legs_json" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        trade_id=501,
                        decision_id=77,
                        candidate_id=701,
                        feature_snapshot_id=801,
                        status="OPEN",
                        trade_source="live",
                        strategy_type="credit_spread_put",
                        underlying="SPX",
                        entry_time=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        exit_time=None,
                        last_mark_ts=datetime(2026, 2, 1, 15, 5, tzinfo=timezone.utc),
                        target_dte=3,
                        expiration=date(2026, 2, 5),
                        contracts=1,
                        contract_multiplier=100,
                        spread_width_points=25.0,
                        entry_credit=2.1,
                        current_exit_cost=1.3,
                        current_pnl=80.0,
                        realized_pnl=None,
                        max_profit=210.0,
                        max_loss=2290.0,
                        take_profit_target=105.0,
                        stop_loss_target=-210.0,
                        exit_reason=None,
                        mark_count=2,
                        legs_json=[
                            {"leg_index": 0, "option_symbol": "SPX_SHORT", "side": "STO", "qty": 1},
                            {"leg_index": 1, "option_symbol": "SPX_LONG", "side": "BTO", "qty": 1},
                        ],
                    )
                ]
            )

        if "FROM trade_candidates" in sql and "realized_pnl IS NOT NULL" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        ts=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        realized_pnl=40.0,
                        hit_tp50=True,
                        hit_tp100=False,
                        spread_side="put",
                        max_loss=8.0,
                        contracts=1,
                    ),
                    SimpleNamespace(
                        ts=datetime(2026, 2, 2, 15, 0, tzinfo=timezone.utc),
                        realized_pnl=-20.0,
                        hit_tp50=False,
                        hit_tp100=False,
                        spread_side="put",
                        max_loss=8.0,
                        contracts=1,
                    ),
                ]
            )

        if "FROM trade_candidates" in sql and "AS resolved_count" in sql and "GROUP BY spread_side" not in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        resolved_count=10,
                        tp50_count=6,
                        tp100_count=3,
                        avg_realized_pnl=12.5,
                    )
                ]
            )

        if "GROUP BY spread_side" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(spread_side="call", resolved_count=4, tp50_count=2, tp100_count=1, avg_realized_pnl=8.0),
                    SimpleNamespace(spread_side="put", resolved_count=6, tp50_count=4, tp100_count=2, avg_realized_pnl=15.5),
                ]
            )

        if "model_versions_count" in sql and "model_predictions_24h_count" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        model_versions_count=2,
                        training_runs_count=1,
                        model_predictions_count=20,
                        model_predictions_24h_count=5,
                        latest_prediction_ts=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                    )
                ]
            )

        if "FROM model_versions" in sql and "metrics_json" in sql and "LIMIT 1" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        model_version_id=11,
                        version="wf_20260201",
                        rollout_status="shadow",
                        is_active=False,
                        created_at=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        promoted_at=None,
                        metrics_json={"tp50_rate_test": 0.55, "expectancy_test": 11.0},
                    )
                ]
            )

        if "FROM training_runs tr" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        training_run_id=31,
                        model_version_id=11,
                        status="COMPLETED",
                        started_at=datetime(2026, 2, 1, 14, 0, tzinfo=timezone.utc),
                        finished_at=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        rows_train=500,
                        rows_test=120,
                        notes="ok",
                        metrics_json={"gate": {"passed": True}},
                    )
                ]
            )

        if "FROM model_versions" in sql and "AND is_active = true" in sql:
            return _FakeExecResult([])

        if "DELETE FROM trade_decisions" in sql:
            if int(query_params.get("decision_id", 0)) == 1:
                return _FakeExecResult([SimpleNamespace(decision_id=1)])
            return _FakeExecResult([])

        if "SELECT" in sql and "quotes_count" in sql and "latest_market_clock_ts" in sql:
            fresh_ts = datetime.now(tz=timezone.utc)
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        quotes_count=100,
                        snapshots_count=50,
                        chain_rows_count=1000,
                        gex_snapshots_count=40,
                        decisions_count=20,
                        feature_snapshots_count=60,
                        trade_candidates_count=120,
                        labeled_candidates_count=80,
                        model_versions_count=2,
                        training_runs_count=1,
                        model_predictions_count=20,
                        trades_count=10,
                        open_trades_count=4,
                        closed_trades_count=6,
                        latest_quote_ts=fresh_ts,
                        latest_snapshot_ts=fresh_ts,
                        latest_gex_ts=fresh_ts,
                        latest_decision_ts=fresh_ts,
                        latest_feature_ts=fresh_ts,
                        latest_candidate_ts=fresh_ts,
                        latest_model_version_ts=fresh_ts,
                        latest_training_run_ts=fresh_ts,
                        latest_prediction_ts=fresh_ts,
                        latest_trade_mark_ts=fresh_ts,
                        latest_trade_entry_ts=fresh_ts,
                        latest_market_clock_ts=fresh_ts,
                    )
                ]
            )

        if "FROM chain_snapshots" in sql and "ORDER BY ts DESC, snapshot_id DESC" in sql and "LIMIT 1" in sql:
            fresh_ts = datetime.now(tz=timezone.utc)
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        snapshot_id=102,
                        ts=fresh_ts,
                        target_dte=3,
                        expiration=date(2026, 2, 5),
                    )
                ]
            )

        if "FROM gex_snapshots" in sql and "ORDER BY ts DESC, snapshot_id DESC" in sql and "LIMIT 1" in sql:
            fresh_ts = datetime.now(tz=timezone.utc)
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        snapshot_id=101,
                        ts=fresh_ts,
                        gex_net=1200.0,
                        zero_gamma_level=6000.0,
                        method="oi_gamma_spot",
                    )
                ]
            )

        if "FROM trade_decisions" in sql and "ORDER BY ts DESC, decision_id DESC" in sql:
            fresh_ts = datetime.now(tz=timezone.utc)
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        decision_id=77,
                        ts=fresh_ts,
                        decision="TRADE",
                        reason=None,
                        score=1.23,
                        target_dte=3,
                        delta_target=0.2,
                        chain_snapshot_id=101,
                        decision_source="rules",
                    )
                ]
            )

        if "SELECT DISTINCT ON (symbol)" in sql:
            fresh_ts = datetime.now(tz=timezone.utc)
            return _FakeExecResult(
                [
                    SimpleNamespace(symbol="SPX", ts=fresh_ts, last=6020.0),
                    SimpleNamespace(symbol="SPY", ts=fresh_ts, last=602.0),
                    SimpleNamespace(symbol="VIX", ts=fresh_ts, last=20.0),
                ]
            )

        if "FROM chain_snapshots" in sql and "LIMIT 20" in sql:
            return _FakeExecResult(
                [
                    SimpleNamespace(
                        snapshot_id=102,
                        ts=datetime(2026, 2, 1, 15, 0, tzinfo=timezone.utc),
                        underlying="SPY",
                        target_dte=3,
                        expiration=date(2026, 2, 5),
                    )
                ]
            )

        if "INSERT INTO auth_audit_log" in sql:
            return _FakeExecResult([])

        raise AssertionError(f"Unhandled SQL in e2e test fake session: {sql}")

    async def commit(self):
        self.commits += 1


class _FakeJob:
    """Simple run-once job stub used by admin run endpoints."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict[str, Any]] = []

    async def run_once(self, *args, **kwargs):  # noqa: ANN002, ANN003 - testing stub
        self.calls.append(dict(kwargs))
        return {"ok": True, "job": self.name, "kwargs": kwargs}


class _FakeTradier:
    """Tradier stub for admin expirations endpoint."""

    async def get_option_expirations(self, symbol: str) -> dict:
        return {"expirations": {"date": ["2026-02-05", "2026-02-07"]}}


def _build_test_client(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "e2e-test-secret")
    monkeypatch.setattr(settings, "auth_register_enabled", True)

    test_app = FastAPI()
    test_app.include_router(auth.router)
    test_app.include_router(public.router)
    test_app.include_router(admin.router)

    fake_session = _RouterAwareSession()

    async def _override_db():
        yield fake_session

    test_app.dependency_overrides[public.get_db_session] = _override_db
    test_app.dependency_overrides[admin.get_db_session] = _override_db

    # Wire all admin-triggered jobs into app.state.
    test_app.state.snapshot_job = _FakeJob("snapshot")
    test_app.state.quote_job = _FakeJob("quotes")
    test_app.state.gex_job = _FakeJob("gex")
    test_app.state.decision_job = _FakeJob("decision")
    test_app.state.feature_builder_job = _FakeJob("feature_builder")
    test_app.state.labeler_job = _FakeJob("labeler")
    test_app.state.trade_pnl_job = _FakeJob("trade_pnl")
    test_app.state.trainer_job = _FakeJob("trainer")
    test_app.state.shadow_inference_job = _FakeJob("shadow_inference")
    test_app.state.promotion_gate_job = _FakeJob("promotion_gate")
    test_app.state.tradier = _FakeTradier()

    return TestClient(test_app), fake_session


def _e2e_auth_headers(client: TestClient) -> dict[str, str]:
    """Log in and return Authorization Bearer headers for e2e tests."""
    r = client.post("/api/auth/login", json={"username": "testuser", "password": "testpass123"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_e2e_public_api_surface(monkeypatch) -> None:
    """Exercise core public API endpoints over HTTP with deterministic fixtures."""
    client, _ = _build_test_client(monkeypatch)

    assert client.get("/health").status_code == 200

    headers = _e2e_auth_headers(client)
    snapshots = client.get("/api/chain-snapshots?limit=2", headers=headers)
    assert snapshots.status_code == 200
    assert len(snapshots.json()["items"]) == 2
    assert {item["underlying"] for item in snapshots.json()["items"]} == {"SPX", "SPY"}

    decisions = client.get("/api/trade-decisions?limit=1", headers=headers)
    assert decisions.status_code == 200
    assert decisions.json()["items"][0]["decision"] == "TRADE"

    trades = client.get("/api/trades?status=OPEN&limit=1", headers=headers)
    assert trades.status_code == 200
    assert trades.json()["items"][0]["status"] == "OPEN"
    assert len(trades.json()["items"][0]["legs"]) == 2

    label_metrics = client.get("/api/label-metrics?lookback_days=90", headers=headers)
    assert label_metrics.status_code == 200
    assert label_metrics.json()["summary"]["resolved"] == 10

    strategy_metrics = client.get("/api/strategy-metrics?lookback_days=90", headers=headers)
    assert strategy_metrics.status_code == 200
    assert strategy_metrics.json()["summary"]["resolved"] == 2

    model_ops = client.get("/api/model-ops", headers=headers)
    assert model_ops.status_code == 200
    assert model_ops.json()["counts"]["model_versions"] == 2

    gex_snapshots = client.get("/api/gex/snapshots?limit=1", headers=headers)
    assert gex_snapshots.status_code == 200
    assert gex_snapshots.json()["items"][0]["underlying"] == "SPX"

    gex_dtes = client.get("/api/gex/dtes?snapshot_id=101", headers=headers)
    assert gex_dtes.status_code == 200
    assert gex_dtes.json()["dte_days"] == [3, 5]

    gex_exps = client.get("/api/gex/expirations?snapshot_id=101", headers=headers)
    assert gex_exps.status_code == 200
    assert len(gex_exps.json()["items"]) == 2

    gex_curve = client.get("/api/gex/curve?snapshot_id=101&dte_days=3", headers=headers)
    assert gex_curve.status_code == 200
    assert len(gex_curve.json()["points"]) == 2

    home = client.get("/", headers=headers)
    assert home.status_code == 200
    assert "IndexSpreadLab (Backend)" in home.text


def test_e2e_admin_auth_and_run_endpoints(monkeypatch) -> None:
    """Validate JWT auth: no token -> 401; with token run endpoints, expirations, preflight, delete."""
    client, fake_session = _build_test_client(monkeypatch)

    unauthorized = client.post("/api/admin/run-snapshot")
    assert unauthorized.status_code == 401

    headers = _e2e_auth_headers(client)
    run_paths = [
        "/api/admin/run-snapshot",
        "/api/admin/run-quotes",
        "/api/admin/run-gex",
        "/api/admin/run-decision",
        "/api/admin/run-feature-builder",
        "/api/admin/run-labeler",
        "/api/admin/run-trade-pnl",
        "/api/admin/run-trainer",
        "/api/admin/run-shadow-inference",
        "/api/admin/run-promotion-gates",
    ]
    for path in run_paths:
        resp = client.post(path, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
    assert client.app.state.gex_job.calls == [{"force": True}]

    expirations = client.get("/api/admin/expirations?symbol=SPY", headers=headers)
    assert expirations.status_code == 200
    assert expirations.json()["symbol"] == "SPY"
    assert len(expirations.json()["expirations"]) == 2

    preflight = client.get("/api/admin/preflight", headers=headers)
    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["counts"]["chain_snapshots"] == 50
    assert payload["latest_snapshot"]["snapshot_id"] == 102
    assert payload["freshness"]["quote_age_min"] is not None
    assert payload["freshness"]["quotes_by_symbol"]["SPX"]["is_stale"] is False
    assert payload["freshness"]["quotes_by_symbol"]["SPY"]["is_stale"] is False
    assert payload["freshness"]["quotes_by_symbol"]["VIX"]["is_stale"] is False
    assert "stale_market_clock" not in payload["warnings"]

    delete_missing = client.delete("/api/admin/trade-decisions/999", headers=headers)
    assert delete_missing.status_code == 404

    delete_ok = client.delete("/api/admin/trade-decisions/1", headers=headers)
    assert delete_ok.status_code == 200
    assert delete_ok.json() == {"deleted": True, "decision_id": 1}
    assert fake_session.commits >= 1


def test_e2e_admin_auth_audit_normalizes_ip(monkeypatch) -> None:
    """Admin auth-audit endpoint should strip /32 suffix from INET host addresses."""
    client, _ = _build_test_client(monkeypatch)
    headers = _e2e_auth_headers(client)

    resp = client.get("/api/admin/auth-audit?limit=1", headers=headers)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 1
    assert len(payload["events"]) == 1
    assert payload["events"][0]["ip_address"] == "73.83.187.53"
