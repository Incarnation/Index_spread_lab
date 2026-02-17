from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
import json
from typing import Any
from zoneinfo import ZoneInfo

from httpx import ASGITransport, AsyncClient
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from spx_backend.config import settings
from spx_backend.jobs.decision_job import DecisionJob
from spx_backend.jobs.feature_builder_job import FeatureBuilderJob
from spx_backend.jobs.gex_job import GexJob
from spx_backend.jobs.labeler_job import LabelerJob
from spx_backend.jobs.promotion_gate_job import PromotionGateJob
from spx_backend.jobs.quote_job import QuoteJob
from spx_backend.jobs.shadow_inference_job import ShadowInferenceJob
from spx_backend.jobs.snapshot_job import SnapshotJob, SnapshotJobConfig, build_snapshot_job
from spx_backend.jobs.trade_pnl_job import TradePnlJob
from spx_backend.jobs.trainer_job import TrainerJob
from spx_backend.web.routers import admin, public

import spx_backend.jobs.decision_job as decision_module
import spx_backend.jobs.feature_builder_job as feature_builder_module
import spx_backend.jobs.gex_job as gex_module
import spx_backend.jobs.labeler_job as labeler_module
import spx_backend.jobs.promotion_gate_job as promotion_gate_module
import spx_backend.jobs.quote_job as quote_module
import spx_backend.jobs.shadow_inference_job as shadow_inference_module
import spx_backend.jobs.snapshot_job as snapshot_module
import spx_backend.jobs.trade_pnl_job as trade_pnl_module
import spx_backend.jobs.trainer_job as trainer_module

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


class _DeterministicTradier:
    """Tradier stub that returns deterministic quotes/expirations/chains."""

    def __init__(self) -> None:
        self._chain_calls: dict[tuple[str, str], int] = {}

    async def get_quotes(self, symbols: list[str] | str) -> dict:
        """Return stable quote payload for configured symbols."""
        wanted = symbols if isinstance(symbols, list) else [s.strip() for s in symbols.split(",") if s.strip()]
        base = {
            "SPX": {"last": 6000.0, "bid": 5999.0, "ask": 6001.0},
            "SPY": {"last": 600.0, "bid": 599.8, "ask": 600.2},
            "VIX": {"last": 20.0, "bid": 19.9, "ask": 20.1},
            "VIX9D": {"last": 18.0, "bid": 17.9, "ask": 18.1},
        }
        quotes: list[dict[str, Any]] = []
        for symbol in wanted:
            q = base.get(symbol, {"last": 100.0, "bid": 99.5, "ask": 100.5})
            quotes.append(
                {
                    "symbol": symbol,
                    "last": q["last"],
                    "bid": q["bid"],
                    "ask": q["ask"],
                    "open": q["last"],
                    "high": q["last"] + 1.0,
                    "low": q["last"] - 1.0,
                    "close": q["last"],
                    "volume": 1000,
                    "change": 0.0,
                    "change_percentage": 0.0,
                    "prevclose": q["last"],
                }
            )
        return {"quotes": {"quote": quotes}}

    async def get_option_expirations(self, symbol: str) -> dict:
        """Return a short ordered expiration ladder for trading-DTE mapping."""
        as_of = datetime.now(tz=ZoneInfo(settings.tz)).date()
        exps = [(as_of + timedelta(days=i)).isoformat() for i in (1, 2, 3, 4)]
        return {"expirations": {"date": exps}}

    async def get_option_chain(self, *, underlying: str, expiration: str, greeks: bool = True) -> dict:  # noqa: ARG002
        """Return deterministic option chain; second call improves mark values."""
        key = (underlying, expiration)
        call_idx = self._chain_calls.get(key, 0)
        self._chain_calls[key] = call_idx + 1
        improved = call_idx >= 1

        if improved:
            short_put_bid, short_put_ask = 0.6, 0.8
            long_put_bid, long_put_ask = 0.2, 0.4
        else:
            short_put_bid, short_put_ask = 2.4, 2.6
            long_put_bid, long_put_ask = 1.2, 1.4

        options = [
            {
                "symbol": f"{underlying}_{expiration}_P6000",
                "underlying": underlying,
                "expiration_date": expiration,
                "strike": 6000.0,
                "option_type": "put",
                "bid": short_put_bid,
                "ask": short_put_ask,
                "last": (short_put_bid + short_put_ask) / 2.0,
                "volume": 100,
                "open_interest": 400,
                "contract_size": 100,
                "greeks": {
                    "delta": -0.20,
                    "gamma": 0.02,
                    "theta": -0.01,
                    "vega": 0.05,
                    "rho": -0.01,
                },
            },
            {
                "symbol": f"{underlying}_{expiration}_P5975",
                "underlying": underlying,
                "expiration_date": expiration,
                "strike": 5975.0,
                "option_type": "put",
                "bid": long_put_bid,
                "ask": long_put_ask,
                "last": (long_put_bid + long_put_ask) / 2.0,
                "volume": 100,
                "open_interest": 350,
                "contract_size": 100,
                "greeks": {
                    "delta": -0.12,
                    "gamma": 0.018,
                    "theta": -0.009,
                    "vega": 0.04,
                    "rho": -0.009,
                },
            },
            {
                "symbol": f"{underlying}_{expiration}_C6025",
                "underlying": underlying,
                "expiration_date": expiration,
                "strike": 6025.0,
                "option_type": "call",
                "bid": 1.8,
                "ask": 2.0,
                "last": 1.9,
                "volume": 80,
                "open_interest": 300,
                "contract_size": 100,
                "greeks": {
                    "delta": 0.22,
                    "gamma": 0.019,
                    "theta": -0.01,
                    "vega": 0.05,
                    "rho": 0.01,
                },
            },
            {
                "symbol": f"{underlying}_{expiration}_C6050",
                "underlying": underlying,
                "expiration_date": expiration,
                "strike": 6050.0,
                "option_type": "call",
                "bid": 1.0,
                "ask": 1.2,
                "last": 1.1,
                "volume": 70,
                "open_interest": 280,
                "contract_size": 100,
                "greeks": {
                    "delta": 0.14,
                    "gamma": 0.017,
                    "theta": -0.008,
                    "vega": 0.04,
                    "rho": 0.009,
                },
            },
        ]
        return {"options": {"option": options}}


class _QuoteFetchFailsTradier(_DeterministicTradier):
    """Tradier stub that raises quote-fetch errors."""

    async def get_quotes(self, symbols: list[str] | str) -> dict:  # noqa: ARG002
        raise RuntimeError("forced_quote_failure")


class _NoExpirationsTradier(_DeterministicTradier):
    """Tradier stub that returns an empty expiration list."""

    async def get_option_expirations(self, symbol: str) -> dict:  # noqa: ARG002
        return {"expirations": {"date": []}}


class _QuoteTimeoutTradier(_DeterministicTradier):
    """Tradier stub that raises timeout-style quote errors."""

    async def get_quotes(self, symbols: list[str] | str) -> dict:  # noqa: ARG002
        raise TimeoutError("forced_timeout")


class _PartialMalformedQuotesTradier(_DeterministicTradier):
    """Tradier stub that emits one invalid quote row and one valid row."""

    async def get_quotes(self, symbols: list[str] | str) -> dict:  # noqa: ARG002
        return {
            "quotes": {
                "quote": [
                    {
                        "symbol": "SPX",
                        "last": 6000.0,
                        "bid": 5999.0,
                        "ask": 6001.0,
                        "open": 6000.0,
                        "high": 6002.0,
                        "low": 5998.0,
                        "close": 6000.0,
                        "volume": 1000,
                        "change": 0.0,
                        "change_percentage": 0.0,
                        "prevclose": 6000.0,
                    },
                    {
                        # Missing symbol triggers NOT NULL violation for isolation test.
                        "last": 20.0,
                        "bid": 19.9,
                        "ask": 20.1,
                    },
                ]
            }
        }


class _PartiallyMalformedChainTradier(_DeterministicTradier):
    """Tradier stub that returns one valid chain and one malformed chain."""

    def __init__(self) -> None:
        super().__init__()
        self._expirations: list[str] = []

    async def get_option_expirations(self, symbol: str) -> dict:  # noqa: ARG002
        as_of = datetime.now(tz=ZoneInfo(settings.tz)).date()
        self._expirations = [(as_of + timedelta(days=2)).isoformat(), (as_of + timedelta(days=3)).isoformat()]
        return {"expirations": {"date": self._expirations}}

    async def get_option_chain(self, *, underlying: str, expiration: str, greeks: bool = True) -> dict:  # noqa: ARG002
        if self._expirations and expiration == self._expirations[1]:
            return {"options": {"option": "malformed_option_payload"}}
        return await super().get_option_chain(underlying=underlying, expiration=expiration, greeks=greeks)


def _configure_workflow_settings(monkeypatch) -> None:
    """Pin workflow settings for deterministic integration behavior."""
    monkeypatch.setattr(settings, "quote_symbols", "SPX,SPY,VIX,VIX9D")
    monkeypatch.setattr(settings, "snapshot_dte_mode", "targets")
    monkeypatch.setattr(settings, "snapshot_dte_targets", "3")
    monkeypatch.setattr(settings, "snapshot_dte_tolerance_days", 0)
    monkeypatch.setattr(settings, "snapshot_strikes_each_side", 0)
    monkeypatch.setattr(settings, "decision_dte_targets", "3")
    monkeypatch.setattr(settings, "decision_delta_targets", "0.20")
    monkeypatch.setattr(settings, "decision_spread_sides", "put")
    monkeypatch.setattr(settings, "decision_spread_side", "put")
    monkeypatch.setattr(settings, "decision_spread_width_points", 25.0)
    monkeypatch.setattr(settings, "trainer_min_rows", 1)
    monkeypatch.setattr(settings, "trainer_min_train_rows", 0)
    monkeypatch.setattr(settings, "trainer_min_test_rows", 0)
    monkeypatch.setattr(settings, "trainer_test_days", 1)
    monkeypatch.setattr(settings, "shadow_inference_lookback_minutes", 10080)


def _patch_job_session_locals(monkeypatch, session_factory) -> None:
    """Patch job modules to use integration test DB session factory."""
    monkeypatch.setattr(quote_module, "SessionLocal", session_factory)
    monkeypatch.setattr(snapshot_module, "SessionLocal", session_factory)
    monkeypatch.setattr(gex_module, "SessionLocal", session_factory)
    monkeypatch.setattr(decision_module, "SessionLocal", session_factory)
    monkeypatch.setattr(feature_builder_module, "SessionLocal", session_factory)
    monkeypatch.setattr(labeler_module, "SessionLocal", session_factory)
    monkeypatch.setattr(trainer_module, "SessionLocal", session_factory)
    monkeypatch.setattr(shadow_inference_module, "SessionLocal", session_factory)
    monkeypatch.setattr(promotion_gate_module, "SessionLocal", session_factory)
    monkeypatch.setattr(trade_pnl_module, "SessionLocal", session_factory)


@asynccontextmanager
async def _build_workflow_client(*, integration_db_session, monkeypatch, tradier):
    """Build an integration app/client wired to test DB and provided Tradier stub."""
    _configure_workflow_settings(monkeypatch)

    session_factory = async_sessionmaker(
        bind=integration_db_session.bind,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    _patch_job_session_locals(monkeypatch, session_factory)

    app = FastAPI()
    app.include_router(public.router)
    app.include_router(admin.router)

    async def _override_db():
        yield integration_db_session

    app.dependency_overrides[public.get_db_session] = _override_db
    app.dependency_overrides[admin.get_db_session] = _override_db
    app.state.tradier = tradier
    app.state.quote_job = QuoteJob(tradier=tradier)
    app.state.snapshot_job = build_snapshot_job(tradier=tradier)
    app.state.gex_job = GexJob()
    app.state.feature_builder_job = FeatureBuilderJob()
    app.state.decision_job = DecisionJob()
    app.state.trade_pnl_job = TradePnlJob()
    app.state.labeler_job = LabelerJob()
    app.state.trainer_job = TrainerJob()
    app.state.shadow_inference_job = ShadowInferenceJob()
    app.state.promotion_gate_job = PromotionGateJob()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
async def workflow_client(integration_db_session, monkeypatch, admin_headers):  # noqa: ARG001
    """Integration app/client with real jobs backed by deterministic Tradier + test DB."""
    async with _build_workflow_client(
        integration_db_session=integration_db_session,
        monkeypatch=monkeypatch,
        tradier=_DeterministicTradier(),
    ) as client:
        yield client


async def _seed_two_chain_snapshots_for_gex(*, session, now_utc: datetime) -> list[int]:
    """Seed two snapshots + option rows and one quote for GEX failure-isolation tests.

    Parameters
    ----------
    session:
        Async SQLAlchemy session bound to the integration test database.
    now_utc:
        Reference timestamp for deterministic ordering.

    Returns
    -------
    list[int]
        Snapshot IDs inserted in descending recency order.
    """
    await session.execute(
        text(
            """
            INSERT INTO underlying_quotes (
              ts, symbol, last, bid, ask, open, high, low, close, source, raw_json
            )
            VALUES (
              :ts, 'SPX', 6000.0, 5999.0, 6001.0, 6000.0, 6002.0, 5998.0, 6000.0, 'tradier', '{}'::jsonb
            )
            """
        ),
        {"ts": now_utc - timedelta(minutes=1)},
    )
    snapshot_ids: list[int] = []
    for idx in range(2):
        ts_value = now_utc - timedelta(minutes=idx + 2)
        expiration = (ts_value + timedelta(days=3)).date()
        snapshot_row = await session.execute(
            text(
                """
                INSERT INTO chain_snapshots (ts, underlying, target_dte, expiration, payload_json, checksum)
                VALUES (:ts, 'SPX', 3, :expiration, '{}'::jsonb, :checksum)
                RETURNING snapshot_id
                """
            ),
            {
                "ts": ts_value,
                "expiration": expiration,
                "checksum": f"gex_seed_{idx}",
            },
        )
        snapshot_id = int(snapshot_row.scalar_one())
        snapshot_ids.append(snapshot_id)
        await session.execute(
            text(
                """
                INSERT INTO option_chain_rows (
                  snapshot_id, option_symbol, underlying, expiration, strike, option_right,
                  bid, ask, last, volume, open_interest, contract_size, gamma, raw_json
                )
                VALUES
                  (:snapshot_id, :opt_put, 'SPX', :expiration, 6000.0, 'P', 2.0, 2.2, 2.1, 10, 100, 100, 0.020, '{}'::jsonb),
                  (:snapshot_id, :opt_call, 'SPX', :expiration, 6025.0, 'C', 1.8, 2.0, 1.9, 10, 100, 100, 0.019, '{}'::jsonb)
                """
            ),
            {
                "snapshot_id": snapshot_id,
                "expiration": expiration,
                "opt_put": f"SPX_P_{snapshot_id}",
                "opt_call": f"SPX_C_{snapshot_id}",
            },
        )
    await session.commit()
    return snapshot_ids


async def _seed_completed_training_run(*, session, version: str, metrics: dict[str, Any]) -> tuple[int, int]:
    """Seed one completed training run + model version for promotion gate tests."""
    model_row = await session.execute(
        text(
            """
            INSERT INTO model_versions (
              model_name, version, algorithm, feature_spec_json,
              data_snapshot_json, metrics_json, rollout_status, is_active, notes
            )
            VALUES (
              :model_name, :version, 'bucket_empirical_v1', '{}'::jsonb,
              '{}'::jsonb, '{}'::jsonb, 'shadow', false, 'seeded_for_gate_tests'
            )
            RETURNING model_version_id
            """
        ),
        {
            "model_name": settings.promotion_gate_model_name,
            "version": version,
        },
    )
    model_version_id = int(model_row.scalar_one())

    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    run_row = await session.execute(
        text(
            """
            INSERT INTO training_runs (
              model_version_id, started_at, finished_at, status, config_json, metrics_json, notes
            )
            VALUES (
              :model_version_id, :started_at, :finished_at, 'COMPLETED',
              '{}'::jsonb, CAST(:metrics_json AS jsonb), 'seeded_completed'
            )
            RETURNING training_run_id
            """
        ),
        {
            "model_version_id": model_version_id,
            "started_at": now_utc - timedelta(hours=1),
            "finished_at": now_utc,
            "metrics_json": json.dumps(metrics, default=str),
        },
    )
    training_run_id = int(run_row.scalar_one())
    await session.commit()
    return training_run_id, model_version_id


@pytest.mark.asyncio
async def test_db_backed_full_admin_workflow_pipeline(workflow_client, admin_headers) -> None:
    """Execute full admin-run workflow with deterministic side effects in test DB."""
    run_quotes = await workflow_client.post("/api/admin/run-quotes", headers=admin_headers)
    assert run_quotes.status_code == 200
    assert run_quotes.json()["skipped"] is False
    assert run_quotes.json()["quotes_inserted"] >= 4

    run_snapshot_1 = await workflow_client.post("/api/admin/run-snapshot", headers=admin_headers)
    assert run_snapshot_1.status_code == 200
    assert run_snapshot_1.json()["skipped"] is False
    assert run_snapshot_1.json()["chain_rows_inserted"] >= 4

    run_gex = await workflow_client.post("/api/admin/run-gex", headers=admin_headers)
    assert run_gex.status_code == 200
    assert run_gex.json()["skipped"] is False
    assert run_gex.json()["computed_snapshots"] >= 1

    run_feature_builder = await workflow_client.post("/api/admin/run-feature-builder", headers=admin_headers)
    assert run_feature_builder.status_code == 200
    assert run_feature_builder.json()["skipped"] is False
    assert run_feature_builder.json()["features_inserted"] >= 1
    assert run_feature_builder.json()["candidates_inserted"] >= 1

    run_decision = await workflow_client.post("/api/admin/run-decision", headers=admin_headers)
    assert run_decision.status_code == 200
    assert run_decision.json()["skipped"] is False
    assert run_decision.json()["decision"] == "TRADE"

    # Second snapshot creates forward marks for labeler/trade-pnl evaluation.
    run_snapshot_2 = await workflow_client.post("/api/admin/run-snapshot", headers=admin_headers)
    assert run_snapshot_2.status_code == 200
    assert run_snapshot_2.json()["skipped"] is False
    assert run_snapshot_2.json()["chain_rows_inserted"] >= 4

    run_trade_pnl = await workflow_client.post("/api/admin/run-trade-pnl", headers=admin_headers)
    assert run_trade_pnl.status_code == 200
    assert run_trade_pnl.json()["skipped"] is False
    assert run_trade_pnl.json()["updated"] >= 1
    assert run_trade_pnl.json()["marks_written"] >= 1

    run_labeler = await workflow_client.post("/api/admin/run-labeler", headers=admin_headers)
    assert run_labeler.status_code == 200
    assert run_labeler.json()["skipped"] is False
    assert run_labeler.json()["resolved"] >= 1

    run_trainer = await workflow_client.post("/api/admin/run-trainer", headers=admin_headers)
    assert run_trainer.status_code == 200
    assert run_trainer.json()["skipped"] is False
    assert run_trainer.json().get("model_version_id") is not None

    run_shadow = await workflow_client.post("/api/admin/run-shadow-inference", headers=admin_headers)
    assert run_shadow.status_code == 200
    assert run_shadow.json()["skipped"] is False
    assert run_shadow.json()["inserted_or_updated"] >= 1

    run_gates = await workflow_client.post("/api/admin/run-promotion-gates", headers=admin_headers)
    assert run_gates.status_code == 200
    assert run_gates.json()["skipped"] is False
    assert isinstance(run_gates.json()["passed"], bool)

    preflight = await workflow_client.get("/api/admin/preflight", headers=admin_headers)
    assert preflight.status_code == 200
    payload = preflight.json()
    assert payload["counts"]["underlying_quotes"] >= 4
    assert payload["counts"]["chain_snapshots"] >= 2
    assert payload["counts"]["option_chain_rows"] >= 8
    assert payload["counts"]["gex_snapshots"] >= 1
    assert payload["counts"]["trade_decisions"] >= 1
    assert payload["counts"]["feature_snapshots"] >= 1
    assert payload["counts"]["trade_candidates"] >= 1
    assert payload["counts"]["model_versions"] >= 1
    assert payload["counts"]["training_runs"] >= 1
    assert payload["counts"]["model_predictions"] >= 1
    assert payload["counts"]["trades"] >= 1

    critical_warnings = {
        "no_chain_snapshots",
        "no_gex_snapshots",
        "no_trade_decisions",
        "no_feature_snapshots",
        "no_trade_candidates",
        "no_model_versions",
        "no_training_runs",
        "no_model_predictions",
        "no_trades",
    }
    assert critical_warnings.isdisjoint(set(payload["warnings"]))


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_quote_fetch_failure_returns_skip(
    integration_db_session,
    monkeypatch,
    admin_headers,
) -> None:
    """Quote endpoint should return quote_fetch_failed when Tradier quotes call fails."""
    async with _build_workflow_client(
        integration_db_session=integration_db_session,
        monkeypatch=monkeypatch,
        tradier=_QuoteFetchFailsTradier(),
    ) as client:
        run_quotes = await client.post("/api/admin/run-quotes", headers=admin_headers)

    assert run_quotes.status_code == 200
    payload = run_quotes.json()
    assert payload["skipped"] is True
    assert payload["reason"] == "quote_fetch_failed"
    assert payload["quotes_inserted"] == 0

    count = (await integration_db_session.execute(text("SELECT COUNT(*) FROM underlying_quotes"))).scalar_one()
    assert int(count) == 0


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_quote_timeout_returns_skip(
    integration_db_session,
    monkeypatch,
    admin_headers,
) -> None:
    """Quote endpoint should map timeout exceptions into a clean skip payload."""
    async with _build_workflow_client(
        integration_db_session=integration_db_session,
        monkeypatch=monkeypatch,
        tradier=_QuoteTimeoutTradier(),
    ) as client:
        run_quotes = await client.post("/api/admin/run-quotes", headers=admin_headers)

    assert run_quotes.status_code == 200
    payload = run_quotes.json()
    assert payload["skipped"] is True
    assert payload["reason"] == "quote_fetch_failed"
    assert payload["quotes_inserted"] == 0

    count = (await integration_db_session.execute(text("SELECT COUNT(*) FROM underlying_quotes"))).scalar_one()
    assert int(count) == 0


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_quote_partial_insert_isolated_per_symbol(
    integration_db_session,
    monkeypatch,
    admin_headers,
) -> None:
    """Quote job should keep valid symbols when one malformed quote row fails."""
    async with _build_workflow_client(
        integration_db_session=integration_db_session,
        monkeypatch=monkeypatch,
        tradier=_PartialMalformedQuotesTradier(),
    ) as client:
        run_quotes = await client.post("/api/admin/run-quotes", headers=admin_headers)

    assert run_quotes.status_code == 200
    payload = run_quotes.json()
    assert payload["skipped"] is False
    assert payload["quotes_inserted"] == 1
    assert payload["quotes_failed"] == 1

    rows = (
        await integration_db_session.execute(
            text(
                """
                SELECT symbol
                FROM underlying_quotes
                ORDER BY quote_id ASC
                """
            )
        )
    ).fetchall()
    assert [row.symbol for row in rows] == ["SPX"]


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_snapshot_malformed_chain_rolls_back_only_failed_expiration(
    integration_db_session,
    monkeypatch,
) -> None:
    """Snapshot job should commit valid expirations while rolling back malformed ones."""
    session_factory = async_sessionmaker(
        bind=integration_db_session.bind,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    monkeypatch.setattr(snapshot_module, "SessionLocal", session_factory)
    tradier = _PartiallyMalformedChainTradier()
    config = SnapshotJobConfig(
        job_name="snapshot_job_test",
        underlying="SPX",
        dte_mode="range",
        dte_targets=[],
        dte_min_days=0,
        dte_max_days=10,
        range_fallback_enabled=False,
        range_fallback_count=2,
        dte_tolerance_days=0,
        strikes_each_side=0,
        allow_outside_rth=True,
    )
    result = await SnapshotJob(tradier=tradier, config=config).run_once(force=True)

    assert result["skipped"] is False
    assert len(result["inserted"]) == 1
    assert len(result["failed_items"]) == 1
    assert result["failed_items"][0]["stage"] == "persist_chain"
    assert result["chain_rows_inserted"] > 0

    snapshots = (
        await integration_db_session.execute(
            text(
                """
                SELECT expiration
                FROM chain_snapshots
                ORDER BY snapshot_id ASC
                """
            )
        )
    ).fetchall()
    assert len(snapshots) == 1
    assert snapshots[0].expiration.isoformat() == tradier._expirations[0]
    bad_count = (
        await integration_db_session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM chain_snapshots
                WHERE expiration = :bad_expiration
                """
            ),
            {"bad_expiration": date.fromisoformat(tradier._expirations[1])},
        )
    ).scalar_one()
    assert int(bad_count) == 0


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_gex_partial_failure_keeps_other_snapshots(
    integration_db_session,
    monkeypatch,
) -> None:
    """GEX job should continue batch processing when one snapshot fails."""
    session_factory = async_sessionmaker(
        bind=integration_db_session.bind,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    monkeypatch.setattr(gex_module, "SessionLocal", session_factory)
    await _seed_two_chain_snapshots_for_gex(session=integration_db_session, now_utc=datetime.now(tz=ZoneInfo("UTC")))

    call_state = {"count": 0}

    async def _flaky_get_spot_price(session, ts, underlying):  # noqa: ANN001
        """Raise for first snapshot then return a valid spot for remaining snapshots."""
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise RuntimeError("forced_spot_lookup_failure")
        return 6000.0

    class _FlakyGexJob(GexJob):
        """GEX job override that fails one snapshot-level spot lookup."""

        async def _get_spot_price(self, session, ts, underlying):  # noqa: ANN001
            """Raise first call to force one snapshot failure path."""
            return await _flaky_get_spot_price(session, ts, underlying)

    job = _FlakyGexJob()
    result = await job.run_once()

    assert result["skipped"] is False
    assert result["computed_snapshots"] == 1
    assert len(result["failed_snapshots"]) == 1

    gex_rows = (
        await integration_db_session.execute(
            text(
                """
                SELECT snapshot_id
                FROM gex_snapshots
                ORDER BY snapshot_id ASC
                """
            )
        )
    ).fetchall()
    assert len(gex_rows) == 1
    failed_snapshot_id = int(result["failed_snapshots"][0]["snapshot_id"])
    assert int(gex_rows[0].snapshot_id) != failed_snapshot_id


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_snapshot_no_expirations_returns_skip(
    integration_db_session,
    monkeypatch,
    admin_headers,
) -> None:
    """Snapshot endpoint should skip cleanly when no expirations are available."""
    async with _build_workflow_client(
        integration_db_session=integration_db_session,
        monkeypatch=monkeypatch,
        tradier=_NoExpirationsTradier(),
    ) as client:
        run_snapshot = await client.post("/api/admin/run-snapshot", headers=admin_headers)

    assert run_snapshot.status_code == 200
    payload = run_snapshot.json()
    assert payload["skipped"] is True
    assert payload["reason"] == "no_expirations"
    assert payload["chain_rows_inserted"] == 0
    assert payload["inserted"] == []
    assert payload["fallback_used"] is False

    snapshots_count = (await integration_db_session.execute(text("SELECT COUNT(*) FROM chain_snapshots"))).scalar_one()
    rows_count = (await integration_db_session.execute(text("SELECT COUNT(*) FROM option_chain_rows"))).scalar_one()
    assert int(snapshots_count) == 0
    assert int(rows_count) == 0


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_shadow_inference_no_model_returns_skip(
    workflow_client,
    integration_db_session,
    admin_headers,
) -> None:
    """Shadow inference should skip with no_shadow_model in a fresh database."""
    run_shadow = await workflow_client.post("/api/admin/run-shadow-inference", headers=admin_headers)
    assert run_shadow.status_code == 200
    payload = run_shadow.json()
    assert payload["skipped"] is True
    assert payload["reason"] == "no_shadow_model"

    count = (await integration_db_session.execute(text("SELECT COUNT(*) FROM model_predictions"))).scalar_one()
    assert int(count) == 0


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_promotion_gate_fail_branch(
    integration_db_session,
    monkeypatch,
    admin_headers,
) -> None:
    """Promotion gate should stay in shadow when checks fail."""
    fail_metrics = {
        "resolved_test": 10,
        "tp50_rate_test": 0.10,
        "expectancy_test": -50.0,
        "max_drawdown_test": 50000.0,
        "tail_loss_proxy_test": -10000.0,
        "avg_margin_usage_test": 10000.0,
    }
    training_run_id, model_version_id = await _seed_completed_training_run(
        session=integration_db_session,
        version="gate_fail_seed",
        metrics=fail_metrics,
    )

    async with _build_workflow_client(
        integration_db_session=integration_db_session,
        monkeypatch=monkeypatch,
        tradier=_DeterministicTradier(),
    ) as client:
        run_gates = await client.post("/api/admin/run-promotion-gates", headers=admin_headers)

    assert run_gates.status_code == 200
    payload = run_gates.json()
    assert payload["skipped"] is False
    assert payload["passed"] is False
    assert payload["rollout_status"] == "shadow"
    assert payload["is_active"] is False
    assert int(payload["training_run_id"]) == training_run_id
    assert int(payload["model_version_id"]) == model_version_id

    run_row = (
        await integration_db_session.execute(
            text(
                """
                SELECT notes, metrics_json
                FROM training_runs
                WHERE training_run_id = :training_run_id
                """
            ),
            {"training_run_id": training_run_id},
        )
    ).fetchone()
    assert run_row is not None
    assert run_row.notes == "gate_failed"
    assert run_row.metrics_json["gate"]["passed"] is False

    model_row = (
        await integration_db_session.execute(
            text(
                """
                SELECT rollout_status, is_active, metrics_json
                FROM model_versions
                WHERE model_version_id = :model_version_id
                """
            ),
            {"model_version_id": model_version_id},
        )
    ).fetchone()
    assert model_row is not None
    assert model_row.rollout_status == "shadow"
    assert model_row.is_active is False
    assert model_row.metrics_json["gate"]["passed"] is False


@pytest.mark.asyncio
@pytest.mark.regression
async def test_db_backed_promotion_gate_pass_branch(
    integration_db_session,
    monkeypatch,
    admin_headers,
) -> None:
    """Promotion gate should advance to canary when checks pass."""
    monkeypatch.setattr(settings, "promotion_gate_auto_activate", False)
    pass_metrics = {
        "resolved_test": 500,
        "tp50_rate_test": 0.70,
        "expectancy_test": 250.0,
        "max_drawdown_test": 1000.0,
        "tail_loss_proxy_test": 100.0,
        "avg_margin_usage_test": 2000.0,
    }
    training_run_id, model_version_id = await _seed_completed_training_run(
        session=integration_db_session,
        version="gate_pass_seed",
        metrics=pass_metrics,
    )

    async with _build_workflow_client(
        integration_db_session=integration_db_session,
        monkeypatch=monkeypatch,
        tradier=_DeterministicTradier(),
    ) as client:
        run_gates = await client.post("/api/admin/run-promotion-gates", headers=admin_headers)

    assert run_gates.status_code == 200
    payload = run_gates.json()
    assert payload["skipped"] is False
    assert payload["passed"] is True
    assert payload["rollout_status"] == "canary"
    assert payload["is_active"] is False
    assert int(payload["training_run_id"]) == training_run_id
    assert int(payload["model_version_id"]) == model_version_id

    run_row = (
        await integration_db_session.execute(
            text(
                """
                SELECT notes, metrics_json
                FROM training_runs
                WHERE training_run_id = :training_run_id
                """
            ),
            {"training_run_id": training_run_id},
        )
    ).fetchone()
    assert run_row is not None
    assert run_row.notes == "gate_passed"
    assert run_row.metrics_json["gate"]["passed"] is True

    model_row = (
        await integration_db_session.execute(
            text(
                """
                SELECT rollout_status, is_active, metrics_json
                FROM model_versions
                WHERE model_version_id = :model_version_id
                """
            ),
            {"model_version_id": model_version_id},
        )
    ).fetchone()
    assert model_row is not None
    assert model_row.rollout_status == "canary"
    assert model_row.is_active is False
    assert model_row.metrics_json["gate"]["passed"] is True
