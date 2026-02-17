from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from spx_backend.config import settings
from spx_backend.jobs.shadow_inference_job import ShadowInferenceJob
from spx_backend.jobs.trainer_job import TrainerJob

import spx_backend.jobs.shadow_inference_job as shadow_module
import spx_backend.jobs.trainer_job as trainer_module

pytestmark = pytest.mark.integration


def _candidate_json(*, spread_side: str, target_dte: int, delta_target: float, entry_credit: float) -> dict:
    """Build minimal candidate payload compatible with feature extraction."""
    return {
        "spread_side": spread_side,
        "target_dte": target_dte,
        "delta_target": delta_target,
        "entry_credit": entry_credit,
        "width_points": 25.0,
        "contracts": 1,
        "context_flags": ["gex_support"],
        "vix_regime": "normal",
        "term_structure_regime": "flat",
        "spy_spx_ratio_regime": "parity",
    }


async def _seed_resolved_candidates_for_trainer(*, session, now_utc: datetime) -> None:
    """Insert deterministic resolved candidates spanning train/test windows."""
    rows = [
        {
            "ts": now_utc - timedelta(days=6),
            "realized_pnl": 80.0,
            "hit_tp50": True,
            "hit_tp100": False,
            "spread_side": "put",
        },
        {
            "ts": now_utc - timedelta(days=4),
            "realized_pnl": -40.0,
            "hit_tp50": False,
            "hit_tp100": False,
            "spread_side": "put",
        },
        {
            "ts": now_utc - timedelta(days=3),
            "realized_pnl": 65.0,
            "hit_tp50": True,
            "hit_tp100": True,
            "spread_side": "call",
        },
        {
            "ts": now_utc - timedelta(hours=20),
            "realized_pnl": 35.0,
            "hit_tp50": True,
            "hit_tp100": False,
            "spread_side": "call",
        },
        {
            "ts": now_utc - timedelta(hours=8),
            "realized_pnl": -25.0,
            "hit_tp50": False,
            "hit_tp100": False,
            "spread_side": "put",
        },
    ]
    for idx, row in enumerate(rows, start=1):
        candidate_json = _candidate_json(
            spread_side=row["spread_side"],
            target_dte=3,
            delta_target=0.20,
            entry_credit=1.5,
        )
        await session.execute(
            text(
                """
                INSERT INTO trade_candidates (
                  ts, candidate_hash, candidate_schema_version, candidate_rank,
                  entry_credit, max_loss, credit_to_width, candidate_json, constraints_json,
                  label_json, label_schema_version, label_status, label_horizon, resolved_at,
                  realized_pnl, hit_tp50_before_sl_or_expiry, hit_tp100_at_expiry
                )
                VALUES (
                  :ts, :candidate_hash, 'cand_v1', 1,
                  :entry_credit, :max_loss, :credit_to_width, CAST(:candidate_json AS jsonb), '{}'::jsonb,
                  CAST(:label_json AS jsonb), 'label_v1', 'resolved', 'to_expiration', :resolved_at,
                  :realized_pnl, :hit_tp50, :hit_tp100
                )
                """
            ),
            {
                "ts": row["ts"],
                "candidate_hash": f"trainer_seed_{idx}",
                "entry_credit": 1.5,
                "max_loss": 8.5,
                "credit_to_width": 1.5 / 25.0,
                "candidate_json": json.dumps(candidate_json, default=str),
                "label_json": json.dumps({"hit_tp100_at_expiry": row["hit_tp100"]}, default=str),
                "resolved_at": row["ts"] + timedelta(minutes=30),
                "realized_pnl": row["realized_pnl"],
                "hit_tp50": row["hit_tp50"],
                "hit_tp100": row["hit_tp100"],
            },
        )
    await session.commit()


async def _seed_shadow_model(*, session) -> int:
    """Insert one shadow model version with deterministic payload."""
    model_payload = {
        "model_type": "bucket_empirical_v1",
        "min_bucket_size": 1,
        "prior_strength": 1.0,
        "utility_weights": {
            "prob_weight": 0.35,
            "tail_penalty": 0.20,
            "margin_penalty": 0.02,
        },
        "global": {
            "count": 10,
            "wins": 6,
            "prob_tp50": 0.60,
            "expected_pnl": 30.0,
            "pnl_std": 12.0,
            "tail_loss_proxy": -5.0,
            "avg_margin_usage": 950.0,
        },
        "buckets": {},
    }
    result = await session.execute(
        text(
            """
            INSERT INTO model_versions (
              model_name, version, algorithm,
              feature_spec_json, data_snapshot_json,
              rollout_status, is_active
            )
            VALUES (
              :model_name, :version, 'bucket_empirical_v1',
              '{}'::jsonb, CAST(:data_snapshot_json AS jsonb),
              'shadow', false
            )
            RETURNING model_version_id
            """
        ),
        {
            "model_name": settings.shadow_inference_model_name,
            "version": "shadow_seed_v1",
            "data_snapshot_json": json.dumps({"model_payload": model_payload}, default=str),
        },
    )
    model_version_id = int(result.scalar_one())
    await session.commit()
    return model_version_id


async def _seed_unscored_candidates_for_shadow(*, session, now_utc: datetime) -> None:
    """Insert candidates that should be scored by shadow inference."""
    candidates = [
        _candidate_json(spread_side="put", target_dte=3, delta_target=0.20, entry_credit=1.8),
        _candidate_json(spread_side="call", target_dte=7, delta_target=0.15, entry_credit=1.4),
    ]
    for idx, candidate_json in enumerate(candidates, start=1):
        await session.execute(
            text(
                """
                INSERT INTO trade_candidates (
                  ts, candidate_hash, candidate_schema_version, candidate_rank,
                  entry_credit, max_loss, credit_to_width, candidate_json, constraints_json
                )
                VALUES (
                  :ts, :candidate_hash, 'cand_v1', 1,
                  :entry_credit, :max_loss, :credit_to_width, CAST(:candidate_json AS jsonb), '{}'::jsonb
                )
                """
            ),
            {
                "ts": now_utc - timedelta(minutes=idx * 5),
                "candidate_hash": f"shadow_seed_{idx}",
                "entry_credit": candidate_json["entry_credit"],
                "max_loss": 7.5,
                "credit_to_width": candidate_json["entry_credit"] / candidate_json["width_points"],
                "candidate_json": json.dumps(candidate_json, default=str),
            },
        )
    await session.commit()


@pytest.mark.asyncio
async def test_trainer_run_once_persists_model_and_training_run(
    integration_db_session,
    monkeypatch,
) -> None:
    """Trainer run_once should complete and persist model/training metadata."""
    monkeypatch.setattr(settings, "trainer_min_rows", 2)
    monkeypatch.setattr(settings, "trainer_min_train_rows", 1)
    monkeypatch.setattr(settings, "trainer_min_test_rows", 1)
    monkeypatch.setattr(settings, "trainer_test_days", 1)
    monkeypatch.setattr(settings, "trainer_min_bucket_size", 1)

    session_factory = async_sessionmaker(
        bind=integration_db_session.bind,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    monkeypatch.setattr(trainer_module, "SessionLocal", session_factory)

    now_utc = datetime.now(tz=timezone.utc)
    await _seed_resolved_candidates_for_trainer(session=integration_db_session, now_utc=now_utc)

    result = await TrainerJob().run_once(force=True)
    assert result["skipped"] is False
    assert int(result["rows_train"]) >= 1
    assert int(result["rows_test"]) >= 1
    model_version_id = int(result["model_version_id"])
    training_run_id = int(result["training_run_id"])

    run_row = (
        await integration_db_session.execute(
            text(
                """
                SELECT model_version_id, status, rows_train, rows_test, metrics_json
                FROM training_runs
                WHERE training_run_id = :training_run_id
                """
            ),
            {"training_run_id": training_run_id},
        )
    ).fetchone()
    assert run_row is not None
    assert int(run_row.model_version_id) == model_version_id
    assert run_row.status == "COMPLETED"
    assert int(run_row.rows_train or 0) >= 1
    assert int(run_row.rows_test or 0) >= 1
    assert isinstance(run_row.metrics_json, dict)
    assert run_row.metrics_json.get("rows_test") is not None

    model_row = (
        await integration_db_session.execute(
            text(
                """
                SELECT rollout_status, is_active, data_snapshot_json
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
    assert isinstance(model_row.data_snapshot_json, dict)
    assert isinstance(model_row.data_snapshot_json.get("model_payload"), dict)


@pytest.mark.asyncio
async def test_trainer_sparse_cv_mode_persists_fold_diagnostics(
    integration_db_session,
    monkeypatch,
) -> None:
    """Trainer should use sparse CV mode and persist fold diagnostics when rows are limited."""
    monkeypatch.setattr(settings, "trainer_min_rows", 100)
    monkeypatch.setattr(settings, "trainer_sparse_cv_enabled", True)
    monkeypatch.setattr(settings, "trainer_sparse_cv_min_rows", 4)
    monkeypatch.setattr(settings, "trainer_sparse_cv_folds", 3)
    monkeypatch.setattr(settings, "trainer_sparse_cv_min_train_rows", 2)
    monkeypatch.setattr(settings, "trainer_sparse_cv_min_test_rows", 1)
    monkeypatch.setattr(settings, "trainer_min_bucket_size", 1)
    monkeypatch.setattr(settings, "trainer_adaptive_prior_enabled", True)
    monkeypatch.setattr(settings, "trainer_adaptive_prior_reference_rows", 200)
    monkeypatch.setattr(settings, "trainer_adaptive_prior_min", 2.0)
    monkeypatch.setattr(settings, "trainer_adaptive_prior_max", 24.0)

    session_factory = async_sessionmaker(
        bind=integration_db_session.bind,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    monkeypatch.setattr(trainer_module, "SessionLocal", session_factory)

    now_utc = datetime.now(tz=timezone.utc)
    await _seed_resolved_candidates_for_trainer(session=integration_db_session, now_utc=now_utc)

    result = await TrainerJob().run_once(force=True)
    assert result["skipped"] is False
    assert result["evaluation_mode"] == "sparse_time_series_cv"
    assert result["metrics"]["evaluation_mode"] == "sparse_time_series_cv"
    assert int(result["metrics"]["cv_folds_used"]) >= 2
    assert isinstance(result["metrics"]["cv_fold_metrics"], list)

    run_row = (
        await integration_db_session.execute(
            text(
                """
                SELECT walkforward_fold, metrics_json, notes
                FROM training_runs
                WHERE training_run_id = :training_run_id
                """
            ),
            {"training_run_id": int(result["training_run_id"])},
        )
    ).fetchone()
    assert run_row is not None
    assert int(run_row.walkforward_fold or 0) >= 2
    assert run_row.notes == "completed_sparse_cv"
    assert run_row.metrics_json["evaluation_mode"] == "sparse_time_series_cv"


@pytest.mark.asyncio
async def test_shadow_inference_run_once_writes_predictions(
    integration_db_session,
    monkeypatch,
) -> None:
    """Shadow inference run_once should persist predictions for eligible candidates."""
    monkeypatch.setattr(settings, "shadow_inference_batch_limit", 100)
    monkeypatch.setattr(settings, "shadow_inference_lookback_minutes", 7 * 24 * 60)
    monkeypatch.setattr(settings, "decision_hybrid_min_probability", 0.5)
    monkeypatch.setattr(settings, "decision_hybrid_min_expected_pnl", 0.0)
    monkeypatch.setattr(settings, "decision_hybrid_min_bucket_count", 0)
    monkeypatch.setattr(settings, "decision_hybrid_max_pnl_std", 10_000.0)

    session_factory = async_sessionmaker(
        bind=integration_db_session.bind,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    monkeypatch.setattr(shadow_module, "SessionLocal", session_factory)

    model_version_id = await _seed_shadow_model(session=integration_db_session)
    now_utc = datetime.now(tz=timezone.utc)
    await _seed_unscored_candidates_for_shadow(session=integration_db_session, now_utc=now_utc)

    result = await ShadowInferenceJob().run_once(force=True)
    assert result["skipped"] is False
    assert int(result["model_version_id"]) == model_version_id
    assert int(result["inserted_or_updated"]) == 2

    prediction_rows = (
        await integration_db_session.execute(
            text(
                """
                SELECT model_version_id, probability_win, expected_value, decision, meta_json
                FROM model_predictions
                ORDER BY prediction_id ASC
                """
            )
        )
    ).fetchall()
    assert len(prediction_rows) == 2
    assert all(int(row.model_version_id) == model_version_id for row in prediction_rows)
    assert all(float(row.probability_win) > 0.0 for row in prediction_rows)
    assert all(float(row.expected_value) >= 0.0 for row in prediction_rows)
    assert {str(row.decision) for row in prediction_rows} == {"TRADE"}
    assert all(isinstance(row.meta_json, dict) for row in prediction_rows)
    assert all("uncertainty_level" in row.meta_json for row in prediction_rows)
