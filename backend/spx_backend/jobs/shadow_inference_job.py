from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.jobs.modeling import extract_candidate_features, predict_with_bucket_model


def _to_json(value: dict | None) -> str | None:
    """Serialize dict payloads for JSONB statements."""
    if value is None:
        return None
    return json.dumps(value, default=str)


def classify_prediction(*, probability_win: float, expected_pnl: float) -> str:
    """Map predicted probability/EV into a trade/skip suggestion."""
    if probability_win < settings.decision_hybrid_min_probability:
        return "SKIP"
    if expected_pnl < settings.decision_hybrid_min_expected_pnl:
        return "SKIP"
    return "TRADE"


@dataclass(frozen=True)
class ShadowInferenceJob:
    """Score candidates in shadow mode and persist to model_predictions."""

    async def _get_shadow_model(self, *, session) -> dict[str, Any] | None:
        """Load latest model eligible for shadow inference."""
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                      model_version_id,
                      model_name,
                      version,
                      rollout_status,
                      data_snapshot_json
                    FROM model_versions
                    WHERE model_name = :model_name
                      AND rollout_status IN ('shadow', 'canary', 'active')
                    ORDER BY created_at DESC, model_version_id DESC
                    LIMIT 1
                    """
                ),
                {"model_name": settings.shadow_inference_model_name},
            )
        ).fetchone()
        if row is None:
            return None
        data_snapshot_json = row.data_snapshot_json if isinstance(row.data_snapshot_json, dict) else {}
        model_payload = data_snapshot_json.get("model_payload")
        if not isinstance(model_payload, dict):
            return None
        return {
            "model_version_id": int(row.model_version_id),
            "model_name": str(row.model_name),
            "version": str(row.version),
            "rollout_status": str(row.rollout_status),
            "model_payload": model_payload,
        }

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        """Run one shadow-inference cycle."""
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        if (not force) and (not settings.shadow_inference_enabled):
            return {"skipped": True, "reason": "shadow_inference_disabled", "now_utc": now_utc.isoformat()}

        lookback_minutes = max(settings.shadow_inference_lookback_minutes, 1)
        window_start = now_utc - timedelta(minutes=lookback_minutes)

        async with SessionLocal() as session:
            model = await self._get_shadow_model(session=session)
            if model is None:
                return {"skipped": True, "reason": "no_shadow_model", "now_utc": now_utc.isoformat()}

            candidate_rows = (
                await session.execute(
                    text(
                        """
                        SELECT
                          tc.candidate_id,
                          tc.candidate_json,
                          tc.max_loss,
                          tc.ts
                        FROM trade_candidates tc
                        LEFT JOIN model_predictions mp
                          ON mp.candidate_id = tc.candidate_id
                         AND mp.model_version_id = :model_version_id
                        WHERE tc.ts >= :window_start
                          AND tc.candidate_json IS NOT NULL
                          AND mp.prediction_id IS NULL
                        ORDER BY tc.ts DESC, tc.candidate_id DESC
                        LIMIT :limit
                        """
                    ),
                    {
                        "model_version_id": model["model_version_id"],
                        "window_start": window_start,
                        "limit": settings.shadow_inference_batch_limit,
                    },
                )
            ).fetchall()

            inserted = 0
            for row in candidate_rows:
                candidate_json = row.candidate_json if isinstance(row.candidate_json, dict) else {}
                features = extract_candidate_features(
                    candidate_json=candidate_json,
                    max_loss_points=(float(row.max_loss) if row.max_loss is not None else None),
                    contract_multiplier=settings.label_contract_multiplier,
                )
                pred = predict_with_bucket_model(model_payload=model["model_payload"], features=features)
                probability_win = float(pred["probability_win"])
                expected_pnl = float(pred["expected_pnl"])
                score_raw = float(pred["utility_score"])
                decision_hint = classify_prediction(probability_win=probability_win, expected_pnl=expected_pnl)

                meta_json = {
                    "model_name": model["model_name"],
                    "model_version": model["version"],
                    "rollout_status": model["rollout_status"],
                    "scoring_source": pred.get("source"),
                    "bucket_key": pred.get("bucket_key"),
                    "bucket_count": pred.get("bucket_count"),
                    "tail_loss_proxy": pred.get("tail_loss_proxy"),
                    "pnl_std": pred.get("pnl_std"),
                    "margin_usage": pred.get("margin_usage"),
                }
                await session.execute(
                    text(
                        """
                        INSERT INTO model_predictions (
                          candidate_id,
                          model_version_id,
                          prediction_schema_version,
                          score_raw,
                          score_calibrated,
                          probability_win,
                          expected_value,
                          threshold_used,
                          rank_in_snapshot,
                          decision,
                          meta_json
                        )
                        VALUES (
                          :candidate_id,
                          :model_version_id,
                          'pred_v1',
                          :score_raw,
                          :score_calibrated,
                          :probability_win,
                          :expected_value,
                          :threshold_used,
                          NULL,
                          :decision,
                          CAST(:meta_json AS jsonb)
                        )
                        ON CONFLICT (candidate_id, model_version_id) DO UPDATE
                        SET score_raw = EXCLUDED.score_raw,
                            score_calibrated = EXCLUDED.score_calibrated,
                            probability_win = EXCLUDED.probability_win,
                            expected_value = EXCLUDED.expected_value,
                            threshold_used = EXCLUDED.threshold_used,
                            decision = EXCLUDED.decision,
                            meta_json = EXCLUDED.meta_json
                        """
                    ),
                    {
                        "candidate_id": int(row.candidate_id),
                        "model_version_id": int(model["model_version_id"]),
                        "score_raw": score_raw,
                        "score_calibrated": score_raw,
                        "probability_win": probability_win,
                        "expected_value": expected_pnl,
                        "threshold_used": settings.decision_hybrid_min_probability,
                        "decision": decision_hint,
                        "meta_json": _to_json(meta_json),
                    },
                )
                inserted += 1

            if inserted > 0:
                # Rank predictions by snapshot for easier downstream selection/debugging.
                await session.execute(
                    text(
                        """
                        UPDATE model_predictions mp
                        SET rank_in_snapshot = ranked.rank_in_snapshot
                        FROM (
                          SELECT
                            mp2.prediction_id,
                            ROW_NUMBER() OVER (
                              PARTITION BY tc.feature_snapshot_id
                              ORDER BY mp2.score_raw DESC, mp2.prediction_id ASC
                            ) AS rank_in_snapshot
                          FROM model_predictions mp2
                          JOIN trade_candidates tc ON tc.candidate_id = mp2.candidate_id
                          WHERE mp2.model_version_id = :model_version_id
                            AND tc.ts >= :window_start
                        ) ranked
                        WHERE mp.prediction_id = ranked.prediction_id
                          AND mp.model_version_id = :model_version_id
                        """
                    ),
                    {"model_version_id": int(model["model_version_id"]), "window_start": window_start},
                )

            await session.commit()
            logger.info(
                "shadow_inference_job: model_version_id={} inserted_or_updated={}",
                model["model_version_id"],
                inserted,
            )
            return {
                "skipped": False,
                "reason": None,
                "model_version_id": int(model["model_version_id"]),
                "model_name": model["model_name"],
                "model_version": model["version"],
                "inserted_or_updated": inserted,
                "window_start_utc": window_start.isoformat(),
            }


def build_shadow_inference_job() -> ShadowInferenceJob:
    """Factory helper for ShadowInferenceJob."""
    return ShadowInferenceJob()

