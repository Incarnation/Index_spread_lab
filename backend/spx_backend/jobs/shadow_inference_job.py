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
from spx_backend.jobs.modeling import (
    extract_candidate_features,
    extract_xgb_features,
    predict_with_bucket_model,
    predict_xgb_entry,
)


def _to_json(value: dict | None) -> str | None:
    """Serialize dict payloads for JSONB statements."""
    if value is None:
        return None
    return json.dumps(value, default=str)


def compute_uncertainty_penalty(*, bucket_count: int, pnl_std: float) -> float:
    """Compute score penalty for low-support or high-variance predictions.

    Parameters
    ----------
    bucket_count:
        Number of historical rows supporting the selected prediction bucket.
    pnl_std:
        Historical PnL standard deviation for the selected bucket.

    Returns
    -------
    float
        Non-negative penalty subtracted from raw utility score.
    """
    min_count = max(settings.decision_hybrid_min_bucket_count, 0)
    max_std = max(float(settings.decision_hybrid_max_pnl_std), 0.0)
    penalty = 0.0
    if bucket_count < min_count:
        penalty += (min_count - bucket_count) * 0.25
    if max_std > 0.0 and pnl_std > max_std:
        penalty += (pnl_std - max_std) * 0.01
    return penalty


def classify_uncertainty_level(*, bucket_count: int, pnl_std: float) -> str:
    """Classify prediction confidence into low/medium/high uncertainty buckets.

    Parameters
    ----------
    bucket_count:
        Number of historical rows supporting the selected prediction bucket.
    pnl_std:
        Historical PnL standard deviation for the selected bucket.

    Returns
    -------
    str
        Uncertainty label used for explainability metadata.
    """
    min_count = max(settings.decision_hybrid_min_bucket_count, 0)
    max_std = max(float(settings.decision_hybrid_max_pnl_std), 0.0)
    if bucket_count < min_count or (max_std > 0.0 and pnl_std > max_std):
        return "high"
    if bucket_count < max(min_count * 2, 1):
        return "medium"
    return "low"


def classify_prediction(*, probability_win: float, expected_pnl: float, bucket_count: int, pnl_std: float) -> str:
    """Map predicted probability, EV, and uncertainty into trade/skip.

    Parameters
    ----------
    probability_win:
        Predicted win probability for TP50 outcome.
    expected_pnl:
        Predicted expected value in dollars.
    bucket_count:
        Historical support count for selected bucket stats.
    pnl_std:
        Historical PnL volatility proxy for selected bucket stats.

    Returns
    -------
    str
        `TRADE` when signal thresholds and uncertainty gates pass, else `SKIP`.
    """
    if bucket_count < settings.decision_hybrid_min_bucket_count:
        return "SKIP"
    if pnl_std > settings.decision_hybrid_max_pnl_std:
        return "SKIP"
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
        """Run one shadow-inference cycle with uncertainty-aware metadata."""
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
            model_type = (model["model_payload"].get("model_type") or "bucket_empirical_v1")
            for row in candidate_rows:
                candidate_json = row.candidate_json if isinstance(row.candidate_json, dict) else {}
                max_loss_pts = float(row.max_loss) if row.max_loss is not None else None
                candidate_ts = row.ts if hasattr(row, "ts") else None

                if model_type == "xgb_entry_v1":
                    features = extract_xgb_features(
                        candidate_json=candidate_json,
                        max_loss_points=max_loss_pts,
                        contract_multiplier=settings.label_contract_multiplier,
                        candidate_ts=candidate_ts,
                    )
                    pred = predict_xgb_entry(model_payload=model["model_payload"], features=features)
                else:
                    features = extract_candidate_features(
                        candidate_json=candidate_json,
                        max_loss_points=max_loss_pts,
                        contract_multiplier=settings.label_contract_multiplier,
                    )
                    pred = predict_with_bucket_model(model_payload=model["model_payload"], features=features)
                probability_win = float(pred["probability_win"])
                expected_pnl = float(pred["expected_pnl"])
                raw_utility = float(pred["utility_score"])

                if model_type == "xgb_entry_v1":
                    # XGBoost models have no bucket-based uncertainty; skip
                    # the bucket_count / pnl_std penalties that assume the
                    # bucket_empirical pipeline.
                    score_raw = raw_utility
                    uncertainty_penalty = 0.0
                    uncertainty_level = "n/a"
                    decision_hint = (
                        "TRADE"
                        if (probability_win >= settings.decision_hybrid_min_probability
                            and expected_pnl >= settings.decision_hybrid_min_expected_pnl)
                        else "SKIP"
                    )
                    bucket_count = 0
                    pnl_std = 0.0
                else:
                    bucket_count = int(pred.get("bucket_count") or 0)
                    pnl_std = float(pred.get("pnl_std") or 0.0)
                    uncertainty_penalty = compute_uncertainty_penalty(bucket_count=bucket_count, pnl_std=pnl_std)
                    score_raw = raw_utility - uncertainty_penalty
                    uncertainty_level = classify_uncertainty_level(bucket_count=bucket_count, pnl_std=pnl_std)
                    decision_hint = classify_prediction(
                        probability_win=probability_win,
                        expected_pnl=expected_pnl,
                        bucket_count=bucket_count,
                        pnl_std=pnl_std,
                    )

                meta_json = {
                    "model_name": model["model_name"],
                    "model_version": model["version"],
                    "rollout_status": model["rollout_status"],
                    "scoring_source": pred.get("source"),
                    "bucket_key": pred.get("bucket_key"),
                    "bucket_level": pred.get("bucket_level"),
                    "bucket_count": bucket_count,
                    "tail_loss_proxy": pred.get("tail_loss_proxy"),
                    "pnl_std": pnl_std,
                    "margin_usage": pred.get("margin_usage"),
                    "uncertainty_level": uncertainty_level,
                    "uncertainty_penalty": uncertainty_penalty,
                    "raw_utility": raw_utility,
                    "min_bucket_count_threshold": settings.decision_hybrid_min_bucket_count,
                    "max_pnl_std_threshold": settings.decision_hybrid_max_pnl_std,
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

