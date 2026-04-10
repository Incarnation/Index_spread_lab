"""Upload a trained XGBoost entry model to the model_versions table.

Supports two deployment paths:
  1. ``--upload``  (default): push model artifact into ``model_versions``
     so the shadow_inference_job can score live candidates on Railway.
  2. ``--batch-predict``: load model from disk, score unscored candidates
     directly from the DB, and write predictions to ``model_predictions``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


def _load_model_artifacts(model_dir: Path) -> dict[str, Any]:
    """Read classifier.json, regressor.json, and metadata.json from disk.

    Returns a dict suitable for embedding in model_versions.data_snapshot_json.
    """
    cls_path = model_dir / "classifier.json"
    reg_path = model_dir / "regressor.json"
    meta_path = model_dir / "metadata.json"

    for p in (cls_path, reg_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    classifier_json = cls_path.read_text()
    regressor_json = reg_path.read_text()
    meta = json.loads(meta_path.read_text())

    return {
        "model_type": "xgb_entry_v1",
        "classifier_json": classifier_json,
        "regressor_json": regressor_json,
        "feature_names": meta.get("feature_names", []),
        "cls_params": meta.get("cls_params", {}),
        "reg_params": meta.get("reg_params", {}),
    }


async def upload_model(
    model_dir: Path,
    model_name: str,
    activate: bool = False,
) -> int:
    """Insert a new model_versions row with the XGBoost payload.

    Parameters
    ----------
    model_dir : directory containing classifier.json / regressor.json / metadata.json.
    model_name : the model_name value (must match shadow_inference_model_name config).
    activate : if True, set rollout_status='active' and is_active=True.

    Returns
    -------
    The new model_version_id.
    """
    from spx_backend.database import SessionLocal

    payload = _load_model_artifacts(model_dir)
    now_utc = datetime.now(tz=timezone.utc)
    version_tag = f"xgb_{now_utc.strftime('%Y%m%d%H%M%S')}"
    rollout = "active" if activate else "shadow"

    data_snapshot: dict[str, Any] = {"model_payload": payload}

    async with SessionLocal() as session:
        row = await session.execute(
            text("""
                INSERT INTO model_versions (
                    model_name, version, algorithm,
                    rollout_status, is_active,
                    data_snapshot_json, feature_spec_json, metrics_json,
                    created_at
                )
                VALUES (
                    :model_name, :version, :algorithm,
                    :rollout_status, :is_active,
                    CAST(:data_snapshot AS jsonb),
                    CAST(:feature_spec AS jsonb),
                    CAST(:metrics AS jsonb),
                    :created_at
                )
                RETURNING model_version_id
            """),
            {
                "model_name": model_name,
                "version": version_tag,
                "algorithm": "xgb_entry_v1",
                "rollout_status": rollout,
                "is_active": activate,
                "data_snapshot": json.dumps(data_snapshot, default=str),
                "feature_spec": json.dumps({"feature_names": payload["feature_names"]}),
                "metrics": json.dumps({}),
                "created_at": now_utc,
            },
        )
        mvid = int(row.scalar_one())

        if activate:
            await session.execute(
                text("""
                    UPDATE model_versions
                    SET is_active = FALSE, rollout_status = 'shadow'
                    WHERE model_name = :model_name
                      AND model_version_id != :mvid
                      AND is_active = TRUE
                """),
                {"model_name": model_name, "mvid": mvid},
            )

        await session.commit()
    return mvid


async def batch_predict(
    model_dir: Path,
    model_name: str,
    limit: int = 5000,
) -> int:
    """Score unscored candidates using the local model and write to DB.

    Parameters
    ----------
    model_dir : directory with model artifacts.
    model_name : model_name to look up the model_version_id.
    limit : max candidates to score per run.

    Returns
    -------
    Number of predictions inserted.
    """
    from spx_backend.database import SessionLocal
    from spx_backend.jobs.modeling import extract_xgb_features, predict_xgb_entry

    payload = _load_model_artifacts(model_dir)

    async with SessionLocal() as session:
        mv_row = (await session.execute(
            text("""
                SELECT model_version_id FROM model_versions
                WHERE model_name = :mn AND algorithm = 'xgb_entry_v1'
                ORDER BY created_at DESC LIMIT 1
            """),
            {"mn": model_name},
        )).fetchone()

        if mv_row is None:
            logger.error("No xgb_entry_v1 model_version found. Run --upload first.")
            return 0
        mvid = int(mv_row.model_version_id)

        candidates = (await session.execute(
            text("""
                SELECT tc.candidate_id, tc.candidate_json, tc.max_loss, tc.ts
                FROM trade_candidates tc
                LEFT JOIN model_predictions mp
                  ON mp.candidate_id = tc.candidate_id
                 AND mp.model_version_id = :mvid
                WHERE tc.candidate_json IS NOT NULL
                  AND mp.prediction_id IS NULL
                ORDER BY tc.ts DESC
                LIMIT :lim
            """),
            {"mvid": mvid, "lim": limit},
        )).fetchall()

        inserted = 0
        for row in candidates:
            cj = row.candidate_json if isinstance(row.candidate_json, dict) else {}
            ts = row.ts if hasattr(row, "ts") else None
            features = extract_xgb_features(cj, candidate_ts=ts)
            pred = predict_xgb_entry(payload, features)

            score_raw = pred["utility_score"]
            await session.execute(
                text("""
                    INSERT INTO model_predictions (
                        candidate_id, model_version_id,
                        prediction_schema_version, score_raw, score_calibrated,
                        probability_win, expected_value, decision, meta_json
                    ) VALUES (
                        :cid, :mvid, 'pred_v1', :sr, :sc,
                        :pw, :ev, :dec, CAST(:meta AS jsonb)
                    )
                    ON CONFLICT (candidate_id, model_version_id) DO UPDATE
                    SET score_raw = EXCLUDED.score_raw,
                        score_calibrated = EXCLUDED.score_calibrated,
                        probability_win = EXCLUDED.probability_win,
                        expected_value = EXCLUDED.expected_value,
                        decision = EXCLUDED.decision,
                        meta_json = EXCLUDED.meta_json
                """),
                {
                    "cid": int(row.candidate_id),
                    "mvid": mvid,
                    "sr": score_raw,
                    "sc": score_raw,
                    "pw": pred["probability_win"],
                    "ev": pred["expected_pnl"],
                    "dec": "TRADE" if pred["probability_win"] >= 0.5 else "SKIP",
                    "meta": json.dumps({"source": "batch_predict", "model_type": "xgb_entry_v1"}),
                },
            )
            inserted += 1

        await session.commit()
    return inserted


def main() -> None:
    """CLI entry point for model upload and batch prediction."""
    parser = argparse.ArgumentParser(description="Upload XGBoost entry model to DB")
    parser.add_argument(
        "--model-dir", type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "xgb_entry_model"),
        help="Directory containing classifier.json, regressor.json, metadata.json",
    )
    parser.add_argument(
        "--model-name", type=str, default="spx_credit_spread_v1",
        help="model_name in model_versions (must match shadow_inference_model_name)",
    )
    parser.add_argument("--activate", action="store_true", help="Set model as active immediately")
    parser.add_argument("--batch-predict", action="store_true", dest="batch",
                        help="Score unscored candidates locally and write to model_predictions")
    parser.add_argument("--limit", type=int, default=5000, help="Max candidates for --batch-predict")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    if args.batch:
        print(f"[BATCH] Scoring up to {args.limit} candidates from {model_dir} ...")
        n = asyncio.run(batch_predict(model_dir, args.model_name, args.limit))
        print(f"[BATCH] Inserted/updated {n} predictions.")
    else:
        print(f"[UPLOAD] Uploading model from {model_dir} ...")
        mvid = asyncio.run(upload_model(model_dir, args.model_name, args.activate))
        status = "ACTIVE" if args.activate else "SHADOW"
        print(f"[UPLOAD] Created model_version_id={mvid} ({status})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
