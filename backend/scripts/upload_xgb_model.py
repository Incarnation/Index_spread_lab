"""Upload a trained XGBoost entry model to the model_versions table.

The script's only deployment path is ``--upload``: push a model artifact
into ``model_versions`` so it is available for offline / future ML re-entry
on the portfolio path.

The legacy ``--batch-predict`` workflow was removed in Track A.7 along
with the ``trade_candidates`` and ``model_predictions`` tables; live
inference will be re-introduced at decision time on the portfolio path
rather than via pre-baked prediction rows.
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


_KNOWN_MODEL_TYPES: tuple[str, ...] = ("xgb_v1", "xgb_entry_v1", "xgb_entry_v2")


def _load_model_artifacts(model_dir: Path) -> dict[str, Any]:
    """Read classifier.json, regressor.json, and metadata.json from disk.

    The ``model_type`` is taken from ``metadata.json`` (written by
    ``xgb_model.save_model``).  When the field is absent we default to
    ``xgb_entry_v1`` for backward compatibility with artifacts trained
    before the metadata stamp was introduced (see C5 in
    OFFLINE_PIPELINE_AUDIT.md).  An unrecognised stamp raises so the
    operator never silently uploads a model under the wrong inference
    contract.

    Returns a dict suitable for embedding in
    ``model_versions.data_snapshot_json``.
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

    model_type = str(meta.get("model_type") or "xgb_entry_v1")
    if model_type not in _KNOWN_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type {model_type!r} in {meta_path}. "
            f"Expected one of {_KNOWN_MODEL_TYPES}."
        )

    return {
        "model_type": model_type,
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
    model_name : the ``model_name`` value to register on the new
        ``model_versions`` row.  Pick a stable identifier so the offline
        ML re-entry path can look the artifact up later (the legacy
        ``shadow_inference_model_name`` config that this used to mirror
        was removed in Track A).
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

    # algorithm and data_snapshot share the same model_type so downstream
    # selection (model_versions.algorithm) and inference (model_payload.model_type)
    # cannot drift apart.
    algorithm = payload["model_type"]
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
                "algorithm": algorithm,
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


def main() -> None:
    """CLI entry point for uploading a model artifact to ``model_versions``."""
    parser = argparse.ArgumentParser(description="Upload XGBoost entry model to DB")
    parser.add_argument(
        "--model-dir", type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "xgb_entry_model"),
        help="Directory containing classifier.json, regressor.json, metadata.json",
    )
    parser.add_argument(
        "--model-name", type=str, default="spx_credit_spread_v1",
        help="model_name to register in the `model_versions` table",
    )
    parser.add_argument("--activate", action="store_true", help="Set model as active immediately")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

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
