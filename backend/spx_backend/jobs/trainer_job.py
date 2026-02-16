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
    predict_with_bucket_model,
    summarize_strategy_quality,
    train_bucket_model,
)


def _to_json(value: dict | None) -> str | None:
    """Serialize dict payloads for JSONB SQL statements."""
    if value is None:
        return None
    return json.dumps(value, default=str)


def build_walkforward_windows(*, now_utc: datetime, lookback_days: int, test_days: int) -> dict[str, datetime]:
    """Build train/test windows for a simple walk-forward split."""
    window_start = now_utc - timedelta(days=max(lookback_days, 1))
    test_start = now_utc - timedelta(days=max(test_days, 1))
    if test_start <= window_start:
        test_start = window_start + (now_utc - window_start) / 2
    if test_start >= now_utc:
        test_start = now_utc - timedelta(seconds=1)
    return {"window_start": window_start, "test_start": test_start}


@dataclass(frozen=True)
class TrainerJob:
    """Weekly walk-forward trainer writing model_versions/training_runs."""

    async def _insert_training_run_start(self, *, session, started_at: datetime, config_json: dict[str, Any]) -> int:
        """Create a RUNNING training run row."""
        result = await session.execute(
            text(
                """
                INSERT INTO training_runs (
                  started_at, run_type, status, config_json
                )
                VALUES (
                  :started_at, 'walk_forward', 'RUNNING', CAST(:config_json AS jsonb)
                )
                RETURNING training_run_id
                """
            ),
            {"started_at": started_at, "config_json": _to_json(config_json)},
        )
        return int(result.scalar_one())

    async def _load_resolved_candidates(self, *, session, window_start: datetime) -> list[dict[str, Any]]:
        """Load resolved labeled candidates for training/evaluation."""
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                      candidate_id,
                      ts,
                      candidate_json,
                      max_loss,
                      realized_pnl,
                      COALESCE(hit_tp50_before_sl_or_expiry, false) AS hit_tp50,
                      COALESCE((label_json->>'hit_tp100_at_expiry')::boolean, false) AS hit_tp100
                    FROM trade_candidates
                    WHERE label_status = 'resolved'
                      AND realized_pnl IS NOT NULL
                      AND ts >= :window_start
                    ORDER BY ts ASC, candidate_id ASC
                    """
                ),
                {"window_start": window_start},
            )
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            candidate_json = row.candidate_json if isinstance(row.candidate_json, dict) else {}
            features = extract_candidate_features(
                candidate_json=candidate_json,
                max_loss_points=(float(row.max_loss) if row.max_loss is not None else None),
                contract_multiplier=settings.label_contract_multiplier,
            )
            out.append(
                {
                    "candidate_id": int(row.candidate_id),
                    "ts": row.ts,
                    "features": features,
                    "realized_pnl": float(row.realized_pnl),
                    "hit_tp50": bool(row.hit_tp50),
                    "hit_tp100": bool(row.hit_tp100),
                    "margin_usage": float(features.get("margin_usage") or 0.0),
                    "spread_side": str(features.get("spread_side") or "unknown"),
                }
            )
        return out

    def _evaluate_model(self, *, model_payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate trained model on held-out rows."""
        if not rows:
            return {
                "rows_test": 0,
                "resolved_test": 0,
                "tp50_test": 0,
                "tp100_test": 0,
                "tp50_rate_test": None,
                "tp100_rate_test": None,
                "expectancy_test": None,
                "max_drawdown_test": None,
                "tail_loss_proxy_test": None,
                "avg_margin_usage_test": None,
                "brier_score_tp50": None,
                "mae_expected_pnl": None,
                "by_side": {},
            }

        brier_terms: list[float] = []
        ev_abs_errors: list[float] = []
        realized_pnls = [float(r["realized_pnl"]) for r in rows]
        margin_usages = [float(r["margin_usage"]) for r in rows]
        hit_tp50 = sum(1 for r in rows if bool(r["hit_tp50"]))
        hit_tp100 = sum(1 for r in rows if bool(r["hit_tp100"]))

        by_side_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            pred = predict_with_bucket_model(model_payload=model_payload, features=row["features"])
            truth = 1.0 if bool(row["hit_tp50"]) else 0.0
            brier_terms.append((float(pred["probability_win"]) - truth) ** 2)
            ev_abs_errors.append(abs(float(pred["expected_pnl"]) - float(row["realized_pnl"])))

            side = str(row["spread_side"] or "unknown")
            slot = by_side_map.setdefault(side, {"pnls": [], "margins": [], "tp50": 0, "tp100": 0})
            slot["pnls"].append(float(row["realized_pnl"]))
            slot["margins"].append(float(row["margin_usage"]))
            if bool(row["hit_tp50"]):
                slot["tp50"] += 1
            if bool(row["hit_tp100"]):
                slot["tp100"] += 1

        summary = summarize_strategy_quality(
            realized_pnls=realized_pnls,
            margin_usages=margin_usages,
            hit_tp50_count=hit_tp50,
            hit_tp100_count=hit_tp100,
        )
        by_side: dict[str, dict[str, Any]] = {}
        for side, payload in by_side_map.items():
            by_side[side] = summarize_strategy_quality(
                realized_pnls=list(payload["pnls"]),
                margin_usages=list(payload["margins"]),
                hit_tp50_count=int(payload["tp50"]),
                hit_tp100_count=int(payload["tp100"]),
            )

        return {
            "rows_test": len(rows),
            "resolved_test": summary["resolved"],
            "tp50_test": summary["tp50"],
            "tp100_test": summary["tp100_at_expiry"],
            "tp50_rate_test": summary["tp50_rate"],
            "tp100_rate_test": summary["tp100_at_expiry_rate"],
            "expectancy_test": summary["expectancy"],
            "max_drawdown_test": summary["max_drawdown"],
            "tail_loss_proxy_test": summary["tail_loss_proxy"],
            "avg_margin_usage_test": summary["avg_margin_usage"],
            "brier_score_tp50": (sum(brier_terms) / len(brier_terms)) if brier_terms else None,
            "mae_expected_pnl": (sum(ev_abs_errors) / len(ev_abs_errors)) if ev_abs_errors else None,
            "by_side": by_side,
        }

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        """Run one walk-forward training cycle."""
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        if (not force) and (not settings.trainer_enabled):
            return {"skipped": True, "reason": "trainer_disabled", "now_utc": now_utc.isoformat()}

        windows = build_walkforward_windows(
            now_utc=now_utc,
            lookback_days=settings.trainer_lookback_days,
            test_days=settings.trainer_test_days,
        )
        config_json = {
            "model_name": settings.trainer_model_name,
            "lookback_days": settings.trainer_lookback_days,
            "test_days": settings.trainer_test_days,
            "min_rows": settings.trainer_min_rows,
            "min_train_rows": settings.trainer_min_train_rows,
            "min_test_rows": settings.trainer_min_test_rows,
            "min_bucket_size": settings.trainer_min_bucket_size,
            "prior_strength": settings.trainer_prior_strength,
            "utility_prob_weight": settings.trainer_utility_prob_weight,
            "utility_tail_penalty": settings.trainer_utility_tail_penalty,
            "utility_margin_penalty": settings.trainer_utility_margin_penalty,
        }

        async with SessionLocal() as session:
            run_id = await self._insert_training_run_start(session=session, started_at=now_utc, config_json=config_json)
            await session.commit()

            try:
                all_rows = await self._load_resolved_candidates(session=session, window_start=windows["window_start"])
                if len(all_rows) < settings.trainer_min_rows:
                    await session.execute(
                        text(
                            """
                            UPDATE training_runs
                            SET status = 'COMPLETED',
                                finished_at = :finished_at,
                                rows_test = :rows_test,
                                metrics_json = CAST(:metrics_json AS jsonb),
                                notes = :notes
                            WHERE training_run_id = :training_run_id
                            """
                        ),
                        {
                            "training_run_id": run_id,
                            "finished_at": now_utc,
                            "rows_test": len(all_rows),
                            "metrics_json": _to_json({"rows_total": len(all_rows), "skipped_reason": "insufficient_rows"}),
                            "notes": "insufficient_rows",
                        },
                    )
                    await session.commit()
                    return {
                        "skipped": True,
                        "reason": "insufficient_rows",
                        "training_run_id": run_id,
                        "rows_total": len(all_rows),
                    }

                train_rows = [r for r in all_rows if r["ts"] < windows["test_start"]]
                test_rows = [r for r in all_rows if r["ts"] >= windows["test_start"]]
                if len(train_rows) < settings.trainer_min_train_rows or len(test_rows) < settings.trainer_min_test_rows:
                    await session.execute(
                        text(
                            """
                            UPDATE training_runs
                            SET status = 'COMPLETED',
                                finished_at = :finished_at,
                                rows_train = :rows_train,
                                rows_test = :rows_test,
                                metrics_json = CAST(:metrics_json AS jsonb),
                                notes = :notes
                            WHERE training_run_id = :training_run_id
                            """
                        ),
                        {
                            "training_run_id": run_id,
                            "finished_at": now_utc,
                            "rows_train": len(train_rows),
                            "rows_test": len(test_rows),
                            "metrics_json": _to_json(
                                {
                                    "rows_total": len(all_rows),
                                    "rows_train": len(train_rows),
                                    "rows_test": len(test_rows),
                                    "skipped_reason": "insufficient_split_rows",
                                }
                            ),
                            "notes": "insufficient_split_rows",
                        },
                    )
                    await session.commit()
                    return {
                        "skipped": True,
                        "reason": "insufficient_split_rows",
                        "training_run_id": run_id,
                        "rows_train": len(train_rows),
                        "rows_test": len(test_rows),
                    }

                model_payload = train_bucket_model(
                    rows=train_rows,
                    min_bucket_size=settings.trainer_min_bucket_size,
                    prior_strength=settings.trainer_prior_strength,
                    utility_prob_weight=settings.trainer_utility_prob_weight,
                    utility_tail_penalty=settings.trainer_utility_tail_penalty,
                    utility_margin_penalty=settings.trainer_utility_margin_penalty,
                )
                eval_metrics = self._evaluate_model(model_payload=model_payload, rows=test_rows)

                model_version_tag = f"wf_{now_utc.strftime('%Y%m%d%H%M%S')}"
                train_start = train_rows[0]["ts"] if train_rows else None
                train_end = train_rows[-1]["ts"] if train_rows else None
                test_start = test_rows[0]["ts"] if test_rows else None
                test_end = test_rows[-1]["ts"] if test_rows else None

                feature_spec_json = {
                    "feature_family": "candidate_bucket_features",
                    "keys": ["spread_side", "target_dte", "delta_bucket", "credit_bucket", "context_regime"],
                    "utility_weights": model_payload.get("utility_weights"),
                }
                data_snapshot_json = {
                    "window_start_utc": windows["window_start"].isoformat(),
                    "test_start_utc": windows["test_start"].isoformat(),
                    "rows_total": len(all_rows),
                    "rows_train": len(train_rows),
                    "rows_test": len(test_rows),
                    "model_payload": model_payload,
                }

                model_insert = await session.execute(
                    text(
                        """
                        INSERT INTO model_versions (
                          model_name, version, algorithm, target_label, prediction_type,
                          feature_schema_version, candidate_schema_version, label_schema_version,
                          feature_spec_json, data_snapshot_json,
                          train_start, train_end, val_start, val_end,
                          metrics_json, rollout_status, is_active, notes
                        )
                        VALUES (
                          :model_name, :version, :algorithm, :target_label, :prediction_type,
                          :feature_schema_version, :candidate_schema_version, :label_schema_version,
                          CAST(:feature_spec_json AS jsonb), CAST(:data_snapshot_json AS jsonb),
                          :train_start, :train_end, :val_start, :val_end,
                          CAST(:metrics_json AS jsonb), :rollout_status, :is_active, :notes
                        )
                        RETURNING model_version_id
                        """
                    ),
                    {
                        "model_name": settings.trainer_model_name,
                        "version": model_version_tag,
                        "algorithm": "bucket_empirical_v1",
                        "target_label": "hit_tp50_before_sl_or_expiry",
                        "prediction_type": "hybrid_utility",
                        "feature_schema_version": settings.feature_schema_version,
                        "candidate_schema_version": settings.candidate_schema_version,
                        "label_schema_version": settings.label_schema_version,
                        "feature_spec_json": _to_json(feature_spec_json),
                        "data_snapshot_json": _to_json(data_snapshot_json),
                        "train_start": train_start,
                        "train_end": train_end,
                        "val_start": test_start,
                        "val_end": test_end,
                        "metrics_json": _to_json(eval_metrics),
                        "rollout_status": "shadow",
                        "is_active": False,
                        "notes": "weekly_walk_forward",
                    },
                )
                model_version_id = int(model_insert.scalar_one())

                leakage_checks_json = {
                    "train_end_before_test_start": bool((train_end is not None) and (test_start is not None) and (train_end < test_start)),
                    "as_of_window_cutoff_utc": windows["test_start"].isoformat(),
                }
                await session.execute(
                    text(
                        """
                        UPDATE training_runs
                        SET model_version_id = :model_version_id,
                            finished_at = :finished_at,
                            status = 'COMPLETED',
                            walkforward_fold = 1,
                            train_window_start = :train_window_start,
                            train_window_end = :train_window_end,
                            test_window_start = :test_window_start,
                            test_window_end = :test_window_end,
                            rows_train = :rows_train,
                            rows_test = :rows_test,
                            leakage_checks_json = CAST(:leakage_checks_json AS jsonb),
                            artifacts_json = CAST(:artifacts_json AS jsonb),
                            metrics_json = CAST(:metrics_json AS jsonb),
                            notes = :notes
                        WHERE training_run_id = :training_run_id
                        """
                    ),
                    {
                        "training_run_id": run_id,
                        "model_version_id": model_version_id,
                        "finished_at": now_utc,
                        "train_window_start": train_start,
                        "train_window_end": train_end,
                        "test_window_start": test_start,
                        "test_window_end": test_end,
                        "rows_train": len(train_rows),
                        "rows_test": len(test_rows),
                        "leakage_checks_json": _to_json(leakage_checks_json),
                        "artifacts_json": _to_json({"model_payload_path": "model_versions.data_snapshot_json.model_payload"}),
                        "metrics_json": _to_json(eval_metrics),
                        "notes": "completed",
                    },
                )
                await session.commit()
                logger.info(
                    "trainer_job: completed run_id={} model_version_id={} rows_train={} rows_test={}",
                    run_id,
                    model_version_id,
                    len(train_rows),
                    len(test_rows),
                )
                return {
                    "skipped": False,
                    "reason": None,
                    "training_run_id": run_id,
                    "model_version_id": model_version_id,
                    "rows_train": len(train_rows),
                    "rows_test": len(test_rows),
                    "metrics": eval_metrics,
                }
            except Exception as exc:
                await session.rollback()
                await session.execute(
                    text(
                        """
                        UPDATE training_runs
                        SET finished_at = :finished_at,
                            status = 'FAILED',
                            metrics_json = CAST(:metrics_json AS jsonb),
                            notes = :notes
                        WHERE training_run_id = :training_run_id
                        """
                    ),
                    {
                        "training_run_id": run_id,
                        "finished_at": now_utc,
                        "metrics_json": _to_json({"error": str(exc)}),
                        "notes": "failed",
                    },
                )
                await session.commit()
                logger.exception("trainer_job: failed run_id={}", run_id)
                return {
                    "skipped": False,
                    "reason": "failed",
                    "training_run_id": run_id,
                    "error": str(exc),
                }


def build_trainer_job() -> TrainerJob:
    """Factory helper for TrainerJob."""
    return TrainerJob()

