from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal


def _to_json(value: dict | None) -> str | None:
    """Serialize dict payloads for JSONB statements."""
    if value is None:
        return None
    return json.dumps(value, default=str)


def evaluate_promotion_gates(
    *,
    metrics: dict[str, Any],
    min_resolved: int,
    min_tp50_rate: float,
    min_expectancy: float,
    max_drawdown: float,
    min_tail_loss_proxy: float,
    max_avg_margin_usage: float,
) -> dict[str, Any]:
    """Evaluate promotion gates and return detailed pass/fail breakdown."""
    resolved = int(metrics.get("resolved_test") or metrics.get("resolved") or 0)
    tp50_rate = float(metrics.get("tp50_rate_test") or metrics.get("tp50_rate") or 0.0)
    expectancy = float(metrics.get("expectancy_test") or metrics.get("expectancy") or 0.0)
    drawdown = float(metrics.get("max_drawdown_test") or metrics.get("max_drawdown") or 0.0)
    tail_loss_proxy = float(metrics.get("tail_loss_proxy_test") or metrics.get("tail_loss_proxy") or 0.0)
    avg_margin_usage = float(metrics.get("avg_margin_usage_test") or metrics.get("avg_margin_usage") or 0.0)

    checks = {
        "resolved_count": {"value": resolved, "threshold": min_resolved, "pass": resolved >= min_resolved},
        "tp50_rate": {"value": tp50_rate, "threshold": min_tp50_rate, "pass": tp50_rate >= min_tp50_rate},
        "expectancy": {"value": expectancy, "threshold": min_expectancy, "pass": expectancy >= min_expectancy},
        "max_drawdown": {"value": drawdown, "threshold": max_drawdown, "pass": drawdown <= max_drawdown},
        "tail_loss_proxy": {
            "value": tail_loss_proxy,
            "threshold": min_tail_loss_proxy,
            "pass": tail_loss_proxy >= min_tail_loss_proxy,
        },
        "avg_margin_usage": {
            "value": avg_margin_usage,
            "threshold": max_avg_margin_usage,
            "pass": avg_margin_usage <= max_avg_margin_usage,
        },
    }
    passed = all(bool(v["pass"]) for v in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "summary": {
            "resolved": resolved,
            "tp50_rate": tp50_rate,
            "expectancy": expectancy,
            "max_drawdown": drawdown,
            "tail_loss_proxy": tail_loss_proxy,
            "avg_margin_usage": avg_margin_usage,
        },
    }


@dataclass(frozen=True)
class PromotionGateJob:
    """Evaluate model promotion gates from latest walk-forward run."""

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        """Evaluate promotion gates and update model rollout status."""
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        if (not force) and (not settings.promotion_gate_enabled):
            return {"skipped": True, "reason": "promotion_gate_disabled", "now_utc": now_utc.isoformat()}

        async with SessionLocal() as session:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT
                          tr.training_run_id,
                          tr.model_version_id,
                          tr.metrics_json AS run_metrics_json,
                          mv.metrics_json AS model_metrics_json,
                          mv.rollout_status,
                          mv.is_active
                        FROM training_runs tr
                        JOIN model_versions mv ON mv.model_version_id = tr.model_version_id
                        WHERE tr.status = 'COMPLETED'
                          AND tr.model_version_id IS NOT NULL
                          AND mv.model_name = :model_name
                        ORDER BY tr.finished_at DESC NULLS LAST, tr.training_run_id DESC
                        LIMIT 1
                        """
                    ),
                    {"model_name": settings.promotion_gate_model_name},
                )
            ).fetchone()
            if row is None:
                return {"skipped": True, "reason": "no_completed_training_run", "now_utc": now_utc.isoformat()}

            training_run_id = int(row.training_run_id)
            model_version_id = int(row.model_version_id)
            run_metrics = row.run_metrics_json if isinstance(row.run_metrics_json, dict) else {}
            model_metrics = row.model_metrics_json if isinstance(row.model_metrics_json, dict) else {}

            gate = evaluate_promotion_gates(
                metrics=run_metrics,
                min_resolved=settings.promotion_gate_min_resolved,
                min_tp50_rate=settings.promotion_gate_min_tp50_rate,
                min_expectancy=settings.promotion_gate_min_expectancy,
                max_drawdown=settings.promotion_gate_max_drawdown,
                min_tail_loss_proxy=settings.promotion_gate_min_tail_loss_proxy,
                max_avg_margin_usage=settings.promotion_gate_max_avg_margin_usage,
            )
            new_rollout_status = "canary" if gate["passed"] else "shadow"
            new_is_active = bool(settings.promotion_gate_auto_activate and gate["passed"])
            promoted_at = now_utc if new_is_active else None

            run_metrics_merged = dict(run_metrics)
            run_metrics_merged["gate"] = gate
            run_metrics_merged["gate_evaluated_at_utc"] = now_utc.isoformat()

            model_metrics_merged = dict(model_metrics)
            model_metrics_merged["gate"] = gate
            model_metrics_merged["gate_evaluated_at_utc"] = now_utc.isoformat()

            await session.execute(
                text(
                    """
                    UPDATE training_runs
                    SET metrics_json = CAST(:metrics_json AS jsonb),
                        notes = :notes
                    WHERE training_run_id = :training_run_id
                    """
                ),
                {
                    "training_run_id": training_run_id,
                    "metrics_json": _to_json(run_metrics_merged),
                    "notes": ("gate_passed" if gate["passed"] else "gate_failed"),
                },
            )
            await session.execute(
                text(
                    """
                    UPDATE model_versions
                    SET metrics_json = CAST(:metrics_json AS jsonb),
                        rollout_status = :rollout_status,
                        is_active = :is_active,
                        promoted_at = :promoted_at
                    WHERE model_version_id = :model_version_id
                    """
                ),
                {
                    "model_version_id": model_version_id,
                    "metrics_json": _to_json(model_metrics_merged),
                    "rollout_status": new_rollout_status,
                    "is_active": new_is_active,
                    "promoted_at": promoted_at,
                },
            )
            await session.commit()

            logger.info(
                "promotion_gate_job: model_version_id={} passed={} rollout_status={} is_active={}",
                model_version_id,
                gate["passed"],
                new_rollout_status,
                new_is_active,
            )
            return {
                "skipped": False,
                "reason": None,
                "training_run_id": training_run_id,
                "model_version_id": model_version_id,
                "passed": bool(gate["passed"]),
                "rollout_status": new_rollout_status,
                "is_active": new_is_active,
                "gate": gate,
            }


def build_promotion_gate_job() -> PromotionGateJob:
    """Factory helper for PromotionGateJob."""
    return PromotionGateJob()

