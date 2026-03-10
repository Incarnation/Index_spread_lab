from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import statistics
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


def _mean_or_none(values: list[float]) -> float | None:
    """Return arithmetic mean for numeric values or None when empty.

    Parameters
    ----------
    values:
        Numeric values collected from fold-level metrics.

    Returns
    -------
    float | None
        Arithmetic mean when values exist, otherwise None.
    """
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _stdev_or_none(values: list[float]) -> float | None:
    """Return population standard deviation for numeric values.

    Parameters
    ----------
    values:
        Numeric values collected from fold-level metrics.

    Returns
    -------
    float | None
        Population standard deviation when at least two values exist,
        otherwise None.
    """
    if len(values) < 2:
        return None
    return float(statistics.pstdev(values))


def build_walkforward_windows(*, now_utc: datetime, lookback_days: int, test_days: int) -> dict[str, datetime]:
    """Build train/test windows for a simple walk-forward split."""
    window_start = now_utc - timedelta(days=max(lookback_days, 1))
    test_start = now_utc - timedelta(days=max(test_days, 1))
    if test_start <= window_start:
        test_start = window_start + (now_utc - window_start) / 2
    if test_start >= now_utc:
        test_start = now_utc - timedelta(seconds=1)
    return {"window_start": window_start, "test_start": test_start}


def build_time_series_cv_folds(
    *,
    rows_count: int,
    fold_count: int,
    min_train_rows: int,
    min_test_rows: int,
) -> list[dict[str, int]]:
    """Build ordered expanding-window CV folds for sparse training data.

    Parameters
    ----------
    rows_count:
        Total number of ordered candidate rows available.
    fold_count:
        Requested number of folds (clamped to >=2).
    min_train_rows:
        Minimum training rows required per fold.
    min_test_rows:
        Minimum test rows required per fold.

    Returns
    -------
    list[dict[str, int]]
        Fold descriptors containing `train_end_idx`, `test_start_idx`,
        `test_end_idx`, and row counts for each valid fold.
    """
    total = max(int(rows_count), 0)
    if total <= 0:
        return []
    folds_requested = max(int(fold_count), 2)
    min_train = max(int(min_train_rows), 1)
    min_test = max(int(min_test_rows), 1)
    folds: list[dict[str, int]] = []
    for fold_index in range(folds_requested):
        train_end_idx = int((fold_index + 1) * total / float(folds_requested + 1))
        test_start_idx = train_end_idx
        test_end_idx = int((fold_index + 2) * total / float(folds_requested + 1))
        if fold_index == folds_requested - 1:
            test_end_idx = total
        train_size = max(train_end_idx, 0)
        test_size = max(test_end_idx - test_start_idx, 0)
        if train_size < min_train or test_size < min_test:
            continue
        folds.append(
            {
                "fold_index": fold_index + 1,
                "train_end_idx": train_end_idx,
                "test_start_idx": test_start_idx,
                "test_end_idx": test_end_idx,
                "rows_train": train_size,
                "rows_test": test_size,
            }
        )
    return folds


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

    async def _mark_training_run_skipped(
        self,
        *,
        session,
        training_run_id: int,
        finished_at: datetime,
        notes: str,
        metrics_json: dict[str, Any],
        rows_train: int | None = None,
        rows_test: int | None = None,
    ) -> None:
        """Persist a skipped training attempt with honest status metadata.

        Parameters
        ----------
        session:
            Async database session used for the update statement.
        training_run_id:
            Primary key of the in-flight training run row.
        finished_at:
            UTC timestamp marking when the skip decision was finalized.
        notes:
            Short machine-readable reason stored alongside the run.
        metrics_json:
            JSON payload explaining why the attempt skipped.
        rows_train:
            Optional training-row count to store when known.
        rows_test:
            Optional test-row or total-row count to store when known.
        """
        await session.execute(
            text(
                """
                UPDATE training_runs
                SET status = 'SKIPPED',
                    finished_at = :finished_at,
                    rows_train = COALESCE(:rows_train, rows_train),
                    rows_test = COALESCE(:rows_test, rows_test),
                    metrics_json = CAST(:metrics_json AS jsonb),
                    notes = :notes
                WHERE training_run_id = :training_run_id
                """
            ),
            {
                "training_run_id": training_run_id,
                "finished_at": finished_at,
                "rows_train": rows_train,
                "rows_test": rows_test,
                "metrics_json": _to_json(metrics_json),
                "notes": notes,
            },
        )

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

    def _aggregate_cv_metrics(self, *, fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate fold-level evaluation metrics into one sparse-CV summary.

        Parameters
        ----------
        fold_metrics:
            Per-fold output payloads from `_evaluate_model` with optional fold
            metadata attached by the caller.

        Returns
        -------
        dict[str, Any]
            Aggregated metrics payload preserving existing gate fields and
            adding fold-level diagnostics plus metric standard deviations.
        """
        if not fold_metrics:
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
                "evaluation_mode": "sparse_time_series_cv",
                "cv_folds_used": 0,
                "cv_fold_metrics": [],
                "cv_metric_std": {},
            }

        total_rows_test = sum(int(m.get("rows_test") or 0) for m in fold_metrics)
        total_resolved = sum(int(m.get("resolved_test") or 0) for m in fold_metrics)
        total_tp50 = sum(int(m.get("tp50_test") or 0) for m in fold_metrics)
        total_tp100 = sum(int(m.get("tp100_test") or 0) for m in fold_metrics)
        float_fields = (
            "tp50_rate_test",
            "tp100_rate_test",
            "expectancy_test",
            "max_drawdown_test",
            "tail_loss_proxy_test",
            "avg_margin_usage_test",
            "brier_score_tp50",
            "mae_expected_pnl",
        )
        means: dict[str, float | None] = {}
        stds: dict[str, float | None] = {}
        for field in float_fields:
            values = [float(metric[field]) for metric in fold_metrics if metric.get(field) is not None]
            means[field] = _mean_or_none(values)
            stds[field] = _stdev_or_none(values)

        # Aggregate by-side performance so UI and gates can compare consistency
        # across folds for call/put cohorts even in sparse training windows.
        side_accumulator: dict[str, dict[str, Any]] = {}
        side_float_fields = ("tp50_rate", "tp100_at_expiry_rate", "expectancy", "max_drawdown", "tail_loss_proxy", "avg_margin_usage")
        for metric in fold_metrics:
            by_side = metric.get("by_side")
            if not isinstance(by_side, dict):
                continue
            for side, side_stats in by_side.items():
                if not isinstance(side_stats, dict):
                    continue
                slot = side_accumulator.setdefault(
                    str(side),
                    {
                        "resolved": 0,
                        "tp50": 0,
                        "tp100_at_expiry": 0,
                        "float_values": {field: [] for field in side_float_fields},
                    },
                )
                slot["resolved"] += int(side_stats.get("resolved") or 0)
                slot["tp50"] += int(side_stats.get("tp50") or 0)
                slot["tp100_at_expiry"] += int(side_stats.get("tp100_at_expiry") or 0)
                for field in side_float_fields:
                    value = side_stats.get(field)
                    if value is not None:
                        slot["float_values"][field].append(float(value))

        by_side_agg: dict[str, dict[str, Any]] = {}
        for side, payload in side_accumulator.items():
            values_map = payload["float_values"]
            by_side_agg[side] = {
                "resolved": payload["resolved"],
                "tp50": payload["tp50"],
                "tp100_at_expiry": payload["tp100_at_expiry"],
                "tp50_rate": _mean_or_none(values_map["tp50_rate"]),
                "tp100_at_expiry_rate": _mean_or_none(values_map["tp100_at_expiry_rate"]),
                "expectancy": _mean_or_none(values_map["expectancy"]),
                "max_drawdown": _mean_or_none(values_map["max_drawdown"]),
                "tail_loss_proxy": _mean_or_none(values_map["tail_loss_proxy"]),
                "avg_margin_usage": _mean_or_none(values_map["avg_margin_usage"]),
            }

        return {
            "rows_test": total_rows_test,
            "resolved_test": total_resolved,
            "tp50_test": total_tp50,
            "tp100_test": total_tp100,
            "tp50_rate_test": means["tp50_rate_test"],
            "tp100_rate_test": means["tp100_rate_test"],
            "expectancy_test": means["expectancy_test"],
            "max_drawdown_test": means["max_drawdown_test"],
            "tail_loss_proxy_test": means["tail_loss_proxy_test"],
            "avg_margin_usage_test": means["avg_margin_usage_test"],
            "brier_score_tp50": means["brier_score_tp50"],
            "mae_expected_pnl": means["mae_expected_pnl"],
            "by_side": by_side_agg,
            "evaluation_mode": "sparse_time_series_cv",
            "cv_folds_used": len(fold_metrics),
            "cv_fold_metrics": fold_metrics,
            "cv_metric_std": stds,
        }

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        """Run one training cycle using walk-forward or sparse-data CV mode.

        Parameters
        ----------
        force:
            When true, bypasses scheduler enable checks.

        Returns
        -------
        dict[str, Any]
            Status payload containing run identifiers, selected evaluation mode,
            row counts, and metrics or skip/failure reasons.
        """
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
            "sparse_cv_enabled": settings.trainer_sparse_cv_enabled,
            "sparse_cv_min_rows": settings.trainer_sparse_cv_min_rows,
            "sparse_cv_folds": settings.trainer_sparse_cv_folds,
            "sparse_cv_min_train_rows": settings.trainer_sparse_cv_min_train_rows,
            "sparse_cv_min_test_rows": settings.trainer_sparse_cv_min_test_rows,
            "min_bucket_size": settings.trainer_min_bucket_size,
            "prior_strength": settings.trainer_prior_strength,
            "adaptive_prior_enabled": settings.trainer_adaptive_prior_enabled,
            "adaptive_prior_reference_rows": settings.trainer_adaptive_prior_reference_rows,
            "adaptive_prior_min": settings.trainer_adaptive_prior_min,
            "adaptive_prior_max": settings.trainer_adaptive_prior_max,
            "utility_prob_weight": settings.trainer_utility_prob_weight,
            "utility_tail_penalty": settings.trainer_utility_tail_penalty,
            "utility_margin_penalty": settings.trainer_utility_margin_penalty,
        }

        async with SessionLocal() as session:
            run_id = await self._insert_training_run_start(session=session, started_at=now_utc, config_json=config_json)
            await session.commit()

            try:
                all_rows = await self._load_resolved_candidates(session=session, window_start=windows["window_start"])
                rows_total = len(all_rows)
                trainer_min_rows = max(settings.trainer_min_rows, 1)
                sparse_cv_min_rows = max(settings.trainer_sparse_cv_min_rows, 1)
                can_run_walk_forward = rows_total >= trainer_min_rows
                can_run_sparse_cv = settings.trainer_sparse_cv_enabled and rows_total >= sparse_cv_min_rows
                if (not can_run_walk_forward) and (not can_run_sparse_cv):
                    rows_required = min(trainer_min_rows, sparse_cv_min_rows) if settings.trainer_sparse_cv_enabled else trainer_min_rows
                    await self._mark_training_run_skipped(
                        session=session,
                        training_run_id=run_id,
                        finished_at=now_utc,
                        rows_test=rows_total,
                        metrics_json={
                            "rows_total": rows_total,
                            "rows_required": rows_required,
                            "skipped_reason": "insufficient_rows",
                        },
                        notes="insufficient_rows",
                    )
                    await session.commit()
                    return {
                        "skipped": True,
                        "reason": "insufficient_rows",
                        "training_run_id": run_id,
                        "rows_total": rows_total,
                        "rows_required": rows_required,
                    }

                evaluation_mode = "walk_forward"
                walkforward_fold_value = 1
                cv_folds: list[dict[str, int]] = []
                train_rows_for_model: list[dict[str, Any]] = []
                rows_train_for_run = 0
                rows_test_for_run = 0
                train_start = None
                train_end = None
                test_start = None
                test_end = None
                eval_metrics: dict[str, Any]

                if (not can_run_walk_forward) and can_run_sparse_cv:
                    evaluation_mode = "sparse_time_series_cv"
                    cv_folds = build_time_series_cv_folds(
                        rows_count=rows_total,
                        fold_count=settings.trainer_sparse_cv_folds,
                        min_train_rows=settings.trainer_sparse_cv_min_train_rows,
                        min_test_rows=settings.trainer_sparse_cv_min_test_rows,
                    )
                    if not cv_folds:
                        await self._mark_training_run_skipped(
                            session=session,
                            training_run_id=run_id,
                            finished_at=now_utc,
                            rows_test=rows_total,
                            metrics_json={
                                "rows_total": rows_total,
                                "sparse_cv_folds_requested": settings.trainer_sparse_cv_folds,
                                "skipped_reason": "insufficient_cv_folds",
                            },
                            notes="insufficient_cv_folds",
                        )
                        await session.commit()
                        return {
                            "skipped": True,
                            "reason": "insufficient_cv_folds",
                            "training_run_id": run_id,
                            "rows_total": rows_total,
                        }

                    fold_metrics: list[dict[str, Any]] = []
                    for fold in cv_folds:
                        fold_train_rows = all_rows[: int(fold["train_end_idx"])]
                        fold_test_rows = all_rows[int(fold["test_start_idx"]) : int(fold["test_end_idx"])]
                        fold_model_payload = train_bucket_model(
                            rows=fold_train_rows,
                            min_bucket_size=settings.trainer_min_bucket_size,
                            prior_strength=settings.trainer_prior_strength,
                            adaptive_prior_enabled=settings.trainer_adaptive_prior_enabled,
                            adaptive_prior_reference_rows=settings.trainer_adaptive_prior_reference_rows,
                            adaptive_prior_min=settings.trainer_adaptive_prior_min,
                            adaptive_prior_max=settings.trainer_adaptive_prior_max,
                            utility_prob_weight=settings.trainer_utility_prob_weight,
                            utility_tail_penalty=settings.trainer_utility_tail_penalty,
                            utility_margin_penalty=settings.trainer_utility_margin_penalty,
                        )
                        fold_eval = self._evaluate_model(model_payload=fold_model_payload, rows=fold_test_rows)
                        fold_eval["fold_index"] = int(fold["fold_index"])
                        fold_eval["rows_train"] = int(fold["rows_train"])
                        fold_eval["rows_test"] = int(fold["rows_test"])
                        fold_metrics.append(fold_eval)

                    eval_metrics = self._aggregate_cv_metrics(fold_metrics=fold_metrics)
                    train_rows_for_model = list(all_rows)
                    rows_train_for_run = len(train_rows_for_model)
                    rows_test_for_run = int(eval_metrics.get("rows_test") or 0)
                    walkforward_fold_value = len(cv_folds)
                    train_start = train_rows_for_model[0]["ts"] if train_rows_for_model else None
                    train_end = train_rows_for_model[-1]["ts"] if train_rows_for_model else None
                    first_test_idx = int(cv_folds[0]["test_start_idx"])
                    last_test_end_idx = int(cv_folds[-1]["test_end_idx"])
                    if 0 <= first_test_idx < len(all_rows):
                        test_start = all_rows[first_test_idx]["ts"]
                    if last_test_end_idx > 0 and last_test_end_idx - 1 < len(all_rows):
                        test_end = all_rows[last_test_end_idx - 1]["ts"]
                else:
                    train_rows = [r for r in all_rows if r["ts"] < windows["test_start"]]
                    test_rows = [r for r in all_rows if r["ts"] >= windows["test_start"]]
                    if len(train_rows) < settings.trainer_min_train_rows or len(test_rows) < settings.trainer_min_test_rows:
                        await self._mark_training_run_skipped(
                            session=session,
                            training_run_id=run_id,
                            finished_at=now_utc,
                            rows_train=len(train_rows),
                            rows_test=len(test_rows),
                            metrics_json={
                                "rows_total": len(all_rows),
                                "rows_train": len(train_rows),
                                "rows_test": len(test_rows),
                                "skipped_reason": "insufficient_split_rows",
                            },
                            notes="insufficient_split_rows",
                        )
                        await session.commit()
                        return {
                            "skipped": True,
                            "reason": "insufficient_split_rows",
                            "training_run_id": run_id,
                            "rows_train": len(train_rows),
                            "rows_test": len(test_rows),
                        }
                    train_rows_for_model = list(train_rows)
                    rows_train_for_run = len(train_rows_for_model)
                    rows_test_for_run = len(test_rows)
                    train_start = train_rows[0]["ts"] if train_rows else None
                    train_end = train_rows[-1]["ts"] if train_rows else None
                    test_start = test_rows[0]["ts"] if test_rows else None
                    test_end = test_rows[-1]["ts"] if test_rows else None

                model_payload = train_bucket_model(
                    rows=train_rows_for_model,
                    min_bucket_size=settings.trainer_min_bucket_size,
                    prior_strength=settings.trainer_prior_strength,
                    adaptive_prior_enabled=settings.trainer_adaptive_prior_enabled,
                    adaptive_prior_reference_rows=settings.trainer_adaptive_prior_reference_rows,
                    adaptive_prior_min=settings.trainer_adaptive_prior_min,
                    adaptive_prior_max=settings.trainer_adaptive_prior_max,
                    utility_prob_weight=settings.trainer_utility_prob_weight,
                    utility_tail_penalty=settings.trainer_utility_tail_penalty,
                    utility_margin_penalty=settings.trainer_utility_margin_penalty,
                )
                if evaluation_mode == "walk_forward":
                    eval_metrics = self._evaluate_model(
                        model_payload=model_payload,
                        rows=[r for r in all_rows if r["ts"] >= windows["test_start"]],
                    )
                    eval_metrics["evaluation_mode"] = "walk_forward"

                model_version_tag = f"wf_{now_utc.strftime('%Y%m%d%H%M%S')}"

                feature_spec_json = {
                    "feature_family": "candidate_bucket_features_v2",
                    "keys": [
                        "spread_side",
                        "target_dte",
                        "delta_bucket",
                        "credit_bucket",
                        "context_regime",
                        "vix_regime",
                        "term_structure_regime",
                        "spy_spx_ratio_regime",
                        "cboe_regime",
                        "cboe_wall_proximity",
                        "vix_delta_interaction_bucket",
                        "dte_credit_interaction_bucket",
                    ],
                    "bucket_hierarchy_levels": ["full", "relaxed_market", "core"],
                    "fallback_order": model_payload.get("fallback_order"),
                    "utility_weights": model_payload.get("utility_weights"),
                }
                data_snapshot_json = {
                    "window_start_utc": windows["window_start"].isoformat(),
                    "test_start_utc": (test_start.isoformat() if isinstance(test_start, datetime) else windows["test_start"].isoformat()),
                    "rows_total": rows_total,
                    "rows_train": rows_train_for_run,
                    "rows_test": rows_test_for_run,
                    "evaluation_mode": evaluation_mode,
                    "cv_folds": cv_folds,
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
                        "notes": ("weekly_sparse_cv" if evaluation_mode == "sparse_time_series_cv" else "weekly_walk_forward"),
                    },
                )
                model_version_id = int(model_insert.scalar_one())

                leakage_checks_json = {
                    "train_end_before_test_start": bool((train_end is not None) and (test_start is not None) and (train_end < test_start)),
                    "as_of_window_cutoff_utc": (
                        test_start.isoformat() if isinstance(test_start, datetime) else windows["test_start"].isoformat()
                    ),
                    "evaluation_mode": evaluation_mode,
                }
                await session.execute(
                    text(
                        """
                        UPDATE training_runs
                        SET model_version_id = :model_version_id,
                            finished_at = :finished_at,
                            status = 'COMPLETED',
                            walkforward_fold = :walkforward_fold,
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
                        "walkforward_fold": walkforward_fold_value,
                        "train_window_start": train_start,
                        "train_window_end": train_end,
                        "test_window_start": test_start,
                        "test_window_end": test_end,
                        "rows_train": rows_train_for_run,
                        "rows_test": rows_test_for_run,
                        "leakage_checks_json": _to_json(leakage_checks_json),
                        "artifacts_json": _to_json({"model_payload_path": "model_versions.data_snapshot_json.model_payload"}),
                        "metrics_json": _to_json(eval_metrics),
                        "notes": ("completed_sparse_cv" if evaluation_mode == "sparse_time_series_cv" else "completed"),
                    },
                )
                await session.commit()
                logger.info(
                    "trainer_job: completed run_id={} model_version_id={} rows_train={} rows_test={} mode={}",
                    run_id,
                    model_version_id,
                    rows_train_for_run,
                    rows_test_for_run,
                    evaluation_mode,
                )
                return {
                    "skipped": False,
                    "reason": None,
                    "training_run_id": run_id,
                    "model_version_id": model_version_id,
                    "rows_train": rows_train_for_run,
                    "rows_test": rows_test_for_run,
                    "evaluation_mode": evaluation_mode,
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

