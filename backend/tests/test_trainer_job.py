from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import spx_backend.jobs.trainer_job as trainer_module
from spx_backend.config import settings
from spx_backend.jobs.trainer_job import TrainerJob, build_time_series_cv_folds, build_walkforward_windows


def test_build_walkforward_windows_orders_train_before_test() -> None:
    now_utc = datetime(2026, 2, 14, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
    windows = build_walkforward_windows(now_utc=now_utc, lookback_days=365, test_days=28)
    assert windows["window_start"] < windows["test_start"] < now_utc


def test_build_walkforward_windows_handles_tiny_ranges() -> None:
    now_utc = datetime(2026, 2, 14, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
    windows = build_walkforward_windows(now_utc=now_utc, lookback_days=1, test_days=10)
    assert windows["window_start"] < windows["test_start"] < now_utc


def test_build_time_series_cv_folds_enforces_monotonic_windows() -> None:
    """CV fold builder should emit ordered train/test windows with valid sizes."""
    folds = build_time_series_cv_folds(
        rows_count=40,
        fold_count=4,
        min_train_rows=6,
        min_test_rows=4,
    )
    assert len(folds) >= 2
    previous_test_end = 0
    for fold in folds:
        assert fold["train_end_idx"] == fold["test_start_idx"]
        assert fold["test_end_idx"] > fold["test_start_idx"]
        assert fold["rows_train"] >= 6
        assert fold["rows_test"] >= 4
        assert fold["test_start_idx"] >= previous_test_end
        previous_test_end = fold["test_end_idx"]


def test_aggregate_cv_metrics_returns_mean_and_std_payload() -> None:
    """Aggregated CV metrics should include means, std values, and fold traces."""
    fold_metrics = [
        {
            "rows_test": 8,
            "resolved_test": 8,
            "tp50_test": 5,
            "tp100_test": 2,
            "tp50_rate_test": 0.625,
            "tp100_rate_test": 0.25,
            "expectancy_test": 20.0,
            "max_drawdown_test": 40.0,
            "tail_loss_proxy_test": -15.0,
            "avg_margin_usage_test": 900.0,
            "brier_score_tp50": 0.22,
            "mae_expected_pnl": 12.0,
            "by_side": {},
            "fold_index": 1,
        },
        {
            "rows_test": 7,
            "resolved_test": 7,
            "tp50_test": 3,
            "tp100_test": 1,
            "tp50_rate_test": 3 / 7,
            "tp100_rate_test": 1 / 7,
            "expectancy_test": 10.0,
            "max_drawdown_test": 55.0,
            "tail_loss_proxy_test": -22.0,
            "avg_margin_usage_test": 980.0,
            "brier_score_tp50": 0.28,
            "mae_expected_pnl": 16.0,
            "by_side": {},
            "fold_index": 2,
        },
    ]
    result = TrainerJob()._aggregate_cv_metrics(fold_metrics=fold_metrics)
    assert result["evaluation_mode"] == "sparse_time_series_cv"
    assert result["cv_folds_used"] == 2
    assert len(result["cv_fold_metrics"]) == 2
    assert result["rows_test"] == 15
    assert result["tp50_test"] == 8
    assert result["tp50_rate_test"] is not None
    assert "tp50_rate_test" in result["cv_metric_std"]


class _FakeExecResult:
    """Minimal SQLAlchemy-like result wrapper for trainer job tests."""

    def __init__(self, *, scalar_result=None):
        """Store scalar values returned by insert-returning statements."""
        self._scalar_result = scalar_result

    def scalar_one(self):
        """Return one scalar value for insert-returning SQL branches."""
        return self._scalar_result


class _CaptureTrainerSession:
    """Fake async DB session that records trainer inserts and skip updates."""

    def __init__(self) -> None:
        """Initialize SQL capture containers for assertions."""
        self.update_sql: list[str] = []
        self.update_params: list[dict] = []

    async def execute(self, stmt, params=None):  # noqa: ANN001 - SQLAlchemy text object in production
        """Capture insert/update statements used by the trainer skip path."""
        sql = str(stmt)
        if "INSERT INTO training_runs" in sql:
            return _FakeExecResult(scalar_result=321)
        if "UPDATE training_runs" in sql:
            self.update_sql.append(sql)
            self.update_params.append(dict(params or {}))
            return _FakeExecResult()
        raise AssertionError(f"Unexpected SQL in trainer test: {sql}")

    async def commit(self) -> None:
        """No-op commit hook for fake session compatibility."""
        return None

    async def rollback(self) -> None:
        """No-op rollback hook for fake session compatibility."""
        return None


class _SessionFactory:
    """Async context-manager factory that returns one fake trainer session."""

    def __init__(self, session: _CaptureTrainerSession):
        """Store the fake session returned during the async context."""
        self._session = session

    def __call__(self):
        """Return self to mimic the production SessionLocal callable."""
        return self

    async def __aenter__(self) -> _CaptureTrainerSession:
        """Yield the fake trainer session to the job code."""
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        """Do not suppress exceptions raised inside the async block."""
        return False


@pytest.mark.asyncio
async def test_run_once_marks_insufficient_rows_as_skipped(monkeypatch) -> None:
    """Trainer should persist SKIPPED when row volume is below training minimums."""
    capture_session = _CaptureTrainerSession()
    monkeypatch.setattr(trainer_module, "SessionLocal", _SessionFactory(capture_session))
    monkeypatch.setattr(settings, "trainer_enabled", True)
    monkeypatch.setattr(settings, "trainer_sparse_cv_enabled", False)
    monkeypatch.setattr(settings, "trainer_min_rows", 5)

    async def _fake_load_resolved_candidates(self, *, session, window_start):  # noqa: ANN001, ARG001
        """Return too few rows so the trainer takes the explicit skip path."""
        return [{"ts": datetime(2026, 2, 14, 15, 0, 0, tzinfo=ZoneInfo("UTC"))}] * 2

    monkeypatch.setattr(TrainerJob, "_load_resolved_candidates", _fake_load_resolved_candidates)

    result = await TrainerJob().run_once(force=True)

    assert result["skipped"] is True
    assert result["reason"] == "insufficient_rows"
    assert capture_session.update_sql
    assert "status = 'SKIPPED'" in capture_session.update_sql[0]
    assert capture_session.update_params[0]["notes"] == "insufficient_rows"
