from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

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
