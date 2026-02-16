from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from spx_backend.jobs.trainer_job import build_walkforward_windows


def test_build_walkforward_windows_orders_train_before_test() -> None:
    now_utc = datetime(2026, 2, 14, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
    windows = build_walkforward_windows(now_utc=now_utc, lookback_days=365, test_days=28)
    assert windows["window_start"] < windows["test_start"] < now_utc


def test_build_walkforward_windows_handles_tiny_ranges() -> None:
    now_utc = datetime(2026, 2, 14, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
    windows = build_walkforward_windows(now_utc=now_utc, lookback_days=1, test_days=10)
    assert windows["window_start"] < windows["test_start"] < now_utc
