from __future__ import annotations

from types import SimpleNamespace

import pytest

from spx_backend.web.routers.admin import admin_run_performance_analytics


class _FakePerformanceAnalyticsJob:
    """Capture run_once invocations for admin endpoint unit tests."""

    def __init__(self) -> None:
        """Initialize call-tracking fields used by assertions."""
        self.calls: list[dict] = []

    async def run_once(self, *, force: bool = False) -> dict:
        """Record invocation arguments and return a deterministic payload."""
        self.calls.append({"force": force})
        return {"ok": True, "force": force}


@pytest.mark.asyncio
async def test_admin_run_performance_analytics_forces_job_run() -> None:
    """POST admin endpoint should invoke analytics run_once(force=True)."""
    job = _FakePerformanceAnalyticsJob()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(performance_analytics_job=job)))

    result = await admin_run_performance_analytics(request=request)

    assert result["ok"] is True
    assert result["force"] is True
    assert job.calls == [{"force": True}]
