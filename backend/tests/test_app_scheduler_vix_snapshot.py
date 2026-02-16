from __future__ import annotations

import pytest

import spx_backend.web.app as app_module
from spx_backend.config import settings


class _FakeScheduler:
    """Capture scheduled jobs without starting real APScheduler threads."""

    def __init__(self, timezone=None):
        """Store constructor args and initialize capture fields."""
        self.timezone = timezone
        self.jobs: list[dict] = []
        self.started = False
        self.shutdown_called = False

    def add_job(self, func, trigger, **kwargs):
        """Capture job metadata for assertions."""
        self.jobs.append({"id": kwargs.get("id"), "trigger": trigger, "func": func, "kwargs": kwargs})

    def start(self):
        """Mark scheduler as started."""
        self.started = True

    def shutdown(self, wait=False):
        """Mark scheduler as shutdown."""
        self.shutdown_called = True


class _FakeJob:
    """Generic async job stub with run counter."""

    def __init__(self, *args, **kwargs):
        """Accept any constructor args used by production job wiring."""
        self.run_calls = 0

    async def run_once(self, *args, **kwargs):
        """Increment call counter and return a minimal result payload."""
        self.run_calls += 1
        return {"ok": True}


class _FakeClockCache:
    """No-op market clock cache stub for app wiring tests."""

    def __init__(self, *args, **kwargs):
        """Accept production constructor args."""
        return None


async def _fake_init_db() -> None:
    """No-op DB initializer used in lifespan tests."""
    return None


@pytest.mark.asyncio
async def test_lifespan_wires_vix_snapshot_job_when_enabled(monkeypatch) -> None:
    """Verify optional VIX snapshot scheduler job is added and exposed on app.state."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()
    vix_snapshot_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: vix_snapshot_job)
    monkeypatch.setattr(app_module, "QuoteJob", _FakeJob)
    monkeypatch.setattr(app_module, "GexJob", _FakeJob)
    monkeypatch.setattr(app_module, "DecisionJob", _FakeJob)
    monkeypatch.setattr(app_module, "FeatureBuilderJob", _FakeJob)
    monkeypatch.setattr(app_module, "LabelerJob", _FakeJob)
    monkeypatch.setattr(app_module, "TradePnlJob", _FakeJob)
    monkeypatch.setattr(app_module, "TrainerJob", _FakeJob)
    monkeypatch.setattr(app_module, "ShadowInferenceJob", _FakeJob)
    monkeypatch.setattr(app_module, "PromotionGateJob", _FakeJob)

    monkeypatch.setattr(settings, "vix_snapshot_enabled", True)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "snapshot_job" in job_ids
        assert "snapshot_job_vix" in job_ids
        assert app_module.app.state.vix_snapshot_job is vix_snapshot_job
        assert vix_snapshot_job.run_calls == 1


@pytest.mark.asyncio
async def test_lifespan_skips_vix_snapshot_job_when_disabled(monkeypatch) -> None:
    """Verify no VIX scheduler job/state is created when feature is disabled."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "QuoteJob", _FakeJob)
    monkeypatch.setattr(app_module, "GexJob", _FakeJob)
    monkeypatch.setattr(app_module, "DecisionJob", _FakeJob)
    monkeypatch.setattr(app_module, "FeatureBuilderJob", _FakeJob)
    monkeypatch.setattr(app_module, "LabelerJob", _FakeJob)
    monkeypatch.setattr(app_module, "TradePnlJob", _FakeJob)
    monkeypatch.setattr(app_module, "TrainerJob", _FakeJob)
    monkeypatch.setattr(app_module, "ShadowInferenceJob", _FakeJob)
    monkeypatch.setattr(app_module, "PromotionGateJob", _FakeJob)

    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "snapshot_job_vix" not in job_ids
        assert app_module.app.state.vix_snapshot_job is None
