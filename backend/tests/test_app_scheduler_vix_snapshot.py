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
        self.listeners: list[dict] = []
        self.started = False
        self.shutdown_called = False

    def add_job(self, func, trigger, **kwargs):
        """Capture job metadata for assertions."""
        self.jobs.append({"id": kwargs.get("id"), "trigger": trigger, "func": func, "kwargs": kwargs})

    def add_listener(self, callback, mask):
        """Capture listener registration metadata for assertions."""
        self.listeners.append({"callback": callback, "mask": mask})

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


class _FailingJob:
    """Async job stub that always raises during run_once."""

    def __init__(self, *args, **kwargs):
        """Accept any constructor args used by production job wiring."""
        self.run_calls = 0

    async def run_once(self, *args, **kwargs):
        """Raise a deterministic startup failure for logging assertions."""
        self.run_calls += 1
        raise RuntimeError("forced_startup_failure")


class _FakeClockCache:
    """No-op market clock cache stub for app wiring tests."""

    def __init__(self, *args, **kwargs):
        """Accept production constructor args."""
        return None


async def _fake_init_db() -> None:
    """No-op DB initializer used in lifespan tests."""
    return None


def _job_by_id(scheduler: _FakeScheduler, job_id: str) -> dict:
    """Return one captured scheduler job by id for focused assertions."""
    for job in scheduler.jobs:
        if job["id"] == job_id:
            return job
    raise AssertionError(f"missing job id: {job_id}")


@pytest.mark.asyncio
async def test_lifespan_wires_vix_snapshot_job_when_enabled(monkeypatch) -> None:
    """Verify optional VIX snapshot scheduler job is added and exposed on app.state."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()
    vix_snapshot_job = _FakeJob()
    spy_snapshot_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: vix_snapshot_job)
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: spy_snapshot_job)
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
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "skip_startup_warmup", False)
    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "quote_job" in job_ids
        assert "quote_job_close" in job_ids
        assert "gex_job" in job_ids
        assert "gex_job_close" in job_ids
        assert "snapshot_job" in job_ids
        assert "snapshot_job_close" in job_ids
        assert "snapshot_job_vix" in job_ids
        assert "snapshot_job_vix_close" in job_ids
        assert app_module.app.state.vix_snapshot_job is vix_snapshot_job
        assert vix_snapshot_job.run_calls == 1
        assert _job_by_id(scheduler, "quote_job_close")["kwargs"]["kwargs"] == {"force": True}
        assert _job_by_id(scheduler, "gex_job_close")["kwargs"]["kwargs"] == {"force": True}
        assert _job_by_id(scheduler, "snapshot_job_close")["kwargs"]["kwargs"] == {"force": True}
        assert _job_by_id(scheduler, "snapshot_job_vix_close")["kwargs"]["kwargs"] == {"force": True}
        assert scheduler.listeners
        for job in scheduler.jobs:
            assert job["kwargs"]["max_instances"] == 1
            assert job["kwargs"]["misfire_grace_time"] == 300


@pytest.mark.asyncio
async def test_lifespan_skips_vix_snapshot_job_when_disabled(monkeypatch) -> None:
    """Verify no VIX scheduler job/state is created when feature is disabled."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()
    spy_snapshot_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: spy_snapshot_job)
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
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "skip_startup_warmup", False)
    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "snapshot_job_vix" not in job_ids
        assert "snapshot_job_vix_close" not in job_ids
        assert app_module.app.state.vix_snapshot_job is None


@pytest.mark.asyncio
async def test_lifespan_wires_cboe_gex_job_when_enabled(monkeypatch) -> None:
    """Verify optional CBOE GEX scheduler job is added and exposed on app.state."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()
    cboe_gex_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_cboe_gex_job", lambda **kwargs: cboe_gex_job)
    monkeypatch.setattr(app_module, "QuoteJob", _FakeJob)
    monkeypatch.setattr(app_module, "GexJob", _FakeJob)
    monkeypatch.setattr(app_module, "DecisionJob", _FakeJob)
    monkeypatch.setattr(app_module, "FeatureBuilderJob", _FakeJob)
    monkeypatch.setattr(app_module, "LabelerJob", _FakeJob)
    monkeypatch.setattr(app_module, "TradePnlJob", _FakeJob)
    monkeypatch.setattr(app_module, "TrainerJob", _FakeJob)
    monkeypatch.setattr(app_module, "ShadowInferenceJob", _FakeJob)
    monkeypatch.setattr(app_module, "PromotionGateJob", _FakeJob)

    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "skip_startup_warmup", False)
    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "cboe_gex_job" in job_ids
        assert "cboe_gex_job_close" in job_ids
        assert app_module.app.state.cboe_gex_job is cboe_gex_job
        assert cboe_gex_job.run_calls == 1
        assert _job_by_id(scheduler, "cboe_gex_job_close")["kwargs"]["kwargs"] == {"force": True}


@pytest.mark.asyncio
async def test_lifespan_wires_spy_snapshot_job_when_enabled(monkeypatch) -> None:
    """Verify optional SPY snapshot scheduler job is added and exposed on app.state."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()
    spy_snapshot_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: spy_snapshot_job)
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

    monkeypatch.setattr(settings, "spy_snapshot_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "skip_startup_warmup", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "snapshot_job" in job_ids
        assert "snapshot_job_close" in job_ids
        assert "snapshot_job_spy" in job_ids
        assert "snapshot_job_spy_close" in job_ids
        assert app_module.app.state.spy_snapshot_job is spy_snapshot_job
        assert spy_snapshot_job.run_calls == 1
        assert _job_by_id(scheduler, "snapshot_job_close")["kwargs"]["kwargs"] == {"force": True}
        assert _job_by_id(scheduler, "snapshot_job_spy_close")["kwargs"]["kwargs"] == {"force": True}


@pytest.mark.asyncio
async def test_lifespan_skips_spy_snapshot_job_when_disabled(monkeypatch) -> None:
    """Verify no SPY scheduler job/state is created when feature is disabled."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()
    vix_snapshot_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: _FakeJob())
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

    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "skip_startup_warmup", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "snapshot_job_spy" not in job_ids
        assert "snapshot_job_spy_close" not in job_ids
        assert app_module.app.state.spy_snapshot_job is None


@pytest.mark.asyncio
async def test_lifespan_logs_startup_warmup_failures_without_crashing(monkeypatch) -> None:
    """Verify startup warm-up exceptions are logged and other jobs still run."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    quote_job = _FakeJob()
    snapshot_job = _FailingJob()
    gex_job = _FakeJob()
    log_calls: list[str] = []

    def _capture_exception(message: str, *args) -> None:
        """Capture formatted startup exception log lines for assertions."""
        log_calls.append(message.format(*args))

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: snapshot_job)
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "QuoteJob", lambda *args, **kwargs: quote_job)
    monkeypatch.setattr(app_module, "GexJob", lambda *args, **kwargs: gex_job)
    monkeypatch.setattr(app_module, "DecisionJob", _FakeJob)
    monkeypatch.setattr(app_module, "FeatureBuilderJob", _FakeJob)
    monkeypatch.setattr(app_module, "LabelerJob", _FakeJob)
    monkeypatch.setattr(app_module, "TradePnlJob", _FakeJob)
    monkeypatch.setattr(app_module, "TrainerJob", _FakeJob)
    monkeypatch.setattr(app_module, "ShadowInferenceJob", _FakeJob)
    monkeypatch.setattr(app_module, "PromotionGateJob", _FakeJob)
    monkeypatch.setattr(app_module.logger, "exception", _capture_exception)

    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "skip_startup_warmup", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        assert quote_job.run_calls == 1
        assert snapshot_job.run_calls == 1
        assert gex_job.run_calls == 1

    assert any("startup_warmup: job_id=snapshot_job status=failed" in call for call in log_calls)
