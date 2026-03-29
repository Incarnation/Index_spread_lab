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

    market_open: bool = True

    def __init__(self, *args, **kwargs):
        """Accept production constructor args."""
        return None

    async def is_open(self, now_et) -> bool:
        """Return deterministic open/closed status for guard-logic assertions."""
        return self.market_open


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
async def test_lifespan_wires_performance_analytics_job_when_enabled(monkeypatch) -> None:
    """Verify analytics refresh job is scheduled and exposed on app.state."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    spx_snapshot_job = _FakeJob()
    performance_analytics_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: spx_snapshot_job)
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_performance_analytics_job", lambda **kwargs: performance_analytics_job)
    monkeypatch.setattr(app_module, "QuoteJob", _FakeJob)
    monkeypatch.setattr(app_module, "GexJob", _FakeJob)
    monkeypatch.setattr(app_module, "DecisionJob", _FakeJob)
    monkeypatch.setattr(app_module, "FeatureBuilderJob", _FakeJob)
    monkeypatch.setattr(app_module, "LabelerJob", _FakeJob)
    monkeypatch.setattr(app_module, "TradePnlJob", _FakeJob)
    monkeypatch.setattr(app_module, "TrainerJob", _FakeJob)
    monkeypatch.setattr(app_module, "ShadowInferenceJob", _FakeJob)
    monkeypatch.setattr(app_module, "PromotionGateJob", _FakeJob)

    monkeypatch.setattr(settings, "performance_analytics_enabled", True)
    monkeypatch.setattr(settings, "performance_analytics_interval_minutes", 5)
    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "skip_startup_warmup", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        job_ids = {job["id"] for job in scheduler.jobs}
        assert "performance_analytics_job" in job_ids
        assert "performance_analytics_job_close" in job_ids
        assert app_module.app.state.performance_analytics_job is performance_analytics_job
        assert performance_analytics_job.run_calls == 1
        assert _job_by_id(scheduler, "performance_analytics_job_close")["kwargs"]["kwargs"] == {"force": True}


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


@pytest.mark.asyncio
async def test_lifespan_schedules_ml_jobs_with_daily_and_weekly_cadence(monkeypatch) -> None:
    """Verify ML jobs use daily-after-close and weekly-after-trainer cron wiring."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: _FakeJob())
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

    monkeypatch.setattr(settings, "skip_startup_warmup", True)
    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "performance_analytics_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", True)
    monkeypatch.setattr(settings, "shadow_inference_enabled", True)
    monkeypatch.setattr(settings, "trainer_enabled", True)
    monkeypatch.setattr(settings, "promotion_gate_enabled", True)
    monkeypatch.setattr(settings, "trainer_weekday", "sat")
    monkeypatch.setattr(settings, "trainer_hour", 9)
    monkeypatch.setattr(settings, "trainer_minute", 0)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler
        labeler_job = _job_by_id(scheduler, "labeler_job")
        assert labeler_job["trigger"] == "cron"
        assert labeler_job["kwargs"]["day_of_week"] == "mon-fri"
        assert labeler_job["kwargs"]["hour"] == 16
        assert labeler_job["kwargs"]["minute"] == 15
        assert labeler_job["kwargs"]["kwargs"] == {"force": True}

        shadow_job = _job_by_id(scheduler, "shadow_inference_job")
        assert shadow_job["trigger"] == "cron"
        assert shadow_job["kwargs"]["day_of_week"] == "mon-fri"
        assert shadow_job["kwargs"]["hour"] == 16
        assert shadow_job["kwargs"]["minute"] == 20
        assert shadow_job["kwargs"]["kwargs"] == {"force": True}

        trainer_job = _job_by_id(scheduler, "trainer_job")
        assert trainer_job["trigger"] == "cron"
        assert trainer_job["kwargs"]["day_of_week"] == "sat"
        assert trainer_job["kwargs"]["hour"] == 9
        assert trainer_job["kwargs"]["minute"] == 0

        promotion_job = _job_by_id(scheduler, "promotion_gate_job")
        assert promotion_job["trigger"] == "cron"
        assert promotion_job["kwargs"]["day_of_week"] == "sat"
        assert promotion_job["kwargs"]["hour"] == 10
        assert promotion_job["kwargs"]["minute"] == 0


@pytest.mark.asyncio
async def test_lifespan_holiday_guard_skips_hard_rth_and_entry_jobs(monkeypatch) -> None:
    """Verify holiday guards skip hard-RTH close ticks and entry-time ticks."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    snapshot_job = _FakeJob()
    decision_job = _FakeJob()
    feature_builder_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(_FakeClockCache, "market_open", False)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: snapshot_job)
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "QuoteJob", _FakeJob)
    monkeypatch.setattr(app_module, "GexJob", _FakeJob)
    monkeypatch.setattr(app_module, "DecisionJob", lambda *args, **kwargs: decision_job)
    monkeypatch.setattr(app_module, "FeatureBuilderJob", lambda *args, **kwargs: feature_builder_job)
    monkeypatch.setattr(app_module, "LabelerJob", _FakeJob)
    monkeypatch.setattr(app_module, "TradePnlJob", _FakeJob)
    monkeypatch.setattr(app_module, "TrainerJob", _FakeJob)
    monkeypatch.setattr(app_module, "ShadowInferenceJob", _FakeJob)
    monkeypatch.setattr(app_module, "PromotionGateJob", _FakeJob)

    monkeypatch.setattr(settings, "skip_startup_warmup", True)
    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "performance_analytics_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", False)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", False)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)
    monkeypatch.setattr(settings, "feature_builder_enabled", True)
    monkeypatch.setattr(settings, "decision_entry_times", "10:00")

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler

        snapshot_close_job = _job_by_id(scheduler, "snapshot_job_close")
        snapshot_close_result = await snapshot_close_job["func"](**snapshot_close_job["kwargs"]["kwargs"])
        assert snapshot_close_result["skipped"] is True
        assert snapshot_close_result["reason"] == "non_trading_day"
        assert snapshot_job.run_calls == 0

        decision_entry_job = _job_by_id(scheduler, "decision_job_1000")
        assert decision_entry_job["kwargs"]["day_of_week"] == "mon-fri"
        decision_entry_result = await decision_entry_job["func"]()
        assert decision_entry_result["skipped"] is True
        assert decision_entry_result["reason"] == "market_closed_or_holiday"
        assert decision_job.run_calls == 0

        feature_entry_job = _job_by_id(scheduler, "feature_builder_job_1000")
        assert feature_entry_job["kwargs"]["day_of_week"] == "mon-fri"
        feature_entry_result = await feature_entry_job["func"]()
        assert feature_entry_result["skipped"] is True
        assert feature_entry_result["reason"] == "market_closed_or_holiday"
        assert feature_builder_job.run_calls == 0


@pytest.mark.asyncio
async def test_lifespan_holiday_guard_skips_daily_after_close_jobs(monkeypatch) -> None:
    """Verify daily after-close ML jobs skip on non-trading days."""
    import apscheduler.schedulers.asyncio as aps_asyncio

    labeler_job = _FakeJob()
    shadow_job = _FakeJob()

    monkeypatch.setattr(aps_asyncio, "AsyncIOScheduler", _FakeScheduler)
    monkeypatch.setattr(app_module, "init_db", _fake_init_db)
    monkeypatch.setattr(app_module, "get_tradier_client", lambda: object())
    monkeypatch.setattr(app_module, "MarketClockCache", _FakeClockCache)
    monkeypatch.setattr(_FakeClockCache, "market_open", False)
    monkeypatch.setattr(app_module, "build_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_spy_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "build_vix_snapshot_job", lambda **kwargs: _FakeJob())
    monkeypatch.setattr(app_module, "QuoteJob", _FakeJob)
    monkeypatch.setattr(app_module, "GexJob", _FakeJob)
    monkeypatch.setattr(app_module, "DecisionJob", _FakeJob)
    monkeypatch.setattr(app_module, "FeatureBuilderJob", _FakeJob)
    monkeypatch.setattr(app_module, "LabelerJob", lambda *args, **kwargs: labeler_job)
    monkeypatch.setattr(app_module, "TradePnlJob", _FakeJob)
    monkeypatch.setattr(app_module, "TrainerJob", _FakeJob)
    monkeypatch.setattr(app_module, "ShadowInferenceJob", lambda *args, **kwargs: shadow_job)
    monkeypatch.setattr(app_module, "PromotionGateJob", _FakeJob)

    monkeypatch.setattr(settings, "skip_startup_warmup", True)
    monkeypatch.setattr(settings, "spy_snapshot_enabled", False)
    monkeypatch.setattr(settings, "vix_snapshot_enabled", False)
    monkeypatch.setattr(settings, "cboe_gex_enabled", False)
    monkeypatch.setattr(settings, "performance_analytics_enabled", False)
    monkeypatch.setattr(settings, "labeler_enabled", True)
    monkeypatch.setattr(settings, "trade_pnl_enabled", False)
    monkeypatch.setattr(settings, "trainer_enabled", False)
    monkeypatch.setattr(settings, "shadow_inference_enabled", True)
    monkeypatch.setattr(settings, "promotion_gate_enabled", False)
    monkeypatch.setattr(settings, "feature_builder_enabled", False)

    async with app_module.lifespan(app_module.app):
        scheduler = app_module.app.state.scheduler

        labeler_schedule = _job_by_id(scheduler, "labeler_job")
        labeler_result = await labeler_schedule["func"](**labeler_schedule["kwargs"]["kwargs"])
        assert labeler_result["skipped"] is True
        assert labeler_result["reason"] == "non_trading_day"
        assert labeler_job.run_calls == 0

        shadow_schedule = _job_by_id(scheduler, "shadow_inference_job")
        shadow_result = await shadow_schedule["func"](**shadow_schedule["kwargs"]["kwargs"])
        assert shadow_result["skipped"] is True
        assert shadow_result["reason"] == "non_trading_day"
        assert shadow_job.run_calls == 0
