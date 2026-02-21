from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from spx_backend.config import settings
from spx_backend.database import init_db
from spx_backend.ingestion.tradier_client import get_tradier_client
from spx_backend.jobs.cboe_gex_job import build_cboe_gex_job
from spx_backend.jobs.decision_job import DecisionJob
from spx_backend.jobs.feature_builder_job import FeatureBuilderJob
from spx_backend.jobs.gex_job import GexJob
from spx_backend.jobs.labeler_job import LabelerJob
from spx_backend.jobs.performance_analytics_job import build_performance_analytics_job
from spx_backend.jobs.promotion_gate_job import PromotionGateJob
from spx_backend.jobs.quote_job import QuoteJob
from spx_backend.jobs.shadow_inference_job import ShadowInferenceJob
from spx_backend.jobs.snapshot_job import build_snapshot_job, build_spy_snapshot_job, build_vix_snapshot_job
from spx_backend.jobs.trainer_job import TrainerJob
from spx_backend.jobs.trade_pnl_job import TradePnlJob
from spx_backend.market_clock import MarketClockCache
from spx_backend.web.routers import admin, auth, public
from spx_backend.web.routers.public import (
    get_performance_analytics,
    get_gex_curve,
    get_label_metrics,
    get_model_ops,
    get_strategy_metrics,
    list_gex_dtes,
    list_gex_expirations,
    list_trades,
)


def _scheduler_event_listener(event: Any) -> None:
    """Log APScheduler job errors/misfires with consistent context.

    Parameters
    ----------
    event:
        APScheduler event object carrying job_id, exception, traceback, and
        other runtime metadata depending on event type.

    Behavior
    --------
    - Logs job exceptions with traceback when available.
    - Logs missed executions as warnings so ingestion gaps are visible.
    """
    job_id = getattr(event, "job_id", "unknown_job")
    exception = getattr(event, "exception", None)
    traceback_value = getattr(event, "traceback", None)
    if exception is not None:
        logger.error("scheduler: job_id={} failed error={} traceback={}", job_id, exception, traceback_value)
        return
    logger.warning("scheduler: job_id={} missed scheduled run", job_id)


def _normalized_rth_interval_minutes(interval_minutes: int, *, job_id: str) -> int:
    """Normalize configured cadence to values representable in cron minute fields.

    Parameters
    ----------
    interval_minutes:
        Desired scheduling cadence loaded from settings.
    job_id:
        Scheduler job identifier used for actionable warning context.

    Returns
    -------
    int
        A positive minute cadence that evenly divides one hour. Invalid values
        fall back to ``10`` minutes so the RTH schedule remains deterministic.
    """
    if interval_minutes > 0 and (60 % interval_minutes) == 0:
        return interval_minutes
    logger.warning(
        "scheduler: job_id={} invalid_interval_minutes={} -> using fallback=10",
        job_id,
        interval_minutes,
    )
    return 10


def _build_rth_regular_trigger(*, interval_minutes: int, timezone: str, job_id: str) -> Any:
    """Build a weekday RTH trigger spanning 09:30 through 15:50 ET.

    Parameters
    ----------
    interval_minutes:
        Desired cadence for regular snapshot/cboe ingestion runs.
    timezone:
        Scheduler timezone string (for example ``America/New_York``).
    job_id:
        Job identifier used to contextualize interval fallback warnings.

    Returns
    -------
    Any
        An APScheduler trigger that fires at 09:30+ cadence in the opening
        hour and then repeats every cadence interval from 10:00 to 15:50.
    """
    from apscheduler.triggers.combining import OrTrigger
    from apscheduler.triggers.cron import CronTrigger

    cadence = _normalized_rth_interval_minutes(interval_minutes, job_id=job_id)
    opening_minutes = ",".join(str(minute) for minute in range(30, 60, cadence))
    session_minutes = ",".join(str(minute) for minute in range(0, 60, cadence))
    return OrTrigger(
        [
            CronTrigger(day_of_week="mon-fri", hour=9, minute=opening_minutes, timezone=timezone),
            CronTrigger(day_of_week="mon-fri", hour="10-15", minute=session_minutes, timezone=timezone),
        ]
    )


def _build_serialized_run_once_runner(job: Any) -> Any:
    """Wrap ``run_once`` so split scheduler triggers execute serially.

    Parameters
    ----------
    job:
        Job instance exposing ``async run_once(force: bool = False)``.

    Returns
    -------
    Any
        An async callable that forwards ``force`` while preventing overlap
        between regular and forced-close runs.
    """
    run_lock = asyncio.Lock()

    async def _run(*, force: bool = False) -> dict[str, Any]:
        """Execute one serialized scheduler tick for the wrapped job."""
        async with run_lock:
            return await job.run_once(force=force)

    return _run


def _schedule_rth_window_job(
    scheduler: Any,
    *,
    job: Any,
    job_id: str,
    interval_minutes: int,
    timezone: str,
    max_job_instances: int,
    misfire_grace_seconds: int,
) -> None:
    """Schedule one job for RTH cadence with a guaranteed 16:00 ET run.

    The regular trigger runs from 09:30 through 15:50 on weekdays at the
    configured cadence. A separate forced-close trigger runs at 16:00 with
    ``force=True`` so the final in-session batch still executes when market
    status flips to closed at the session boundary.
    """
    run_callable = _build_serialized_run_once_runner(job)
    scheduler.add_job(
        run_callable,
        trigger=_build_rth_regular_trigger(interval_minutes=interval_minutes, timezone=timezone, job_id=job_id),
        id=job_id,
        replace_existing=True,
        max_instances=max_job_instances,
        misfire_grace_time=misfire_grace_seconds,
    )
    scheduler.add_job(
        run_callable,
        "cron",
        day_of_week="mon-fri",
        hour=16,
        minute=0,
        kwargs={"force": True},
        id=f"{job_id}_close",
        replace_existing=True,
        max_instances=max_job_instances,
        misfire_grace_time=misfire_grace_seconds,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, scheduler, and background jobs.

    The lifespan context wires jobs, starts APScheduler, runs one warm-up
    ingestion cycle, then yields control to the ASGI app. On shutdown it
    attempts a graceful scheduler stop.
    """
    await init_db()

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED

    scheduler = AsyncIOScheduler(timezone=settings.tz)
    scheduler.add_listener(_scheduler_event_listener, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    tradier = get_tradier_client()
    clock_cache = MarketClockCache(tradier=tradier, ttl_seconds=settings.market_clock_cache_seconds)
    max_job_instances = 1
    misfire_grace_seconds = 300

    snapshot_job = build_snapshot_job(tradier=tradier, clock_cache=clock_cache)
    spy_snapshot_job = build_spy_snapshot_job(tradier=tradier, clock_cache=clock_cache) if settings.spy_snapshot_enabled else None
    vix_snapshot_job = build_vix_snapshot_job(tradier=tradier, clock_cache=clock_cache) if settings.vix_snapshot_enabled else None
    quote_job = QuoteJob(tradier=tradier, clock_cache=clock_cache)
    gex_job = GexJob(clock_cache=clock_cache)
    cboe_gex_job = build_cboe_gex_job(clock_cache=clock_cache) if settings.cboe_gex_enabled else None
    decision_job = DecisionJob(clock_cache=clock_cache)
    feature_builder_job = FeatureBuilderJob(clock_cache=clock_cache)
    labeler_job = LabelerJob()
    trade_pnl_job = TradePnlJob(clock_cache=clock_cache)
    trainer_job = TrainerJob()
    shadow_inference_job = ShadowInferenceJob()
    promotion_gate_job = PromotionGateJob()
    performance_analytics_job = build_performance_analytics_job() if settings.performance_analytics_enabled else None

    _schedule_rth_window_job(
        scheduler,
        job=snapshot_job,
        job_id="snapshot_job",
        interval_minutes=settings.snapshot_interval_minutes,
        timezone=settings.tz,
        max_job_instances=max_job_instances,
        misfire_grace_seconds=misfire_grace_seconds,
    )
    if vix_snapshot_job is not None:
        _schedule_rth_window_job(
            scheduler,
            job=vix_snapshot_job,
            job_id="snapshot_job_vix",
            interval_minutes=settings.vix_snapshot_interval_minutes,
            timezone=settings.tz,
            max_job_instances=max_job_instances,
            misfire_grace_seconds=misfire_grace_seconds,
        )
    if spy_snapshot_job is not None:
        _schedule_rth_window_job(
            scheduler,
            job=spy_snapshot_job,
            job_id="snapshot_job_spy",
            interval_minutes=settings.spy_snapshot_interval_minutes,
            timezone=settings.tz,
            max_job_instances=max_job_instances,
            misfire_grace_seconds=misfire_grace_seconds,
        )
    _schedule_rth_window_job(
        scheduler,
        job=quote_job,
        job_id="quote_job",
        interval_minutes=settings.quote_interval_minutes,
        timezone=settings.tz,
        max_job_instances=max_job_instances,
        misfire_grace_seconds=misfire_grace_seconds,
    )
    _schedule_rth_window_job(
        scheduler,
        job=gex_job,
        job_id="gex_job",
        interval_minutes=settings.gex_interval_minutes,
        timezone=settings.tz,
        max_job_instances=max_job_instances,
        misfire_grace_seconds=misfire_grace_seconds,
    )
    if cboe_gex_job is not None:
        _schedule_rth_window_job(
            scheduler,
            job=cboe_gex_job,
            job_id="cboe_gex_job",
            interval_minutes=settings.cboe_gex_interval_minutes,
            timezone=settings.tz,
            max_job_instances=max_job_instances,
            misfire_grace_seconds=misfire_grace_seconds,
        )
    for hour, minute in settings.decision_entry_times_list():
        if settings.feature_builder_enabled:
            scheduler.add_job(
                feature_builder_job.run_once,
                "cron",
                hour=hour,
                minute=minute,
                id=f"feature_builder_job_{hour:02d}{minute:02d}",
                replace_existing=True,
                max_instances=max_job_instances,
                misfire_grace_time=misfire_grace_seconds,
            )
        scheduler.add_job(
            decision_job.run_once,
            "cron",
            hour=hour,
            minute=minute,
            id=f"decision_job_{hour:02d}{minute:02d}",
            replace_existing=True,
            max_instances=max_job_instances,
            misfire_grace_time=misfire_grace_seconds,
        )
    if settings.labeler_enabled:
        scheduler.add_job(
            labeler_job.run_once,
            "interval",
            minutes=settings.labeler_interval_minutes,
            id="labeler_job",
            replace_existing=True,
            max_instances=max_job_instances,
            misfire_grace_time=misfire_grace_seconds,
        )
    if settings.trade_pnl_enabled:
        scheduler.add_job(
            trade_pnl_job.run_once,
            "interval",
            minutes=settings.trade_pnl_interval_minutes,
            id="trade_pnl_job",
            replace_existing=True,
            max_instances=max_job_instances,
            misfire_grace_time=misfire_grace_seconds,
        )
    if performance_analytics_job is not None:
        _schedule_rth_window_job(
            scheduler,
            job=performance_analytics_job,
            job_id="performance_analytics_job",
            interval_minutes=settings.performance_analytics_interval_minutes,
            timezone=settings.tz,
            max_job_instances=max_job_instances,
            misfire_grace_seconds=misfire_grace_seconds,
        )
    if settings.trainer_enabled:
        scheduler.add_job(
            trainer_job.run_once,
            "cron",
            day_of_week=settings.trainer_weekday,
            hour=settings.trainer_hour,
            minute=settings.trainer_minute,
            id="trainer_job",
            replace_existing=True,
            max_instances=max_job_instances,
            misfire_grace_time=misfire_grace_seconds,
        )
    if settings.shadow_inference_enabled:
        scheduler.add_job(
            shadow_inference_job.run_once,
            "interval",
            minutes=settings.shadow_inference_interval_minutes,
            id="shadow_inference_job",
            replace_existing=True,
            max_instances=max_job_instances,
            misfire_grace_time=misfire_grace_seconds,
        )
    if settings.promotion_gate_enabled:
        scheduler.add_job(
            promotion_gate_job.run_once,
            "interval",
            minutes=settings.promotion_gate_interval_minutes,
            id="promotion_gate_job",
            replace_existing=True,
            max_instances=max_job_instances,
            misfire_grace_time=misfire_grace_seconds,
        )
    scheduler.start()

    app.state.scheduler = scheduler
    app.state.tradier = tradier
    app.state.clock_cache = clock_cache
    app.state.snapshot_job = snapshot_job
    app.state.spy_snapshot_job = spy_snapshot_job
    app.state.vix_snapshot_job = vix_snapshot_job
    app.state.quote_job = quote_job
    app.state.gex_job = gex_job
    app.state.cboe_gex_job = cboe_gex_job
    app.state.decision_job = decision_job
    app.state.feature_builder_job = feature_builder_job
    app.state.labeler_job = labeler_job
    app.state.trade_pnl_job = trade_pnl_job
    app.state.trainer_job = trainer_job
    app.state.shadow_inference_job = shadow_inference_job
    app.state.promotion_gate_job = promotion_gate_job
    app.state.performance_analytics_job = performance_analytics_job

    # Run ingestion jobs once immediately on boot (unless skip_startup_warmup). Log every failure explicitly.
    if not settings.skip_startup_warmup:
        warmup_jobs: list[tuple[str, Any]] = [
            ("quote_job", quote_job.run_once),
            ("snapshot_job", snapshot_job.run_once),
        ]
        if spy_snapshot_job is not None:
            warmup_jobs.append(("snapshot_job_spy", spy_snapshot_job.run_once))
        if vix_snapshot_job is not None:
            warmup_jobs.append(("snapshot_job_vix", vix_snapshot_job.run_once))
        warmup_jobs.append(("gex_job", gex_job.run_once))
        if cboe_gex_job is not None:
            warmup_jobs.append(("cboe_gex_job", cboe_gex_job.run_once))
        if performance_analytics_job is not None:
            warmup_jobs.append(("performance_analytics_job", performance_analytics_job.run_once))
        for job_name, run_once_callable in warmup_jobs:
            try:
                await run_once_callable()
                logger.info("startup_warmup: job_id={} status=ok", job_name)
            except Exception as exc:
                # Do not crash startup, but always emit actionable diagnostics.
                logger.exception("startup_warmup: job_id={} status=failed error={}", job_name, exc)
    else:
        logger.info("startup_warmup: skipped (SKIP_STARTUP_WARMUP=true)")

    yield

    # Graceful shutdown.
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass


app = FastAPI(title="IndexSpreadLab (Backend)", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(public.router)
app.include_router(admin.router)

__all__ = [
    "app",
    "lifespan",
    # Re-exported endpoint symbols for tests/backward compatibility.
    "list_trades",
    "get_label_metrics",
    "get_performance_analytics",
    "get_model_ops",
    "get_strategy_metrics",
    "list_gex_dtes",
    "list_gex_expirations",
    "get_gex_curve",
]
