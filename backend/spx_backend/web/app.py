from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from spx_backend.config import settings
from spx_backend.database import init_db
from spx_backend.ingestion.tradier_client import get_tradier_client
from spx_backend.jobs.decision_job import DecisionJob
from spx_backend.jobs.feature_builder_job import FeatureBuilderJob
from spx_backend.jobs.gex_job import GexJob
from spx_backend.jobs.labeler_job import LabelerJob
from spx_backend.jobs.promotion_gate_job import PromotionGateJob
from spx_backend.jobs.quote_job import QuoteJob
from spx_backend.jobs.shadow_inference_job import ShadowInferenceJob
from spx_backend.jobs.snapshot_job import build_snapshot_job, build_spy_snapshot_job, build_vix_snapshot_job
from spx_backend.jobs.trainer_job import TrainerJob
from spx_backend.jobs.trade_pnl_job import TradePnlJob
from spx_backend.market_clock import MarketClockCache
from spx_backend.web.routers import admin, public
from spx_backend.web.routers.public import (
    get_gex_curve,
    get_label_metrics,
    get_model_ops,
    get_strategy_metrics,
    list_gex_dtes,
    list_gex_expirations,
    list_trades,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, scheduler, and background jobs."""
    await init_db()

    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler(timezone=settings.tz)
    tradier = get_tradier_client()
    clock_cache = MarketClockCache(tradier=tradier, ttl_seconds=settings.market_clock_cache_seconds)

    snapshot_job = build_snapshot_job(tradier=tradier, clock_cache=clock_cache)
    spy_snapshot_job = build_spy_snapshot_job(tradier=tradier, clock_cache=clock_cache) if settings.spy_snapshot_enabled else None
    vix_snapshot_job = build_vix_snapshot_job(tradier=tradier, clock_cache=clock_cache) if settings.vix_snapshot_enabled else None
    quote_job = QuoteJob(tradier=tradier, clock_cache=clock_cache)
    gex_job = GexJob()
    decision_job = DecisionJob(clock_cache=clock_cache)
    feature_builder_job = FeatureBuilderJob(clock_cache=clock_cache)
    labeler_job = LabelerJob()
    trade_pnl_job = TradePnlJob(clock_cache=clock_cache)
    trainer_job = TrainerJob()
    shadow_inference_job = ShadowInferenceJob()
    promotion_gate_job = PromotionGateJob()

    scheduler.add_job(
        snapshot_job.run_once,
        "interval",
        minutes=settings.snapshot_interval_minutes,
        id="snapshot_job",
        replace_existing=True,
    )
    if vix_snapshot_job is not None:
        scheduler.add_job(
            vix_snapshot_job.run_once,
            "interval",
            minutes=settings.vix_snapshot_interval_minutes,
            id="snapshot_job_vix",
            replace_existing=True,
        )
    if spy_snapshot_job is not None:
        scheduler.add_job(
            spy_snapshot_job.run_once,
            "interval",
            minutes=settings.spy_snapshot_interval_minutes,
            id="snapshot_job_spy",
            replace_existing=True,
        )
    scheduler.add_job(
        quote_job.run_once,
        "interval",
        minutes=settings.quote_interval_minutes,
        id="quote_job",
        replace_existing=True,
    )
    scheduler.add_job(
        gex_job.run_once,
        "interval",
        minutes=settings.gex_interval_minutes,
        id="gex_job",
        replace_existing=True,
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
            )
        scheduler.add_job(
            decision_job.run_once,
            "cron",
            hour=hour,
            minute=minute,
            id=f"decision_job_{hour:02d}{minute:02d}",
            replace_existing=True,
        )
    if settings.labeler_enabled:
        scheduler.add_job(
            labeler_job.run_once,
            "interval",
            minutes=settings.labeler_interval_minutes,
            id="labeler_job",
            replace_existing=True,
        )
    if settings.trade_pnl_enabled:
        scheduler.add_job(
            trade_pnl_job.run_once,
            "interval",
            minutes=settings.trade_pnl_interval_minutes,
            id="trade_pnl_job",
            replace_existing=True,
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
        )
    if settings.shadow_inference_enabled:
        scheduler.add_job(
            shadow_inference_job.run_once,
            "interval",
            minutes=settings.shadow_inference_interval_minutes,
            id="shadow_inference_job",
            replace_existing=True,
        )
    if settings.promotion_gate_enabled:
        scheduler.add_job(
            promotion_gate_job.run_once,
            "interval",
            minutes=settings.promotion_gate_interval_minutes,
            id="promotion_gate_job",
            replace_existing=True,
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
    app.state.decision_job = decision_job
    app.state.feature_builder_job = feature_builder_job
    app.state.labeler_job = labeler_job
    app.state.trade_pnl_job = trade_pnl_job
    app.state.trainer_job = trainer_job
    app.state.shadow_inference_job = shadow_inference_job
    app.state.promotion_gate_job = promotion_gate_job

    # Run once immediately on boot (useful for confirming wiring).
    try:
        await quote_job.run_once()
        await snapshot_job.run_once()
        if spy_snapshot_job is not None:
            await spy_snapshot_job.run_once()
        if vix_snapshot_job is not None:
            await vix_snapshot_job.run_once()
        await gex_job.run_once()
    except Exception:
        # Don't crash the web app if the first snapshot fails.
        pass

    yield

    # Graceful shutdown.
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass


app = FastAPI(title="SPX Tools (Backend)", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(public.router)
app.include_router(admin.router)

__all__ = [
    "app",
    "lifespan",
    # Re-exported endpoint symbols for tests/backward compatibility.
    "list_trades",
    "get_label_metrics",
    "get_model_ops",
    "get_strategy_metrics",
    "list_gex_dtes",
    "list_gex_expirations",
    "get_gex_curve",
]
