from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from spx_backend.config import settings
from spx_backend.database import init_db
from spx_backend.scheduler_builder import build_scheduler
from spx_backend.web.routers import admin, auth, portfolio, public
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, scheduler, and background jobs.

    Delegates all scheduler construction and job registration to
    ``build_scheduler`` so this function stays focused on lifecycle
    orchestration (init, start, warmup, shutdown).
    """
    if settings.skip_init_db:
        logger.info("init_db: skipped (SKIP_INIT_DB=true)")
    else:
        await init_db()

    ctx = build_scheduler()
    ctx.scheduler.start()
    ctx.attach_to_app_state(app)

    if not settings.skip_startup_warmup:
        await ctx.run_warmup()
    else:
        logger.info("startup_warmup: skipped (SKIP_STARTUP_WARMUP=true)")

    yield

    try:
        ctx.scheduler.shutdown(wait=False)
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
app.include_router(portfolio.router)

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
