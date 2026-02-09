from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.db import get_db_session
from spx_backend.db_init import init_db
from spx_backend.ingestion.tradier_client import TradierClient, get_tradier_client
from spx_backend.jobs.gex_job import GexJob
from spx_backend.jobs.quote_job import QuoteJob, build_quote_job
from spx_backend.jobs.snapshot_job import SnapshotJob, _parse_expirations, build_snapshot_job
from spx_backend.market_clock import MarketClockCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema and start scheduler.
    await init_db()

    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler(timezone=settings.tz)
    tradier = get_tradier_client()
    clock_cache = MarketClockCache(tradier=tradier, ttl_seconds=settings.market_clock_cache_seconds)

    snapshot_job = SnapshotJob(tradier=tradier, clock_cache=clock_cache)
    quote_job = QuoteJob(tradier=tradier, clock_cache=clock_cache)
    gex_job = GexJob()

    scheduler.add_job(
        snapshot_job.run_once,
        "interval",
        minutes=settings.snapshot_interval_minutes,
        id="snapshot_job",
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
    scheduler.start()

    app.state.scheduler = scheduler
    app.state.tradier = tradier
    app.state.clock_cache = clock_cache
    app.state.snapshot_job = snapshot_job
    app.state.quote_job = quote_job
    app.state.gex_job = gex_job

    # Run once immediately on boot (useful for confirming wiring).
    try:
        await quote_job.run_once()
        await snapshot_job.run_once()
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/chain-snapshots")
async def list_chain_snapshots(limit: int = 50, db: AsyncSession = Depends(get_db_session)) -> dict:
    limit = max(1, min(limit, 500))
    r = await db.execute(
        text(
            """
            SELECT snapshot_id, ts, underlying, target_dte, expiration, checksum
            FROM chain_snapshots
            ORDER BY ts DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    )
    rows = r.fetchall()
    return {
        "items": [
            {
                "snapshot_id": row.snapshot_id,
                "ts": row.ts.isoformat(),
                "underlying": row.underlying,
                "target_dte": row.target_dte,
                "expiration": str(row.expiration),
                "checksum": row.checksum,
            }
            for row in rows
        ]
    }


@app.get("/api/gex/snapshots")
async def list_gex_snapshots(limit: int = 50, db: AsyncSession = Depends(get_db_session)) -> dict:
    limit = max(1, min(limit, 500))
    r = await db.execute(
        text(
            """
            SELECT snapshot_id, ts, underlying, spot_price, gex_net, gex_calls, gex_puts, gex_abs, zero_gamma_level, method
            FROM gex_snapshots
            ORDER BY ts DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    )
    rows = r.fetchall()
    return {
        "items": [
            {
                "snapshot_id": row.snapshot_id,
                "ts": row.ts.isoformat(),
                "underlying": row.underlying,
                "spot_price": row.spot_price,
                "gex_net": row.gex_net,
                "gex_calls": row.gex_calls,
                "gex_puts": row.gex_puts,
                "gex_abs": row.gex_abs,
                "zero_gamma_level": row.zero_gamma_level,
                "method": row.method,
            }
            for row in rows
        ]
    }


@app.get("/api/gex/dtes")
async def list_gex_dtes(snapshot_id: int, db: AsyncSession = Depends(get_db_session)) -> dict:
    r = await db.execute(
        text(
            """
            SELECT DISTINCT dte_days
            FROM gex_by_expiry_strike
            WHERE snapshot_id = :snapshot_id AND dte_days IS NOT NULL
            ORDER BY dte_days ASC
            """
        ),
        {"snapshot_id": snapshot_id},
    )
    dtes = [row.dte_days for row in r.fetchall()]
    return {"snapshot_id": snapshot_id, "dte_days": dtes}


@app.get("/api/gex/curve")
async def get_gex_curve(snapshot_id: int, dte_days: int | None = None, db: AsyncSession = Depends(get_db_session)) -> dict:
    if dte_days is None:
        r = await db.execute(
            text(
                """
                SELECT strike, gex_net, gex_calls, gex_puts
                FROM gex_by_strike
                WHERE snapshot_id = :snapshot_id
                ORDER BY strike ASC
                """
            ),
            {"snapshot_id": snapshot_id},
        )
    else:
        r = await db.execute(
            text(
                """
                SELECT strike, gex_net, gex_calls, gex_puts
                FROM gex_by_expiry_strike
                WHERE snapshot_id = :snapshot_id AND dte_days = :dte_days
                ORDER BY strike ASC
                """
            ),
            {"snapshot_id": snapshot_id, "dte_days": dte_days},
        )
    rows = r.fetchall()
    return {
        "snapshot_id": snapshot_id,
        "dte_days": dte_days,
        "points": [
            {
                "strike": row.strike,
                "gex_net": row.gex_net,
                "gex_calls": row.gex_calls,
                "gex_puts": row.gex_puts,
            }
            for row in rows
        ],
    }


def _require_admin(x_api_key: str | None = Header(default=None)) -> None:
    # If ADMIN_API_KEY is not set, allow local/dev usage without auth.
    if settings.admin_api_key:
        if not x_api_key or x_api_key != settings.admin_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/api/admin/run-snapshot")
async def admin_run_snapshot(request: Request, _: None = Depends(_require_admin)) -> dict:
    # Force run once even outside RTH (useful for testing).
    job: SnapshotJob = getattr(request.app.state, "snapshot_job", build_snapshot_job())
    result = await job.run_once(force=True)
    return result


@app.post("/api/admin/run-quotes")
async def admin_run_quotes(request: Request, _: None = Depends(_require_admin)) -> dict:
    job: QuoteJob = getattr(request.app.state, "quote_job", build_quote_job())
    result = await job.run_once(force=True)
    return result


@app.post("/api/admin/run-gex")
async def admin_run_gex(request: Request, _: None = Depends(_require_admin)) -> dict:
    job: GexJob = getattr(request.app.state, "gex_job", GexJob())
    result = await job.run_once()
    return result


@app.get("/api/admin/expirations")
async def admin_list_expirations(request: Request, symbol: str = "SPX", _: None = Depends(_require_admin)) -> dict:
    client: TradierClient = getattr(request.app.state, "tradier", get_tradier_client())
    resp = await client.get_option_expirations(symbol)
    exps = _parse_expirations(resp)
    return {"symbol": symbol, "expirations": [e.isoformat() for e in exps]}


@app.get("/", response_class=HTMLResponse)
async def home(db: AsyncSession = Depends(get_db_session)) -> HTMLResponse:
    r = await db.execute(
        text(
            """
            SELECT snapshot_id, ts, underlying, target_dte, expiration
            FROM chain_snapshots
            ORDER BY ts DESC
            LIMIT 20
            """
        )
    )
    rows = r.fetchall()
    items = "\n".join(
        f"<li>#{row.snapshot_id} {row.ts} {row.underlying} dte={row.target_dte} exp={row.expiration}</li>"
        for row in rows
    )
    html = f"""
    <html>
      <head><title>SPX Tools (Backend)</title></head>
      <body style="font-family: system-ui; max-width: 900px; margin: 40px auto;">
        <h2>SPX Tools (Backend)</h2>
        <p>Server time: {datetime.utcnow().isoformat()}Z</p>
        <h3>Latest chain snapshots</h3>
        <ol>{items}</ol>
        <p><a href="/health">/health</a> · <a href="/api/chain-snapshots">/api/chain-snapshots</a></p>
      </body>
    </html>
    """
    return HTMLResponse(html)

