from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.db import get_db_session
from spx_backend.db_init import init_db
from spx_backend.ingestion.tradier_client import TradierClient, get_tradier_client
from spx_backend.jobs.decision_job import DecisionJob, build_decision_job
from spx_backend.jobs.feature_builder_job import FeatureBuilderJob, build_feature_builder_job
from spx_backend.jobs.gex_job import GexJob
from spx_backend.jobs.labeler_job import LabelerJob, build_labeler_job
from spx_backend.jobs.quote_job import QuoteJob, build_quote_job
from spx_backend.jobs.snapshot_job import SnapshotJob, _parse_expirations, build_snapshot_job
from spx_backend.market_clock import MarketClockCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, scheduler, and background jobs."""
    # Initialize DB schema and start scheduler.
    await init_db()

    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    scheduler = AsyncIOScheduler(timezone=settings.tz)
    tradier = get_tradier_client()
    clock_cache = MarketClockCache(tradier=tradier, ttl_seconds=settings.market_clock_cache_seconds)

    snapshot_job = SnapshotJob(tradier=tradier, clock_cache=clock_cache)
    quote_job = QuoteJob(tradier=tradier, clock_cache=clock_cache)
    gex_job = GexJob()
    decision_job = DecisionJob(clock_cache=clock_cache)
    feature_builder_job = FeatureBuilderJob(clock_cache=clock_cache)
    labeler_job = LabelerJob()

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
    scheduler.start()

    app.state.scheduler = scheduler
    app.state.tradier = tradier
    app.state.clock_cache = clock_cache
    app.state.snapshot_job = snapshot_job
    app.state.quote_job = quote_job
    app.state.gex_job = gex_job
    app.state.decision_job = decision_job
    app.state.feature_builder_job = feature_builder_job
    app.state.labeler_job = labeler_job

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
    """Simple health check."""
    return {"status": "ok"}


@app.get("/api/chain-snapshots")
async def list_chain_snapshots(limit: int = 50, db: AsyncSession = Depends(get_db_session)) -> dict:
    """Return recent chain snapshot metadata."""
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


@app.get("/api/trade-decisions")
async def list_trade_decisions(limit: int = 50, db: AsyncSession = Depends(get_db_session)) -> dict:
    """Return recent trade decisions."""
    limit = max(1, min(limit, 500))
    r = await db.execute(
        text(
            """
            SELECT decision_id, ts, target_dte, entry_slot, delta_target,
                   decision, reason, score, chain_snapshot_id, decision_source,
                   ruleset_version, chosen_legs_json, strategy_params_json
            FROM trade_decisions
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
                "decision_id": row.decision_id,
                "ts": row.ts.isoformat(),
                "target_dte": row.target_dte,
                "entry_slot": row.entry_slot,
                "delta_target": row.delta_target,
                "decision": row.decision,
                "reason": row.reason,
                "score": row.score,
                "chain_snapshot_id": row.chain_snapshot_id,
                "decision_source": row.decision_source,
                "ruleset_version": row.ruleset_version,
                "chosen_legs_json": row.chosen_legs_json,
                "strategy_params_json": row.strategy_params_json,
            }
            for row in rows
        ]
    }


@app.get("/api/gex/snapshots")
async def list_gex_snapshots(limit: int = 50, db: AsyncSession = Depends(get_db_session)) -> dict:
    """Return recent GEX snapshot aggregates."""
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
    """Return available DTEs for a GEX snapshot batch (same ts + underlying)."""
    r = await db.execute(
        text(
            """
            WITH anchor AS (
              SELECT ts, underlying
              FROM gex_snapshots
              WHERE snapshot_id = :snapshot_id
            ),
            batch AS (
              SELECT gs.snapshot_id
              FROM gex_snapshots gs
              JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying
            )
            SELECT DISTINCT gbes.dte_days
            FROM gex_by_expiry_strike gbes
            JOIN batch b ON b.snapshot_id = gbes.snapshot_id
            WHERE gbes.dte_days IS NOT NULL
            ORDER BY gbes.dte_days ASC
            """
        ),
        {"snapshot_id": snapshot_id},
    )
    dtes = [row.dte_days for row in r.fetchall()]
    return {"snapshot_id": snapshot_id, "dte_days": dtes}


@app.get("/api/gex/expirations")
async def list_gex_expirations(snapshot_id: int, db: AsyncSession = Depends(get_db_session)) -> dict:
    """Return available expirations for a GEX snapshot batch (same ts + underlying)."""
    r = await db.execute(
        text(
            """
            WITH anchor AS (
              SELECT ts, underlying
              FROM gex_snapshots
              WHERE snapshot_id = :snapshot_id
            ),
            batch AS (
              SELECT gs.snapshot_id
              FROM gex_snapshots gs
              JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying
            )
            SELECT DISTINCT gbes.expiration, gbes.dte_days
            FROM gex_by_expiry_strike gbes
            JOIN batch b ON b.snapshot_id = gbes.snapshot_id
            ORDER BY gbes.expiration ASC
            """
        ),
        {"snapshot_id": snapshot_id},
    )
    rows = r.fetchall()
    return {
        "snapshot_id": snapshot_id,
        "items": [
            {
                "expiration": row.expiration.isoformat(),
                "dte_days": row.dte_days,
            }
            for row in rows
        ],
    }


@app.get("/api/gex/curve")
async def get_gex_curve(
    snapshot_id: int,
    dte_days: int | None = None,
    expirations_csv: str | None = None,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return GEX curve by strike for a snapshot batch (optional DTE/custom expirations)."""
    selected_expirations: list[date] = []
    if expirations_csv is not None:
        for part in expirations_csv.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                selected_expirations.append(date.fromisoformat(p))
            except ValueError:
                continue
        if not selected_expirations:
            return {"snapshot_id": snapshot_id, "dte_days": dte_days, "expirations": [], "points": []}

    if selected_expirations:
        stmt = text(
            """
            WITH anchor AS (
              SELECT ts, underlying
              FROM gex_snapshots
              WHERE snapshot_id = :snapshot_id
            ),
            batch AS (
              SELECT gs.snapshot_id
              FROM gex_snapshots gs
              JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying
            )
            SELECT gbes.strike,
                   SUM(gbes.gex_net) AS gex_net,
                   SUM(gbes.gex_calls) AS gex_calls,
                   SUM(gbes.gex_puts) AS gex_puts
            FROM gex_by_expiry_strike gbes
            JOIN batch b ON b.snapshot_id = gbes.snapshot_id
            WHERE gbes.expiration IN :expirations
            GROUP BY gbes.strike
            ORDER BY gbes.strike ASC
            """
        ).bindparams(bindparam("expirations", expanding=True))
        r = await db.execute(
            stmt,
            {"snapshot_id": snapshot_id, "expirations": selected_expirations},
        )
    elif dte_days is None:
        r = await db.execute(
            text(
                """
                WITH anchor AS (
                  SELECT ts, underlying
                  FROM gex_snapshots
                  WHERE snapshot_id = :snapshot_id
                ),
                batch AS (
                  SELECT gs.snapshot_id
                  FROM gex_snapshots gs
                  JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying
                )
                SELECT gbes.strike,
                       SUM(gbes.gex_net) AS gex_net,
                       SUM(gbes.gex_calls) AS gex_calls,
                       SUM(gbes.gex_puts) AS gex_puts
                FROM gex_by_expiry_strike gbes
                JOIN batch b ON b.snapshot_id = gbes.snapshot_id
                GROUP BY gbes.strike
                ORDER BY gbes.strike ASC
                """
            ),
            {"snapshot_id": snapshot_id},
        )
    else:
        r = await db.execute(
            text(
                """
                WITH anchor AS (
                  SELECT ts, underlying
                  FROM gex_snapshots
                  WHERE snapshot_id = :snapshot_id
                ),
                batch AS (
                  SELECT gs.snapshot_id
                  FROM gex_snapshots gs
                  JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying
                )
                SELECT gbes.strike,
                       SUM(gbes.gex_net) AS gex_net,
                       SUM(gbes.gex_calls) AS gex_calls,
                       SUM(gbes.gex_puts) AS gex_puts
                FROM gex_by_expiry_strike gbes
                JOIN batch b ON b.snapshot_id = gbes.snapshot_id
                WHERE gbes.dte_days = :dte_days
                GROUP BY gbes.strike
                ORDER BY gbes.strike ASC
                """
            ),
            {"snapshot_id": snapshot_id, "dte_days": dte_days},
        )
    rows = r.fetchall()
    if (not rows) and (dte_days is None) and (not selected_expirations):
        # Compatibility fallback for older data that only has gex_by_strike rows.
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
        rows = r.fetchall()
    return {
        "snapshot_id": snapshot_id,
        "dte_days": dte_days,
        "expirations": [e.isoformat() for e in selected_expirations],
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
    """Enforce admin API key if configured."""
    # If ADMIN_API_KEY is not set, allow local/dev usage without auth.
    if settings.admin_api_key:
        if not x_api_key or x_api_key != settings.admin_api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/api/admin/run-snapshot")
async def admin_run_snapshot(request: Request, _: None = Depends(_require_admin)) -> dict:
    """Force a snapshot run immediately."""
    # Force run once even outside RTH (useful for testing).
    job: SnapshotJob = getattr(request.app.state, "snapshot_job", build_snapshot_job())
    result = await job.run_once(force=True)
    return result


@app.post("/api/admin/run-quotes")
async def admin_run_quotes(request: Request, _: None = Depends(_require_admin)) -> dict:
    """Force a quote run immediately."""
    job: QuoteJob = getattr(request.app.state, "quote_job", build_quote_job())
    result = await job.run_once(force=True)
    return result


@app.post("/api/admin/run-gex")
async def admin_run_gex(request: Request, _: None = Depends(_require_admin)) -> dict:
    """Force a GEX run immediately."""
    job: GexJob = getattr(request.app.state, "gex_job", GexJob())
    result = await job.run_once()
    return result


@app.post("/api/admin/run-decision")
async def admin_run_decision(request: Request, _: None = Depends(_require_admin)) -> dict:
    """Force a decision run immediately."""
    job: DecisionJob = getattr(request.app.state, "decision_job", build_decision_job())
    result = await job.run_once(force=True)
    return result


@app.post("/api/admin/run-feature-builder")
async def admin_run_feature_builder(request: Request, _: None = Depends(_require_admin)) -> dict:
    """Force feature builder run immediately."""
    job: FeatureBuilderJob = getattr(request.app.state, "feature_builder_job", build_feature_builder_job())
    result = await job.run_once(force=True)
    return result


@app.post("/api/admin/run-labeler")
async def admin_run_labeler(request: Request, _: None = Depends(_require_admin)) -> dict:
    """Force labeler run immediately."""
    job: LabelerJob = getattr(request.app.state, "labeler_job", build_labeler_job())
    result = await job.run_once(force=True)
    return result


@app.delete("/api/admin/trade-decisions/{decision_id}")
async def admin_delete_trade_decision(decision_id: int, db: AsyncSession = Depends(get_db_session), _: None = Depends(_require_admin)) -> dict:
    """Delete one trade decision row by ID."""
    r = await db.execute(
        text(
            """
            DELETE FROM trade_decisions
            WHERE decision_id = :decision_id
            RETURNING decision_id
            """
        ),
        {"decision_id": decision_id},
    )
    row = r.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="trade_decision_not_found")
    await db.commit()
    return {"deleted": True, "decision_id": row.decision_id}


@app.get("/api/admin/expirations")
async def admin_list_expirations(request: Request, symbol: str = "SPX", _: None = Depends(_require_admin)) -> dict:
    """List expirations from Tradier for debugging."""
    client: TradierClient = getattr(request.app.state, "tradier", get_tradier_client())
    resp = await client.get_option_expirations(symbol)
    exps = _parse_expirations(resp)
    return {"symbol": symbol, "expirations": [e.isoformat() for e in exps]}


@app.get("/api/admin/preflight")
async def admin_preflight(db: AsyncSession = Depends(get_db_session), _: None = Depends(_require_admin)) -> dict:
    """Return one-call pipeline health summary."""

    def _iso(ts) -> str | None:
        return ts.isoformat() if ts is not None else None

    r = await db.execute(
        text(
            """
            SELECT
              (SELECT COUNT(*) FROM underlying_quotes) AS quotes_count,
              (SELECT COUNT(*) FROM chain_snapshots) AS snapshots_count,
              (SELECT COUNT(*) FROM option_chain_rows) AS chain_rows_count,
              (SELECT COUNT(*) FROM gex_snapshots) AS gex_snapshots_count,
              (SELECT COUNT(*) FROM trade_decisions) AS decisions_count,
              (SELECT COUNT(*) FROM feature_snapshots) AS feature_snapshots_count,
              (SELECT COUNT(*) FROM trade_candidates) AS trade_candidates_count,
              (SELECT COUNT(*) FROM trade_candidates WHERE label_status = 'resolved') AS labeled_candidates_count,
              (SELECT MAX(ts) FROM underlying_quotes) AS latest_quote_ts,
              (SELECT MAX(ts) FROM chain_snapshots) AS latest_snapshot_ts,
              (SELECT MAX(ts) FROM gex_snapshots) AS latest_gex_ts,
              (SELECT MAX(ts) FROM trade_decisions) AS latest_decision_ts,
              (SELECT MAX(ts) FROM feature_snapshots) AS latest_feature_ts,
              (SELECT MAX(ts) FROM trade_candidates) AS latest_candidate_ts,
              (SELECT MAX(ts) FROM market_clock_audit) AS latest_market_clock_ts
            """
        )
    )
    summary = r.fetchone()

    latest_snapshot_row = (
        await db.execute(
            text(
                """
                SELECT snapshot_id, ts, target_dte, expiration
                FROM chain_snapshots
                ORDER BY ts DESC, snapshot_id DESC
                LIMIT 1
                """
            )
        )
    ).fetchone()

    latest_gex_row = (
        await db.execute(
            text(
                """
                SELECT snapshot_id, ts, gex_net, zero_gamma_level, method
                FROM gex_snapshots
                ORDER BY ts DESC, snapshot_id DESC
                LIMIT 1
                """
            )
        )
    ).fetchone()

    latest_decision_row = (
        await db.execute(
            text(
                """
                SELECT decision_id, ts, decision, reason, score, target_dte, delta_target, chain_snapshot_id, decision_source
                FROM trade_decisions
                ORDER BY ts DESC, decision_id DESC
                LIMIT 1
                """
            )
        )
    ).fetchone()

    latest_quotes = (
        await db.execute(
            text(
                """
                SELECT DISTINCT ON (symbol) symbol, ts, last
                FROM underlying_quotes
                ORDER BY symbol, ts DESC
                """
            )
        )
    ).fetchall()

    counts = {
        "underlying_quotes": int(summary.quotes_count or 0),
        "chain_snapshots": int(summary.snapshots_count or 0),
        "option_chain_rows": int(summary.chain_rows_count or 0),
        "gex_snapshots": int(summary.gex_snapshots_count or 0),
        "trade_decisions": int(summary.decisions_count or 0),
        "feature_snapshots": int(summary.feature_snapshots_count or 0),
        "trade_candidates": int(summary.trade_candidates_count or 0),
        "labeled_candidates": int(summary.labeled_candidates_count or 0),
    }
    latest = {
        "quote_ts": _iso(summary.latest_quote_ts),
        "snapshot_ts": _iso(summary.latest_snapshot_ts),
        "gex_ts": _iso(summary.latest_gex_ts),
        "decision_ts": _iso(summary.latest_decision_ts),
        "feature_ts": _iso(summary.latest_feature_ts),
        "candidate_ts": _iso(summary.latest_candidate_ts),
        "market_clock_ts": _iso(summary.latest_market_clock_ts),
    }

    warnings: list[str] = []
    if counts["chain_snapshots"] == 0:
        warnings.append("no_chain_snapshots")
    if counts["gex_snapshots"] == 0:
        warnings.append("no_gex_snapshots")
    if counts["trade_decisions"] == 0:
        warnings.append("no_trade_decisions")
    if counts["feature_snapshots"] == 0:
        warnings.append("no_feature_snapshots")
    if counts["trade_candidates"] == 0:
        warnings.append("no_trade_candidates")

    return {
        "now_utc": f"{datetime.utcnow().isoformat()}Z",
        "counts": counts,
        "latest": latest,
        "latest_snapshot": (
            None
            if latest_snapshot_row is None
            else {
                "snapshot_id": latest_snapshot_row.snapshot_id,
                "ts": _iso(latest_snapshot_row.ts),
                "target_dte": latest_snapshot_row.target_dte,
                "expiration": str(latest_snapshot_row.expiration),
            }
        ),
        "latest_gex": (
            None
            if latest_gex_row is None
            else {
                "snapshot_id": latest_gex_row.snapshot_id,
                "ts": _iso(latest_gex_row.ts),
                "gex_net": latest_gex_row.gex_net,
                "zero_gamma_level": latest_gex_row.zero_gamma_level,
                "method": latest_gex_row.method,
            }
        ),
        "latest_decision": (
            None
            if latest_decision_row is None
            else {
                "decision_id": latest_decision_row.decision_id,
                "ts": _iso(latest_decision_row.ts),
                "decision": latest_decision_row.decision,
                "reason": latest_decision_row.reason,
                "score": latest_decision_row.score,
                "target_dte": latest_decision_row.target_dte,
                "delta_target": latest_decision_row.delta_target,
                "chain_snapshot_id": latest_decision_row.chain_snapshot_id,
                "decision_source": latest_decision_row.decision_source,
            }
        ),
        "latest_quotes_by_symbol": [
            {"symbol": row.symbol, "ts": _iso(row.ts), "last": row.last}
            for row in latest_quotes
        ],
        "warnings": warnings,
    }


@app.get("/", response_class=HTMLResponse)
async def home(db: AsyncSession = Depends(get_db_session)) -> HTMLResponse:
    """Small HTML page listing recent snapshots."""
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

