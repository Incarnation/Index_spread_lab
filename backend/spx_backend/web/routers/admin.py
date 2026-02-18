from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.web.routers.auth import UserOut, _normalize_ip, get_current_user, require_admin
from spx_backend.database import get_db_session
from spx_backend.ingestion.tradier_client import TradierClient, get_tradier_client
from spx_backend.jobs.decision_job import DecisionJob, build_decision_job
from spx_backend.jobs.feature_builder_job import FeatureBuilderJob, build_feature_builder_job
from spx_backend.jobs.gex_job import GexJob
from spx_backend.jobs.labeler_job import LabelerJob, build_labeler_job
from spx_backend.jobs.promotion_gate_job import PromotionGateJob, build_promotion_gate_job
from spx_backend.jobs.quote_job import QuoteJob, build_quote_job
from spx_backend.jobs.shadow_inference_job import ShadowInferenceJob, build_shadow_inference_job
from spx_backend.jobs.snapshot_job import SnapshotJob, _parse_expirations, build_snapshot_job
from spx_backend.jobs.trainer_job import TrainerJob, build_trainer_job
from spx_backend.jobs.trade_pnl_job import TradePnlJob, build_trade_pnl_job

router = APIRouter()


def _minutes_since(ts: datetime | None, now_utc: datetime) -> float | None:
    """Return elapsed minutes between now_utc and ts.

    Parameters
    ----------
    ts:
        Optional timestamp to evaluate.
    now_utc:
        Current UTC timestamp used as the freshness reference.

    Returns
    -------
    float | None
        Rounded minute age when ts is present, otherwise None.
    """
    if ts is None:
        return None
    ts_utc = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    delta_seconds = max(0.0, (now_utc - ts_utc).total_seconds())
    return round(delta_seconds / 60.0, 2)


def _is_stale(age_minutes: float | None, threshold_minutes: float) -> bool:
    """Evaluate whether a freshness age exceeds its staleness threshold."""
    return age_minutes is not None and age_minutes > threshold_minutes


@router.post("/api/admin/run-snapshot")
async def admin_run_snapshot(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force a snapshot run immediately."""
    # Force run once even outside RTH (useful for testing).
    job: SnapshotJob = getattr(request.app.state, "snapshot_job", build_snapshot_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-quotes")
async def admin_run_quotes(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force a quote run immediately."""
    job: QuoteJob = getattr(request.app.state, "quote_job", build_quote_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-gex")
async def admin_run_gex(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force a GEX run immediately."""
    job: GexJob = getattr(request.app.state, "gex_job", GexJob())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-decision")
async def admin_run_decision(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force a decision run immediately."""
    job: DecisionJob = getattr(request.app.state, "decision_job", build_decision_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-feature-builder")
async def admin_run_feature_builder(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force feature builder run immediately."""
    job: FeatureBuilderJob = getattr(request.app.state, "feature_builder_job", build_feature_builder_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-labeler")
async def admin_run_labeler(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force labeler run immediately."""
    job: LabelerJob = getattr(request.app.state, "labeler_job", build_labeler_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-trade-pnl")
async def admin_run_trade_pnl(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force trade mark-to-market run immediately."""
    job: TradePnlJob = getattr(request.app.state, "trade_pnl_job", build_trade_pnl_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-trainer")
async def admin_run_trainer(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force weekly trainer run immediately."""
    job: TrainerJob = getattr(request.app.state, "trainer_job", build_trainer_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-shadow-inference")
async def admin_run_shadow_inference(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force shadow inference run immediately."""
    job: ShadowInferenceJob = getattr(request.app.state, "shadow_inference_job", build_shadow_inference_job())
    result = await job.run_once(force=True)
    return result


@router.post("/api/admin/run-promotion-gates")
async def admin_run_promotion_gates(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Force promotion gate evaluation immediately."""
    job: PromotionGateJob = getattr(request.app.state, "promotion_gate_job", build_promotion_gate_job())
    result = await job.run_once(force=True)
    return result


@router.delete("/api/admin/trade-decisions/{decision_id}")
async def admin_delete_trade_decision(
    decision_id: int,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
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


@router.get("/api/admin/expirations")
async def admin_list_expirations(
    request: Request,
    symbol: str = "SPX",
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """List expirations from Tradier for debugging."""
    client: TradierClient = getattr(request.app.state, "tradier", get_tradier_client())
    resp = await client.get_option_expirations(symbol)
    exps = _parse_expirations(resp)
    return {"symbol": symbol, "expirations": [e.isoformat() for e in exps]}


@router.get("/api/admin/auth-audit")
async def admin_auth_audit(
    current_user: UserOut = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
    limit: int = 100,
    offset: int = 0,
    event_type: str | None = None,
    user_id: int | None = None,
) -> dict:
    """
    List auth audit log entries (login success/failure, logout, session_expiry).
    Admin-only. Supports pagination and optional filters by event_type and user_id.
    """
    if limit <= 0 or limit > 500:
        limit = 100
    if offset < 0:
        offset = 0
    # Build WHERE clause from optional filters.
    conditions = []
    params: dict = {"limit": limit, "offset": offset}
    if event_type:
        conditions.append("event_type = :event_type")
        params["event_type"] = event_type
    if user_id is not None:
        conditions.append("user_id = :user_id")
        params["user_id"] = user_id
    where_sql = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    count_sql = f"SELECT COUNT(*) FROM auth_audit_log{where_sql}"
    r_count = await db.execute(text(count_sql), params)
    total = r_count.scalar_one()
    list_sql = f"""
        SELECT id, event_type, user_id, username, occurred_at, ip_address::text, user_agent, country, geo_json, details
        FROM auth_audit_log
        {where_sql}
        ORDER BY occurred_at DESC
        LIMIT :limit OFFSET :offset
    """
    r_list = await db.execute(text(list_sql), params)
    rows = r_list.fetchall()
    events = [
        {
            "id": row.id,
            "event_type": row.event_type,
            "user_id": row.user_id,
            "username": row.username,
            "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
            "ip_address": _normalize_ip(str(row.ip_address)) if row.ip_address else None,
            "user_agent": row.user_agent,
            "country": row.country,
            "geo_json": getattr(row, "geo_json", None),
            "details": row.details,
        }
        for row in rows
    ]
    return {"total": total, "limit": limit, "offset": offset, "events": events}


@router.get("/api/admin/preflight")
async def admin_preflight(
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return one-call pipeline health summary with freshness diagnostics."""

    def _iso(ts) -> str | None:
        """Convert nullable timestamps into ISO strings for preflight payloads."""
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
              (SELECT COUNT(*) FROM model_versions) AS model_versions_count,
              (SELECT COUNT(*) FROM training_runs) AS training_runs_count,
              (SELECT COUNT(*) FROM model_predictions) AS model_predictions_count,
              (SELECT COUNT(*) FROM trades) AS trades_count,
              (SELECT COUNT(*) FROM trades WHERE status = 'OPEN') AS open_trades_count,
              (SELECT COUNT(*) FROM trades WHERE status = 'CLOSED') AS closed_trades_count,
              (SELECT MAX(ts) FROM underlying_quotes) AS latest_quote_ts,
              (SELECT MAX(ts) FROM chain_snapshots) AS latest_snapshot_ts,
              (SELECT MAX(ts) FROM gex_snapshots) AS latest_gex_ts,
              (SELECT MAX(ts) FROM trade_decisions) AS latest_decision_ts,
              (SELECT MAX(ts) FROM feature_snapshots) AS latest_feature_ts,
              (SELECT MAX(ts) FROM trade_candidates) AS latest_candidate_ts,
              (SELECT MAX(created_at) FROM model_versions) AS latest_model_version_ts,
              (SELECT MAX(finished_at) FROM training_runs) AS latest_training_run_ts,
              (SELECT MAX(created_at) FROM model_predictions) AS latest_prediction_ts,
              (SELECT MAX(last_mark_ts) FROM trades) AS latest_trade_mark_ts,
              (SELECT MAX(entry_time) FROM trades) AS latest_trade_entry_ts,
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
        "model_versions": int(summary.model_versions_count or 0),
        "training_runs": int(summary.training_runs_count or 0),
        "model_predictions": int(summary.model_predictions_count or 0),
        "trades": int(summary.trades_count or 0),
        "open_trades": int(summary.open_trades_count or 0),
        "closed_trades": int(summary.closed_trades_count or 0),
    }
    latest = {
        "quote_ts": _iso(summary.latest_quote_ts),
        "snapshot_ts": _iso(summary.latest_snapshot_ts),
        "gex_ts": _iso(summary.latest_gex_ts),
        "decision_ts": _iso(summary.latest_decision_ts),
        "feature_ts": _iso(summary.latest_feature_ts),
        "candidate_ts": _iso(summary.latest_candidate_ts),
        "model_version_ts": _iso(summary.latest_model_version_ts),
        "training_run_ts": _iso(summary.latest_training_run_ts),
        "prediction_ts": _iso(summary.latest_prediction_ts),
        "trade_mark_ts": _iso(summary.latest_trade_mark_ts),
        "trade_entry_ts": _iso(summary.latest_trade_entry_ts),
        "market_clock_ts": _iso(summary.latest_market_clock_ts),
    }
    now_utc = datetime.now(tz=timezone.utc)
    quote_stale_threshold_min = max(float(settings.quote_interval_minutes) * 3.0, 10.0)
    snapshot_stale_threshold_min = max(float(settings.snapshot_interval_minutes) * 3.0, 15.0)
    gex_stale_threshold_min = max(float(settings.gex_interval_minutes) * 3.0, 15.0)
    market_clock_stale_threshold_min = max((float(settings.market_clock_cache_seconds) / 60.0) * 3.0, 10.0)
    quote_age_min = _minutes_since(summary.latest_quote_ts, now_utc)
    snapshot_age_min = _minutes_since(summary.latest_snapshot_ts, now_utc)
    gex_age_min = _minutes_since(summary.latest_gex_ts, now_utc)
    market_clock_age_min = _minutes_since(summary.latest_market_clock_ts, now_utc)
    quote_freshness_by_symbol: dict[str, dict[str, object]] = {}
    latest_quotes_by_symbol = [{"symbol": row.symbol, "ts": _iso(row.ts), "last": row.last} for row in latest_quotes]
    quote_ts_by_symbol = {row.symbol: row.ts for row in latest_quotes}
    for symbol in ("SPX", "SPY", "VIX"):
        symbol_ts = quote_ts_by_symbol.get(symbol)
        symbol_age = _minutes_since(symbol_ts, now_utc)
        quote_freshness_by_symbol[symbol] = {
            "age_min": symbol_age,
            "is_stale": _is_stale(symbol_age, quote_stale_threshold_min),
            "ts": _iso(symbol_ts),
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
    if counts["model_versions"] == 0:
        warnings.append("no_model_versions")
    if counts["training_runs"] == 0:
        warnings.append("no_training_runs")
    if counts["model_predictions"] == 0:
        warnings.append("no_model_predictions")
    if counts["trades"] == 0:
        warnings.append("no_trades")
    if _is_stale(quote_age_min, quote_stale_threshold_min):
        warnings.append("stale_quotes_overall")
    if _is_stale(snapshot_age_min, snapshot_stale_threshold_min):
        warnings.append("stale_snapshots")
    if _is_stale(gex_age_min, gex_stale_threshold_min):
        warnings.append("stale_gex")
    if _is_stale(market_clock_age_min, market_clock_stale_threshold_min):
        warnings.append("stale_market_clock")
    for symbol in ("SPX", "SPY", "VIX"):
        symbol_freshness = quote_freshness_by_symbol[symbol]
        if symbol_freshness["ts"] is None:
            warnings.append(f"missing_quote_{symbol.lower()}")
            continue
        if bool(symbol_freshness["is_stale"]):
            warnings.append(f"stale_quote_{symbol.lower()}")

    return {
        "now_utc": now_utc.isoformat(),
        "counts": counts,
        "latest": latest,
        "freshness": {
            "quote_age_min": quote_age_min,
            "snapshot_age_min": snapshot_age_min,
            "gex_age_min": gex_age_min,
            "market_clock_age_min": market_clock_age_min,
            "quote_stale_threshold_min": quote_stale_threshold_min,
            "snapshot_stale_threshold_min": snapshot_stale_threshold_min,
            "gex_stale_threshold_min": gex_stale_threshold_min,
            "market_clock_stale_threshold_min": market_clock_stale_threshold_min,
            "quotes_by_symbol": quote_freshness_by_symbol,
        },
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
        "latest_quotes_by_symbol": latest_quotes_by_symbol,
        "warnings": warnings,
    }

