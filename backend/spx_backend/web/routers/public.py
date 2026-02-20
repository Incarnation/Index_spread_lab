from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.database import get_db_session
from spx_backend.jobs.modeling import compute_margin_usage_dollars, summarize_strategy_quality
from spx_backend.web.routers.auth import UserOut, get_current_user

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Simple health check."""
    return {"status": "ok"}


@router.get("/api/chain-snapshots")
async def list_chain_snapshots(
    limit: int = 50,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
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


@router.get("/api/trade-decisions")
async def list_trade_decisions(
    limit: int = 50,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
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


@router.get("/api/trades")
async def list_trades(
    limit: int = 100,
    status: str | None = None,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return recent trades with live PnL fields and leg metadata."""
    limit = max(1, min(limit, 500))
    normalized_status = status.strip().upper() if status is not None else None
    if normalized_status not in {None, "OPEN", "CLOSED", "ROLLED"}:
        raise HTTPException(status_code=400, detail="invalid_status")

    status_where_clause = ""
    query_params: dict[str, object] = {"limit": limit}
    if normalized_status is not None:
        status_where_clause = "WHERE t.status = :status"
        query_params["status"] = normalized_status

    r = await db.execute(
        text(
            f"""
            SELECT
              t.trade_id,
              t.decision_id,
              t.candidate_id,
              t.feature_snapshot_id,
              t.status,
              t.trade_source,
              t.strategy_type,
              t.underlying,
              t.entry_time,
              t.exit_time,
              t.last_mark_ts,
              t.target_dte,
              t.expiration,
              t.contracts,
              t.contract_multiplier,
              t.spread_width_points,
              t.entry_credit,
              t.current_exit_cost,
              t.current_pnl,
              t.realized_pnl,
              t.max_profit,
              t.max_loss,
              t.take_profit_target,
              t.stop_loss_target,
              t.exit_reason,
              (
                SELECT COUNT(*)
                FROM trade_marks tm
                WHERE tm.trade_id = t.trade_id
              ) AS mark_count,
              COALESCE(
                (
                  SELECT jsonb_agg(
                    jsonb_build_object(
                      'leg_index', tl.leg_index,
                      'option_symbol', tl.option_symbol,
                      'side', tl.side,
                      'qty', tl.qty,
                      'entry_price', tl.entry_price,
                      'exit_price', tl.exit_price,
                      'strike', tl.strike,
                      'expiration', tl.expiration,
                      'option_right', tl.option_right
                    )
                    ORDER BY tl.leg_index ASC
                  )
                  FROM trade_legs tl
                  WHERE tl.trade_id = t.trade_id
                ),
                '[]'::jsonb
              ) AS legs_json
            FROM trades t
            {status_where_clause}
            ORDER BY t.entry_time DESC, t.trade_id DESC
            LIMIT :limit
            """
        ),
        query_params,
    )
    rows = r.fetchall()
    return {
        "items": [
            {
                "trade_id": row.trade_id,
                "decision_id": row.decision_id,
                "candidate_id": row.candidate_id,
                "feature_snapshot_id": row.feature_snapshot_id,
                "status": row.status,
                "trade_source": row.trade_source,
                "strategy_type": row.strategy_type,
                "underlying": row.underlying,
                "entry_time": row.entry_time.isoformat(),
                "exit_time": (row.exit_time.isoformat() if row.exit_time is not None else None),
                "last_mark_ts": (row.last_mark_ts.isoformat() if row.last_mark_ts is not None else None),
                "target_dte": row.target_dte,
                "expiration": (str(row.expiration) if row.expiration is not None else None),
                "contracts": row.contracts,
                "contract_multiplier": row.contract_multiplier,
                "spread_width_points": row.spread_width_points,
                "entry_credit": row.entry_credit,
                "current_exit_cost": row.current_exit_cost,
                "current_pnl": row.current_pnl,
                "realized_pnl": row.realized_pnl,
                "max_profit": row.max_profit,
                "max_loss": row.max_loss,
                "take_profit_target": row.take_profit_target,
                "stop_loss_target": row.stop_loss_target,
                "exit_reason": row.exit_reason,
                "mark_count": int(row.mark_count or 0),
                "legs": list(row.legs_json or []),
            }
            for row in rows
        ]
    }


@router.get("/api/label-metrics")
async def get_label_metrics(
    lookback_days: int = 90,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return TP50/TP100 label metrics overall and by spread side."""
    if lookback_days <= 0 or lookback_days > 3650:
        raise HTTPException(status_code=400, detail="invalid_lookback_days")

    window_start = datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=lookback_days)

    summary_row = (
        await db.execute(
            text(
                """
                SELECT
                  COUNT(*)::bigint AS resolved_count,
                  SUM(CASE WHEN COALESCE(hit_tp50_before_sl_or_expiry, false) THEN 1 ELSE 0 END)::bigint AS tp50_count,
                  SUM(CASE WHEN COALESCE((label_json->>'hit_tp100_at_expiry')::boolean, false) THEN 1 ELSE 0 END)::bigint AS tp100_count,
                  AVG(realized_pnl) AS avg_realized_pnl
                FROM trade_candidates
                WHERE label_status = 'resolved'
                  AND ts >= :window_start
                """
            ),
            {"window_start": window_start},
        )
    ).fetchone()

    side_rows = (
        await db.execute(
            text(
                """
                SELECT
                  COALESCE(candidate_json->>'spread_side', 'unknown') AS spread_side,
                  COUNT(*)::bigint AS resolved_count,
                  SUM(CASE WHEN COALESCE(hit_tp50_before_sl_or_expiry, false) THEN 1 ELSE 0 END)::bigint AS tp50_count,
                  SUM(CASE WHEN COALESCE((label_json->>'hit_tp100_at_expiry')::boolean, false) THEN 1 ELSE 0 END)::bigint AS tp100_count,
                  AVG(realized_pnl) AS avg_realized_pnl
                FROM trade_candidates
                WHERE label_status = 'resolved'
                  AND ts >= :window_start
                GROUP BY spread_side
                ORDER BY spread_side ASC
                """
            ),
            {"window_start": window_start},
        )
    ).fetchall()

    def _rate(numerator: int, denominator: int) -> float | None:
        """Safely compute a ratio, returning None when denominator is zero."""
        if denominator <= 0:
            return None
        return numerator / denominator

    total_resolved = int(summary_row.resolved_count or 0) if summary_row is not None else 0
    total_tp50 = int(summary_row.tp50_count or 0) if summary_row is not None else 0
    total_tp100 = int(summary_row.tp100_count or 0) if summary_row is not None else 0
    total_avg_pnl = float(summary_row.avg_realized_pnl) if (summary_row is not None and summary_row.avg_realized_pnl is not None) else None

    by_side = []
    for row in side_rows:
        resolved_count = int(row.resolved_count or 0)
        tp50_count = int(row.tp50_count or 0)
        tp100_count = int(row.tp100_count or 0)
        by_side.append(
            {
                "spread_side": row.spread_side,
                "resolved": resolved_count,
                "tp50": tp50_count,
                "tp100_at_expiry": tp100_count,
                "tp50_rate": _rate(tp50_count, resolved_count),
                "tp100_at_expiry_rate": _rate(tp100_count, resolved_count),
                "avg_realized_pnl": (float(row.avg_realized_pnl) if row.avg_realized_pnl is not None else None),
            }
        )

    return {
        "lookback_days": lookback_days,
        "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
        "summary": {
            "resolved": total_resolved,
            "tp50": total_tp50,
            "tp100_at_expiry": total_tp100,
            "tp50_rate": _rate(total_tp50, total_resolved),
            "tp100_at_expiry_rate": _rate(total_tp100, total_resolved),
            "avg_realized_pnl": total_avg_pnl,
        },
        "by_side": by_side,
    }


@router.get("/api/strategy-metrics")
async def get_strategy_metrics(
    lookback_days: int = 90,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return v2 strategy-quality and risk metrics overall + by side."""
    if lookback_days <= 0 or lookback_days > 3650:
        raise HTTPException(status_code=400, detail="invalid_lookback_days")

    window_start = datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=lookback_days)
    rows = (
        await db.execute(
            text(
                """
                SELECT
                  ts,
                  realized_pnl,
                  COALESCE(hit_tp50_before_sl_or_expiry, false) AS hit_tp50,
                  COALESCE((label_json->>'hit_tp100_at_expiry')::boolean, false) AS hit_tp100,
                  COALESCE(candidate_json->>'spread_side', 'unknown') AS spread_side,
                  max_loss,
                  COALESCE((candidate_json->>'contracts')::int, 1) AS contracts
                FROM trade_candidates
                WHERE label_status = 'resolved'
                  AND ts >= :window_start
                  AND realized_pnl IS NOT NULL
                ORDER BY ts ASC
                """
            ),
            {"window_start": window_start},
        )
    ).fetchall()

    realized_pnls: list[float] = []
    margins: list[float] = []
    tp50_count = 0
    tp100_count = 0
    by_side_data: dict[str, dict[str, object]] = {}

    for row in rows:
        pnl = float(row.realized_pnl)
        contracts = int(row.contracts or 1)
        max_loss_points = float(row.max_loss) if row.max_loss is not None else None
        margin_usage = compute_margin_usage_dollars(
            max_loss_points=max_loss_points,
            contracts=contracts,
            contract_multiplier=settings.label_contract_multiplier,
        )
        realized_pnls.append(pnl)
        margins.append(margin_usage)
        if bool(row.hit_tp50):
            tp50_count += 1
        if bool(row.hit_tp100):
            tp100_count += 1

        spread_side = str(row.spread_side or "unknown")
        bucket = by_side_data.setdefault(spread_side, {"pnls": [], "margins": [], "tp50": 0, "tp100": 0})
        cast_pnls = bucket["pnls"]
        cast_margins = bucket["margins"]
        if isinstance(cast_pnls, list):
            cast_pnls.append(pnl)
        if isinstance(cast_margins, list):
            cast_margins.append(margin_usage)
        if bool(row.hit_tp50):
            bucket["tp50"] = int(bucket["tp50"]) + 1
        if bool(row.hit_tp100):
            bucket["tp100"] = int(bucket["tp100"]) + 1

    summary = summarize_strategy_quality(
        realized_pnls=realized_pnls,
        margin_usages=margins,
        hit_tp50_count=tp50_count,
        hit_tp100_count=tp100_count,
    )
    by_side = []
    for spread_side in sorted(by_side_data.keys()):
        payload = by_side_data[spread_side]
        side_summary = summarize_strategy_quality(
            realized_pnls=list(payload["pnls"]) if isinstance(payload["pnls"], list) else [],
            margin_usages=list(payload["margins"]) if isinstance(payload["margins"], list) else [],
            hit_tp50_count=int(payload["tp50"]),
            hit_tp100_count=int(payload["tp100"]),
        )
        by_side.append({"spread_side": spread_side, **side_summary})

    return {
        "lookback_days": lookback_days,
        "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
        "summary": summary,
        "by_side": by_side,
    }


@router.get("/api/model-ops")
async def get_model_ops(
    model_name: str | None = None,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return latest model/training/gate status for dashboard monitoring."""
    selected_model_name = model_name.strip() if isinstance(model_name, str) and model_name.strip() else settings.trainer_model_name

    counts_row = (
        await db.execute(
            text(
                """
                SELECT
                  (SELECT COUNT(*) FROM model_versions WHERE model_name = :model_name) AS model_versions_count,
                  (SELECT COUNT(*) FROM training_runs tr
                    JOIN model_versions mv ON mv.model_version_id = tr.model_version_id
                    WHERE mv.model_name = :model_name) AS training_runs_count,
                  (SELECT COUNT(*) FROM model_predictions mp
                    JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
                    WHERE mv.model_name = :model_name) AS model_predictions_count,
                  (SELECT COUNT(*) FROM model_predictions mp
                    JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
                    WHERE mv.model_name = :model_name AND mp.created_at >= :since_24h) AS model_predictions_24h_count,
                  (SELECT MAX(mp.created_at) FROM model_predictions mp
                    JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
                    WHERE mv.model_name = :model_name) AS latest_prediction_ts
                """
            ),
            {"model_name": selected_model_name, "since_24h": datetime.now(tz=ZoneInfo("UTC")) - timedelta(hours=24)},
        )
    ).fetchone()

    latest_model_row = (
        await db.execute(
            text(
                """
                SELECT
                  model_version_id,
                  version,
                  rollout_status,
                  is_active,
                  created_at,
                  promoted_at,
                  metrics_json
                FROM model_versions
                WHERE model_name = :model_name
                ORDER BY created_at DESC, model_version_id DESC
                LIMIT 1
                """
            ),
            {"model_name": selected_model_name},
        )
    ).fetchone()

    latest_training_row = (
        await db.execute(
            text(
                """
                SELECT
                  tr.training_run_id,
                  tr.model_version_id,
                  tr.status,
                  tr.started_at,
                  tr.finished_at,
                  tr.rows_train,
                  tr.rows_test,
                  tr.notes,
                  tr.metrics_json
                FROM training_runs tr
                JOIN model_versions mv ON mv.model_version_id = tr.model_version_id
                WHERE mv.model_name = :model_name
                ORDER BY tr.finished_at DESC NULLS LAST, tr.training_run_id DESC
                LIMIT 1
                """
            ),
            {"model_name": selected_model_name},
        )
    ).fetchone()

    active_model_row = (
        await db.execute(
            text(
                """
                SELECT
                  model_version_id,
                  version,
                  rollout_status,
                  is_active,
                  created_at,
                  promoted_at
                FROM model_versions
                WHERE model_name = :model_name
                  AND is_active = true
                ORDER BY promoted_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            ),
            {"model_name": selected_model_name},
        )
    ).fetchone()

    def _iso(ts: Any) -> str | None:
        """Convert nullable timestamps into ISO strings for API responses."""
        return ts.isoformat() if ts is not None else None

    model_metrics = latest_model_row.metrics_json if (latest_model_row is not None and isinstance(latest_model_row.metrics_json, dict)) else {}
    model_metrics_summary = {
        "tp50_rate_test": model_metrics.get("tp50_rate_test"),
        "expectancy_test": model_metrics.get("expectancy_test"),
        "max_drawdown_test": model_metrics.get("max_drawdown_test"),
        "tail_loss_proxy_test": model_metrics.get("tail_loss_proxy_test"),
        "avg_margin_usage_test": model_metrics.get("avg_margin_usage_test"),
    }

    training_metrics = (
        latest_training_row.metrics_json if (latest_training_row is not None and isinstance(latest_training_row.metrics_json, dict)) else {}
    )
    gate = training_metrics.get("gate") if isinstance(training_metrics.get("gate"), dict) else None
    warnings: list[str] = []
    if counts_row is not None:
        if int(counts_row.model_versions_count or 0) == 0:
            warnings.append("no_model_versions")
        if int(counts_row.training_runs_count or 0) == 0:
            warnings.append("no_training_runs")
        if int(counts_row.model_predictions_count or 0) == 0:
            warnings.append("no_model_predictions")

    return {
        "model_name": selected_model_name,
        "counts": {
            "model_versions": int(counts_row.model_versions_count or 0) if counts_row is not None else 0,
            "training_runs": int(counts_row.training_runs_count or 0) if counts_row is not None else 0,
            "model_predictions": int(counts_row.model_predictions_count or 0) if counts_row is not None else 0,
            "model_predictions_24h": int(counts_row.model_predictions_24h_count or 0) if counts_row is not None else 0,
        },
        "latest_prediction_ts": (_iso(counts_row.latest_prediction_ts) if counts_row is not None else None),
        "latest_model_version": (
            None
            if latest_model_row is None
            else {
                "model_version_id": int(latest_model_row.model_version_id),
                "version": str(latest_model_row.version),
                "rollout_status": str(latest_model_row.rollout_status),
                "is_active": bool(latest_model_row.is_active),
                "created_at_utc": _iso(latest_model_row.created_at),
                "promoted_at_utc": _iso(latest_model_row.promoted_at),
                "metrics": model_metrics_summary,
            }
        ),
        "active_model_version": (
            None
            if active_model_row is None
            else {
                "model_version_id": int(active_model_row.model_version_id),
                "version": str(active_model_row.version),
                "rollout_status": str(active_model_row.rollout_status),
                "is_active": bool(active_model_row.is_active),
                "created_at_utc": _iso(active_model_row.created_at),
                "promoted_at_utc": _iso(active_model_row.promoted_at),
            }
        ),
        "latest_training_run": (
            None
            if latest_training_row is None
            else {
                "training_run_id": int(latest_training_row.training_run_id),
                "model_version_id": int(latest_training_row.model_version_id),
                "status": str(latest_training_row.status),
                "started_at_utc": _iso(latest_training_row.started_at),
                "finished_at_utc": _iso(latest_training_row.finished_at),
                "rows_train": int(latest_training_row.rows_train or 0),
                "rows_test": int(latest_training_row.rows_test or 0),
                "notes": latest_training_row.notes,
                "gate": gate,
            }
        ),
        "warnings": warnings,
    }


@router.get("/api/gex/snapshots")
async def list_gex_snapshots(
    limit: int = 50,
    underlying: str | None = None,
    source: str | None = None,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return recent GEX snapshot aggregates with optional symbol filtering.

    Parameters
    ----------
    limit:
        Maximum number of rows to return (clamped to 1..500).
    underlying:
        Optional underlying symbol filter (for example ``SPX``, ``SPY``, ``VIX``).
        Matching is case-insensitive, whitespace-tolerant, and applied in SQL.
    source:
        Optional source filter (for example ``TRADIER`` or ``CBOE``). When
        omitted, snapshots from all available sources are returned.

    Returns
    -------
    dict
        JSON payload with one ``items`` list containing snapshot metadata and
        aggregate GEX fields used by the dashboard selector.
    """
    limit = max(1, min(limit, 500))
    normalized_underlying = (underlying or "").strip().upper()
    normalized_source = (source or "").strip().upper()
    sql = """
        SELECT snapshot_id, ts, underlying, source, spot_price, gex_net, gex_calls, gex_puts, gex_abs, zero_gamma_level, method
        FROM gex_snapshots
    """
    params: dict[str, object] = {"limit": limit}
    where_clauses: list[str] = []
    if normalized_underlying:
        where_clauses.append("UPPER(TRIM(underlying)) = :underlying")
        params["underlying"] = normalized_underlying
    if normalized_source:
        where_clauses.append("UPPER(TRIM(source)) = :source")
        params["source"] = normalized_source
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " ORDER BY ts DESC, snapshot_id DESC LIMIT :limit"
    r = await db.execute(text(sql), params)
    rows = r.fetchall()
    return {
        "items": [
            {
                "snapshot_id": row.snapshot_id,
                "ts": row.ts.isoformat(),
                "underlying": row.underlying,
                "source": row.source,
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


@router.get("/api/gex/dtes")
async def list_gex_dtes(
    snapshot_id: int,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return available DTEs for a GEX snapshot batch (same ts + underlying + source)."""
    r = await db.execute(
        text(
            """
            WITH anchor AS (
              SELECT ts, underlying, source
              FROM gex_snapshots
              WHERE snapshot_id = :snapshot_id
            ),
            batch AS (
              SELECT gs.snapshot_id
              FROM gex_snapshots gs
              JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying AND gs.source = a.source
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


@router.get("/api/gex/expirations")
async def list_gex_expirations(
    snapshot_id: int,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return available expirations for a GEX snapshot batch (same ts + underlying + source)."""
    r = await db.execute(
        text(
            """
            WITH anchor AS (
              SELECT ts, underlying, source
              FROM gex_snapshots
              WHERE snapshot_id = :snapshot_id
            ),
            batch AS (
              SELECT gs.snapshot_id
              FROM gex_snapshots gs
              JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying AND gs.source = a.source
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


@router.get("/api/gex/curve")
async def get_gex_curve(
    snapshot_id: int,
    dte_days: int | None = None,
    expirations_csv: str | None = None,
    current_user: UserOut = Depends(get_current_user),
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
              SELECT ts, underlying, source
              FROM gex_snapshots
              WHERE snapshot_id = :snapshot_id
            ),
            batch AS (
              SELECT gs.snapshot_id
              FROM gex_snapshots gs
              JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying AND gs.source = a.source
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
                  SELECT ts, underlying, source
                  FROM gex_snapshots
                  WHERE snapshot_id = :snapshot_id
                ),
                batch AS (
                  SELECT gs.snapshot_id
                  FROM gex_snapshots gs
                  JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying AND gs.source = a.source
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
                  SELECT ts, underlying, source
                  FROM gex_snapshots
                  WHERE snapshot_id = :snapshot_id
                ),
                batch AS (
                  SELECT gs.snapshot_id
                  FROM gex_snapshots gs
                  JOIN anchor a ON gs.ts = a.ts AND gs.underlying = a.underlying AND gs.source = a.source
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


@router.get("/", response_class=HTMLResponse)
async def home(
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
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
      <head><title>IndexSpreadLab (Backend)</title></head>
      <body style="font-family: system-ui; max-width: 900px; margin: 40px auto;">
        <h2>IndexSpreadLab (Backend)</h2>
        <p>Server time: {datetime.utcnow().isoformat()}Z</p>
        <h3>Latest chain snapshots</h3>
        <ol>{items}</ol>
        <p><a href="/health">/health</a> · <a href="/api/chain-snapshots">/api/chain-snapshots</a></p>
      </body>
    </html>
    """
    return HTMLResponse(html)

