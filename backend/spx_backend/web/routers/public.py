from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.database import SessionLocal, get_db_session
from spx_backend.jobs.modeling import compute_max_drawdown
from spx_backend.web.routers.auth import UserOut, get_current_user

router = APIRouter()


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Compute a ratio with None for zero/negative denominators."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _profit_factor(win_pnl_sum: float, loss_pnl_sum: float) -> float | None:
    """Compute profit factor from aggregated win/loss PnL sums."""
    loss_abs = abs(loss_pnl_sum)
    if loss_abs <= 0:
        return None
    return win_pnl_sum / loss_abs


def _summarize_mode_rows(rows: list[Any]) -> dict[str, float | int | None]:
    """Summarize one mode's equity rows into headline KPI values.

    Parameters
    ----------
    rows:
        Equity rows for a single mode and lookback window.

    Returns
    -------
    dict[str, float | int | None]
        Trade counts and core PnL quality metrics derived from aggregated rows.
    """
    if not rows:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "net_pnl": 0.0,
            "win_pnl_sum": 0.0,
            "loss_pnl_sum": 0.0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "avg_pnl": None,
            "profit_factor": None,
            "expectancy": None,
            "max_drawdown": None,
        }
    trade_count = int(sum(int(row.trade_count or 0) for row in rows))
    win_count = int(sum(int(row.win_count or 0) for row in rows))
    loss_count = int(sum(int(row.loss_count or 0) for row in rows))
    net_pnl = float(sum(float(row.pnl_sum or 0.0) for row in rows))
    win_pnl_sum = float(sum(float(row.win_pnl_sum or 0.0) for row in rows))
    loss_pnl_sum = float(sum(float(row.loss_pnl_sum or 0.0) for row in rows))
    daily_pnls = [float(row.pnl_sum or 0.0) for row in rows]
    avg_pnl = _safe_ratio(net_pnl, float(trade_count))
    return {
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "net_pnl": net_pnl,
        "win_pnl_sum": win_pnl_sum,
        "loss_pnl_sum": loss_pnl_sum,
        "win_rate": _safe_ratio(float(win_count), float(trade_count)),
        "avg_win": _safe_ratio(win_pnl_sum, float(win_count)),
        "avg_loss": _safe_ratio(loss_pnl_sum, float(loss_count)),
        "avg_pnl": avg_pnl,
        "profit_factor": _profit_factor(win_pnl_sum, loss_pnl_sum),
        "expectancy": avg_pnl,
        "max_drawdown": (float(compute_max_drawdown(daily_pnls)) if daily_pnls else None),
    }


@router.get("/health")
async def health(request: Request) -> dict:
    """Deep health check -- DB connectivity, scheduler state, pipeline freshness.

    Returns ``status`` of ``healthy``, ``degraded``, or ``unhealthy`` along
    with per-check detail so load balancers and operators can assess system
    readiness at a glance.  No authentication required.
    """
    checks: dict[str, dict] = {}

    # -- DB connectivity --
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        checks["database"] = {"ok": False, "error": str(exc)[:200]}

    # -- Scheduler heartbeat --
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        running = getattr(scheduler, "running", False)
        checks["scheduler"] = {"ok": bool(running)}
    else:
        checks["scheduler"] = {"ok": False, "error": "scheduler not found"}

    # -- Pipeline freshness (only when DB is up) --
    if checks["database"]["ok"]:
        freshness_checks = [
            ("underlying_quotes", "ts", settings.staleness_quotes_max_minutes),
            ("chain_snapshots", "ts", settings.staleness_snapshots_max_minutes),
            ("trade_decisions", "ts", settings.staleness_decisions_max_minutes),
        ]
        try:
            async with SessionLocal() as session:
                for table, col, threshold in freshness_checks:
                    row = await session.execute(
                        text(f"SELECT MAX({col}) AS latest FROM {table}")
                    )
                    latest = row.scalar()
                    if latest is None:
                        checks[table] = {"ok": True, "age_minutes": None, "note": "empty"}
                        continue
                    latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=ZoneInfo("UTC"))
                    age = (datetime.now(ZoneInfo("UTC")) - latest_utc).total_seconds() / 60.0
                    checks[table] = {
                        "ok": age <= threshold,
                        "age_minutes": round(age, 1),
                        "threshold_minutes": threshold,
                    }
        except Exception as exc:
            checks["freshness"] = {"ok": False, "error": str(exc)[:200]}

    any_unhealthy = not checks["database"]["ok"]
    any_degraded = any(not c["ok"] for c in checks.values() if c is not checks.get("database"))
    status = "unhealthy" if any_unhealthy else ("degraded" if any_degraded else "healthy")

    return {"status": status, "checks": checks}


@router.get("/api/pipeline-status")
async def pipeline_status(
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Sanitized pipeline freshness for all authenticated users.

    Returns per-source age and a warnings list so the Overview page can
    show green/amber/red indicators without requiring admin privileges.
    Unlike ``/api/admin/preflight``, this omits row counts and admin-only
    detail.
    """
    sources = [
        ("underlying_quotes", "ts", settings.staleness_quotes_max_minutes),
        ("chain_snapshots", "ts", settings.staleness_snapshots_max_minutes),
        ("gex_snapshots", "ts", settings.staleness_gex_max_minutes),
        ("trade_decisions", "ts", settings.staleness_decisions_max_minutes),
    ]
    freshness: dict[str, dict] = {}
    warnings: list[str] = []
    try:
        async with SessionLocal() as session:
            now_utc = datetime.now(ZoneInfo("UTC"))
            for table, col, threshold in sources:
                row = await session.execute(
                    text(f"SELECT MAX({col}) AS latest FROM {table}")
                )
                latest = row.scalar()
                if latest is None:
                    freshness[table] = {"age_minutes": None, "stale": False}
                    continue
                latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=ZoneInfo("UTC"))
                age = (now_utc - latest_utc).total_seconds() / 60.0
                stale = age > threshold
                freshness[table] = {
                    "age_minutes": round(age, 1),
                    "threshold_minutes": threshold,
                    "stale": stale,
                }
                if stale:
                    warnings.append(f"{table} stale ({round(age)}m > {threshold}m)")
    except Exception as exc:
        warnings.append(f"freshness check error: {str(exc)[:100]}")

    return {"freshness": freshness, "warnings": warnings}


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
              -- `candidate_id` and `feature_snapshot_id` were dropped from
              -- the `trades` table by migration 015 when their referenced
              -- tables (`trade_candidates`, `feature_snapshots`) were
              -- removed in Track A.7.  The fields are no longer surfaced.
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
                # `candidate_id` / `feature_snapshot_id` were removed from
                # the response when their backing columns were dropped by
                # migration 015 (Track A.7).
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


### REMOVED IN ONLINE-ML DECOMMISSION ###################################
# ``GET /api/label-metrics`` and ``GET /api/strategy-metrics`` were
# removed because both endpoints aggregated outcomes from the
# decommissioned ``trade_candidates`` table (scheduled to be dropped in
# the A.7 schema-cleanup migration).  Per-strategy / TP50-TP100 metrics
# can be reconstructed from ``trades`` + ``trade_marks`` if needed for
# the offline ML re-entry path.
########################################################################


@router.get("/api/performance-analytics")
async def get_performance_analytics(
    lookback_days: int = 90,
    mode: str = "combined",
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return aggregate dashboard analytics for realized/combined PnL modes."""
    if lookback_days <= 0 or lookback_days > 3650:
        raise HTTPException(status_code=400, detail="invalid_lookback_days")
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"realized", "combined"}:
        raise HTTPException(status_code=400, detail="invalid_mode")

    latest_snapshot = (
        await db.execute(
            text(
                """
                SELECT
                  analytics_snapshot_id,
                  as_of_ts,
                  source_trade_count,
                  source_closed_count,
                  source_open_count
                FROM trade_performance_snapshots
                ORDER BY as_of_ts DESC, analytics_snapshot_id DESC
                LIMIT 1
                """
            )
        )
    ).fetchone()
    if latest_snapshot is None:
        return {
            "lookback_days": lookback_days,
            "mode": normalized_mode,
            "window_start_utc": None,
            "as_of_utc": None,
            "snapshot": None,
            "summary": None,
            "equity_curve": [],
            "breakdowns": {
                "side": [],
                "dte_bucket": [],
                "delta_bucket": [],
                "weekday": [],
                "hour": [],
                "source": [],
            },
        }

    window_start_date = (datetime.now(tz=ZoneInfo("UTC")) - timedelta(days=lookback_days)).date()
    mode_rows = (
        await db.execute(
            text(
                """
                SELECT
                  mode,
                  bucket_date,
                  trade_count,
                  win_count,
                  loss_count,
                  pnl_sum,
                  win_pnl_sum,
                  loss_pnl_sum
                FROM trade_performance_equity_curve
                WHERE analytics_snapshot_id = :analytics_snapshot_id
                  AND mode IN ('realized', 'combined')
                  AND bucket_date >= :window_start_date
                ORDER BY mode ASC, bucket_date ASC
                """
            ),
            {
                "analytics_snapshot_id": latest_snapshot.analytics_snapshot_id,
                "window_start_date": window_start_date,
            },
        )
    ).fetchall()

    by_mode: dict[str, list[Any]] = {"realized": [], "combined": []}
    for row in mode_rows:
        row_mode = str(row.mode).lower()
        if row_mode in by_mode:
            by_mode[row_mode].append(row)

    realized_summary = _summarize_mode_rows(by_mode["realized"])
    combined_summary = _summarize_mode_rows(by_mode["combined"])
    selected_summary = realized_summary if normalized_mode == "realized" else combined_summary

    selected_curve_rows = by_mode[normalized_mode]
    equity_curve: list[dict[str, Any]] = []
    cumulative = 0.0
    peak = 0.0
    for row in selected_curve_rows:
        daily_pnl = float(row.pnl_sum or 0.0)
        cumulative += daily_pnl
        peak = max(peak, cumulative)
        equity_curve.append(
            {
                "date": row.bucket_date.isoformat(),
                "daily_pnl": daily_pnl,
                "cumulative_pnl": cumulative,
                "drawdown": peak - cumulative,
                "trade_count": int(row.trade_count or 0),
                "win_count": int(row.win_count or 0),
                "loss_count": int(row.loss_count or 0),
            }
        )

    breakdown_rows = (
        await db.execute(
            text(
                """
                SELECT
                  dimension_type,
                  dimension_value,
                  SUM(trade_count)::bigint AS trade_count,
                  SUM(win_count)::bigint AS win_count,
                  SUM(loss_count)::bigint AS loss_count,
                  SUM(pnl_sum) AS pnl_sum,
                  SUM(win_pnl_sum) AS win_pnl_sum,
                  SUM(loss_pnl_sum) AS loss_pnl_sum
                FROM trade_performance_breakdowns
                WHERE analytics_snapshot_id = :analytics_snapshot_id
                  AND mode = :mode
                  AND bucket_date >= :window_start_date
                GROUP BY dimension_type, dimension_value
                ORDER BY dimension_type ASC, pnl_sum DESC, dimension_value ASC
                """
            ),
            {
                "analytics_snapshot_id": latest_snapshot.analytics_snapshot_id,
                "mode": normalized_mode,
                "window_start_date": window_start_date,
            },
        )
    ).fetchall()

    breakdowns: dict[str, list[dict[str, Any]]] = {
        "side": [],
        "dte_bucket": [],
        "delta_bucket": [],
        "weekday": [],
        "hour": [],
        "source": [],
    }
    for row in breakdown_rows:
        dimension_type = str(row.dimension_type)
        if dimension_type not in breakdowns:
            continue
        trade_count = int(row.trade_count or 0)
        win_count = int(row.win_count or 0)
        loss_count = int(row.loss_count or 0)
        pnl_sum = float(row.pnl_sum or 0.0)
        win_pnl_sum = float(row.win_pnl_sum or 0.0)
        loss_pnl_sum = float(row.loss_pnl_sum or 0.0)
        avg_pnl = _safe_ratio(pnl_sum, float(trade_count))
        breakdowns[dimension_type].append(
            {
                "bucket": row.dimension_value,
                "trade_count": trade_count,
                "win_count": win_count,
                "loss_count": loss_count,
                "net_pnl": pnl_sum,
                "win_rate": _safe_ratio(float(win_count), float(trade_count)),
                "avg_win": _safe_ratio(win_pnl_sum, float(win_count)),
                "avg_loss": _safe_ratio(loss_pnl_sum, float(loss_count)),
                "avg_pnl": avg_pnl,
                "expectancy": avg_pnl,
                "profit_factor": _profit_factor(win_pnl_sum, loss_pnl_sum),
            }
        )

    window_start_utc = datetime.combine(window_start_date, datetime.min.time(), tzinfo=ZoneInfo("UTC"))
    return {
        "lookback_days": lookback_days,
        "mode": normalized_mode,
        "window_start_utc": window_start_utc.isoformat().replace("+00:00", "Z"),
        "as_of_utc": latest_snapshot.as_of_ts.isoformat().replace("+00:00", "Z"),
        "snapshot": {
            "analytics_snapshot_id": int(latest_snapshot.analytics_snapshot_id),
            "source_trade_count": int(latest_snapshot.source_trade_count or 0),
            "source_closed_count": int(latest_snapshot.source_closed_count or 0),
            "source_open_count": int(latest_snapshot.source_open_count or 0),
        },
        "summary": {
            "trade_count": int(selected_summary["trade_count"] or 0),
            "win_count": int(selected_summary["win_count"] or 0),
            "loss_count": int(selected_summary["loss_count"] or 0),
            "net_pnl": float(selected_summary["net_pnl"] or 0.0),
            "realized_net_pnl": float(realized_summary["net_pnl"] or 0.0),
            "unrealized_net_pnl": float((combined_summary["net_pnl"] or 0.0) - (realized_summary["net_pnl"] or 0.0)),
            "combined_net_pnl": float(combined_summary["net_pnl"] or 0.0),
            "win_rate": selected_summary["win_rate"],
            "avg_win": selected_summary["avg_win"],
            "avg_loss": selected_summary["avg_loss"],
            "avg_pnl": selected_summary["avg_pnl"],
            "expectancy": selected_summary["expectancy"],
            "profit_factor": selected_summary["profit_factor"],
            "max_drawdown": selected_summary["max_drawdown"],
        },
        "equity_curve": equity_curve,
        "breakdowns": breakdowns,
    }


### REMOVED IN ONLINE-ML DECOMMISSION ###################################
# ``GET /api/model-ops`` was removed because it surfaced freshness/health
# of the decommissioned ``training_runs`` and ``model_predictions``
# tables (scheduled to be dropped in the A.7 schema-cleanup migration).
# ``model_versions`` -- which is preserved for offline ML re-entry --
# is still surfaced via ``/api/admin/preflight``'s ``model_versions``
# count + ``model_version_ts``.
########################################################################




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


### REMOVED IN ONLINE-ML DECOMMISSION ###################################
# The following endpoints were removed because they read from the
# decommissioned ``model_predictions`` and ``trade_candidates`` tables
# (both scheduled to be dropped in the A.7 schema-cleanup migration):
#
#   GET /api/model-predictions     - paginated model predictions
#   GET /api/model-accuracy        - precision/recall/accuracy windows
#   GET /api/model-calibration     - calibration curve bins
#   GET /api/model-pnl-attribution - model-vs-baseline PnL attribution
#
# The frontend ML monitoring page that consumed these endpoints
# (``ModelMonitorPage``) is being removed in Track A.6.  Future ML
# re-entry will plug into the portfolio path rather than restoring
# this monitoring surface unchanged.
########################################################################


@router.get("/api/backtest-results")
async def get_backtest_results(
    current_user: UserOut = Depends(get_current_user),
) -> dict:
    """Return precomputed backtest results if available.

    Reads from a JSON file generated by the offline backtest_entry.py script.
    Returns an empty strategies list if no results file exists yet.
    """
    import json
    from pathlib import Path

    results_path = Path(__file__).resolve().parents[3] / "scripts" / "backtest_results.json"
    if not results_path.exists():
        return {"strategies": [], "generated_at": None}
    try:
        raw = json.loads(results_path.read_text())
        return {"strategies": raw.get("strategies", []), "generated_at": raw.get("generated_at")}
    except Exception:
        return {"strategies": [], "generated_at": None}


def _try_float(val: object) -> float | None:
    """Attempt to convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


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

