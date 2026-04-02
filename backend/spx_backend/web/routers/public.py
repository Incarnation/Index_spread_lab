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
from spx_backend.jobs.modeling import compute_margin_usage_dollars, compute_max_drawdown, summarize_strategy_quality
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


@router.get("/api/model-ops")
async def get_model_ops(
    model_name: str | None = None,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return latest model version, training attempt, and prediction activity."""
    selected_model_name = model_name.strip() if isinstance(model_name, str) and model_name.strip() else settings.trainer_model_name

    counts_row = (
        await db.execute(
            text(
                """
                SELECT
                  (SELECT COUNT(*) FROM model_versions WHERE model_name = :model_name) AS model_versions_count,
                  (SELECT COUNT(*)
                    FROM training_runs tr
                    LEFT JOIN model_versions mv ON mv.model_version_id = tr.model_version_id
                    WHERE COALESCE(NULLIF(tr.config_json->>'model_name', ''), mv.model_name) = :model_name
                  ) AS training_runs_count,
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
                LEFT JOIN model_versions mv ON mv.model_version_id = tr.model_version_id
                WHERE COALESCE(NULLIF(tr.config_json->>'model_name', ''), mv.model_name) = :model_name
                ORDER BY COALESCE(tr.finished_at, tr.started_at) DESC NULLS LAST, tr.training_run_id DESC
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
    skip_reason = training_metrics.get("skipped_reason") if isinstance(training_metrics.get("skipped_reason"), str) else None
    warnings: list[str] = []
    if counts_row is not None:
        if int(counts_row.model_versions_count or 0) == 0:
            warnings.append("no_model_versions")
        if int(counts_row.training_runs_count or 0) == 0:
            warnings.append("no_training_runs")
        if int(counts_row.model_predictions_count or 0) == 0:
            warnings.append("no_model_predictions")
    if latest_training_row is not None:
        latest_training_status = str(latest_training_row.status or "").upper()
        if latest_training_status == "SKIPPED":
            warnings.append("latest_training_skipped")
        elif latest_training_status == "FAILED":
            warnings.append("latest_training_failed")

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
                "model_version_id": (
                    int(latest_training_row.model_version_id)
                    if latest_training_row.model_version_id is not None
                    else None
                ),
                "status": str(latest_training_row.status),
                "started_at_utc": _iso(latest_training_row.started_at),
                "finished_at_utc": _iso(latest_training_row.finished_at),
                "rows_train": int(latest_training_row.rows_train or 0),
                "rows_test": int(latest_training_row.rows_test or 0),
                "notes": latest_training_row.notes,
                "skip_reason": skip_reason,
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


@router.get("/api/model-predictions")
async def list_model_predictions(
    limit: int = 50,
    offset: int = 0,
    model_version_id: int | None = None,
    decision: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return paginated model predictions joined with candidate outcomes.

    Joins model_predictions -> trade_candidates (for outcome labels) and
    model_versions (for model metadata). Supports filtering by model version,
    decision hint, and date range.
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    where_parts: list[str] = []
    params: dict[str, object] = {"limit": limit, "offset": offset}

    if model_version_id is not None:
        where_parts.append("mp.model_version_id = :mv_id")
        params["mv_id"] = model_version_id
    if decision:
        where_parts.append("COALESCE(mp.meta_json->>'decision_hint', mp.decision) = :decision")
        params["decision"] = decision.strip().upper()
    if date_from:
        where_parts.append("mp.created_at >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where_parts.append("mp.created_at <= :date_to")
        params["date_to"] = date_to

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    r = await db.execute(
        text(
            f"""
            SELECT
              mp.prediction_id,
              mp.candidate_id,
              mp.model_version_id,
              mv.model_name,
              mv.version AS model_version,
              mp.probability_win,
              mp.expected_value,
              mp.score_raw,
              mp.meta_json,
              mp.decision,
              mp.created_at,
              tc.realized_pnl,
              tc.hit_tp50_before_sl_or_expiry AS hit_tp50,
              tc.label_json->>'hold_realized_pnl' AS hold_realized_pnl,
              tc.label_json->>'hold_hit_tp50' AS hold_hit_tp50,
              tc.label_json->>'hold_exit_reason' AS hold_exit_reason,
              tc.label_status
            FROM model_predictions mp
            JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
            LEFT JOIN trade_candidates tc ON tc.candidate_id = mp.candidate_id
            {where_clause}
            ORDER BY mp.created_at DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    )
    rows = r.fetchall()

    count_r = await db.execute(
        text(
            f"""
            SELECT COUNT(*) AS total
            FROM model_predictions mp
            JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
            LEFT JOIN trade_candidates tc ON tc.candidate_id = mp.candidate_id
            {where_clause}
            """
        ),
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
    )
    total = int((count_r.fetchone() or (0,))[0])

    items = []
    for row in rows:
        meta = row.meta_json if isinstance(row.meta_json, dict) else {}
        items.append({
            "prediction_id": row.prediction_id,
            "candidate_id": row.candidate_id,
            "model_version_id": row.model_version_id,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "probability_win": float(row.probability_win) if row.probability_win is not None else None,
            "expected_value": float(row.expected_value) if row.expected_value is not None else None,
            "score_raw": float(row.score_raw) if row.score_raw is not None else None,
            "decision_hint": meta.get("decision_hint") or getattr(row, "decision", None),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "hold_realized_pnl": _try_float(row.hold_realized_pnl),
            "hold_hit_tp50": row.hold_hit_tp50,
            "hold_exit_reason": row.hold_exit_reason,
            "realized_pnl": float(row.realized_pnl) if row.realized_pnl is not None else None,
            "label_status": row.label_status,
        })

    return {"total": total, "limit": limit, "offset": offset, "items": items}


@router.get("/api/model-accuracy")
async def get_model_accuracy(
    model_name: str | None = None,
    window: str = "week",
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return accuracy metrics aggregated over time windows.

    Computes precision, recall, accuracy for the TRADE decision by comparing
    model predictions against actual trade outcomes (hold_realized_pnl >= 0 = win).
    """
    selected_model = (model_name or "").strip() or settings.shadow_inference_model_name
    trunc = "week" if window == "week" else "month"

    r = await db.execute(
        text(
            f"""
            WITH pred_outcomes AS (
              SELECT
                date_trunc(:trunc, mp.created_at) AS period,
                COALESCE(mp.meta_json->>'decision_hint', mp.decision) AS decision_hint,
                CASE
                  WHEN tc.label_status = 'resolved' AND tc.realized_pnl IS NOT NULL
                    THEN CASE WHEN tc.realized_pnl >= 0 THEN true ELSE false END
                  ELSE NULL
                END AS actual_win,
                tc.realized_pnl
              FROM model_predictions mp
              JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
              LEFT JOIN trade_candidates tc ON tc.candidate_id = mp.candidate_id
              WHERE mv.model_name = :model_name
                AND tc.label_status = 'resolved'
            )
            SELECT
              period,
              COUNT(*) AS total,
              SUM(CASE WHEN decision_hint = 'TRADE' AND actual_win = true THEN 1 ELSE 0 END) AS true_positive,
              SUM(CASE WHEN decision_hint = 'TRADE' AND actual_win = false THEN 1 ELSE 0 END) AS false_positive,
              SUM(CASE WHEN decision_hint = 'SKIP' AND actual_win = false THEN 1 ELSE 0 END) AS true_negative,
              SUM(CASE WHEN decision_hint = 'SKIP' AND actual_win = true THEN 1 ELSE 0 END) AS false_negative,
              AVG(CASE WHEN decision_hint = 'TRADE' THEN realized_pnl END) AS avg_pnl_traded,
              AVG(CASE WHEN decision_hint = 'SKIP' THEN realized_pnl END) AS avg_pnl_skipped
            FROM pred_outcomes
            WHERE period IS NOT NULL
            GROUP BY period
            ORDER BY period ASC
            """
        ),
        {"model_name": selected_model, "trunc": trunc},
    )
    rows = r.fetchall()

    windows = []
    for row in rows:
        tp = int(row.true_positive or 0)
        fp = int(row.false_positive or 0)
        tn = int(row.true_negative or 0)
        fn = int(row.false_negative or 0)
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        windows.append({
            "period": row.period.isoformat() if row.period else None,
            "total": total,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "avg_pnl_traded": float(row.avg_pnl_traded) if row.avg_pnl_traded is not None else None,
            "avg_pnl_skipped": float(row.avg_pnl_skipped) if row.avg_pnl_skipped is not None else None,
        })

    return {"model_name": selected_model, "window": trunc, "windows": windows}


@router.get("/api/model-calibration")
async def get_model_calibration(
    model_name: str | None = None,
    bins: int = 10,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return calibration curve data: predicted probability bins vs observed win rate.

    Bins predictions by probability_win decile and computes the observed outcome
    rate per bin to assess model calibration.
    """
    selected_model = (model_name or "").strip() or settings.shadow_inference_model_name
    bins = max(2, min(bins, 20))
    bin_width = 1.0 / bins

    r = await db.execute(
        text(
            """
            SELECT
              mp.probability_win,
              CASE
                WHEN tc.label_status = 'resolved' AND tc.realized_pnl IS NOT NULL
                  THEN CASE WHEN tc.realized_pnl >= 0 THEN 1.0 ELSE 0.0 END
                ELSE NULL
              END AS actual_outcome
            FROM model_predictions mp
            JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
            LEFT JOIN trade_candidates tc ON tc.candidate_id = mp.candidate_id
            WHERE mv.model_name = :model_name
              AND mp.probability_win IS NOT NULL
              AND tc.label_status = 'resolved'
            """
        ),
        {"model_name": selected_model},
    )
    rows = r.fetchall()

    bin_data: list[dict[str, list[float]]] = [{"predicted": [], "actual": []} for _ in range(bins)]
    for row in rows:
        prob = float(row.probability_win)
        actual = float(row.actual_outcome) if row.actual_outcome is not None else None
        if actual is None:
            continue
        bin_idx = min(int(prob / bin_width), bins - 1)
        bin_data[bin_idx]["predicted"].append(prob)
        bin_data[bin_idx]["actual"].append(actual)

    result_bins = []
    for i, bd in enumerate(bin_data):
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width
        count = len(bd["predicted"])
        predicted_avg = sum(bd["predicted"]) / count if count > 0 else (bin_lower + bin_upper) / 2
        observed_rate = sum(bd["actual"]) / count if count > 0 else None
        result_bins.append({
            "bin_lower": round(bin_lower, 3),
            "bin_upper": round(bin_upper, 3),
            "predicted_avg": round(predicted_avg, 4),
            "observed_rate": round(observed_rate, 4) if observed_rate is not None else None,
            "count": count,
        })

    return {"model_name": selected_model, "bins": result_bins}


@router.get("/api/model-pnl-attribution")
async def get_model_pnl_attribution(
    model_name: str | None = None,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return PnL attribution comparing model-filtered trades vs baseline.

    baseline_pnl: total PnL if all candidates were traded.
    model_pnl: total PnL of candidates the model said TRADE.
    saved_pnl: losses avoided by skipping (negative PnL candidates that were SKIPped).
    missed_pnl: wins missed by skipping (positive PnL candidates that were SKIPped).
    """
    selected_model = (model_name or "").strip() or settings.shadow_inference_model_name

    r = await db.execute(
        text(
            """
            SELECT
              COALESCE(mp.meta_json->>'decision_hint', mp.decision) AS decision_hint,
              tc.realized_pnl
            FROM model_predictions mp
            JOIN model_versions mv ON mv.model_version_id = mp.model_version_id
            LEFT JOIN trade_candidates tc ON tc.candidate_id = mp.candidate_id
            WHERE mv.model_name = :model_name
              AND tc.label_status = 'resolved'
              AND tc.realized_pnl IS NOT NULL
            """
        ),
        {"model_name": selected_model},
    )
    rows = r.fetchall()

    baseline_pnl = 0.0
    model_pnl = 0.0
    saved_pnl = 0.0
    missed_pnl = 0.0
    trade_count = 0
    skip_count = 0

    for row in rows:
        pnl = float(row.realized_pnl)
        hint = (row.decision_hint or "").upper()
        baseline_pnl += pnl
        if hint == "TRADE":
            model_pnl += pnl
            trade_count += 1
        else:
            skip_count += 1
            if pnl < 0:
                saved_pnl += abs(pnl)
            else:
                missed_pnl += pnl

    return {
        "model_name": selected_model,
        "baseline_pnl": round(baseline_pnl, 2),
        "model_pnl": round(model_pnl, 2),
        "saved_pnl": round(saved_pnl, 2),
        "missed_pnl": round(missed_pnl, 2),
        "net_impact": round(model_pnl - baseline_pnl, 2),
        "trade_count": trade_count,
        "skip_count": skip_count,
        "total_candidates": trade_count + skip_count,
    }


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

