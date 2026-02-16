from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.database import get_db_session

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Simple health check."""
    return {"status": "ok"}


@router.get("/api/chain-snapshots")
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


@router.get("/api/trade-decisions")
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


@router.get("/api/trades")
async def list_trades(limit: int = 100, status: str | None = None, db: AsyncSession = Depends(get_db_session)) -> dict:
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
async def get_label_metrics(lookback_days: int = 90, db: AsyncSession = Depends(get_db_session)) -> dict:
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


@router.get("/api/gex/snapshots")
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


@router.get("/api/gex/dtes")
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


@router.get("/api/gex/expirations")
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


@router.get("/api/gex/curve")
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


@router.get("/", response_class=HTMLResponse)
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

