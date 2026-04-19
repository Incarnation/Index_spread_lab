"""Portfolio management API endpoints.

Exposes portfolio state, history, trades, and configuration for the
capital-budgeted trading dashboard.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.database import get_db_session
from spx_backend.services.event_signals import EventSignalDetector
from spx_backend.web.routers.auth import get_current_user

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.get("/status")
async def get_portfolio_status(
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Return current-day portfolio state and active event signals.

    Reads the latest ``portfolio_state`` row, counts today's trades from
    ``portfolio_trades``, and evaluates live event signals.
    """
    today = date.today()

    row = await session.execute(text(
        "SELECT id, date, equity_start, equity_end, month_start_equity, "
        "trades_placed, lots_per_trade, daily_pnl, monthly_stop_active, event_signals "
        "FROM portfolio_state ORDER BY date DESC LIMIT 1"
    ))
    state = row.fetchone()

    trade_count_row = await session.execute(text(
        "SELECT COUNT(*) FROM portfolio_trades pt "
        "JOIN portfolio_state ps ON pt.portfolio_state_id = ps.id "
        "WHERE ps.date = :today"
    ), {"today": today})
    trades_today = int((trade_count_row.fetchone() or [0])[0])

    equity = float(state.equity_end or state.equity_start or settings.portfolio_starting_capital) if state else settings.portfolio_starting_capital
    month_start = float(state.month_start_equity or settings.portfolio_starting_capital) if state else settings.portfolio_starting_capital
    drawdown_pct = (1 - equity / month_start) * 100 if month_start > 0 else 0.0
    lots = max(1, int(equity / settings.portfolio_lot_per_equity))

    detector = EventSignalDetector()
    try:
        signals = await detector.detect(today)
    except Exception:
        signals = []

    return {
        "date": str(state.date) if state else str(today),
        "equity": round(equity, 2),
        "month_start_equity": round(month_start, 2),
        "drawdown_pct": round(drawdown_pct, 2),
        "lots_per_trade": lots,
        "trades_today": trades_today,
        "max_trades_per_day": settings.portfolio_max_trades_per_day,
        "max_trades_per_run": settings.portfolio_max_trades_per_run,
        "monthly_stop_active": bool(state.monthly_stop_active) if state else False,
        "daily_pnl": round(float(state.daily_pnl or 0), 2) if state else 0.0,
        "event_signals": signals,
    }


@router.get("/history")
async def get_portfolio_history(
    days: int = Query(90, ge=1, le=730),
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Return daily portfolio state history for equity charting.

    Parameters
    ----------
    days : Number of days of history to return (default 90).
    """
    cutoff = date.today() - timedelta(days=days)
    rows = await session.execute(text(
        "SELECT date, equity_start, equity_end, trades_placed, lots_per_trade, "
        "daily_pnl, monthly_stop_active, event_signals "
        "FROM portfolio_state WHERE date >= :cutoff ORDER BY date"
    ), {"cutoff": cutoff})

    items = []
    for r in rows.fetchall():
        items.append({
            "date": str(r.date),
            "equity_start": r.equity_start,
            "equity_end": r.equity_end,
            "trades_placed": r.trades_placed,
            "lots_per_trade": r.lots_per_trade,
            "daily_pnl": r.daily_pnl,
            "monthly_stop_active": r.monthly_stop_active,
            "event_signals": r.event_signals,
        })

    return {"days": days, "items": items}


@router.get("/trades")
async def get_portfolio_trades(
    limit: int = Query(100, ge=1, le=1000),
    source: str | None = Query(None),
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """Return portfolio trades with enrichment from the trades table.

    Parameters
    ----------
    limit : Max rows to return.
    source : Filter by trade_source ('scheduled' or 'event').
    """
    source_clause = ""
    params: dict[str, Any] = {"limit": limit}
    if source and source in ("scheduled", "event"):
        source_clause = "AND pt.trade_source = :source"
        params["source"] = source

    rows = await session.execute(text(f"""
        SELECT pt.id, pt.trade_id, pt.trade_source, pt.event_signal,
               pt.lots, pt.margin_committed, pt.realized_pnl,
               pt.equity_before, pt.equity_after, pt.created_at,
               ps.date,
               t.strategy_type, t.entry_credit, t.status, t.target_dte, t.expiration
        FROM portfolio_trades pt
        JOIN portfolio_state ps ON pt.portfolio_state_id = ps.id
        LEFT JOIN trades t ON pt.trade_id = t.trade_id
        WHERE 1=1 {source_clause}
        ORDER BY pt.created_at DESC
        LIMIT :limit
    """), params)

    items = []
    for r in rows.fetchall():
        items.append({
            "id": r.id,
            "trade_id": r.trade_id,
            "date": str(r.date),
            "trade_source": r.trade_source,
            "event_signal": r.event_signal,
            "lots": r.lots,
            "margin_committed": r.margin_committed,
            "realized_pnl": r.realized_pnl,
            "equity_before": r.equity_before,
            "equity_after": r.equity_after,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "strategy_type": r.strategy_type,
            "entry_credit": float(r.entry_credit) if r.entry_credit is not None else None,
            "trade_status": r.status,
            "target_dte": r.target_dte,
            "expiration": str(r.expiration) if r.expiration else None,
        })

    return {"items": items, "limit": limit, "source_filter": source}


@router.get("/config")
async def get_portfolio_config(
    _user=Depends(get_current_user),
) -> dict[str, Any]:
    """Return current portfolio and event configuration (read-only, no secrets)."""
    return {
        "portfolio": {
            "starting_capital": settings.portfolio_starting_capital,
            "max_trades_per_day": settings.portfolio_max_trades_per_day,
            "max_trades_per_run": settings.portfolio_max_trades_per_run,
            "monthly_drawdown_limit": settings.portfolio_monthly_drawdown_limit,
            "lot_per_equity": settings.portfolio_lot_per_equity,
            "max_equity_risk_pct": settings.portfolio_max_equity_risk_pct,
            "max_margin_pct": settings.portfolio_max_margin_pct,
            "calls_only": settings.portfolio_calls_only,
        },
        "event": {
            "enabled": settings.event_enabled,
            "budget_mode": settings.event_budget_mode,
            "max_trades": settings.event_max_trades,
            "spx_drop_threshold": settings.event_spx_drop_threshold,
            "spx_drop_2d_threshold": settings.event_spx_drop_2d_threshold,
            "spx_drop_min": settings.event_spx_drop_min,
            "spx_drop_max": settings.event_spx_drop_max,
            "vix_spike_threshold": settings.event_vix_spike_threshold,
            "vix_elevated_threshold": settings.event_vix_elevated_threshold,
            "term_inversion_threshold": settings.event_term_inversion_threshold,
            "side_preference": settings.event_side_preference,
            "min_dte": settings.event_min_dte,
            "max_dte": settings.event_max_dte,
            "min_delta": settings.event_min_delta,
            "max_delta": settings.event_max_delta,
            "rally_avoidance": settings.event_rally_avoidance,
            "rally_threshold": settings.event_rally_threshold,
        },
        "decision": {
            "entry_times": settings.decision_entry_times,
            "dte_targets": settings.decision_dte_targets,
            "delta_targets": settings.decision_delta_targets,
            "spread_width_points": settings.decision_spread_width_points,
        },
    }
