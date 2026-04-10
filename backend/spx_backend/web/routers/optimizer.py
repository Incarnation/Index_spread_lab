"""API endpoints for the optimizer dashboard.

Serves optimizer run history, backtest results, Pareto frontier,
regime breakdowns, config comparisons, and walk-forward validation
from the ``optimizer_runs`` / ``optimizer_results`` / ``optimizer_walkforward``
PostgreSQL tables.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.database import get_db_session
from spx_backend.web.routers.auth import UserOut, get_current_user

router = APIRouter()


@router.get("/api/optimizer/runs")
async def list_optimizer_runs(
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """List all optimizer runs with summary stats, newest first."""
    result = await db.execute(text("""
        SELECT r.run_id, r.run_name, r.git_hash, r.config_file,
               r.optimizer_mode, r.started_at, r.finished_at,
               r.num_configs, r.status, r.metadata,
               COUNT(res.id) AS result_count,
               MAX(res.sharpe) AS best_sharpe,
               MAX(res.return_pct) AS best_return_pct,
               SUM(CASE WHEN res.is_pareto THEN 1 ELSE 0 END) AS pareto_count
        FROM optimizer_runs r
        LEFT JOIN optimizer_results res ON res.run_id = r.run_id
        GROUP BY r.id
        ORDER BY r.started_at DESC
        LIMIT :limit OFFSET :offset
    """), {"limit": limit, "offset": offset})

    runs = []
    for row in result.mappings():
        runs.append({
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "git_hash": row["git_hash"],
            "config_file": row["config_file"],
            "optimizer_mode": row["optimizer_mode"],
            "started_at": str(row["started_at"]) if row["started_at"] else None,
            "finished_at": str(row["finished_at"]) if row["finished_at"] else None,
            "num_configs": row["num_configs"],
            "status": row["status"],
            "result_count": row["result_count"],
            "best_sharpe": row["best_sharpe"],
            "best_return_pct": row["best_return_pct"],
            "pareto_count": row["pareto_count"],
        })

    count_result = await db.execute(text("SELECT COUNT(*) FROM optimizer_runs"))
    total = count_result.scalar()

    return {"runs": runs, "total": total}


@router.get("/api/optimizer/runs/{run_id}")
async def get_optimizer_run(
    run_id: str,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Get details for a single optimizer run."""
    result = await db.execute(text("""
        SELECT * FROM optimizer_runs WHERE run_id = :run_id
    """), {"run_id": run_id})
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {
        "run_id": row["run_id"],
        "run_name": row["run_name"],
        "git_hash": row["git_hash"],
        "config_file": row["config_file"],
        "optimizer_mode": row["optimizer_mode"],
        "started_at": str(row["started_at"]) if row["started_at"] else None,
        "finished_at": str(row["finished_at"]) if row["finished_at"] else None,
        "num_configs": row["num_configs"],
        "status": row["status"],
        "metadata": row["metadata"],
    }


@router.get("/api/optimizer/runs/{run_id}/results")
async def get_optimizer_results(
    run_id: str,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    sort_by: str = Query("sharpe", regex="^(sharpe|return_pct|max_dd_pct|win_rate|total_trades)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    min_sharpe: float | None = None,
    min_win_rate: float | None = None,
    min_trades: int | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Paginated optimizer results with filtering and sorting."""
    run_check = await db.execute(
        text("SELECT 1 FROM optimizer_runs WHERE run_id = :run_id"), {"run_id": run_id}
    )
    if not run_check.first():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    where_clauses = ["run_id = :run_id"]
    params: dict[str, Any] = {"run_id": run_id, "limit": limit, "offset": offset}

    if min_sharpe is not None:
        where_clauses.append("sharpe >= :min_sharpe")
        params["min_sharpe"] = min_sharpe
    if min_win_rate is not None:
        where_clauses.append("win_rate >= :min_win_rate")
        params["min_win_rate"] = min_win_rate
    if min_trades is not None:
        where_clauses.append("total_trades >= :min_trades")
        params["min_trades"] = min_trades

    where_sql = " AND ".join(where_clauses)
    order_dir = "DESC" if sort_order == "desc" else "ASC"

    result = await db.execute(text(f"""
        SELECT * FROM optimizer_results
        WHERE {where_sql}
        ORDER BY {sort_by} {order_dir} NULLS LAST
        LIMIT :limit OFFSET :offset
    """), params)

    rows = [dict(r) for r in result.mappings()]

    count_result = await db.execute(text(f"""
        SELECT COUNT(*) FROM optimizer_results WHERE {where_sql}
    """), params)
    total = count_result.scalar()

    return {"results": rows, "total": total}


@router.get("/api/optimizer/runs/{run_id}/pareto")
async def get_pareto_frontier(
    run_id: str,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return only Pareto-optimal configs for a run (Sharpe vs max DD)."""
    run_check = await db.execute(
        text("SELECT 1 FROM optimizer_runs WHERE run_id = :run_id"), {"run_id": run_id}
    )
    if not run_check.first():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    result = await db.execute(text("""
        SELECT * FROM optimizer_results
        WHERE run_id = :run_id AND is_pareto = TRUE
        ORDER BY sharpe DESC
    """), {"run_id": run_id})

    rows = [dict(r) for r in result.mappings()]
    return {"pareto": rows, "count": len(rows)}


@router.get("/api/optimizer/runs/{run_id}/equity-curve")
async def get_equity_curve(
    run_id: str,
    result_id: int = Query(..., description="ID of the specific result row"),
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return the equity curve for a specific config.

    Requires re-running the backtest for the requested config, or returning
    a stored curve if available.  For now returns the config details so the
    frontend can request a re-run if needed.
    """
    result = await db.execute(text("""
        SELECT * FROM optimizer_results
        WHERE id = :result_id AND run_id = :run_id
    """), {"result_id": result_id, "run_id": run_id})
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")
    return {"config": dict(row), "equity_curve": None, "note": "On-demand curve generation not yet implemented"}


@router.get("/api/optimizer/regime-breakdown")
async def get_regime_breakdown(
    result_id: int = Query(..., description="ID of the result to analyze"),
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return regime-based performance breakdown for a config.

    Buckets daily results by VIX level, SPX return, DTE, and day-of-week.
    Requires the daily-level data which is not stored in optimizer_results.
    Returns the config for frontend display.
    """
    result = await db.execute(text("""
        SELECT * FROM optimizer_results WHERE id = :result_id
    """), {"result_id": result_id})
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")
    return {
        "config": dict(row),
        "regime_breakdown": None,
        "note": "On-demand regime analysis not yet implemented",
    }


@router.get("/api/optimizer/compare")
async def compare_configs(
    ids: str = Query(..., description="Comma-separated result IDs to compare"),
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Side-by-side comparison of up to 10 optimizer configs."""
    id_list = [int(x.strip()) for x in ids.split(",") if x.strip().isdigit()]
    if len(id_list) < 2 or len(id_list) > 10:
        raise HTTPException(status_code=400, detail="Provide 2-10 comma-separated IDs")

    placeholders = ", ".join(f":id_{i}" for i in range(len(id_list)))
    params = {f"id_{i}": v for i, v in enumerate(id_list)}

    result = await db.execute(text(f"""
        SELECT * FROM optimizer_results WHERE id IN ({placeholders})
    """), params)

    rows = [dict(r) for r in result.mappings()]
    if not rows:
        raise HTTPException(status_code=404, detail="No matching results found")

    # Identify differing config columns
    config_cols = [c for c in rows[0].keys() if c.startswith(("p_", "t_", "e_", "r_"))]
    diffs = []
    for col in config_cols:
        values = [r.get(col) for r in rows]
        if len(set(str(v) for v in values)) > 1:
            diffs.append(col)

    return {
        "configs": rows,
        "differing_columns": diffs,
    }


@router.get("/api/optimizer/walkforward/{run_id}")
async def get_walkforward_results(
    run_id: str,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Walk-forward validation results for a run."""
    run_check = await db.execute(
        text("SELECT 1 FROM optimizer_runs WHERE run_id = :run_id"), {"run_id": run_id}
    )
    if not run_check.first():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    result = await db.execute(text("""
        SELECT * FROM optimizer_walkforward
        WHERE run_id = :run_id
        ORDER BY window_label
    """), {"run_id": run_id})

    rows = [dict(r) for r in result.mappings()]
    return {"walkforward": rows, "count": len(rows)}
