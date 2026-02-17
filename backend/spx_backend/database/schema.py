from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from spx_backend.database.connection import engine

# All application tables (order matches db_schema / reset scripts).
ALL_APP_TABLES = [
    "users",
    "option_instruments",
    "chain_snapshots",
    "option_chain_rows",
    "gex_snapshots",
    "gex_by_strike",
    "gex_by_expiry_strike",
    "context_snapshots",
    "underlying_quotes",
    "market_clock_audit",
    "strategy_versions",
    "model_versions",
    "training_runs",
    "feature_snapshots",
    "trade_candidates",
    "model_predictions",
    "backtest_runs",
    "strategy_recommendations",
    "trade_decisions",
    "orders",
    "fills",
    "trades",
    "trade_legs",
    "trade_marks",
]

# Tables dropped by ML-only reset (db_reset_ml_tables.sql).
ML_RESET_TABLES = [
    "strategy_versions",
    "model_versions",
    "training_runs",
    "feature_snapshots",
    "trade_candidates",
    "model_predictions",
    "backtest_runs",
    "strategy_recommendations",
    "trade_decisions",
    "orders",
    "fills",
    "trades",
    "trade_legs",
    "trade_marks",
]


def _sql_dir() -> Path:
    """
    Return SQL directory.

    SQL files live under `spx_backend/database/sql/`.
    """
    return Path(__file__).resolve().parent / "sql"


async def _execute_sql_file(path: Path) -> None:
    """Execute SQL file statement-by-statement for asyncpg compatibility."""
    sql = path.read_text(encoding="utf-8")
    async with engine.begin() as conn:
        # Execute statement-by-statement (asyncpg won't reliably accept multi-statement executes).
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            await conn.exec_driver_sql(stmt)


async def init_db() -> None:
    """Initialize full schema (idempotent creates)."""
    schema_path = _sql_dir() / "db_schema.sql"
    await _execute_sql_file(schema_path)


async def reset_ml_tables() -> None:
    """Drop and recreate ML/decision/trade tables."""
    reset_path = _sql_dir() / "db_reset_ml_tables.sql"
    await _execute_sql_file(reset_path)
    await init_db()


async def reset_all_tables() -> None:
    """Drop and recreate all application tables."""
    reset_path = _sql_dir() / "db_reset_all_tables.sql"
    await _execute_sql_file(reset_path)
    await init_db()


async def verify_table_counts(
    table_names: list[str] | None = None,
) -> dict[str, int]:
    """
    Query row counts for the given tables (or all app tables if None).
    Returns a dict of table_name -> count. Used after reset to confirm no leftover data.
    """
    tables = table_names if table_names is not None else ALL_APP_TABLES
    counts: dict[str, int] = {}
    async with engine.connect() as conn:
        for name in tables:
            # Table names are from our constants only (no user input).
            stmt = text("SELECT count(*) FROM " + name)
            result = await conn.execute(stmt)
            counts[name] = result.scalar_one()
    return counts

