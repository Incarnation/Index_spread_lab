"""Export production DB tables to CSV/Parquet for offline training pipeline.

Connects to the production PostgreSQL database using DATABASE_URL from
the .env file and exports the following table families:

Market context (consumed by the training pipeline):
- ``context_snapshots``        -> ``context_snapshots_export.csv``
- ``chains``                   -> per-day Parquet under ``production_exports/chains/``
- ``underlying_parquet``       -> per-symbol Parquet under ``production_exports/underlying/``
- ``calendar_merge``           -> merges ``economic_events`` into ``economic_calendar.csv``

Trade forensics (re-entry into ML, audits, manual review):
- ``trade_decisions``          -> ``production_exports/trade_decisions.csv``
- ``trades`` + ``trade_legs``  -> per-table CSVs under ``production_exports/trades/``
- ``trade_marks``              -> per-day Parquet under ``production_exports/trade_marks/`` (chunked)
- ``orders`` + ``fills``       -> CSVs under ``production_exports/orders/``
                                  ``orders.request_json/response_json`` redacted (H7 / PII)

GEX archaeology:
- ``gex_snapshots``            -> per-day Parquet under ``production_exports/gex_snapshots/``
- ``gex_by_strike``            -> per-day Parquet under ``production_exports/gex_by_strike/`` (chunked)
- ``gex_by_expiry_strike``     -> per-day Parquet under ``production_exports/gex_by_expiry_strike/`` (chunked)

Portfolio path:
- ``portfolio_state``          -> ``production_exports/portfolio_state.csv``
- ``portfolio_trades``         -> ``production_exports/portfolio_trades.csv``

Optimizer history:
- ``optimizer_runs``           -> ``production_exports/optimizer_runs.csv``
- ``optimizer_results``        -> ``production_exports/optimizer_results.csv``
- ``optimizer_walkforward``    -> ``production_exports/optimizer_walkforward.csv``

The legacy ``trade_candidates`` exporter was removed in Track A.7 along
with the underlying table (migration 015).  ML re-entry will synthesize
training rows directly from ``trade_decisions`` + ``trades`` +
``trade_marks`` (now exported here per H7 in OFFLINE_PIPELINE_AUDIT.md).

PII / secrets policy (H7):
- ``users``, ``auth_audit_log``, and any ``user_id`` columns are
  intentionally NOT exported.
- ``orders.request_json/response_json`` is redacted via
  ``_redact_order_payload`` to drop broker account IDs, API tokens,
  and IP addresses while preserving the order parameters needed for
  forensics.

Atomicity (L9):
- All CSV/Parquet writes go through ``_atomic_write_csv`` /
  ``_atomic_write_parquet`` (temp + ``os.replace``) so a crash
  mid-write never leaves a torn file for downstream consumers.

Usage:
    python scripts/export_production_data.py [--start DATE] [--end DATE]
    python scripts/export_production_data.py --tables chains
    python scripts/export_production_data.py --tables underlying_parquet
    python scripts/export_production_data.py --tables trade_decisions
    python scripts/export_production_data.py --tables trade_marks --chunk-size 50000

Defaults to exporting all available data when --start/--end are omitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import load_project_env

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


_BACKEND = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND.parent / "data"

# Legacy CSV destinations.  These remain explicitly addressable via the
# CLI (``--tables underlying_quotes`` / ``economic_events``) for
# operators with existing tooling, but are no longer included in
# ``--tables all`` because no offline consumer reads them (L4).
UNDERLYING_QUOTES_CSV = DATA_DIR / "underlying_quotes_export.csv"
CONTEXT_SNAPSHOTS_CSV = DATA_DIR / "context_snapshots_export.csv"
ECONOMIC_EVENTS_CSV = DATA_DIR / "economic_events_export.csv"
PRODUCTION_CHAINS_DIR = DATA_DIR / "production_exports" / "chains"
PRODUCTION_UNDERLYING_DIR = DATA_DIR / "production_exports" / "underlying"
ECONOMIC_CALENDAR_CSV = DATA_DIR / "economic_calendar.csv"

# H7 destinations.  Forensics + future ML re-entry artifacts.  All
# under ``data/production_exports/`` so a `data/README.md` consumer
# sees a single subtree containing every "captured from prod" file.
_EXPORT_ROOT = DATA_DIR / "production_exports"
TRADE_DECISIONS_CSV = _EXPORT_ROOT / "trade_decisions.csv"
TRADES_CSV = _EXPORT_ROOT / "trades" / "trades.csv"
TRADE_LEGS_CSV = _EXPORT_ROOT / "trades" / "trade_legs.csv"
TRADE_MARKS_DIR = _EXPORT_ROOT / "trade_marks"  # per-day Parquet (chunked)
ORDERS_CSV = _EXPORT_ROOT / "orders" / "orders.csv"
FILLS_CSV = _EXPORT_ROOT / "orders" / "fills.csv"
GEX_SNAPSHOTS_DIR = _EXPORT_ROOT / "gex_snapshots"
GEX_BY_STRIKE_DIR = _EXPORT_ROOT / "gex_by_strike"
GEX_BY_EXPIRY_STRIKE_DIR = _EXPORT_ROOT / "gex_by_expiry_strike"
PORTFOLIO_STATE_CSV = _EXPORT_ROOT / "portfolio_state.csv"
PORTFOLIO_TRADES_CSV = _EXPORT_ROOT / "portfolio_trades.csv"
OPTIMIZER_RUNS_CSV = _EXPORT_ROOT / "optimizer_runs.csv"
OPTIMIZER_RESULTS_CSV = _EXPORT_ROOT / "optimizer_results.csv"
OPTIMIZER_WALKFORWARD_CSV = _EXPORT_ROOT / "optimizer_walkforward.csv"

# Default chunk size for streaming reads on heavy tables (trade_marks,
# gex_by_strike, gex_by_expiry_strike).  Tunable via ``--chunk-size``;
# 100k rows is a few hundred MB of pandas memory at typical column
# widths and keeps the postgres cursor responsive.
_DEFAULT_CHUNK_SIZE = 100_000

# Allowlist of safe keys to keep when redacting order payloads.  The
# tradier_client puts the raw order-submit body into ``request_json``
# and the broker response (with order ID, status, error messages) into
# ``response_json``.  Account IDs are sensitive PII; explicit
# allowlists are safer than blocklists because new fields default to
# "redacted".  See H7 in OFFLINE_PIPELINE_AUDIT.md.
_ORDER_REQUEST_KEEP_KEYS = frozenset({
    "class", "symbol", "duration", "type", "price", "side", "quantity",
    "option_symbol", "underlying", "expiration", "strike", "right",
    "ratio_quantity", "tag",
})
_ORDER_RESPONSE_KEEP_KEYS = frozenset({
    "id", "status", "partner_id", "errors", "fault_string",
    "type", "duration", "price", "avg_fill_price", "exec_quantity",
    "last_fill_price", "last_fill_quantity", "remaining_quantity",
    "transaction_date", "create_date", "filled_at", "side", "quantity",
})


def _redact_walk(value: Any, keep_keys: frozenset[str]) -> Any:
    """Recursively filter dicts/lists to ``keep_keys`` only.

    Internal worker for ``_redact_order_payload``.  Critically, string
    values are *not* JSON-parsed here — that only happens at the
    top-level call, so a leaf field like ``"side": "sell_to_open"`` is
    preserved verbatim instead of being passed to ``json.loads`` and
    discarded as invalid JSON.

    - dict   -> dict whose keys are filtered to ``keep_keys`` and whose
                values are recursively walked.
    - list   -> list of recursively-walked elements.
    - other  -> returned unchanged (numbers, strings, bool, None).
    """
    if isinstance(value, dict):
        return {
            k: _redact_walk(v, keep_keys)
            for k, v in value.items()
            if k in keep_keys
        }
    if isinstance(value, list):
        return [_redact_walk(item, keep_keys) for item in value]
    return value


def _redact_order_payload(value: Any, keep_keys: frozenset[str]) -> Any:
    """Filter an ``orders.{request,response}_json`` payload to ``keep_keys`` (H7).

    Used before exporting so the resulting CSV is safe to share /
    archive without leaking broker account IDs, API tokens, IP
    addresses, or anything else the live integration may write.

    The top-level value may arrive as either a parsed JSON object
    (``dict``/``list``) or a JSON string (some pandas drivers return
    JSONB columns as strings).  We try ``json.loads`` once at the top
    level and then delegate to ``_redact_walk`` for recursion.  If
    parsing fails we return ``None`` rather than risk leaking a raw
    HTTP error body or stack trace.

    Returns:
        - dict / list -> recursively filtered to ``keep_keys``
        - None        -> for None input or unparseable strings
        - other       -> primitives passed through unchanged

    An empty ``{}`` after redaction is preserved rather than dropped so
    the consumer sees that *some* payload was there but contained
    nothing on the allowlist (vs "no payload was ever written" = None).
    """
    if value is None:
        return None
    # Top-level: if the DB driver handed us a JSON string, parse once.
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            # Non-JSON top-level string is treated as an opaque blob and
            # dropped (could be an HTTP error body with PII).
            return None
    # If the top-level was a primitive (number, bool, parsed-from-str
    # int, etc.), just hand it back; primitives can't leak structured
    # PII.  Otherwise recurse with the safe walker.
    if not isinstance(value, (dict, list)):
        return value
    return _redact_walk(value, keep_keys)


def _sync_url(async_url: str) -> str:
    """Convert an asyncpg DATABASE_URL to a synchronous psycopg2 URL.

    The production .env uses ``postgresql+asyncpg://...``.  This helper
    replaces the driver portion so we can use a blocking engine in this
    standalone script.
    """
    return async_url.replace("postgresql+asyncpg://", "postgresql://")


def _atomic_write_csv(df: pd.DataFrame, dest: Path) -> None:
    """Write a DataFrame to ``dest`` atomically (L9 fix).

    Pandas ``to_csv`` writes incrementally; a crash mid-write leaves a
    truncated file that downstream consumers (the training pipeline,
    backtest_strategy.py, regime_analysis.py) silently treat as
    complete.  Writing to ``<dest>.tmp`` first and then ``os.replace``ing
    into place is atomic on POSIX/macOS/NTFS — readers see either the
    old version or the new one, never a half-written file.

    The caller is responsible for ensuring ``dest`` lives in a writable
    directory.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(str(tmp), index=False)
    os.replace(str(tmp), str(dest))


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write a DataFrame to ``dest`` (Parquet) atomically (L9 fix).

    Same intent as ``_atomic_write_csv`` but for Parquet outputs (per-day
    chains, per-symbol underlying).  ``pyarrow`` writes the file in one
    shot, but a SIGKILL/OOM during write can still leave a torn file —
    the temp+rename pattern guarantees readers never observe one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_parquet(str(tmp), index=False)
    os.replace(str(tmp), str(dest))


def _load_env() -> str:
    """Load .env and return the DATABASE_URL.

    Delegates to the shared ``_env.load_project_env()`` helper, then
    reads DATABASE_URL from the environment.

    Returns:
        The DATABASE_URL string.

    Raises:
        SystemExit: If DATABASE_URL is not set after loading .env.
    """
    load_project_env()

    url = os.getenv("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL not found in environment or .env")
        sys.exit(1)
    return url


def export_underlying_quotes(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = UNDERLYING_QUOTES_CSV,
) -> int:
    """Export underlying_quotes (SPX, SPY, VIX, VIX9D, VVIX, SKEW) to CSV.

    When a date filter is applied, new rows are merged with the existing
    CSV (if present) and deduplicated by (ts, symbol) so incremental
    exports don't destroy previously exported data.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output:
        Destination CSV path.

    Returns
    -------
    int
        Total number of rows in the final output.
    """
    clauses = ["symbol IN ('SPX', 'SPY', 'VIX', 'VIX9D', 'VVIX', 'SKEW')"]
    params: dict = {}
    if start:
        clauses.append("ts >= :start")
        params["start"] = start
    if end:
        clauses.append("ts < :end")
        params["end"] = end

    where = " AND ".join(clauses)
    query = text(
        f"SELECT ts, symbol, last, bid, ask, volume "
        f"FROM underlying_quotes WHERE {where} ORDER BY ts"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if output.exists() and (start or end):
        existing = pd.read_csv(str(output))
        df = pd.concat([existing, df], ignore_index=True)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
        df = df.drop_duplicates(subset=["ts", "symbol"], keep="last")
        df = df.sort_values("ts").reset_index(drop=True)

    _atomic_write_csv(df, output)
    return len(df)


def export_context_snapshots(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = CONTEXT_SNAPSHOTS_CSV,
) -> int:
    """Export context_snapshots to CSV, merging with any existing export.

    When a date filter is applied, new rows are merged with the existing
    CSV (if present) and deduplicated by timestamp so incremental exports
    don't destroy previously exported data.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output:
        Destination CSV path.

    Returns
    -------
    int
        Total number of rows in the final output.
    """
    clauses = ["1=1"]
    params: dict = {}
    if start:
        clauses.append("ts >= :start")
        params["start"] = start
    if end:
        clauses.append("ts < :end")
        params["end"] = end

    where = " AND ".join(clauses)
    query = text(
        f"SELECT ts, underlying, spx_price, spy_price, vix, vix9d, "
        f"term_structure, vvix, skew, gex_net, zero_gamma_level "
        f"FROM context_snapshots WHERE {where} ORDER BY ts"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if output.exists() and (start or end):
        existing = pd.read_csv(str(output))
        df = pd.concat([existing, df], ignore_index=True)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
        df = df.drop_duplicates(subset=["ts"], keep="last")
        df = df.sort_values("ts").reset_index(drop=True)

    _atomic_write_csv(df, output)
    return len(df)


def export_economic_events(
    engine,
    *,
    output: Path = ECONOMIC_EVENTS_CSV,
) -> int:
    """Export economic_events to CSV.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    output:
        Destination CSV path.

    Returns
    -------
    int
        Number of rows exported.
    """
    query = text(
        "SELECT date, event_type, has_projections, is_triple_witching "
        "FROM economic_events ORDER BY date"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    _atomic_write_csv(df, output)
    return len(df)


def export_chain_data(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = PRODUCTION_CHAINS_DIR,
) -> tuple[int, int]:
    """Export chain_snapshots + option_chain_rows as per-day Parquet files.

    Joins the two tables and writes one Parquet file per trading day to
    ``output_dir/{YYYYMMDD}.parquet``.  Each file contains the columns
    needed by the offline training pipeline's production adapters.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output_dir:
        Destination directory for per-day Parquet files.

    Returns
    -------
    tuple[int, int]
        (number of days exported, total rows across all days).
    """
    clauses = ["cs.underlying = 'SPX'"]
    params: dict = {}
    if start:
        clauses.append("cs.ts >= :start")
        params["start"] = start
    if end:
        clauses.append("cs.ts < :end")
        params["end"] = end

    where = " AND ".join(clauses)
    query = text(
        f"SELECT cs.ts, ocr.option_symbol, ocr.expiration, ocr.strike, "
        f"ocr.option_right, ocr.bid, ocr.ask, ocr.open_interest, "
        f"ocr.delta, ocr.gamma "
        f"FROM chain_snapshots cs "
        f"JOIN option_chain_rows ocr ON cs.snapshot_id = ocr.snapshot_id "
        f"WHERE {where} "
        f"ORDER BY cs.ts, ocr.option_symbol"
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return 0, 0

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["day"] = df["ts"].dt.strftime("%Y%m%d")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    days_exported = 0

    for day_str, day_df in df.groupby("day"):
        out_path = output_dir / f"{day_str}.parquet"
        _atomic_write_parquet(day_df.drop(columns=["day"]), out_path)
        total_rows += len(day_df)
        days_exported += 1

    return days_exported, total_rows


def export_underlying_parquet(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = PRODUCTION_UNDERLYING_DIR,
) -> dict[str, int]:
    """Export underlying_quotes as per-symbol Parquet files.

    Writes one Parquet file per symbol to ``output_dir/{SYMBOL}_1min.parquet``
    with columns ``[ts, close]``, matching the FRD parquet format so the
    training pipeline's ``load_frd_quotes`` can read them directly.

    When a date filter is applied, new rows are merged with the existing
    parquet (if present) and deduplicated by timestamp so incremental
    exports don't destroy previously exported data.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    start:
        ISO date string for lower bound (inclusive).  None = no lower bound.
    end:
        ISO date string for upper bound (exclusive).  None = no upper bound.
    output_dir:
        Destination directory for per-symbol Parquet files.

    Returns
    -------
    dict[str, int]
        Mapping of symbol name to total number of rows in each output file.
    """
    symbols = ("SPX", "SPY", "VIX", "VIX9D", "VVIX", "SKEW")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for symbol in symbols:
        clauses = [f"symbol = '{symbol}'"]
        params: dict = {}
        if start:
            clauses.append("ts >= :start")
            params["start"] = start
        if end:
            clauses.append("ts < :end")
            params["end"] = end

        where = " AND ".join(clauses)
        query = text(
            f"SELECT ts, last AS close FROM underlying_quotes "
            f"WHERE {where} ORDER BY ts"
        )

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            results[symbol] = 0
            continue

        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        out_path = output_dir / f"{symbol}_1min.parquet"
        if out_path.exists() and (start or end):
            existing = pd.read_parquet(out_path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["ts"], keep="last")
            df = df.sort_values("ts").reset_index(drop=True)

        _atomic_write_parquet(df, out_path)
        results[symbol] = len(df)

    return results


def export_economic_calendar_merged(
    engine,
    *,
    output: Path = ECONOMIC_CALENDAR_CSV,
) -> int:
    """Export economic_events and merge with existing economic_calendar.csv.

    Reads the production ``economic_events`` table and the existing
    ``economic_calendar.csv`` (historical events), deduplicates, and writes
    the merged result back.  This ensures the training pipeline always has
    a complete calendar covering both historical and production periods.

    Parameters
    ----------
    engine:
        SQLAlchemy synchronous engine.
    output:
        Destination CSV path (typically ``data/economic_calendar.csv``).

    Returns
    -------
    int
        Total number of rows in the merged output.
    """
    query = text(
        "SELECT date, event_type, has_projections, is_triple_witching "
        "FROM economic_events ORDER BY date"
    )
    with engine.connect() as conn:
        prod_df = pd.read_sql(query, conn)

    # Load existing historical calendar if present
    if output.exists():
        hist_df = pd.read_csv(str(output))
    else:
        hist_df = pd.DataFrame(columns=["date", "event_type", "has_projections", "is_triple_witching"])

    merged = pd.concat([hist_df, prod_df], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates(subset=["date", "event_type"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)

    _atomic_write_csv(merged, output)
    return len(merged)


# ---------------------------------------------------------------------------
# H7 exporters: trade forensics, GEX archaeology, portfolio + optimizer
# ---------------------------------------------------------------------------


def _date_filter_clause(
    column: str,
    start: str | None,
    end: str | None,
) -> tuple[str, dict[str, str]]:
    """Build a parameterised ``WHERE`` snippet for a TIMESTAMPTZ column.

    Returns ``(where_fragment, params)`` where the fragment is suitable
    for direct interpolation into a query and ``params`` is the dict
    that should be passed to ``conn.execute(...)``.
    """
    clauses: list[str] = ["1=1"]
    params: dict[str, str] = {}
    if start:
        clauses.append(f"{column} >= :start")
        params["start"] = start
    if end:
        clauses.append(f"{column} < :end")
        params["end"] = end
    return " AND ".join(clauses), params


def export_trade_decisions(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = TRADE_DECISIONS_CSV,
) -> int:
    """Export ``trade_decisions`` to CSV (H7).

    No ``user_id``-style PII columns exist on this table, but
    ``chosen_legs_json`` and ``strategy_params_json`` may include
    operator-supplied notes.  Both are kept verbatim because the
    downstream ML re-entry expects them.
    """
    where, params = _date_filter_clause("ts", start, end)
    query = text(
        "SELECT decision_id, ts, target_dte, entry_slot, delta_target, "
        "chosen_legs_json, strategy_params_json, ruleset_version, score, "
        "decision, reason, chain_snapshot_id, strategy_version_id, "
        "model_version_id, decision_source "
        f"FROM trade_decisions WHERE {where} ORDER BY ts"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    _atomic_write_csv(df, output)
    return len(df)


def export_trades(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = TRADES_CSV,
) -> int:
    """Export ``trades`` to CSV, scoped by entry_time window (H7)."""
    where, params = _date_filter_clause("entry_time", start, end)
    query = text(
        "SELECT trade_id, decision_id, entry_snapshot_id, last_snapshot_id, "
        "backtest_run_id, trade_source, strategy_version_id, "
        "model_version_id, strategy_type, status, underlying, entry_time, "
        "exit_time, last_mark_ts, target_dte, expiration, contracts, "
        "contract_multiplier, spread_width_points, entry_credit, max_profit, "
        "max_loss, take_profit_target, stop_loss_target, current_exit_cost, "
        "current_pnl, realized_pnl, exit_reason "
        f"FROM trades WHERE {where} ORDER BY entry_time"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    _atomic_write_csv(df, output)
    return len(df)


def export_trade_legs(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = TRADE_LEGS_CSV,
) -> int:
    """Export ``trade_legs`` joined to ``trades.entry_time`` for windowing.

    The legs table itself has no timestamp, so the date filter is
    applied via the parent ``trades`` row.  Without this join an
    incremental export would have to dump every leg every time.
    """
    where, params = _date_filter_clause("t.entry_time", start, end)
    query = text(
        "SELECT tl.trade_id, tl.leg_index, tl.option_symbol, tl.side, "
        "tl.qty, tl.entry_price, tl.exit_price, tl.strike, tl.expiration, "
        "tl.option_right "
        "FROM trade_legs tl "
        "JOIN trades t ON t.trade_id = tl.trade_id "
        f"WHERE {where} ORDER BY tl.trade_id, tl.leg_index"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    _atomic_write_csv(df, output)
    return len(df)


def export_trade_marks(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = TRADE_MARKS_DIR,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[int, int]:
    """Export ``trade_marks`` as per-day Parquet files using chunked reads.

    ``trade_marks`` has the highest row count among trade tables (one
    row per mark per trade per ~5 minutes); a naive ``read_sql`` of a
    full year can OOM the script.  The H7 fix streams the table in
    chunks via ``read_sql_query(..., chunksize=...)`` and groups by
    day before writing each Parquet file atomically.

    Day grouping happens inside each chunk; days that span multiple
    chunks get appended via the same atomic-write pattern (read
    existing + concat + write back temp + replace), which is safe but
    slower than the single-shot path.

    Returns:
        ``(days_written, total_rows)``.
    """
    where, params = _date_filter_clause("ts", start, end)
    query = text(
        "SELECT mark_id, trade_id, snapshot_id, ts, short_mid, long_mid, "
        "exit_cost, pnl, status, created_at "
        f"FROM trade_marks WHERE {where} ORDER BY ts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    days_written: set[str] = set()

    with engine.connect().execution_options(stream_results=True) as conn:
        chunks: Iterable[pd.DataFrame] = pd.read_sql_query(
            query, conn, params=params, chunksize=chunk_size,
        )
        for chunk in chunks:
            if chunk.empty:
                continue
            chunk["ts"] = pd.to_datetime(chunk["ts"], utc=True)
            chunk["day"] = chunk["ts"].dt.strftime("%Y%m%d")
            for day_str, day_df in chunk.groupby("day"):
                out_path = output_dir / f"{day_str}.parquet"
                payload = day_df.drop(columns=["day"])
                if out_path.exists() and day_str in days_written:
                    # This day was already touched in an earlier chunk;
                    # merge rather than overwrite.
                    existing = pd.read_parquet(out_path)
                    payload = pd.concat([existing, payload], ignore_index=True)
                _atomic_write_parquet(payload, out_path)
                days_written.add(day_str)
                total_rows += len(day_df)
    return len(days_written), total_rows


def export_orders(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = ORDERS_CSV,
) -> int:
    """Export ``orders`` to CSV with PII-safe JSONB redaction (H7).

    ``request_json`` (sent to the broker) and ``response_json``
    (returned by the broker) are filtered through
    ``_redact_order_payload`` so account IDs, API tokens, and any
    payload key not on the curated allowlist are dropped before the
    file leaves the production environment.
    """
    where, params = _date_filter_clause("submitted_at", start, end)
    query = text(
        "SELECT order_id, decision_id, status, submitted_at, updated_at, "
        "request_json, response_json "
        f"FROM orders WHERE {where} ORDER BY submitted_at"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if not df.empty:
        df["request_json"] = df["request_json"].apply(
            lambda v: json.dumps(_redact_order_payload(v, _ORDER_REQUEST_KEEP_KEYS)),
        )
        df["response_json"] = df["response_json"].apply(
            lambda v: json.dumps(_redact_order_payload(v, _ORDER_RESPONSE_KEEP_KEYS)),
        )
    _atomic_write_csv(df, output)
    return len(df)


def export_fills(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = FILLS_CSV,
) -> int:
    """Export ``fills`` to CSV (H7).

    No PII columns; fills only contain option_symbol/qty/price.
    """
    where, params = _date_filter_clause("ts", start, end)
    query = text(
        "SELECT fill_id, order_id, ts, option_symbol, qty, price "
        f"FROM fills WHERE {where} ORDER BY ts"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    _atomic_write_csv(df, output)
    return len(df)


def export_gex_snapshots(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = GEX_SNAPSHOTS_DIR,
) -> tuple[int, int]:
    """Export ``gex_snapshots`` as per-day Parquet files (H7).

    Lower row volume than the per-strike tables (one row per snapshot)
    so chunking isn't required, but per-day partitioning keeps file
    sizes manageable and matches the chains export layout.
    """
    where, params = _date_filter_clause("ts", start, end)
    query = text(
        "SELECT snapshot_id, ts, underlying, source, spot_price, gex_net, "
        "gex_calls, gex_puts, gex_abs, zero_gamma_level, method "
        f"FROM gex_snapshots WHERE {where} ORDER BY ts"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return 0, 0
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["day"] = df["ts"].dt.strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    days_written = 0
    for day_str, day_df in df.groupby("day"):
        out_path = output_dir / f"{day_str}.parquet"
        _atomic_write_parquet(day_df.drop(columns=["day"]), out_path)
        days_written += 1
    return days_written, len(df)


def _export_gex_per_strike_chunked(
    engine,
    *,
    table: str,
    columns: str,
    start: str | None,
    end: str | None,
    output_dir: Path,
    chunk_size: int,
) -> tuple[int, int]:
    """Shared chunked exporter for the two heavy per-strike GEX tables.

    Both ``gex_by_strike`` and ``gex_by_expiry_strike`` join to
    ``chain_snapshots`` for the timestamp; without that join we'd have
    to dump every row every run.  Streams via ``stream_results=True``
    to avoid OOM on the cursor side.
    """
    where, params = _date_filter_clause("cs.ts", start, end)
    query = text(
        f"SELECT cs.ts, {columns} "
        f"FROM {table} t "
        f"JOIN chain_snapshots cs ON cs.snapshot_id = t.snapshot_id "
        f"WHERE {where} ORDER BY cs.ts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    days_written: set[str] = set()

    with engine.connect().execution_options(stream_results=True) as conn:
        chunks: Iterable[pd.DataFrame] = pd.read_sql_query(
            query, conn, params=params, chunksize=chunk_size,
        )
        for chunk in chunks:
            if chunk.empty:
                continue
            chunk["ts"] = pd.to_datetime(chunk["ts"], utc=True)
            chunk["day"] = chunk["ts"].dt.strftime("%Y%m%d")
            for day_str, day_df in chunk.groupby("day"):
                out_path = output_dir / f"{day_str}.parquet"
                payload = day_df.drop(columns=["day"])
                if out_path.exists() and day_str in days_written:
                    existing = pd.read_parquet(out_path)
                    payload = pd.concat([existing, payload], ignore_index=True)
                _atomic_write_parquet(payload, out_path)
                days_written.add(day_str)
                total_rows += len(day_df)
    return len(days_written), total_rows


def export_gex_by_strike(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = GEX_BY_STRIKE_DIR,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[int, int]:
    """Export ``gex_by_strike`` as per-day Parquet (chunked)."""
    return _export_gex_per_strike_chunked(
        engine,
        table="gex_by_strike",
        columns="t.snapshot_id, t.strike, t.gex_net, t.gex_calls, "
                "t.gex_puts, t.oi_total, t.method",
        start=start, end=end, output_dir=output_dir, chunk_size=chunk_size,
    )


def export_gex_by_expiry_strike(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path = GEX_BY_EXPIRY_STRIKE_DIR,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> tuple[int, int]:
    """Export ``gex_by_expiry_strike`` as per-day Parquet (chunked)."""
    return _export_gex_per_strike_chunked(
        engine,
        table="gex_by_expiry_strike",
        columns="t.snapshot_id, t.expiration, t.dte_days, t.strike, "
                "t.gex_net, t.gex_calls, t.gex_puts, t.oi_total, t.method",
        start=start, end=end, output_dir=output_dir, chunk_size=chunk_size,
    )


def export_portfolio_state(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = PORTFOLIO_STATE_CSV,
) -> int:
    """Export ``portfolio_state`` to CSV (H7).

    Date column on this table is ``date`` (DATE not TIMESTAMPTZ); the
    helper handles both via parameter substitution.
    """
    clauses: list[str] = ["1=1"]
    params: dict[str, str] = {}
    if start:
        clauses.append("date >= :start")
        params["start"] = start[:10]
    if end:
        clauses.append("date < :end")
        params["end"] = end[:10]
    where = " AND ".join(clauses)
    query = text(
        "SELECT id, date, equity_start, equity_end, month_start_equity, "
        "trades_placed, lots_per_trade, daily_pnl, monthly_stop_active, "
        "event_signals, created_at "
        f"FROM portfolio_state WHERE {where} ORDER BY date"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    _atomic_write_csv(df, output)
    return len(df)


def export_portfolio_trades(
    engine,
    *,
    start: str | None = None,
    end: str | None = None,
    output: Path = PORTFOLIO_TRADES_CSV,
) -> int:
    """Export ``portfolio_trades`` to CSV, scoped by created_at (H7)."""
    where, params = _date_filter_clause("created_at", start, end)
    query = text(
        "SELECT id, trade_id, portfolio_state_id, trade_source, event_signal, "
        "lots, margin_committed, realized_pnl, equity_before, equity_after, "
        "created_at "
        f"FROM portfolio_trades WHERE {where} ORDER BY created_at"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    _atomic_write_csv(df, output)
    return len(df)


def export_optimizer_history(
    engine,
    *,
    output_runs: Path = OPTIMIZER_RUNS_CSV,
    output_results: Path = OPTIMIZER_RESULTS_CSV,
    output_walkforward: Path = OPTIMIZER_WALKFORWARD_CSV,
) -> dict[str, int]:
    """Export the three optimizer-history tables (H7).

    These tables are not large enough to need chunking (a few thousand
    rows per typical sweep) but exporting all three together makes it
    easy to reproduce dashboard analytics offline.

    No date filter — the typical use case is "give me everything"
    because operators want the full optimizer ledger.
    """
    queries = [
        ("optimizer_runs", "SELECT * FROM optimizer_runs ORDER BY started_at NULLS LAST", output_runs),
        ("optimizer_results", "SELECT * FROM optimizer_results ORDER BY id", output_results),
        ("optimizer_walkforward", "SELECT * FROM optimizer_walkforward ORDER BY id", output_walkforward),
    ]
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for name, sql, dest in queries:
            df = pd.read_sql(text(sql), conn)
            _atomic_write_csv(df, dest)
            counts[name] = len(df)
    return counts


def main() -> None:
    """CLI entry point for production data export."""
    parser = argparse.ArgumentParser(
        description="Export production DB tables to CSV/Parquet for offline training."
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date (inclusive, ISO format, e.g. 2026-01-01)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (exclusive, ISO format, e.g. 2026-04-01)",
    )
    parser.add_argument(
        "--tables", default="all",
        choices=[
            # ``all`` excludes ``underlying_quotes`` and
            # ``economic_events`` per L4 (no offline consumer reads
            # them); request them explicitly when needed.
            "all",
            "underlying_quotes", "context_snapshots",
            "economic_events",
            "chains", "underlying_parquet", "calendar_merge",
            # H7 additions
            "trade_decisions", "trades", "trade_legs", "trade_marks",
            "orders", "fills",
            "gex_snapshots", "gex_by_strike", "gex_by_expiry_strike",
            "portfolio_state", "portfolio_trades",
            "optimizer_history",
        ],
        help="Which table(s) to export (default: all).  ``all`` covers "
             "the full forensics + ML re-entry surface but skips the "
             "two unused legacy CSVs (underlying_quotes, "
             "economic_events) which now require explicit selection.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE,
        help=f"Rows per streaming chunk for heavy tables (trade_marks, "
             f"gex_by_strike, gex_by_expiry_strike).  Default: "
             f"{_DEFAULT_CHUNK_SIZE:,}.  Lower for memory-constrained "
             f"machines; higher for faster throughput on big boxes.",
    )
    parser.add_argument(
        "--check-gex", action="store_true",
        help="Only check if GEX data exists in context_snapshots (no export)",
    )
    args = parser.parse_args()

    db_url = _load_env()
    sync_url = _sync_url(db_url)
    engine = create_engine(sync_url)

    if args.check_gex:
        print("\n" + "=" * 60)
        print("GEX DATA CHECK")
        print("=" * 60)
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT COUNT(*) AS total, "
                "COUNT(gex_net) AS gex_rows, "
                "MIN(ts) FILTER (WHERE gex_net IS NOT NULL) AS gex_first, "
                "MAX(ts) FILTER (WHERE gex_net IS NOT NULL) AS gex_last "
                "FROM context_snapshots"
            ))
            row = result.fetchone()
            print(f"  Total context_snapshots rows: {row[0]}")
            print(f"  Rows with gex_net non-null:   {row[1]}")
            if row[1] and row[1] > 0:
                print(f"  GEX data range: {row[2]} to {row[3]}")
                print("\n  GEX data IS available in production DB.")
            else:
                print("\n  GEX data is NOT available (all gex_net values are NULL).")
        print("=" * 60)
        return

    date_msg = ""
    if args.start or args.end:
        date_msg = f" [{args.start or '...'} to {args.end or '...'}]"

    print(f"\nExporting production data{date_msg}")
    print("=" * 60)

    # Per L4: ``underlying_quotes`` and ``economic_events`` are not in
    # ``all`` because no offline consumer uses them.  An operator who
    # truly needs them can pass ``--tables underlying_quotes`` etc.
    if args.tables == "underlying_quotes":
        n = export_underlying_quotes(engine, start=args.start, end=args.end)
        print(f"  underlying_quotes: {n} rows -> {UNDERLYING_QUOTES_CSV}")

    if args.tables in ("all", "context_snapshots"):
        n = export_context_snapshots(engine, start=args.start, end=args.end)
        print(f"  context_snapshots: {n} rows -> {CONTEXT_SNAPSHOTS_CSV}")

    if args.tables == "economic_events":
        n = export_economic_events(engine)
        print(f"  economic_events: {n} rows -> {ECONOMIC_EVENTS_CSV}")

    if args.tables in ("all", "chains"):
        days, rows = export_chain_data(engine, start=args.start, end=args.end)
        print(f"  chains: {rows} rows across {days} days -> {PRODUCTION_CHAINS_DIR}/")

    if args.tables in ("all", "underlying_parquet"):
        sym_counts = export_underlying_parquet(engine, start=args.start, end=args.end)
        for sym, n in sym_counts.items():
            if n > 0:
                print(f"  underlying_parquet/{sym}: {n} rows -> {PRODUCTION_UNDERLYING_DIR}/{sym}_1min.parquet")

    if args.tables in ("all", "calendar_merge"):
        n = export_economic_calendar_merged(engine)
        print(f"  economic_calendar (merged): {n} rows -> {ECONOMIC_CALENDAR_CSV}")

    # H7 additions ------------------------------------------------
    if args.tables in ("all", "trade_decisions"):
        n = export_trade_decisions(engine, start=args.start, end=args.end)
        print(f"  trade_decisions: {n} rows -> {TRADE_DECISIONS_CSV}")

    if args.tables in ("all", "trades"):
        n = export_trades(engine, start=args.start, end=args.end)
        print(f"  trades: {n} rows -> {TRADES_CSV}")

    if args.tables in ("all", "trade_legs"):
        n = export_trade_legs(engine, start=args.start, end=args.end)
        print(f"  trade_legs: {n} rows -> {TRADE_LEGS_CSV}")

    if args.tables in ("all", "trade_marks"):
        days, rows = export_trade_marks(
            engine, start=args.start, end=args.end, chunk_size=args.chunk_size,
        )
        print(f"  trade_marks: {rows} rows across {days} days -> {TRADE_MARKS_DIR}/")

    if args.tables in ("all", "orders"):
        n = export_orders(engine, start=args.start, end=args.end)
        print(f"  orders: {n} rows -> {ORDERS_CSV} (PII-redacted)")

    if args.tables in ("all", "fills"):
        n = export_fills(engine, start=args.start, end=args.end)
        print(f"  fills: {n} rows -> {FILLS_CSV}")

    if args.tables in ("all", "gex_snapshots"):
        days, rows = export_gex_snapshots(engine, start=args.start, end=args.end)
        print(f"  gex_snapshots: {rows} rows across {days} days -> {GEX_SNAPSHOTS_DIR}/")

    if args.tables in ("all", "gex_by_strike"):
        days, rows = export_gex_by_strike(
            engine, start=args.start, end=args.end, chunk_size=args.chunk_size,
        )
        print(f"  gex_by_strike: {rows} rows across {days} days -> {GEX_BY_STRIKE_DIR}/")

    if args.tables in ("all", "gex_by_expiry_strike"):
        days, rows = export_gex_by_expiry_strike(
            engine, start=args.start, end=args.end, chunk_size=args.chunk_size,
        )
        print(f"  gex_by_expiry_strike: {rows} rows across {days} days -> {GEX_BY_EXPIRY_STRIKE_DIR}/")

    if args.tables in ("all", "portfolio_state"):
        n = export_portfolio_state(engine, start=args.start, end=args.end)
        print(f"  portfolio_state: {n} rows -> {PORTFOLIO_STATE_CSV}")

    if args.tables in ("all", "portfolio_trades"):
        n = export_portfolio_trades(engine, start=args.start, end=args.end)
        print(f"  portfolio_trades: {n} rows -> {PORTFOLIO_TRADES_CSV}")

    if args.tables in ("all", "optimizer_history"):
        counts = export_optimizer_history(engine)
        for tbl, n in counts.items():
            print(f"  {tbl}: {n} rows")

    print("=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
