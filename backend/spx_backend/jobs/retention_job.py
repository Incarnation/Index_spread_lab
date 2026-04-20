"""Retention job -- purge old chain_snapshots and their cascaded children.

Deletes ``chain_snapshots`` rows older than the configured retention window
in batches to avoid long-running locks.  Cascading foreign keys automatically
remove the associated ``option_chain_rows``, ``gex_snapshots``,
``gex_by_strike``, and ``gex_by_expiry_strike`` rows.

Snapshots still referenced by OPEN trades are excluded as a safety measure
even though the FK constraints use ON DELETE SET NULL.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import engine


async def run_once(*, force: bool = False) -> dict:
    """Delete chain_snapshots older than ``retention_days``.

    Parameters
    ----------
    force : Bypass the ``retention_enabled`` guard.

    Returns
    -------
    dict with ``deleted_snapshots`` count and ``skipped`` flag.
    """
    if not (settings.retention_enabled or force):
        return {"skipped": True, "reason": "disabled"}

    cutoff = datetime.now(ZoneInfo("UTC")) - timedelta(days=settings.retention_days)
    batch_size = max(1, settings.retention_batch_size)
    total_deleted = 0

    # L5 (audit): observe per-table cascade row counts before/after the
    # purge so logs document the cascade rather than just the parent
    # delete count. The pre-counts are an estimate from pg_class.reltuples
    # (stat-only, no full table scan) so this stays cheap on the 9.6 GB
    # gex_by_expiry_strike table.
    cascade_tables = (
        "option_chain_rows",
        "gex_snapshots",
        "gex_by_strike",
        "gex_by_expiry_strike",
    )
    pre_counts: dict[str, int] = {}
    async with engine.begin() as conn:
        for table_name in cascade_tables:
            row = (
                await conn.execute(
                    text(
                        "SELECT reltuples::bigint AS approx FROM pg_class WHERE relname = :tname"
                    ),
                    {"tname": table_name},
                )
            ).fetchone()
            pre_counts[table_name] = int(row.approx) if row and row.approx is not None else 0

    while True:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    DELETE FROM chain_snapshots
                    WHERE snapshot_id IN (
                        SELECT cs.snapshot_id FROM chain_snapshots cs
                        WHERE cs.ts < :cutoff
                          AND cs.snapshot_id NOT IN (
                              SELECT entry_snapshot_id FROM trades
                              WHERE status = 'OPEN' AND entry_snapshot_id IS NOT NULL
                              UNION
                              SELECT last_snapshot_id FROM trades
                              WHERE status = 'OPEN' AND last_snapshot_id IS NOT NULL
                          )
                        ORDER BY cs.ts ASC
                        LIMIT :batch
                    )
                    """
                ),
                {"cutoff": cutoff, "batch": batch_size},
            )
            deleted = result.rowcount
        total_deleted += deleted
        if deleted < batch_size:
            break

    # L5 (audit): post-purge cascade snapshot. ANALYZE first (L4) so
    # pg_class.reltuples is fresh on the affected children.
    post_counts: dict[str, int] = {}
    async with engine.begin() as conn:
        for table_name in cascade_tables:
            # L4 (audit): explicit ANALYZE on the cascaded children.
            # Autovacuum will eventually catch up, but on the 15.4M-row
            # gex_by_expiry_strike table that "eventually" can be hours;
            # we want planner stats fresh for downstream readers right
            # after a large delete.
            await conn.execute(text(f"ANALYZE {table_name}"))
            row = (
                await conn.execute(
                    text(
                        "SELECT reltuples::bigint AS approx FROM pg_class WHERE relname = :tname"
                    ),
                    {"tname": table_name},
                )
            ).fetchone()
            post_counts[table_name] = int(row.approx) if row and row.approx is not None else 0

    cascade_deltas = {
        table_name: pre_counts[table_name] - post_counts[table_name]
        for table_name in cascade_tables
    }
    logger.info(
        "retention_job: deleted {} chain_snapshots older than {} (cutoff={}); "
        "cascade approx-deltas={}",
        total_deleted, f"{settings.retention_days}d", cutoff.isoformat(), cascade_deltas,
    )
    return {
        "skipped": False,
        "deleted_snapshots": total_deleted,
        "cutoff": cutoff.isoformat(),
        # L5 (audit): expose per-table cascade approximations for
        # observability dashboards / job-history overlays.
        "cascade_pre_counts": pre_counts,
        "cascade_post_counts": post_counts,
        "cascade_approx_deltas": cascade_deltas,
    }
