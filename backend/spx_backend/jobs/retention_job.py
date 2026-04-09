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

    logger.info(
        "retention_job: deleted {} chain_snapshots older than {} (cutoff={})",
        total_deleted, f"{settings.retention_days}d", cutoff.isoformat(),
    )
    return {
        "skipped": False,
        "deleted_snapshots": total_deleted,
        "cutoff": cutoff.isoformat(),
    }
