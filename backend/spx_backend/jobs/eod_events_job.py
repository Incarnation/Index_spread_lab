"""EOD job that seeds/refreshes the ``economic_events`` table.

Runs daily after market close.  On each invocation it upserts the full
calendar (FOMC, CPI, NFP, OPEX for 2021-2027) via INSERT ... ON CONFLICT
DO NOTHING, so repeated runs are cheap no-ops once the table is populated.

The authoritative event data lives in
``backend/scripts/generate_economic_calendar.py`` and is imported here
so there is a single source of truth.
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text

from spx_backend.database.connection import engine


def _generate_rows() -> list[dict]:
    """Build calendar rows using the same logic as generate_economic_calendar.py.

    Imported lazily to keep the top-level import lightweight. The
    ``generate_rows()`` function returns dicts with keys: date,
    event_type, has_projections, is_triple_witching.
    """
    import sys
    from pathlib import Path

    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from generate_economic_calendar import generate_rows
    return generate_rows()


class EodEventsJob:
    """Upsert economic calendar events into Postgres.

    Parameters
    ----------
    None -- this job is self-contained; it reads from the hardcoded
    calendar module and writes to the ``economic_events`` table.
    """

    async def run_once(self, *, force: bool = False) -> None:
        """Seed or refresh economic_events.

        Parameters
        ----------
        force:
            Accepted for interface consistency with other jobs but has
            no behavioral effect -- every run is idempotent.
        """
        rows = _generate_rows()
        if not rows:
            logger.warning("eod_events: generate_rows returned empty list")
            return

        upsert_sql = text("""
            INSERT INTO economic_events (date, event_type, has_projections, is_triple_witching)
            VALUES (:date, :event_type, :has_projections, :is_triple_witching)
            ON CONFLICT (date, event_type) DO NOTHING
        """)

        inserted = 0
        async with engine.begin() as conn:
            for row in rows:
                result = await conn.execute(upsert_sql, {
                    "date": row["date"],
                    "event_type": row["event_type"],
                    "has_projections": row["has_projections"],
                    "is_triple_witching": row["is_triple_witching"],
                })
                inserted += result.rowcount

        logger.info(
            "eod_events: upserted {inserted} new rows ({total} total in calendar)",
            inserted=inserted,
            total=len(rows),
        )
