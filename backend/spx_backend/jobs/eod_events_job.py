"""EOD job that seeds/refreshes the ``economic_events`` table.

Runs daily after market close. On each invocation it upserts the full
calendar (FOMC, CPI, NFP, OPEX for 2021-2027). Per audit M7 the upsert
now uses ``ON CONFLICT DO UPDATE`` (was ``DO NOTHING``) so vendor
revisions of ``has_projections`` / ``is_triple_witching`` flow through
without manual SQL. ``updated_at`` is bumped on every conflict via
migration 021's column so operators can see when the row last drifted.

The authoritative event data lives in
``spx_backend.services.economic_calendar`` (the runtime job imports
``generate_rows`` directly from there). The CLI tool at
``backend/scripts/generate_economic_calendar.py`` is a thin wrapper
that calls the same service to produce a static CSV for offline
analysis -- it is NOT the source of truth, just a serializer.
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import text

from spx_backend.database.connection import engine
from spx_backend.services.economic_calendar import generate_rows as _generate_rows


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
            L8 (audit) fix: now actually honored. When ``True`` the
            DISTINCT-FROM WHERE clause is dropped so every row touched
            bumps ``updated_at`` -- useful for manual reseeds where the
            operator wants ``updated_at`` to reflect the reseed event
            even on no-op rows. The default ``False`` keeps the
            efficient skip-no-op behavior introduced in M7.
        """
        rows = _generate_rows()
        if not rows:
            logger.warning("eod_events: generate_rows returned empty list")
            return

        # M7 (audit): switched from ON CONFLICT DO NOTHING to DO UPDATE so
        # vendor flag flips (e.g., a previously non-triple-witching OPEX
        # later flagged ``is_triple_witching=true``) propagate to the row.
        # WHERE clause skips no-op writes so ``updated_at`` only bumps
        # when something actually changed; ``inserted_count`` /
        # ``updated_count`` are returned separately for observability.
        # L8 (audit): when ``force`` is set we omit the WHERE clause so
        # the DO UPDATE always fires -- semantically equivalent to a
        # full reseed but transactional and conflict-safe.
        if force:
            upsert_sql = text(
                """
                INSERT INTO economic_events
                    (date, event_type, has_projections, is_triple_witching, updated_at)
                VALUES
                    (:date, :event_type, :has_projections, :is_triple_witching, :updated_at)
                ON CONFLICT (date, event_type) DO UPDATE SET
                    has_projections    = EXCLUDED.has_projections,
                    is_triple_witching = EXCLUDED.is_triple_witching,
                    updated_at         = EXCLUDED.updated_at
                RETURNING (xmax = 0) AS was_inserted
                """
            )
        else:
            upsert_sql = text(
                """
                INSERT INTO economic_events
                    (date, event_type, has_projections, is_triple_witching, updated_at)
                VALUES
                    (:date, :event_type, :has_projections, :is_triple_witching, :updated_at)
                ON CONFLICT (date, event_type) DO UPDATE SET
                    has_projections    = EXCLUDED.has_projections,
                    is_triple_witching = EXCLUDED.is_triple_witching,
                    updated_at         = EXCLUDED.updated_at
                WHERE
                    economic_events.has_projections    IS DISTINCT FROM EXCLUDED.has_projections
                 OR economic_events.is_triple_witching IS DISTINCT FROM EXCLUDED.is_triple_witching
                RETURNING (xmax = 0) AS was_inserted
                """
            )

        now_utc = datetime.now(timezone.utc)
        inserted_count = 0
        updated_count = 0
        async with engine.begin() as conn:
            for row in rows:
                result = await conn.execute(
                    upsert_sql,
                    {
                        "date": row["date"],
                        "event_type": row["event_type"],
                        "has_projections": row["has_projections"],
                        "is_triple_witching": row["is_triple_witching"],
                        "updated_at": now_utc,
                    },
                )
                fetched = result.fetchone()
                if fetched is None:
                    # No row returned: either a true no-op (skip via WHERE) or
                    # an unchanged conflict that the WHERE filter elided.
                    continue
                if bool(fetched.was_inserted):
                    inserted_count += 1
                else:
                    updated_count += 1

        logger.info(
            "eod_events: inserted={inserted} updated={updated} total_calendar={total}",
            inserted=inserted_count,
            updated=updated_count,
            total=len(rows),
        )
