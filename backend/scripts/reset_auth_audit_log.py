#!/usr/bin/env python3
"""Clear all rows from the ``auth_audit_log`` table.

Usage (from repo root)::

    cd backend && PYTHONPATH=. python scripts/reset_auth_audit_log.py --confirm
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from sqlalchemy import text

from spx_backend.database.connection import SessionLocal

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


async def _run() -> None:
    """Truncate the ``auth_audit_log`` table and report the row count."""
    async with SessionLocal() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM auth_audit_log"))
        count = result.scalar_one() or 0
        await session.execute(text("TRUNCATE auth_audit_log RESTART IDENTITY"))
        await session.commit()
        logger.info("Auth audit log reset: removed %d row(s). Table is now empty.", count)


def main() -> None:
    """Parse CLI arguments and reset the audit log if confirmed."""
    parser = argparse.ArgumentParser(
        description="Truncate the auth_audit_log table.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required safety flag to actually perform the truncation.",
    )
    args = parser.parse_args()

    if not args.confirm:
        logger.warning("Dry run -- pass --confirm to actually truncate.")
        sys.exit(0)

    asyncio.run(_run())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
