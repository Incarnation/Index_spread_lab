#!/usr/bin/env python3
"""Promote a user to admin (``is_admin = true``).

Usage (from repo root)::

    cd backend && PYTHONPATH=. python scripts/set_admin.py alice
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


async def _run(username: str) -> None:
    """Set ``is_admin = true`` for *username*.

    Args:
        username: The user to promote.

    Raises:
        SystemExit: If no matching user is found.
    """
    async with SessionLocal() as session:
        result = await session.execute(
            text("UPDATE users SET is_admin = true WHERE username = :u RETURNING id"),
            {"u": username},
        )
        row = result.fetchone()
        await session.commit()
        if row:
            logger.info("User %r is now an admin (id=%s).", username, row[0])
        else:
            logger.error("No user found with username %r. Nothing changed.", username)
            sys.exit(1)


def main() -> None:
    """Parse CLI arguments and promote the specified user."""
    parser = argparse.ArgumentParser(
        description="Promote a user to admin (is_admin = true).",
    )
    parser.add_argument("username", help="Username to promote to admin.")
    args = parser.parse_args()

    asyncio.run(_run(args.username))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
