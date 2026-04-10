#!/usr/bin/env python3
"""Create an allowed user directly in the DB (bcrypt-hashed password).

Usage (from repo root)::

    cd backend && PYTHONPATH=. python scripts/create_allowed_users.py \\
        --username alice --password '<password>'

Idempotent: skips insert if the username already exists.
New users are created with ``is_admin = false``; use ``set_admin.py``
to promote if needed.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from passlib.context import CryptContext
from sqlalchemy import text

from spx_backend.database.connection import SessionLocal

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash_password(password: str) -> str:
    """Return a bcrypt hash of *password* (same scheme as the auth router).

    Args:
        password: Plain-text password to hash.

    Returns:
        Bcrypt hash string.
    """
    return pwd_ctx.hash(password)


async def _run(username: str, password: str) -> None:
    """Insert a single user if they do not already exist.

    Args:
        username: Desired username.
        password: Plain-text password (will be hashed before storage).
    """
    async with SessionLocal() as session:
        exists = await session.execute(
            text("SELECT 1 FROM users WHERE username = :u"), {"u": username}
        )
        if exists.scalar():
            logger.info("User %r already exists, skipping.", username)
            return
        password_hash = _hash_password(password)
        await session.execute(
            text(
                "INSERT INTO users (username, password_hash, is_admin) "
                "VALUES (:username, :password_hash, :is_admin)"
            ),
            {"username": username, "password_hash": password_hash, "is_admin": False},
        )
        await session.commit()
        logger.info("Created user %r.", username)


def main() -> None:
    """Parse CLI arguments and create the requested user."""
    parser = argparse.ArgumentParser(
        description="Create an allowed user in the database (bcrypt-hashed).",
    )
    parser.add_argument("--username", required=True, help="Username to create.")
    parser.add_argument("--password", required=True, help="Password for the new user.")
    args = parser.parse_args()

    asyncio.run(_run(args.username, args.password))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
