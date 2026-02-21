#!/usr/bin/env python3
"""
Set a user as admin (is_admin = true).

Usage (from repo root, backend uses .env for DATABASE_URL):
  cd backend && PYTHONPATH=. python scripts/set_admin.py <username>

Example:
  cd backend && PYTHONPATH=. python scripts/set_admin.py eric_huang
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.database.connection import SessionLocal


async def _run(username: str) -> None:
    async with SessionLocal() as session:  # type: AsyncSession
        result = await session.execute(
            text("UPDATE users SET is_admin = true WHERE username = :u RETURNING id"),
            {"u": username},
        )
        row = result.fetchone()
        await session.commit()
        if row:
            print(f"User {username!r} is now an admin (id={row[0]}).")
        else:
            print(f"No user found with username {username!r}. Nothing changed.")
            sys.exit(1)


def main() -> None:
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        print("Usage: python scripts/set_admin.py <username>", file=sys.stderr)
        sys.exit(2)
    username = sys.argv[1].strip()
    asyncio.run(_run(username))


if __name__ == "__main__":
    main()
