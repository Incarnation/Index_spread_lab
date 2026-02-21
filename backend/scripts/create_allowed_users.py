#!/usr/bin/env python3
"""
Create allowed users directly in the DB (bcrypt-hashed passwords).

Usage (from repo root):
  cd backend && PYTHONPATH=. python scripts/create_allowed_users.py

Idempotent: skips insert if username already exists.
New users are created with is_admin = false; use set_admin.py to promote if needed.
"""

from __future__ import annotations

import asyncio
import sys

from passlib.context import CryptContext
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.database.connection import SessionLocal

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Users to create: (username, password)
USERS = [
    ("lainey", "lainey"),
    ("liz", "liz"),
]


def _hash_password(password: str) -> str:
    """Return bcrypt hash of password (same as auth router)."""
    return pwd_ctx.hash(password)


async def _run() -> None:
    async with SessionLocal() as session:  # type: AsyncSession
        for username, password in USERS:
            password_hash = _hash_password(password)
            exists = await session.execute(
                text("SELECT 1 FROM users WHERE username = :u"), {"u": username}
            )
            if exists.scalar():
                print(f"User {username!r} already exists, skipping.")
                continue
            await session.execute(
                text(
                    "INSERT INTO users (username, password_hash, is_admin) VALUES (:username, :password_hash, :is_admin)"
                ),
                {"username": username, "password_hash": password_hash, "is_admin": False},
            )
            print(f"Created user {username!r}.")
        await session.commit()
    print("Done.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
    sys.exit(0)
