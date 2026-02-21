#!/usr/bin/env python3
"""
Clear all rows from the auth_audit_log table (reset the audit log).

Usage (from repo root; backend uses .env for DATABASE_URL):
  cd backend && PYTHONPATH=. python scripts/reset_auth_audit_log.py
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.database.connection import SessionLocal


async def _run() -> None:
    async with SessionLocal() as session:  # type: AsyncSession
        result = await session.execute(text("SELECT COUNT(*) FROM auth_audit_log"))
        count = result.scalar_one() or 0
        await session.execute(text("TRUNCATE auth_audit_log RESTART IDENTITY"))
        await session.commit()
        print(f"Auth audit log reset: removed {count} row(s). Table is now empty.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
    sys.exit(0)
