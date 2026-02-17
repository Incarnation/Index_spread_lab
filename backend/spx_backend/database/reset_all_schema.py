from __future__ import annotations

import asyncio
import sys

from spx_backend.database.schema import ALL_APP_TABLES, reset_all_tables, verify_table_counts


def main() -> None:
    """CLI entrypoint for destructive full schema reset. Verifies all tables are empty afterward."""
    async def _run() -> None:
        await reset_all_tables()
        counts = await verify_table_counts(ALL_APP_TABLES)
        print("Post-reset table row counts (expected 0 for all):")
        non_zero: list[str] = []
        for name in ALL_APP_TABLES:
            n = counts.get(name, 0)
            status = "OK" if n == 0 else "LEFTOVER"
            if n != 0:
                non_zero.append(name)
            print(f"  {name}: {n}  [{status}]")
        if non_zero:
            print("\nWARNING: tables still have data:", non_zero, file=sys.stderr)
            sys.exit(1)
        print("All tables empty.")

    asyncio.run(_run())


if __name__ == "__main__":
    main()

