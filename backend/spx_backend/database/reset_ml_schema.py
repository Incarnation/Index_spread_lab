from __future__ import annotations

import asyncio
import sys

from spx_backend.database.schema import ML_RESET_TABLES, reset_ml_tables, verify_table_counts


def main() -> None:
    """CLI entrypoint for destructive ML schema reset. Verifies reset tables are empty afterward."""
    async def _run() -> None:
        await reset_ml_tables()
        counts = await verify_table_counts(ML_RESET_TABLES)
        print("Post-reset table row counts for ML/decision/trade tables (expected 0):")
        non_zero: list[str] = []
        for name in ML_RESET_TABLES:
            n = counts.get(name, 0)
            status = "OK" if n == 0 else "LEFTOVER"
            if n != 0:
                non_zero.append(name)
            print(f"  {name}: {n}  [{status}]")
        if non_zero:
            print("\nWARNING: reset tables still have data:", non_zero, file=sys.stderr)
            sys.exit(1)
        print("All reset tables empty.")
        all_counts = await verify_table_counts()
        print("\nAll app tables (ingestion may have data):")
        for name in all_counts:
            print(f"  {name}: {all_counts[name]}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()

