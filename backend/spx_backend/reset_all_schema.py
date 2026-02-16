from __future__ import annotations

import asyncio

from spx_backend.db_init import reset_all_tables


def main() -> None:
    """CLI entrypoint for destructive full schema reset."""
    asyncio.run(reset_all_tables())


if __name__ == "__main__":
    main()
