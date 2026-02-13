from __future__ import annotations

import asyncio

from spx_backend.db_init import reset_ml_tables


def main() -> None:
    """CLI entrypoint for destructive ML schema reset."""
    asyncio.run(reset_ml_tables())


if __name__ == "__main__":
    main()
