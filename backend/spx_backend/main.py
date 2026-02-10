from __future__ import annotations

import os

from spx_backend.config import settings


def main() -> None:
    """Start Uvicorn, honoring Railway-provided PORT."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("spx_backend.web.app:app", host="0.0.0.0", port=port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()

