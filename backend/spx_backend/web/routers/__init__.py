from __future__ import annotations

"""Router package exports for public, admin, and portfolio API route groups."""

from spx_backend.web.routers import admin, portfolio, public

__all__ = ["public", "admin", "portfolio"]

