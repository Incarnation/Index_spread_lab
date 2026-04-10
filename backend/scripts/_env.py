"""Centralised .env loader for offline scripts.

Provides a single ``load_project_env()`` call that loads environment
variables from the project ``.env`` files, checking ``backend/.env``
first then the repository root ``.env``.  Both files are loaded when
they exist so that variables split across the two are merged (matching
the Pydantic ``Settings(env_file=(".env", "../.env"))`` behaviour).
"""

from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND.parent


def load_project_env() -> None:
    """Load environment variables from project ``.env`` files.

    Loads ``backend/.env`` first (higher priority), then
    ``<repo-root>/.env``.  Both are loaded when present so variables
    split across the two are merged.  If neither exists the call is a
    harmless no-op (the environment is expected to be pre-configured,
    e.g. in CI or Railway).

    Uses ``override=False`` so variables already present in the
    environment (or loaded from a higher-priority file) are not
    clobbered.
    """
    from dotenv import load_dotenv

    for candidate in (_BACKEND / ".env", _REPO_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
