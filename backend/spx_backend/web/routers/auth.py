"""
Auth router: login, registration, and JWT-based current-user dependency.

Public endpoints: POST /api/auth/login, POST /api/auth/register.
Protected: GET /api/auth/me (requires valid Bearer token).
get_current_user is reused by public and admin routers to protect all other /api/* routes.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.database import get_db_session

router = APIRouter()
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Minimum lengths for validation.
USERNAME_MIN = 3
USERNAME_MAX = 64
PASSWORD_MIN = 8


class UserOut(BaseModel):
    """User payload returned in login/register/me responses."""

    id: int
    username: str


class LoginBody(BaseModel):
    """Request body for login."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterBody(BaseModel):
    """Request body for registration."""

    username: str = Field(..., min_length=USERNAME_MIN, max_length=USERNAME_MAX)
    password: str = Field(..., min_length=PASSWORD_MIN)


# Bcrypt truncates at 72 bytes; pass through consistently for hash and verify.
BCRYPT_MAX_PASSWORD_BYTES = 72


def _truncate_password_for_bcrypt(password: str) -> str:
    """Return password truncated to 72 utf-8 bytes so bcrypt never raises ValueError."""
    encoded = password.encode("utf-8")
    if len(encoded) <= BCRYPT_MAX_PASSWORD_BYTES:
        return password
    return encoded[:BCRYPT_MAX_PASSWORD_BYTES].decode("utf-8", errors="ignore")


def _hash_password(password: str) -> str:
    """Return bcrypt hash of password."""
    return pwd_ctx.hash(_truncate_password_for_bcrypt(password))


def _verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain password matches hash."""
    return pwd_ctx.verify(_truncate_password_for_bcrypt(plain), hashed)


def _issue_token(user_id: int) -> str:
    """Encode JWT for user_id; raises if jwt_secret not configured."""
    if not (getattr(settings, "jwt_secret", None) and settings.jwt_secret):
        raise HTTPException(
            status_code=503,
            detail="Auth not configured (JWT_SECRET not set)",
        )
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    # PyJWT 2.x returns str; no need for decode if we pass str to frontend.


async def _get_user_by_id(db: AsyncSession, user_id: int) -> UserOut | None:
    """Load user by id from DB; return None if not found."""
    r = await db.execute(
        text("SELECT id, username FROM users WHERE id = :id"),
        {"id": user_id},
    )
    row = r.fetchone()
    if not row:
        return None
    return UserOut(id=row.id, username=row.username)


async def get_current_user(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncSession = Depends(get_db_session),
) -> UserOut:
    """
    Dependency: parse Authorization Bearer token, decode JWT, load user from DB.
    Raises 401 if missing or invalid.
    """
    if not authorization or not authorization.strip().lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.strip()[7:].strip()  # after "Bearer "
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    if not (getattr(settings, "jwt_secret", None) and settings.jwt_secret):
        raise HTTPException(status_code=503, detail="Auth not configured")
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        user_id = int(sub)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await _get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("/api/auth/login")
async def login(
    body: LoginBody,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Authenticate by username and password; return JWT and user info.
    Returns 401 on invalid credentials.
    """
    r = await db.execute(
        text("SELECT id, username, password_hash FROM users WHERE username = :username"),
        {"username": body.username.strip()},
    )
    row = r.fetchone()
    if not row or not _verify_password(body.password, row.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user_id = row.id
    username = row.username
    access_token = _issue_token(user_id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user_id, "username": username},
    }


@router.post("/api/auth/register")
async def register(
    body: RegisterBody,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Create a new user and return JWT and user info (same shape as login).
    Disabled when AUTH_REGISTER_ENABLED is false. Returns 409 if username exists.
    """
    if not settings.auth_register_enabled:
        raise HTTPException(status_code=403, detail="Registration is disabled")
    username = body.username.strip()
    if len(username) < USERNAME_MIN or len(username) > USERNAME_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Username must be {USERNAME_MIN}-{USERNAME_MAX} characters",
        )
    password_hash = _hash_password(body.password)
    try:
        await db.execute(
            text(
                "INSERT INTO users (username, password_hash) VALUES (:username, :password_hash)"
            ),
            {"username": username, "password_hash": password_hash},
        )
        await db.commit()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Username already taken")
    r = await db.execute(
        text("SELECT id, username FROM users WHERE username = :username"),
        {"username": username},
    )
    row = r.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="User created but not found")
    user_id, _ = row
    access_token = _issue_token(user_id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user_id, "username": username},
    }


@router.get("/api/auth/me", response_model=UserOut)
async def me(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    """Return the current authenticated user."""
    return current_user
