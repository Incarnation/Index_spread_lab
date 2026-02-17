"""
Auth router: login, registration, logout, and JWT-based current-user dependency.

Public endpoints: POST /api/auth/login, POST /api/auth/register.
Protected: GET /api/auth/me, POST /api/auth/logout (require valid Bearer token).
get_current_user is reused by public and admin routers to protect all other /api/* routes.
Auth events (login success/failure, logout, session_expiry) are written to auth_audit_log.
"""

from __future__ import annotations

import ipaddress
import json
from datetime import datetime, timezone, timedelta

import httpx
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from spx_backend.config import settings
from spx_backend.database import get_db_session

# Free IP-to-country/region API (no key). 45 req/min limit; we only call on auth events.
# Free tier is HTTP only; HTTPS requires their pro service. No fields param = full JSON response.
GEOIP_API_URL = "http://ip-api.com/json/{ip}"
GEOIP_TIMEOUT_SEC = 2

# Map 2-letter codes to full country names when we get country from proxy header (e.g. CF-IPCountry).
COUNTRY_CODE_TO_NAME: dict[str, str] = {
    "US": "United States",
    "CA": "Canada",
    "GB": "United Kingdom",
    "AU": "Australia",
    "DE": "Germany",
    "FR": "France",
    "IN": "India",
    "JP": "Japan",
    "CN": "China",
    "BR": "Brazil",
    "MX": "Mexico",
    "ES": "Spain",
    "IT": "Italy",
    "NL": "Netherlands",
    "KR": "South Korea",
    "SG": "Singapore",
    "TW": "Taiwan",
    "HK": "Hong Kong",
}

router = APIRouter()
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Event types for auth_audit_log.
EVENT_LOGIN_SUCCESS = "login_success"
EVENT_LOGIN_FAILURE = "login_failure"
EVENT_LOGOUT = "logout"
EVENT_SESSION_EXPIRY = "session_expiry"

# Minimum lengths for validation.
USERNAME_MIN = 3
USERNAME_MAX = 64
PASSWORD_MIN = 8


class UserOut(BaseModel):
    """User payload returned in login/register/me responses."""

    id: int
    username: str
    is_admin: bool = False


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


def _request_metadata(request: Request) -> dict:
    """
    Extract IP, User-Agent, and country from request for audit logging.
    Country uses CF-IPCountry (Cloudflare) or X-Geo-Country if present; otherwise use geo lookup (see _request_metadata_with_geo).
    """
    # Prefer forwarded-for first proxy; client.host is direct client.
    forwarded = request.headers.get("x-forwarded-for")
    ip_raw = (forwarded.split(",")[0].strip()) if forwarded else request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    country = request.headers.get("cf-ipcountry") or request.headers.get("x-geo-country")
    return {"ip": ip_raw, "user_agent": user_agent, "country": country}


def _normalize_ip(ip: str | None) -> str | None:
    """Return IP string without CIDR suffix (e.g. 73.83.187.53/32 -> 73.83.187.53), or None if invalid."""
    if not ip or not ip.strip():
        return None
    s = ip.strip().split("/")[0].strip()
    return s if s else None


def _is_private_ip(ip_str: str) -> bool:
    """Return True if the IP is private or loopback (no point querying geo API)."""
    try:
        addr = ipaddress.ip_address(ip_str)
        return addr.is_private or addr.is_loopback
    except ValueError:
        return True


def _lookup_geo_by_ip_result(data: dict) -> tuple[str | None, str | None, dict | None]:
    """
    Extract (country name, region name, full response) from ip-api.com response.
    Full response is stored in auth_audit_log.geo_json for UI display.
    """
    if data.get("status") != "success":
        return (None, None, None)
    country = (data.get("country") or "").strip() or None
    region = (data.get("regionName") or "").strip() or None
    # Store full JSON for audit (exclude internal keys if any; ip-api returns query, status, country, etc.)
    return (country, region, dict(data))


async def _lookup_geo_by_ip(ip: str) -> tuple[str | None, str | None, dict | None]:
    """
    Resolve country, region, and full geo JSON for a public IP using ip-api.com (free, no key).
    Returns (country, region, full_geo_dict) or (None, None, None). Does not raise.
    """
    if _is_private_ip(ip):
        return (None, None, None)
    url = GEOIP_API_URL.format(ip=ip)
    try:
        async with httpx.AsyncClient(timeout=GEOIP_TIMEOUT_SEC) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return (None, None, None)
            data = r.json()
            return _lookup_geo_by_ip_result(data)
    except Exception:
        return (None, None, None)


async def _request_metadata_with_geo(request: Request) -> dict:
    """
    Like _request_metadata but fills country (full name) and region (state) from IP when
    CF-IPCountry / X-Geo-Country are missing. Uses free ip-api.com; best-effort.
    When country comes from header as 2-letter code, expand to full name via COUNTRY_CODE_TO_NAME.
    """
    meta = _request_metadata(request)
    header_country = meta.get("country")
    if header_country:
        # Expand 2-letter code to full name if we have it; otherwise keep as-is.
        code = (header_country or "").strip().upper()
        if len(code) == 2 and code in COUNTRY_CODE_TO_NAME:
            meta["country"] = COUNTRY_CODE_TO_NAME[code]
        return meta
    ip = _normalize_ip(meta.get("ip"))
    if ip:
        country, region, geo_details = await _lookup_geo_by_ip(ip)
        if country:
            meta["country"] = country
        if geo_details:
            meta["geo_details"] = geo_details
    return meta


async def _insert_audit_event(
    db: AsyncSession,
    event_type: str,
    *,
    user_id: int | None = None,
    username: str | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
    country: str | None = None,
    geo_json: dict | None = None,
    details: dict | None = None,
) -> None:
    """
    Insert one row into auth_audit_log. Does not commit; caller must commit.
    geo_json: full ip-api.com response (continent, country, city, lat, lon, isp, etc.).
    """
    details_json = json.dumps(details) if details else None
    geo_json_str = json.dumps(geo_json) if geo_json else None
    await db.execute(
        text("""
            INSERT INTO auth_audit_log (event_type, user_id, username, ip_address, user_agent, country, geo_json, details)
            VALUES (:event_type, :user_id, :username, CAST(:ip AS inet), :user_agent, :country, CAST(:geo_json AS jsonb), CAST(:details AS jsonb))
        """),
        {
            "event_type": event_type,
            "user_id": user_id,
            "username": username or None,
            "ip": ip or None,
            "user_agent": user_agent or None,
            "country": country or None,
            "geo_json": geo_json_str,
            "details": details_json,
        },
    )


async def _get_user_by_id(db: AsyncSession, user_id: int) -> UserOut | None:
    """Load user by id from DB; return None if not found."""
    r = await db.execute(
        text("SELECT id, username, COALESCE(is_admin, false) AS is_admin FROM users WHERE id = :id"),
        {"id": user_id},
    )
    row = r.fetchone()
    if not row:
        return None
    return UserOut(id=row.id, username=row.username, is_admin=getattr(row, "is_admin", False))


async def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncSession = Depends(get_db_session),
) -> UserOut:
    """
    Dependency: parse Authorization Bearer token, decode JWT, load user from DB.
    On invalid or expired token, records session_expiry in auth_audit_log then raises 401.
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
        # Record session_expiry for audit (user unknown from invalid/expired token).
        # Best-effort: if table missing or insert fails, still raise 401.
        try:
            meta = await _request_metadata_with_geo(request)
            await _insert_audit_event(
                db,
                EVENT_SESSION_EXPIRY,
                user_id=None,
                username=None,
                ip=meta.get("ip"),
                user_agent=meta.get("user_agent"),
                country=meta.get("country"),
                geo_json=meta.get("geo_details"),
            )
            await db.commit()
        except Exception:
            await db.rollback()
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


async def require_admin(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    """
    Dependency: requires current user to have is_admin True.
    Raises 403 if the user is not an admin. Use for admin-only endpoints (e.g. auth audit).
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return current_user


@router.post("/api/auth/login")
async def login(
    request: Request,
    body: LoginBody,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Authenticate by username and password; return JWT and user info.
    Records login_success or login_failure in auth_audit_log. Returns 401 on invalid credentials.
    """
    meta = await _request_metadata_with_geo(request)
    username_clean = body.username.strip()
    r = await db.execute(
        text("SELECT id, username, password_hash, COALESCE(is_admin, false) AS is_admin FROM users WHERE username = :username"),
        {"username": username_clean},
    )
    row = r.fetchone()
    if not row or not _verify_password(body.password, row.password_hash):
        await _insert_audit_event(
            db,
            EVENT_LOGIN_FAILURE,
            username=username_clean,
            ip=meta.get("ip"),
            user_agent=meta.get("user_agent"),
            country=meta.get("country"),
            geo_json=meta.get("geo_details"),
        )
        await db.commit()
        raise HTTPException(status_code=401, detail="Invalid username or password")
    user_id = row.id
    username = row.username
    is_admin = getattr(row, "is_admin", False)
    await _insert_audit_event(
        db,
        EVENT_LOGIN_SUCCESS,
        user_id=user_id,
        username=username,
        ip=meta.get("ip"),
        user_agent=meta.get("user_agent"),
        country=meta.get("country"),
        geo_json=meta.get("geo_details"),
    )
    await db.commit()
    access_token = _issue_token(user_id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user_id, "username": username, "is_admin": is_admin},
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
        text("SELECT id, username, COALESCE(is_admin, false) AS is_admin FROM users WHERE username = :username"),
        {"username": username},
    )
    row = r.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="User created but not found")
    user_id = row.id
    is_admin = getattr(row, "is_admin", False)
    access_token = _issue_token(user_id)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user_id, "username": username, "is_admin": is_admin},
    }


@router.get("/api/auth/me", response_model=UserOut)
async def me(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    """Return the current authenticated user."""
    return current_user


@router.post("/api/auth/logout")
async def logout(
    request: Request,
    current_user: UserOut = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """
    Record logout in auth_audit_log. Call this before clearing the token on the client.
    Returns 204 No Content.
    """
    meta = await _request_metadata_with_geo(request)
    await _insert_audit_event(
        db,
        EVENT_LOGOUT,
        user_id=current_user.id,
        username=current_user.username,
        ip=meta.get("ip"),
        user_agent=meta.get("user_agent"),
        country=meta.get("country"),
        geo_json=meta.get("geo_details"),
    )
    await db.commit()
    from fastapi.responses import Response
    return Response(status_code=204)
