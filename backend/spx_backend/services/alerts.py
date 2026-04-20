"""Shared SendGrid alert dispatcher with per-key cooldown.

Background
----------
This module is now the single SendGrid entry point for the backend.
All callers go through :func:`send_alert` for DB-backed cooldown
semantics; per-job alerts (``trade_pnl_job``, ``staleness_monitor_job``,
``scheduler_builder`` job-failure emails, etc.) pick a stable
``cooldown_key`` and let this module gate the dispatch.

Historical note: prior to Refactor #3 (audit Wave 4), each consumer
rolled its own cooldown dict (e.g. ``staleness_monitor_job._send_alert``
and ``scheduler_builder._send_job_failure_email``). Those local helpers
were removed once the DB-backed registry below proved out across all
call sites; new callers must NOT add a private cooldown dict.

H7 (audit) -- DB-backed cooldown
--------------------------------
Cooldowns previously lived only in a process-local dict
(``_last_alert_at``). With Railway's rolling deploys (and any restart
inside a noisy alert window) the cooldown silently reset every deploy,
re-paging the operator. Migration 022 added the ``alert_cooldowns``
table; ``send_alert`` now consults / persists ``last_alert_ts`` there
first, falling back to the in-process dict only when the DB is
unreachable so an outage of the metadata DB never silences alerts.

The two existing call sites stay as-is in this batch; opportunistic
migration into ``send_alert`` can happen in a later cleanup plan.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database.connection import engine


# Module-level cooldown registry. Used as a *fallback* when the
# DB-backed table is unavailable (H7) and as the legacy path for
# tests that don't spin up a database. Keyed by caller-provided
# ``cooldown_key`` so distinct alert types (and distinct trade_ids
# within an alert type) each get their own gating window. Callers
# choose the key shape (e.g. ``"split_brain:trade_id=42"``).
_last_alert_at: dict[str, datetime] = {}


async def _read_db_cooldown(cooldown_key: str) -> datetime | None:
    """Return last-alert timestamp from ``alert_cooldowns``, or None.

    Returns
    -------
    datetime | None
        The persisted ``last_alert_ts`` for ``cooldown_key``, or None
        when the row is missing OR the DB is unreachable. Callers must
        treat a None return as "no DB cooldown record" and fall back to
        the in-process dict so a metadata-DB outage cannot silence
        alerts (audit H7).
    """
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                text(
                    """
                    SELECT last_alert_ts
                    FROM alert_cooldowns
                    WHERE cooldown_key = :key
                    """
                ),
                {"key": cooldown_key},
            )
            result = row.fetchone()
            return result.last_alert_ts if result else None
    except Exception as exc:
        logger.warning(
            "alerts._read_db_cooldown: DB read failed key={} error={}; "
            "falling back to in-process dict",
            cooldown_key, exc,
        )
        return None


async def _write_db_cooldown(cooldown_key: str, now_utc: datetime) -> None:
    """Persist ``last_alert_ts`` for ``cooldown_key``. Best-effort.

    Failures are logged at warning level and swallowed; the in-process
    dict in ``_last_alert_at`` is always written first by ``send_alert``
    so a write failure here only loses cross-restart cooldown
    durability, not the within-process gate.
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO alert_cooldowns (cooldown_key, last_alert_ts)
                    VALUES (:key, :ts)
                    ON CONFLICT (cooldown_key) DO UPDATE SET
                        last_alert_ts = EXCLUDED.last_alert_ts
                    """
                ),
                {"key": cooldown_key, "ts": now_utc},
            )
    except Exception as exc:
        logger.warning(
            "alerts._write_db_cooldown: DB write failed key={} error={}; "
            "in-process cooldown still active",
            cooldown_key, exc,
        )


async def send_alert(
    *,
    subject: str,
    body_html: str,
    cooldown_key: str,
    cooldown_minutes: int = 60,
) -> bool:
    """Send a SendGrid HTML alert email, gated by per-key cooldown.

    Returns ``True`` when the email was sent successfully, ``False``
    when skipped (cooldown active, SendGrid not configured, or send
    failed).  Never raises -- alert-channel failures must not bring
    down the trading pipeline.

    Parameters
    ----------
    subject:
        Email subject line.  Should be short and unique enough to
        survive inbox grouping (e.g. include the alert type).
    body_html:
        HTML body content.  Callers are responsible for any escaping;
        this function does not sanitize input.
    cooldown_key:
        Key used to look up the last-sent timestamp.  Two calls with
        the same key within ``cooldown_minutes`` will collapse into
        one delivered email; the second call returns ``False`` after
        a debug log.  Pass distinct keys for distinct alertable units
        (e.g. ``f"split_brain:{trade_id}"``).
    cooldown_minutes:
        Minimum minutes between deliveries for a given ``cooldown_key``.
        Defaults to 60.

    Returns
    -------
    bool
        ``True`` only on confirmed-sent (HTTP 2xx).  ``False`` for any
        other outcome; the caller should treat ``False`` as "the alert
        was not delivered to the operator on this call" and rely on
        log lines for forensic detail.
    """
    # Fast-path skip: SendGrid creds missing.  Same convention as the
    # existing call sites; we log a warning the first time per process
    # but do not block the caller.
    if not settings.sendgrid_api_key or not settings.email_alert_recipient:
        logger.warning(
            "alerts.send_alert: SendGrid not configured; subject={}",
            subject,
        )
        return False

    # Per-key cooldown gate. UTC so server clock changes don't
    # accidentally re-open the gate by reading a different timezone.
    # H7 (audit): consult the DB-backed ``alert_cooldowns`` table first
    # so cooldowns survive deploys / pod restarts. If the DB is
    # unreachable we fall back to the in-process dict so an outage of
    # the metadata DB never silences an alert.
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    db_last = await _read_db_cooldown(cooldown_key)
    in_proc_last = _last_alert_at.get(cooldown_key)
    # Prefer whichever is more recent; either gate alone is sufficient.
    last = max(filter(None, (db_last, in_proc_last)), default=None)
    if last is not None:
        elapsed_minutes = (now_utc - last).total_seconds() / 60.0
        if elapsed_minutes < cooldown_minutes:
            logger.debug(
                "alerts.send_alert: cooldown active key={} elapsed={:.1f}min "
                "threshold={}min subject={} source={}",
                cooldown_key, elapsed_minutes, cooldown_minutes, subject,
                "db" if db_last == last else "in_proc",
            )
            return False

    # Lazy import: sendgrid is an optional runtime dep; failing to
    # import should not block the caller's main path.
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
    except ImportError:
        logger.error("alerts.send_alert: sendgrid package not installed")
        return False

    message = Mail(
        from_email=settings.email_alert_sender,
        to_emails=settings.email_alert_recipient,
        subject=subject,
        html_content=body_html,
    )

    try:
        sg = SendGridAPIClient(settings.sendgrid_api_key)
        response = sg.send(message)
    except Exception as exc:
        # Catch broadly: SendGridAPIClient can raise auth, network,
        # serialisation, and rate-limit errors.  Any of them must not
        # propagate into the trading pipeline.
        logger.error(
            "alerts.send_alert: SendGrid send failed key={} subject={} error={}",
            cooldown_key, subject, exc,
        )
        return False

    status = getattr(response, "status_code", None)
    if status in (200, 201, 202):
        # Always write the in-process dict (cheap, infallible) BEFORE the
        # DB write so a transient DB blip can't immediately cause a
        # within-process duplicate page. The DB write is best-effort.
        _last_alert_at[cooldown_key] = now_utc
        await _write_db_cooldown(cooldown_key, now_utc)
        logger.info(
            "alerts.send_alert: delivered key={} subject={} status={}",
            cooldown_key, subject, status,
        )
        return True

    logger.warning(
        "alerts.send_alert: unexpected status key={} subject={} status={}",
        cooldown_key, subject, status,
    )
    return False


def reset_cooldowns() -> None:
    """Clear the in-process cooldown registry. Test-only helper.

    Note: does NOT clear the DB-backed ``alert_cooldowns`` table; tests
    that need a clean DB cooldown state should truncate that table or
    use distinct ``cooldown_key`` values per test.
    """
    _last_alert_at.clear()
