"""Shared SendGrid alert dispatcher with per-key cooldown.

Background
----------
Two existing modules already speak to SendGrid:

* ``spx_backend.jobs.staleness_monitor_job._send_alert`` -- single
  cooldown gate read from ``settings.staleness_cooldown_minutes``.
* ``spx_backend.scheduler_builder._send_job_failure_email`` -- per-job
  cooldown using a module-level dict keyed by ``job_id``.

This module unifies the cooldown-keyed pattern so new callers (the
split-brain handler in ``trade_pnl_job`` and any future tier-3 / data
integrity alerts) do not have to roll their own SendGrid plumbing or
forget to add cooldown logic.

The two existing call sites stay as-is in this batch; opportunistic
migration into ``send_alert`` can happen in a later cleanup plan.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger

from spx_backend.config import settings


# Module-level cooldown registry.  Keyed by caller-provided
# ``cooldown_key`` so distinct alert types (and distinct trade_ids
# within an alert type) each get their own gating window.  Callers
# choose the key shape (e.g. ``"split_brain:trade_id=42"``).
_last_alert_at: dict[str, datetime] = {}


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

    # Per-key cooldown gate.  We use UTC so server clock changes don't
    # accidentally re-open the gate by reading a different timezone.
    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    last = _last_alert_at.get(cooldown_key)
    if last is not None:
        elapsed_minutes = (now_utc - last).total_seconds() / 60.0
        if elapsed_minutes < cooldown_minutes:
            logger.debug(
                "alerts.send_alert: cooldown active key={} elapsed={:.1f}min "
                "threshold={}min subject={}",
                cooldown_key, elapsed_minutes, cooldown_minutes, subject,
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
        _last_alert_at[cooldown_key] = now_utc
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
    """Clear the cooldown registry.  Test-only helper."""
    _last_alert_at.clear()
