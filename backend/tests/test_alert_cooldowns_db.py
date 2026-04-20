"""Wave 5 (audit) H7 contract test: DB-backed alert cooldowns + in-proc fallback.

Pre-H7 behavior: cooldowns lived only in a process-local dict, so a
Railway rolling deploy (or any process restart inside a noisy alert
window) silently reset them and re-paged the operator.

Post-H7 behavior:

1. Before sending, ``send_alert`` queries ``alert_cooldowns`` (DB) AND
   reads ``_last_alert_at`` (in-process), and skips delivery if EITHER
   reports a recent alert within ``cooldown_minutes``.
2. After a successful send, ``send_alert`` writes BOTH the in-process
   dict AND the DB row (best-effort) so the cooldown survives a
   restart.
3. If the DB is unreachable, ``_read_db_cooldown`` returns ``None``
   instead of raising, so the in-process dict still gates duplicate
   alerts within a single process.

These tests exercise that contract end-to-end with mocks of the engine
and the SendGrid client; no real DB or network access is required.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import spx_backend.services.alerts as alerts_module


@pytest.fixture(autouse=True)
def _patch_settings_and_reset():
    """Pin SendGrid creds + reset the in-process cooldown dict."""
    with patch("spx_backend.services.alerts.settings") as s:
        s.sendgrid_api_key = "fake_sendgrid_key"
        s.email_alert_sender = "alerts@example.com"
        s.email_alert_recipient = "ops@example.com"
        alerts_module.reset_cooldowns()
        yield s
    alerts_module.reset_cooldowns()


def _patched_sendgrid(status_code: int):
    """Build a fake SendGridAPIClient + Mail pair returning the given status."""
    fake_response = SimpleNamespace(status_code=status_code)
    sendgrid_client = MagicMock()
    sendgrid_client.send = MagicMock(return_value=fake_response)

    sendgrid_client_class = MagicMock(return_value=sendgrid_client)
    fake_mail = MagicMock()

    return sendgrid_client_class, fake_mail


def _patch_engine_db_round_trip(*, last_ts_in_db: datetime | None):
    """Build a context manager that patches engine.connect and engine.begin.

    ``last_ts_in_db`` controls what the SELECT returns. None means "no
    DB row exists for this key".
    """

    select_row = (
        SimpleNamespace(last_alert_ts=last_ts_in_db)
        if last_ts_in_db is not None
        else None
    )

    select_result = MagicMock()
    select_result.fetchone = MagicMock(return_value=select_row)

    select_conn = AsyncMock()
    select_conn.execute = AsyncMock(return_value=select_result)

    select_ctx = AsyncMock()
    select_ctx.__aenter__ = AsyncMock(return_value=select_conn)
    select_ctx.__aexit__ = AsyncMock(return_value=False)

    write_conn = AsyncMock()
    write_conn.execute = AsyncMock()

    write_ctx = AsyncMock()
    write_ctx.__aenter__ = AsyncMock(return_value=write_conn)
    write_ctx.__aexit__ = AsyncMock(return_value=False)

    fake_engine = MagicMock()
    fake_engine.connect = MagicMock(return_value=select_ctx)
    fake_engine.begin = MagicMock(return_value=write_ctx)

    return fake_engine, select_conn, write_conn


# ---------------------------------------------------------------------------
# 1. Successful send writes both in-process AND DB cooldown rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_send_writes_in_proc_and_db_cooldown():
    """A 202 response must persist the cooldown to BOTH stores (H7)."""
    fake_engine, select_conn, write_conn = _patch_engine_db_round_trip(last_ts_in_db=None)

    sg_class, _ = _patched_sendgrid(status_code=202)

    with (
        patch("spx_backend.services.alerts.engine", fake_engine),
        patch("sendgrid.SendGridAPIClient", sg_class, create=True),
        patch("sendgrid.helpers.mail.Mail", create=True),
    ):
        delivered = await alerts_module.send_alert(
            subject="[test] hi",
            body_html="<p>hi</p>",
            cooldown_key="test:h7:fresh_send",
            cooldown_minutes=60,
        )

    assert delivered is True
    assert "test:h7:fresh_send" in alerts_module._last_alert_at
    write_conn.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# 2. Recent DB cooldown alone is enough to skip; SendGrid must not be called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_db_cooldown_blocks_send_even_with_empty_in_proc_dict():
    """A recent DB row must gate delivery even when the in-proc dict is empty.

    This is the *cross-deploy* path: the process restarted (so the
    in-proc dict is empty) but the DB still remembers the most recent
    page. Without the H7 fix this scenario would re-page on every
    restart.
    """
    recent = datetime.now(timezone.utc) - timedelta(minutes=10)
    fake_engine, _, write_conn = _patch_engine_db_round_trip(last_ts_in_db=recent)

    sg_class, _ = _patched_sendgrid(status_code=202)

    with (
        patch("spx_backend.services.alerts.engine", fake_engine),
        patch("sendgrid.SendGridAPIClient", sg_class, create=True) as sg_patch,
        patch("sendgrid.helpers.mail.Mail", create=True),
    ):
        delivered = await alerts_module.send_alert(
            subject="[test] hi",
            body_html="<p>hi</p>",
            cooldown_key="test:h7:db_only_block",
            cooldown_minutes=60,
        )

    assert delivered is False
    sg_patch.assert_not_called()  # SendGrid never invoked because we tripped the cooldown
    write_conn.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. In-process dict alone gates within the same process
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_proc_dict_blocks_send_when_db_returns_none():
    """A recent in-proc entry must gate delivery even if the DB has no row."""
    alerts_module._last_alert_at["test:h7:in_proc_only"] = (
        datetime.now(timezone.utc) - timedelta(minutes=10)
    )

    fake_engine, _, write_conn = _patch_engine_db_round_trip(last_ts_in_db=None)
    sg_class, _ = _patched_sendgrid(status_code=202)

    with (
        patch("spx_backend.services.alerts.engine", fake_engine),
        patch("sendgrid.SendGridAPIClient", sg_class, create=True) as sg_patch,
        patch("sendgrid.helpers.mail.Mail", create=True),
    ):
        delivered = await alerts_module.send_alert(
            subject="[test] hi",
            body_html="<p>hi</p>",
            cooldown_key="test:h7:in_proc_only",
            cooldown_minutes=60,
        )

    assert delivered is False
    sg_patch.assert_not_called()
    write_conn.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. DB unreachable -> graceful fallback to in-process dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_db_unreachable_falls_back_to_in_proc_dict():
    """When the DB SELECT raises, ``send_alert`` must NOT raise.

    It should fall back to the in-process dict and proceed to send
    when no in-proc cooldown exists. The DB write is also best-effort
    (writes are wrapped in try/except in production code).
    """
    write_conn = AsyncMock()
    write_conn.execute = AsyncMock(side_effect=Exception("DB write boom"))

    fake_engine = MagicMock()
    # Both connect() and begin() raise so neither read nor write succeeds.
    fake_engine.connect = MagicMock(
        side_effect=Exception("DB connect boom")
    )
    fake_engine.begin = MagicMock(
        side_effect=Exception("DB begin boom")
    )

    sg_class, _ = _patched_sendgrid(status_code=202)

    with (
        patch("spx_backend.services.alerts.engine", fake_engine),
        patch("sendgrid.SendGridAPIClient", sg_class, create=True),
        patch("sendgrid.helpers.mail.Mail", create=True),
    ):
        delivered = await alerts_module.send_alert(
            subject="[test] hi",
            body_html="<p>hi</p>",
            cooldown_key="test:h7:db_outage",
            cooldown_minutes=60,
        )

    # Send still succeeds because cooldown gate has no in-proc entry
    # and the DB read returned None on its first failure.
    assert delivered is True
    # In-proc dict was updated to gate the next call within the cooldown.
    assert "test:h7:db_outage" in alerts_module._last_alert_at


# ---------------------------------------------------------------------------
# 5. SendGrid creds missing short-circuits before any cooldown / send work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_sendgrid_credentials_returns_false_quickly():
    """No SendGrid key means ``send_alert`` returns False without DB I/O."""
    with patch("spx_backend.services.alerts.settings") as s:
        s.sendgrid_api_key = ""  # missing
        s.email_alert_sender = "alerts@example.com"
        s.email_alert_recipient = "ops@example.com"

        delivered = await alerts_module.send_alert(
            subject="[test] hi",
            body_html="<p>hi</p>",
            cooldown_key="test:h7:no_creds",
            cooldown_minutes=60,
        )
        assert delivered is False
