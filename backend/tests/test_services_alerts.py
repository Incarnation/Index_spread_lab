"""Unit tests for ``spx_backend.services.alerts.send_alert``.

Coverage targets:

* Cooldown gate: a second call within ``cooldown_minutes`` collapses
  into the first delivery (no SendGrid call, returns ``False``).
* Per-key isolation: distinct ``cooldown_key`` values do not share a
  cooldown window (one trade tripping a split-brain alert must not
  silence a different trade's alert).
* Missing-creds short-circuit: empty ``sendgrid_api_key`` or empty
  ``email_alert_recipient`` returns ``False`` without attempting to
  import or invoke SendGrid.
* Success path (HTTP 202): updates the cooldown timestamp so subsequent
  calls within the window are gated.
* Failure path (SendGrid raises): returns ``False`` and does NOT update
  the cooldown timestamp, so the next call retries instead of being
  silenced for the next 60 minutes.

The cooldown registry is module-level state so each test calls
``alerts.reset_cooldowns()`` in setup to keep tests order-independent.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.config import settings
from spx_backend.services import alerts
from spx_backend.services.alerts import send_alert


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cooldowns():
    """Clear cooldown registry around every test to avoid bleed."""
    alerts.reset_cooldowns()
    yield
    alerts.reset_cooldowns()


@pytest.fixture
def configured_sendgrid(monkeypatch):
    """Set the three SendGrid-related settings so ``send_alert`` proceeds."""
    monkeypatch.setattr(settings, "sendgrid_api_key", "SG.test")
    monkeypatch.setattr(settings, "email_alert_recipient", "ops@example.com")
    monkeypatch.setattr(settings, "email_alert_sender", "alerts@example.com")


def _patched_sendgrid(*, status_code: int = 202, raise_exc: Exception | None = None):
    """Build a ``patch.dict`` context that injects fake sendgrid modules.

    Returns the ``patch.dict`` context plus a handle on the mock client
    instance so callers can assert on send-call counts and arguments.
    """
    mock_response = MagicMock(status_code=status_code)
    mock_client_instance = MagicMock()
    if raise_exc is not None:
        mock_client_instance.send.side_effect = raise_exc
    else:
        mock_client_instance.send.return_value = mock_response

    mock_client_class = MagicMock(return_value=mock_client_instance)

    fake_sg = MagicMock(SendGridAPIClient=mock_client_class)
    fake_helpers = MagicMock()
    fake_mail = MagicMock(Mail=MagicMock())

    ctx = patch.dict(
        "sys.modules",
        {
            "sendgrid": fake_sg,
            "sendgrid.helpers": fake_helpers,
            "sendgrid.helpers.mail": fake_mail,
        },
    )
    return ctx, mock_client_instance


# ---------------------------------------------------------------------
# Missing-creds short-circuit
# ---------------------------------------------------------------------


class TestMissingCredsShortCircuit:
    """When SendGrid creds are missing, return False without SDK access."""

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_false(self, monkeypatch):
        """Empty API key -> short-circuit with no SendGrid import attempt."""
        monkeypatch.setattr(settings, "sendgrid_api_key", "")
        monkeypatch.setattr(settings, "email_alert_recipient", "ops@example.com")
        monkeypatch.setattr(settings, "email_alert_sender", "alerts@example.com")

        ctx, client = _patched_sendgrid()
        with ctx:
            result = await send_alert(
                subject="x",
                body_html="<p>x</p>",
                cooldown_key="k",
            )

        assert result is False
        client.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_recipient_returns_false(self, monkeypatch):
        """Empty recipient -> short-circuit with no SendGrid import attempt."""
        monkeypatch.setattr(settings, "sendgrid_api_key", "SG.test")
        monkeypatch.setattr(settings, "email_alert_recipient", "")
        monkeypatch.setattr(settings, "email_alert_sender", "alerts@example.com")

        ctx, client = _patched_sendgrid()
        with ctx:
            result = await send_alert(
                subject="x",
                body_html="<p>x</p>",
                cooldown_key="k",
            )

        assert result is False
        client.send.assert_not_called()


# ---------------------------------------------------------------------
# Success path + cooldown gating
# ---------------------------------------------------------------------


class TestSuccessAndCooldown:
    """Confirmed deliveries (202) update the cooldown registry; subsequent
    same-key calls within the window are suppressed; distinct keys are
    independent."""

    @pytest.mark.asyncio
    async def test_success_updates_cooldown_timestamp(self, configured_sendgrid):
        """A 202 response stamps ``_last_alert_at`` for the cooldown_key."""
        ctx, client = _patched_sendgrid(status_code=202)
        with ctx:
            result = await send_alert(
                subject="hello",
                body_html="<p>body</p>",
                cooldown_key="alert:42",
            )

        assert result is True
        client.send.assert_called_once()
        assert "alert:42" in alerts._last_alert_at

    @pytest.mark.asyncio
    async def test_second_call_within_window_is_gated(self, configured_sendgrid):
        """Same cooldown_key within ``cooldown_minutes`` -> second call returns
        False and never invokes SendGrid."""
        ctx1, client1 = _patched_sendgrid(status_code=202)
        with ctx1:
            first = await send_alert(
                subject="hello",
                body_html="<p>body</p>",
                cooldown_key="alert:42",
                cooldown_minutes=60,
            )

        assert first is True

        ctx2, client2 = _patched_sendgrid(status_code=202)
        with ctx2:
            second = await send_alert(
                subject="hello again",
                body_html="<p>body</p>",
                cooldown_key="alert:42",
                cooldown_minutes=60,
            )

        assert second is False
        client2.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_distinct_keys_are_independent(self, configured_sendgrid):
        """Two different cooldown_keys both fire even back-to-back."""
        for key in ("alert:1", "alert:2"):
            ctx, client = _patched_sendgrid(status_code=202)
            with ctx:
                result = await send_alert(
                    subject=f"alert {key}",
                    body_html="<p>body</p>",
                    cooldown_key=key,
                )
            assert result is True
            client.send.assert_called_once()

        assert "alert:1" in alerts._last_alert_at
        assert "alert:2" in alerts._last_alert_at

    @pytest.mark.asyncio
    async def test_cooldown_expires_after_window(self, configured_sendgrid):
        """Once the cooldown window has passed, a fresh call delivers again."""
        ctx1, client1 = _patched_sendgrid(status_code=202)
        with ctx1:
            await send_alert(
                subject="first",
                body_html="<p>body</p>",
                cooldown_key="alert:42",
                cooldown_minutes=15,
            )

        # Roll the registry clock back so the next call sees the window
        # as having elapsed without us having to actually sleep.
        old_ts = alerts._last_alert_at["alert:42"]
        alerts._last_alert_at["alert:42"] = old_ts - timedelta(minutes=20)

        ctx2, client2 = _patched_sendgrid(status_code=202)
        with ctx2:
            second = await send_alert(
                subject="second",
                body_html="<p>body</p>",
                cooldown_key="alert:42",
                cooldown_minutes=15,
            )

        assert second is True
        client2.send.assert_called_once()


# ---------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------


class TestFailurePath:
    """When SendGrid raises or returns a non-2xx, return False and DO
    NOT update the cooldown timestamp -- otherwise a transient send
    failure would silence the alert for the entire next cooldown
    window."""

    @pytest.mark.asyncio
    async def test_send_raises_returns_false_and_no_cooldown_stamp(
        self, configured_sendgrid,
    ):
        """If SendGrid.send() raises, the registry remains unchanged so the
        very next call retries (rather than waiting 60 minutes)."""
        ctx, client = _patched_sendgrid(raise_exc=RuntimeError("boom"))
        with ctx:
            result = await send_alert(
                subject="x",
                body_html="<p>x</p>",
                cooldown_key="alert:99",
            )

        assert result is False
        client.send.assert_called_once()
        assert "alert:99" not in alerts._last_alert_at

    @pytest.mark.asyncio
    async def test_non_2xx_status_returns_false_no_cooldown_stamp(
        self, configured_sendgrid,
    ):
        """SendGrid status codes outside the 2xx happy band should also
        skip the cooldown update so the operator gets a retry."""
        ctx, client = _patched_sendgrid(status_code=500)
        with ctx:
            result = await send_alert(
                subject="x",
                body_html="<p>x</p>",
                cooldown_key="alert:500",
            )

        assert result is False
        client.send.assert_called_once()
        assert "alert:500" not in alerts._last_alert_at


# ---------------------------------------------------------------------
# Time semantics
# ---------------------------------------------------------------------


class TestCooldownTimeSemantics:
    """The cooldown registry stores UTC times so server timezone shifts
    can't accidentally re-open the gate.  This test pins the registry
    to a known UTC-aware datetime and checks the arithmetic."""

    @pytest.mark.asyncio
    async def test_registry_stores_timezone_aware_utc(self, configured_sendgrid):
        ctx, _ = _patched_sendgrid(status_code=202)
        with ctx:
            await send_alert(
                subject="x",
                body_html="<p>x</p>",
                cooldown_key="alert:tz",
            )
        ts = alerts._last_alert_at["alert:tz"]
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None
        # Exact tz object equality is brittle across zoneinfo versions, so
        # check the UTC offset instead.
        assert ts.utcoffset() == ZoneInfo("UTC").utcoffset(ts)
