"""Unit tests for SmsNotifier — source filtering, message formatting, error handling."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from spx_backend.services.sms_notifier import SmsNotifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_notifier(monkeypatch, *, enabled: bool = True, sources: str = "all",
                   to_numbers: str = "+15551234567") -> SmsNotifier:
    """Build an SmsNotifier with a mocked Twilio client.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture.
    enabled:
        Value for ``sms_enabled``.
    sources:
        Value for ``sms_notify_sources``.
    to_numbers:
        Comma-separated recipient numbers.

    Returns
    -------
    SmsNotifier
        An instance whose ``_client`` is a ``MagicMock``.
    """
    from spx_backend.config import settings
    monkeypatch.setattr(settings, "sms_enabled", enabled)
    monkeypatch.setattr(settings, "sms_notify_sources", sources)
    monkeypatch.setattr(settings, "twilio_to_numbers", to_numbers)
    monkeypatch.setattr(settings, "twilio_from_number", "+15559999999")
    monkeypatch.setattr(settings, "twilio_account_sid", "ACtest")
    monkeypatch.setattr(settings, "twilio_auth_token", "testtoken")

    with patch("spx_backend.services.sms_notifier.SmsNotifier.__init__", return_value=None):
        notifier = SmsNotifier.__new__(SmsNotifier)
    notifier._client = MagicMock() if enabled else None
    return notifier


def _sample_trade_opened() -> dict[str, Any]:
    """Return a representative trade-opened info dict."""
    return {
        "spread_side": "put",
        "target_dte": 3,
        "expiration": "2026-04-11",
        "short": {
            "symbol": "SPXW 260411P05700",
            "strike": 5700,
            "entry_price": 12.50,
            "delta": -0.18,
        },
        "long": {
            "symbol": "SPXW 260411P05675",
            "strike": 5675,
            "entry_price": 8.30,
        },
        "credit": 4.20,
        "width_points": 25,
        "contracts": 2,
        "max_loss": 4160,
        "source": "portfolio_event",
        "event_signals": ["spx_drop_1d"],
    }


def _sample_trade_closed() -> dict[str, Any]:
    """Return a representative trade-closed info dict."""
    return {
        "trade_id": 1234,
        "strategy_type": "credit_vertical_put",
        "exit_reason": "TAKE_PROFIT_50",
        "entry_credit": 4.20,
        "exit_cost": 0.85,
        "realized_pnl": 670.00,
        "contracts": 2,
        "source": "portfolio_scheduled",
    }


# ---------------------------------------------------------------------------
# _should_notify tests
# ---------------------------------------------------------------------------

class TestShouldNotify:
    """Verify source-filter logic for _should_notify."""

    def test_disabled_returns_false(self, monkeypatch):
        """sms_enabled=False short-circuits regardless of source."""
        notifier = _make_notifier(monkeypatch, enabled=False)
        assert notifier._should_notify("event") is False

    def test_all_passes_any_source(self, monkeypatch):
        """sms_notify_sources='all' matches every source string."""
        notifier = _make_notifier(monkeypatch, sources="all")
        assert notifier._should_notify("portfolio_event") is True
        assert notifier._should_notify("portfolio_scheduled") is True
        assert notifier._should_notify("rules") is True

    def test_event_filter(self, monkeypatch):
        """sms_notify_sources='event' matches only sources containing 'event'."""
        notifier = _make_notifier(monkeypatch, sources="event")
        assert notifier._should_notify("portfolio_event") is True
        assert notifier._should_notify("portfolio_scheduled") is False
        assert notifier._should_notify("rules") is False

    def test_scheduled_filter(self, monkeypatch):
        """sms_notify_sources='scheduled' matches only non-event sources."""
        notifier = _make_notifier(monkeypatch, sources="scheduled")
        assert notifier._should_notify("portfolio_scheduled") is True
        assert notifier._should_notify("rules") is True
        assert notifier._should_notify("portfolio_event") is False

    def test_client_none_with_enabled_returns_false(self, monkeypatch):
        """sms_enabled=True but _client=None (init failure) returns False."""
        notifier = _make_notifier(monkeypatch, enabled=True)
        notifier._client = None
        assert notifier._should_notify("event") is False

    def test_unknown_sources_value_returns_false(self, monkeypatch):
        """Unrecognised sms_notify_sources value returns False (safe default)."""
        notifier = _make_notifier(monkeypatch, sources="bogus")
        assert notifier._should_notify("event") is False
        assert notifier._should_notify("scheduled") is False

    def test_empty_source_with_scheduled_filter_returns_false(self, monkeypatch):
        """Empty source string does not match the 'scheduled' filter."""
        notifier = _make_notifier(monkeypatch, sources="scheduled")
        assert notifier._should_notify("") is False


# ---------------------------------------------------------------------------
# Message formatting tests
# ---------------------------------------------------------------------------

class TestFormatTradeOpened:
    """Verify trade-opened SMS body formatting."""

    def test_contains_spread_side_and_dte(self):
        """Message header includes spread side and DTE."""
        info = _sample_trade_opened()
        body = SmsNotifier._format_trade_opened(info)
        assert "PUT Credit Spread" in body
        assert "3 DTE" in body

    def test_contains_leg_details(self):
        """Message includes short and long leg symbols and strikes."""
        info = _sample_trade_opened()
        body = SmsNotifier._format_trade_opened(info)
        assert "SPXW 260411P05700" in body
        assert "5700" in body
        assert "SPXW 260411P05675" in body

    def test_contains_credit_and_width(self):
        """Message includes credit, width, contracts, and max loss."""
        info = _sample_trade_opened()
        body = SmsNotifier._format_trade_opened(info)
        assert "$4.20" in body
        assert "25pt" in body
        assert "Contracts: 2" in body
        assert "$4,160" in body

    def test_contains_source_and_signals(self):
        """Message includes trade source and event signals."""
        info = _sample_trade_opened()
        body = SmsNotifier._format_trade_opened(info)
        assert "portfolio_event" in body
        assert "spx_drop_1d" in body

    def test_missing_legs_shows_placeholders(self):
        """Missing short/long dicts produce '?' placeholders throughout."""
        info = {"spread_side": "call", "source": "scheduled"}
        body = SmsNotifier._format_trade_opened(info)
        assert "Short: ? @ ?" in body
        assert "Long:  ? @ ?" in body

    def test_none_credit_and_max_loss_show_placeholder(self):
        """None credit and max_loss display '?' instead of crashing."""
        info = _sample_trade_opened()
        info["credit"] = None
        info["max_loss"] = None
        body = SmsNotifier._format_trade_opened(info)
        assert "Credit: ?" in body
        assert "Max Loss: ?" in body

    def test_event_signals_as_string(self):
        """A single string event_signals value is used directly (no join)."""
        info = _sample_trade_opened()
        info["event_signals"] = "vix_spike"
        body = SmsNotifier._format_trade_opened(info)
        assert "(vix_spike)" in body

    def test_empty_symbol_shows_placeholder(self):
        """Empty string symbol from builder falls back to '?'."""
        info = _sample_trade_opened()
        info["short"]["symbol"] = ""
        info["long"]["symbol"] = ""
        body = SmsNotifier._format_trade_opened(info)
        assert "Short: ? @ " in body
        assert "Long:  ? @ " in body


class TestFormatTradeClosed:
    """Verify trade-closed SMS body formatting."""

    def test_contains_exit_reason_and_trade_id(self):
        """Message header includes exit reason and trade ID."""
        info = _sample_trade_closed()
        body = SmsNotifier._format_trade_closed(info)
        assert "TAKE_PROFIT_50" in body
        assert "#1234" in body

    def test_contains_pnl_info(self):
        """Message includes entry credit, exit cost, and realized PnL."""
        info = _sample_trade_closed()
        body = SmsNotifier._format_trade_closed(info)
        assert "Entry Credit: $4.20" in body
        assert "Exit Cost: $0.85" in body
        assert "+$670.00" in body

    def test_none_exit_reason_defaults_to_closed(self):
        """None exit_reason falls back to 'CLOSED' in both header and footer."""
        info = _sample_trade_closed()
        info["exit_reason"] = None
        body = SmsNotifier._format_trade_closed(info)
        assert "TRADE CLOSED — CLOSED" in body
        assert "Exit Reason: CLOSED" in body

    def test_negative_pnl_no_plus_sign(self):
        """Negative PnL omits the '+' prefix."""
        info = _sample_trade_closed()
        info["realized_pnl"] = -150.00
        body = SmsNotifier._format_trade_closed(info)
        assert "Realized PnL: $-150.00" in body
        assert "+$" not in body

    def test_missing_optional_fields(self):
        """When entry_credit, exit_cost, realized_pnl are None, their lines are omitted."""
        info = {
            "trade_id": 99,
            "strategy_type": "credit_spread",
            "exit_reason": "EXPIRED",
            "source": "scheduled",
        }
        body = SmsNotifier._format_trade_closed(info)
        assert "Entry Credit" not in body
        assert "Exit Cost" not in body
        assert "Realized PnL" not in body
        assert "EXPIRED" in body


# ---------------------------------------------------------------------------
# notify_trade_opened tests
# ---------------------------------------------------------------------------

class TestNotifyTradeOpened:
    """Verify async notify_trade_opened behaviour."""

    @pytest.mark.asyncio
    async def test_sends_to_all_recipients(self, monkeypatch):
        """SMS is sent to each number in twilio_to_numbers."""
        notifier = _make_notifier(monkeypatch, to_numbers="+15551111111,+15552222222")
        info = _sample_trade_opened()
        await notifier.notify_trade_opened(info)
        create_mock = notifier._client.messages.create
        assert create_mock.call_count == 2
        to_numbers_called = {call.kwargs["to"] for call in create_mock.call_args_list}
        assert to_numbers_called == {"+15551111111", "+15552222222"}

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, monkeypatch):
        """No Twilio call when sms_enabled=False."""
        notifier = _make_notifier(monkeypatch, enabled=False)
        info = _sample_trade_opened()
        await notifier.notify_trade_opened(info)
        # _client is None when disabled, so no call possible

    @pytest.mark.asyncio
    async def test_skips_when_source_filtered(self, monkeypatch):
        """No SMS when source doesn't match the filter."""
        notifier = _make_notifier(monkeypatch, sources="event")
        info = _sample_trade_opened()
        info["source"] = "portfolio_scheduled"
        await notifier.notify_trade_opened(info)
        notifier._client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_decision_source_fallback(self, monkeypatch):
        """Falls back to decision_source when source key is absent."""
        notifier = _make_notifier(monkeypatch, sources="event")
        info = _sample_trade_opened()
        del info["source"]
        info["decision_source"] = "portfolio_event"
        await notifier.notify_trade_opened(info)
        assert notifier._client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# notify_trade_closed tests
# ---------------------------------------------------------------------------

class TestNotifyTradeClosed:
    """Verify async notify_trade_closed behaviour."""

    @pytest.mark.asyncio
    async def test_sends_sms(self, monkeypatch):
        """Twilio API is called for a closed trade."""
        notifier = _make_notifier(monkeypatch)
        info = _sample_trade_closed()
        await notifier.notify_trade_closed(info)
        assert notifier._client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, monkeypatch):
        """No Twilio call when sms_enabled=False."""
        notifier = _make_notifier(monkeypatch, enabled=False)
        info = _sample_trade_closed()
        await notifier.notify_trade_closed(info)

    @pytest.mark.asyncio
    async def test_skips_when_source_filtered(self, monkeypatch):
        """No SMS when source doesn't match the filter."""
        notifier = _make_notifier(monkeypatch, sources="event")
        info = _sample_trade_closed()
        info["source"] = "portfolio_scheduled"
        await notifier.notify_trade_closed(info)
        notifier._client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_recipients_does_not_raise(self, monkeypatch):
        """Empty TWILIO_TO_NUMBERS logs a warning but doesn't crash."""
        notifier = _make_notifier(monkeypatch, to_numbers="")
        info = _sample_trade_closed()
        await notifier.notify_trade_closed(info)
        notifier._client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_twilio_exception_swallowed(self, monkeypatch):
        """A Twilio SDK exception on closed trade is logged but not raised."""
        notifier = _make_notifier(monkeypatch)
        notifier._client.messages.create.side_effect = Exception("Twilio error")
        info = _sample_trade_closed()
        await notifier.notify_trade_closed(info)

    @pytest.mark.asyncio
    async def test_trade_source_fallback(self, monkeypatch):
        """Falls back to trade_source when source key is absent."""
        notifier = _make_notifier(monkeypatch, sources="event")
        info = _sample_trade_closed()
        del info["source"]
        info["trade_source"] = "portfolio_event"
        await notifier.notify_trade_closed(info)
        assert notifier._client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Verify that Twilio errors are swallowed and never raised."""

    @pytest.mark.asyncio
    async def test_twilio_exception_swallowed(self, monkeypatch):
        """A Twilio SDK exception is logged but never propagated."""
        notifier = _make_notifier(monkeypatch)
        notifier._client.messages.create.side_effect = Exception("Twilio network error")
        info = _sample_trade_opened()
        await notifier.notify_trade_opened(info)

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, monkeypatch):
        """If first recipient fails, second still receives the SMS."""
        notifier = _make_notifier(monkeypatch, to_numbers="+15551111111,+15552222222")
        call_count = {"n": 0}

        def _side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("first fails")
            return MagicMock()

        notifier._client.messages.create.side_effect = _side_effect
        info = _sample_trade_opened()
        await notifier.notify_trade_opened(info)
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_no_recipients_does_not_raise(self, monkeypatch):
        """Empty TWILIO_TO_NUMBERS logs a warning but doesn't crash."""
        notifier = _make_notifier(monkeypatch, to_numbers="")
        info = _sample_trade_opened()
        await notifier.notify_trade_opened(info)
        notifier._client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_format_error_swallowed_on_open(self, monkeypatch):
        """Non-numeric credit doesn't crash notify_trade_opened."""
        notifier = _make_notifier(monkeypatch)
        info = _sample_trade_opened()
        info["credit"] = "not-a-number"
        await notifier.notify_trade_opened(info)
        notifier._client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_format_error_swallowed_on_close(self, monkeypatch):
        """Non-numeric entry_credit doesn't crash notify_trade_closed."""
        notifier = _make_notifier(monkeypatch)
        info = _sample_trade_closed()
        info["entry_credit"] = "bad"
        await notifier.notify_trade_closed(info)
        notifier._client.messages.create.assert_not_called()
