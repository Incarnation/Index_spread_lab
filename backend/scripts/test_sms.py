"""Smoke test: send realistic trade-opened and trade-closed SMS via Twilio.

Sends the exact same message format the production SmsNotifier uses so you
can verify how real alerts will look on your phone.

Usage (from repo root or backend/):
    python3.11 backend/scripts/test_sms.py
    # or
    cd backend && python3.11 scripts/test_sms.py

Reads credentials from .env via pydantic-settings (same path as the app).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the backend package is importable when run from repo root.
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from spx_backend.config import settings
from spx_backend.services.sms_notifier import SmsNotifier

SAMPLE_TRADE_OPENED: dict = {
    "spread_side": "put",
    "target_dte": 5,
    "expiration": "2026-04-13",
    "short": {
        "symbol": "SPXW260413P05500",
        "strike": 5500,
        "entry_price": 1.20,
        "delta": 0.10,
    },
    "long": {
        "symbol": "SPXW260413P05490",
        "strike": 5490,
        "entry_price": 0.45,
    },
    "credit": 0.75,
    "width_points": 10,
    "contracts": 1,
    "max_loss": 925,
    "source": "scheduled",
}

SAMPLE_TRADE_CLOSED: dict = {
    "trade_id": 42,
    "strategy_type": "credit_spread",
    "exit_reason": "TP",
    "entry_credit": 0.75,
    "exit_cost": 0.38,
    "realized_pnl": 37.00,
    "contracts": 1,
    "source": "scheduled",
}


def _send_message(client, label: str, body: str, recipients: list[str]) -> None:
    """Send *body* to every recipient and print results.

    Parameters
    ----------
    client:
        Initialised ``twilio.rest.Client``.
    label:
        Human-readable tag printed in console output (e.g. "TRADE OPENED").
    body:
        Full SMS body text.
    recipients:
        List of E.164 phone numbers.
    """
    for number in recipients:
        print(f"[{label}] Sending to {number} ...")
        try:
            msg = client.messages.create(
                body=body,
                from_=settings.twilio_from_number,
                to=number,
            )
            print(f"  OK  sid={msg.sid}  status={msg.status}")
        except Exception as exc:
            print(f"  FAIL  {exc}")
            sys.exit(1)


def main() -> None:
    """Validate Twilio config and send sample trade-opened + trade-closed SMS."""
    print("=" * 50)
    print("Twilio SMS Smoke Test")
    print("=" * 50)
    print(f"  SMS_ENABLED:        {settings.sms_enabled}")
    print(f"  ACCOUNT_SID:        {settings.twilio_account_sid[:8]}...")
    print(f"  FROM_NUMBER:        {settings.twilio_from_number}")
    print(f"  TO_NUMBERS:         {settings.twilio_to_numbers}")
    print(f"  NOTIFY_SOURCES:     {settings.sms_notify_sources}")
    print()

    if not settings.sms_enabled:
        print("ERROR: SMS_ENABLED is false. Set it to true in .env and retry.")
        sys.exit(1)

    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        print("ERROR: TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN is empty.")
        sys.exit(1)

    recipients = settings.twilio_to_numbers_list()
    if not recipients:
        print("ERROR: TWILIO_TO_NUMBERS is empty. Add at least one recipient.")
        sys.exit(1)

    from twilio.rest import Client

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

    opened_body = SmsNotifier._format_trade_opened(SAMPLE_TRADE_OPENED)
    closed_body = SmsNotifier._format_trade_closed(SAMPLE_TRADE_CLOSED)

    print("-" * 50)
    print("Message 1 (Trade Opened):")
    print("-" * 50)
    print(opened_body)
    print()
    print("-" * 50)
    print("Message 2 (Trade Closed):")
    print("-" * 50)
    print(closed_body)
    print()

    _send_message(client, "TRADE OPENED", opened_body, recipients)
    print()
    _send_message(client, "TRADE CLOSED", closed_body, recipients)

    print()
    print("Done! Check your phone for both messages.")


if __name__ == "__main__":
    main()
