"""Async SMS trade notifications via Twilio.

Sends fire-and-forget text messages when trades are opened or closed.
All Twilio errors are logged and swallowed — SMS failures never block
the trading pipeline.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from spx_backend.config import settings


class SmsNotifier:
    """Twilio-backed SMS notifier for trade lifecycle events.

    Constructed once at startup and shared across jobs.  When
    ``settings.sms_enabled`` is False the public methods return
    immediately without initialising the Twilio client.

    Parameters
    ----------
    None — reads all configuration from the ``settings`` singleton.
    """

    def __init__(self) -> None:
        self._client = None
        if settings.sms_enabled:
            try:
                from twilio.rest import Client
                self._client = Client(
                    settings.twilio_account_sid,
                    settings.twilio_auth_token,
                )
            except Exception:
                logger.exception("sms_notifier: failed to initialise Twilio client")

    def _should_notify(self, source: str) -> bool:
        """Check whether this trade source passes the config filter.

        Parameters
        ----------
        source:
            Trade origin tag — typically ``'event'``, ``'scheduled'``,
            ``'portfolio_event'``, ``'portfolio_scheduled'``, or ``'rules'``.

        Returns
        -------
        bool
            True when SMS should be sent for this source.
        """
        if not settings.sms_enabled or self._client is None:
            return False
        allowed = settings.sms_notify_sources.strip().lower()
        if allowed == "all":
            return True
        if allowed == "event":
            return "event" in source.lower()
        if allowed == "scheduled":
            return bool(source) and "event" not in source.lower()
        logger.warning("sms_notifier: unknown sms_notify_sources={!r}, skipping", allowed)
        return False

    def _send_sync(self, body: str) -> None:
        """Deliver *body* to every configured recipient, swallowing errors.

        This is the synchronous implementation called from a thread pool
        by ``_send`` so the Twilio network I/O never blocks the event loop.
        """
        recipients = settings.twilio_to_numbers_list()
        if not recipients:
            logger.warning("sms_notifier: no recipients configured (TWILIO_TO_NUMBERS empty)")
            return
        for number in recipients:
            try:
                self._client.messages.create(
                    body=body,
                    from_=settings.twilio_from_number,
                    to=number,
                )
                logger.info("sms_notifier: sent to={}", number)
            except Exception:
                logger.exception("sms_notifier: failed to send to={}", number)

    async def _send(self, body: str) -> None:
        """Offload the synchronous Twilio SDK call to a thread pool.

        Prevents blocking the async event loop while waiting for Twilio
        API round-trips (~200-500ms per recipient).
        """
        await asyncio.to_thread(self._send_sync, body)

    @staticmethod
    def _format_trade_opened(info: dict[str, Any]) -> str:
        """Build the SMS body for a newly opened trade.

        Parameters
        ----------
        info:
            Dict with keys from the candidate / chosen_legs_json at trade
            creation time.  Expected keys: spread_side, target_dte,
            expiration, short (dict with symbol, strike, entry_price,
            delta), long (same), credit, width_points, contracts,
            max_loss, source, event_signals.
        """
        side = (info.get("spread_side") or "").upper()
        dte = info.get("target_dte", "?")
        exp = info.get("expiration", "?")

        short = info.get("short") or {}
        long = info.get("long") or {}
        short_sym = short.get("symbol") or "?"
        short_strike = short.get("strike", "?")
        short_price = short.get("entry_price")
        short_delta = short.get("delta")
        long_sym = long.get("symbol") or "?"
        long_strike = long.get("strike", "?")
        long_price = long.get("entry_price")

        credit = info.get("credit")
        width = info.get("width_points")
        contracts = info.get("contracts", 1)
        max_loss = info.get("max_loss")

        source = info.get("source", "")
        signals = info.get("event_signals")

        lines = [
            "NEW TRADE OPENED",
            f"{side} Credit Spread | {dte} DTE",
            f"Expiration: {exp}",
            "",
        ]

        short_line = f"Short: {short_sym} @ {short_strike}"
        if short_price is not None:
            short_line += f" ${short_price:.2f}"
        if short_delta is not None:
            short_line += f" (delta {short_delta})"
        lines.append(short_line)

        long_line = f"Long:  {long_sym} @ {long_strike}"
        if long_price is not None:
            long_line += f" ${long_price:.2f}"
        lines.append(long_line)

        lines.append("")
        credit_str = f"${credit:.2f}" if credit is not None else "?"
        width_str = f"{width}pt" if width is not None else "?"
        lines.append(f"Credit: {credit_str} | Width: {width_str}")

        max_loss_str = f"${max_loss:,.0f}" if max_loss is not None else "?"
        lines.append(f"Contracts: {contracts} | Max Loss: {max_loss_str}")

        if source:
            source_line = f"Source: {source}"
            if signals:
                sig_str = signals if isinstance(signals, str) else ",".join(signals)
                source_line += f" ({sig_str})"
            lines.append("")
            lines.append(source_line)

        return "\n".join(lines)

    @staticmethod
    def _format_trade_closed(info: dict[str, Any]) -> str:
        """Build the SMS body for a closed trade.

        Parameters
        ----------
        info:
            Dict with keys: trade_id, strategy_type, exit_reason,
            entry_credit, exit_cost, realized_pnl, contracts.
        """
        exit_reason = (info.get("exit_reason") or "CLOSED").upper()
        strategy = (info.get("strategy_type") or "Credit Spread").replace("_", " ").title()
        trade_id = info.get("trade_id", "?")

        entry_credit = info.get("entry_credit")
        exit_cost = info.get("exit_cost")
        realized_pnl = info.get("realized_pnl")

        pnl_sign = "+" if (realized_pnl or 0) >= 0 else ""

        lines = [
            f"TRADE CLOSED — {exit_reason}",
            f"{strategy} | Trade #{trade_id}",
            "",
        ]

        if entry_credit is not None:
            lines.append(f"Entry Credit: ${entry_credit:.2f}")
        if exit_cost is not None:
            lines.append(f"Exit Cost: ${exit_cost:.2f}")
        if realized_pnl is not None:
            lines.append(f"Realized PnL: {pnl_sign}${realized_pnl:,.2f}")

        lines.append("")
        lines.append(f"Exit Reason: {exit_reason}")

        return "\n".join(lines)

    async def notify_trade_opened(self, info: dict[str, Any]) -> None:
        """Send SMS for a newly opened trade, if source filter passes.

        Parameters
        ----------
        info:
            Trade data dict.  Must include ``source`` for filtering.
            See ``_format_trade_opened`` for full expected schema.
        """
        source = info.get("source") or info.get("decision_source") or ""
        if not self._should_notify(source):
            return
        try:
            body = self._format_trade_opened(info)
        except Exception:
            logger.exception("sms_notifier: failed to format trade-opened SMS")
            return
        logger.info("sms_notifier: sending trade-opened SMS (source={})", source)
        await self._send(body)

    async def notify_trade_closed(self, info: dict[str, Any]) -> None:
        """Send SMS when a trade is closed (TP, SL, or expiration).

        Parameters
        ----------
        info:
            Trade close data dict.  Must include ``source`` for filtering.
            See ``_format_trade_closed`` for full expected schema.
        """
        source = info.get("source") or info.get("trade_source") or ""
        if not self._should_notify(source):
            return
        try:
            body = self._format_trade_closed(info)
        except Exception:
            logger.exception("sms_notifier: failed to format trade-closed SMS")
            return
        logger.info("sms_notifier: sending trade-closed SMS (source={})", source)
        await self._send(body)
