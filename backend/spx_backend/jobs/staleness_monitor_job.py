"""Staleness monitor: check pipeline data freshness and alert via SendGrid.

Runs periodically during RTH and checks the most recent timestamps in key
tables.  When any source is stale beyond its configured threshold, an email
alert is sent.  A cooldown period prevents duplicate alerts.  The job skips
when the market is closed so it does not fire false alerts on evenings,
weekends, or exchange holidays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth


@dataclass
class StalenessMonitorJob:
    """Check pipeline freshness and send email alerts when data is stale."""

    clock_cache: MarketClockCache | None = None
    _last_alert_ts: datetime | None = field(default=None, repr=False)

    async def _market_open(self, now_et: datetime) -> bool:
        """Check market-open state from clock cache with RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    async def _check_freshness(self, session) -> list[dict]:
        """Query latest timestamps from key tables and compare to thresholds.

        Returns
        -------
        list[dict]
            One entry per stale source with keys: source, latest_ts,
            age_minutes, threshold_minutes.
        """
        checks = [
            ("underlying_quotes", "ts", settings.staleness_quotes_max_minutes),
            ("chain_snapshots", "ts", settings.staleness_snapshots_max_minutes),
            ("gex_snapshots", "ts", settings.staleness_gex_max_minutes),
            ("trade_decisions", "ts", settings.staleness_decisions_max_minutes),
        ]
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        stale: list[dict] = []
        for table, ts_col, threshold in checks:
            row = await session.execute(
                text(f"SELECT MAX({ts_col}) AS latest FROM {table}"),
            )
            result = row.fetchone()
            latest = result.latest if result else None
            if latest is None:
                stale.append({
                    "source": table,
                    "latest_ts": None,
                    "age_minutes": None,
                    "threshold_minutes": threshold,
                })
                continue
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=ZoneInfo("UTC"))
            age = (now_utc - latest).total_seconds() / 60.0
            if age > threshold:
                stale.append({
                    "source": table,
                    "latest_ts": latest.isoformat(),
                    "age_minutes": round(age, 1),
                    "threshold_minutes": threshold,
                })
        return stale

    def _should_alert(self, now_utc: datetime) -> bool:
        """Respect cooldown: return True only if enough time has passed since last alert."""
        if self._last_alert_ts is None:
            return True
        elapsed = (now_utc - self._last_alert_ts).total_seconds() / 60.0
        return elapsed >= settings.staleness_cooldown_minutes

    async def _send_alert(self, stale_sources: list[dict]) -> bool:
        """Send a staleness alert email via SendGrid.

        Parameters
        ----------
        stale_sources:
            List of stale source dicts from ``_check_freshness``.

        Returns
        -------
        bool
            True if the email was sent successfully.
        """
        if not settings.sendgrid_api_key or not settings.email_alert_recipient:
            logger.warning("staleness_monitor: SendGrid API key or recipient not configured")
            return False

        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
        except ImportError:
            logger.error("staleness_monitor: sendgrid package not installed")
            return False

        lines = ["The following data sources are stale:\n"]
        for s in stale_sources:
            age = f"{s['age_minutes']:.0f} min" if s["age_minutes"] is not None else "no data"
            lines.append(
                f"  - {s['source']}: {age} old (threshold: {s['threshold_minutes']} min)"
            )
        body = "\n".join(lines)

        message = Mail(
            from_email=settings.email_alert_sender,
            to_emails=settings.email_alert_recipient,
            subject="[IndexSpreadLab] Pipeline Staleness Alert",
            plain_text_content=body,
        )
        try:
            sg = SendGridAPIClient(settings.sendgrid_api_key)
            response = sg.send(message)
            logger.info(
                "staleness_monitor: alert sent status={} recipient={}",
                response.status_code, settings.email_alert_recipient,
            )
            return response.status_code in (200, 201, 202)
        except Exception as exc:
            logger.error("staleness_monitor: email send failed error={}", exc)
            return False

    async def run_once(self, *, force: bool = False) -> dict:
        """Run one staleness check cycle.

        Skips when the market is closed (evenings, weekends, holidays) to
        avoid false alerts during periods when data staleness is expected.

        Parameters
        ----------
        force:
            Bypass enable and RTH checks.

        Returns
        -------
        dict
            Status payload with stale sources and alert result.
        """
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        now_utc = now_et.astimezone(ZoneInfo("UTC"))

        if (not force) and (not settings.staleness_alert_enabled):
            return {"skipped": True, "reason": "staleness_alert_disabled"}

        if not force:
            if not await self._market_open(now_et):
                return {"skipped": True, "reason": "market_closed"}

        async with SessionLocal() as session:
            stale_sources = await self._check_freshness(session)

        if not stale_sources:
            logger.info("staleness_monitor: all sources fresh")
            return {"skipped": False, "stale_count": 0, "alerted": False}

        logger.warning(
            "staleness_monitor: {} stale source(s): {}",
            len(stale_sources),
            ", ".join(s["source"] for s in stale_sources),
        )

        alerted = False
        if self._should_alert(now_utc):
            alerted = await self._send_alert(stale_sources)
            if alerted:
                self._last_alert_ts = now_utc
        else:
            logger.info("staleness_monitor: alert suppressed (cooldown)")

        return {
            "skipped": False,
            "stale_count": len(stale_sources),
            "stale_sources": stale_sources,
            "alerted": alerted,
        }


def build_staleness_monitor_job(
    clock_cache: MarketClockCache | None = None,
) -> StalenessMonitorJob:
    """Factory helper for StalenessMonitorJob."""
    return StalenessMonitorJob(clock_cache=clock_cache)
