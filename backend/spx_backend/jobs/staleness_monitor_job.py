"""Staleness monitor: check pipeline data freshness and alert via SendGrid.

Runs periodically during RTH and checks the most recent timestamps in key
tables. When any source is stale beyond its configured threshold, an email
alert is sent. A cooldown period prevents duplicate alerts. The job skips
when the market is closed so it does not fire false alerts on evenings,
weekends, or exchange holidays.

Audit Wave 3 / 4 changes
------------------------
* H8: ``_check_freshness`` now groups time-series tables by their
  natural partitioning key (``underlying`` for chain_snapshots /
  gex_snapshots, ``symbol`` for underlying_quotes). A single stale
  underlying / symbol fires a stale entry; previously the global
  MAX masked per-underlying staleness whenever any other underlying
  was fresh.
* M12: a CBOE-source-specific GEX freshness check is layered on top of
  the standard gex_snapshots check using ``source = 'CBOE'`` and a
  separate threshold (``staleness_cboe_gex_max_minutes``) so a mzdata
  vendor outage pages independently of Tradier-side GEX.
* Refactor #3: ``_send_alert`` and the local ``_should_alert`` /
  ``_last_alert_ts`` cooldown have been removed; alert delivery now
  goes through ``services.alerts.send_alert`` (DB-backed cooldown
  per H7), keyed by alert family + offending source key.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal
from spx_backend.market_clock import MarketClockCache, is_rth
from spx_backend.services import alerts


# Per-table check spec used by ``_check_freshness``. ``group_col`` is
# the column to GROUP BY for per-bucket MAX(ts); ``cboe_only`` flags
# the M12 CBOE-specific gex_snapshots variant.
_FRESHNESS_CHECKS: list[tuple[str, str, str | None, int, str | None]] = [
    # (logical name, table, group_col, threshold_minutes_attr_name, source_filter)
    ("underlying_quotes", "underlying_quotes", "symbol", 0, None),
    ("chain_snapshots",   "chain_snapshots",   "underlying", 0, None),
    ("gex_snapshots",     "gex_snapshots",     "underlying", 0, None),
    # M12 (audit): CBOE-specific gex_snapshots freshness.
    ("gex_snapshots_cboe", "gex_snapshots",    "underlying", 0, "CBOE"),
    # trade_decisions has no per-underlying axis; keep as a global
    # MAX(ts) check (group_col=None).
    ("trade_decisions",   "trade_decisions",   None, 0, None),
]


@dataclass
class StalenessMonitorJob:
    """Check pipeline freshness and send email alerts when data is stale."""

    clock_cache: MarketClockCache | None = None

    async def _market_open(self, now_et: datetime) -> bool:
        """Check market-open state from clock cache with RTH fallback."""
        if self.clock_cache:
            return await self.clock_cache.is_open(now_et)
        return is_rth(now_et)

    def _threshold_for(self, source: str) -> int:
        """Return the staleness threshold (minutes) for a logical source.

        Resolved at call time so test overrides of ``settings.*`` flow
        through without re-instantiating the job.
        """
        return {
            "underlying_quotes":   settings.staleness_quotes_max_minutes,
            "chain_snapshots":     settings.staleness_snapshots_max_minutes,
            "gex_snapshots":       settings.staleness_gex_max_minutes,
            "gex_snapshots_cboe":  settings.staleness_cboe_gex_max_minutes,
            "trade_decisions":     settings.staleness_decisions_max_minutes,
        }[source]

    async def _check_freshness(self, session) -> list[dict]:
        """Query latest timestamps from key tables and compare to thresholds.

        H8 (audit): time-series tables are GROUPed BY their natural
        partitioning key (``underlying`` or ``symbol``) so a single
        stale bucket fires a stale entry. Buckets that report no data
        at all are *also* surfaced as stale rows with ``age_minutes=None``;
        the operator should know SPX is not flowing even on cold start.

        Returns
        -------
        list[dict]
            One entry per stale (source, bucket) pair with keys: source,
            bucket (the underlying / symbol or "*" for global checks),
            latest_ts, age_minutes, threshold_minutes.
        """
        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        stale: list[dict] = []
        for logical_name, table, group_col, _placeholder, source_filter in _FRESHNESS_CHECKS:
            threshold = self._threshold_for(logical_name)
            if group_col is None:
                # Global MAX(ts) (e.g. trade_decisions).
                where = ""
                params: dict = {}
                if source_filter is not None:
                    where = "WHERE source = :src"
                    params["src"] = source_filter
                row = await session.execute(
                    text(f"SELECT MAX(ts) AS latest FROM {table} {where}"),
                    params,
                )
                result = row.fetchone()
                latest = result.latest if result else None
                stale_entry = self._build_stale_entry(
                    logical_name, "*", latest, now_utc, threshold,
                )
                if stale_entry is not None:
                    stale.append(stale_entry)
                continue

            # Per-bucket MAX(ts). The GROUP BY guarantees one row per
            # active bucket; we use the configured *expected_buckets*
            # set (chain_snapshots/gex_snapshots: ``cboe_gex_underlyings``
            # / ``snapshot_underlying``; underlying_quotes: ``quote_symbols``)
            # to also detect *missing* buckets that have never reported.
            where = ""
            params = {}
            if source_filter is not None:
                where = "WHERE source = :src"
                params["src"] = source_filter
            row = await session.execute(
                text(
                    f"""
                    SELECT {group_col} AS bucket, MAX(ts) AS latest
                    FROM {table}
                    {where}
                    GROUP BY {group_col}
                    """
                ),
                params,
            )
            observed = {r.bucket: r.latest for r in row.fetchall()}
            expected_buckets = self._expected_buckets_for(logical_name)
            for bucket in expected_buckets | set(observed):
                latest = observed.get(bucket)
                stale_entry = self._build_stale_entry(
                    logical_name, bucket, latest, now_utc, threshold,
                )
                if stale_entry is not None:
                    stale.append(stale_entry)
        return stale

    def _expected_buckets_for(self, logical_name: str) -> set[str]:
        """Return the set of buckets the operator expects to see fresh.

        We compute this at call time so an env-var change to
        ``cboe_gex_underlyings`` or ``quote_symbols`` is picked up
        without redeploy. Empty set means "no expected buckets" --
        rely entirely on observed rows in that case.
        """
        if logical_name in ("chain_snapshots", "gex_snapshots", "gex_snapshots_cboe"):
            # Snapshot writers run for cboe_gex_underlyings (CBOE) and
            # snapshot_underlying (Tradier). Union them so either source
            # missing flags as stale.
            cboe = {s.strip().upper() for s in settings.cboe_gex_underlyings.split(",") if s.strip()}
            tradier = {settings.snapshot_underlying.strip().upper()} if settings.snapshot_underlying else set()
            # M12 (audit): CBOE-source check should only expect the CBOE
            # configured symbols; otherwise it would mark Tradier-only
            # underlyings as stale CBOE rows.
            if logical_name == "gex_snapshots_cboe":
                return cboe
            return cboe | tradier
        if logical_name == "underlying_quotes":
            return {s.strip().upper() for s in settings.quote_symbols.split(",") if s.strip()}
        return set()

    def _build_stale_entry(
        self,
        source: str,
        bucket: str,
        latest: datetime | None,
        now_utc: datetime,
        threshold: int,
    ) -> dict | None:
        """Convert a (source, bucket, latest) tuple into a stale-entry dict, or None.

        Returns None when the bucket is fresh (within threshold) and has
        observed data. A missing bucket (latest is None) always returns
        a stale entry so cold-start gaps surface immediately.
        """
        if latest is None:
            return {
                "source": source,
                "bucket": bucket,
                "latest_ts": None,
                "age_minutes": None,
                "threshold_minutes": threshold,
            }
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=ZoneInfo("UTC"))
        age = (now_utc - latest).total_seconds() / 60.0
        if age > threshold:
            return {
                "source": source,
                "bucket": bucket,
                "latest_ts": latest.isoformat(),
                "age_minutes": round(age, 1),
                "threshold_minutes": threshold,
            }
        return None

    def _format_alert_body(self, stale_sources: list[dict]) -> str:
        """Render a human-readable HTML body for the staleness alert.

        Switched from plain text to HTML in Refactor #3 because
        ``services.alerts.send_alert`` takes ``body_html``; the table
        layout makes per-bucket rows easier to scan than the previous
        wall of text.
        """
        rows_html = []
        for s in stale_sources:
            age = (
                f"{s['age_minutes']:.0f} min"
                if s["age_minutes"] is not None
                else "no data"
            )
            rows_html.append(
                f"<tr>"
                f"<td>{s['source']}</td>"
                f"<td>{s['bucket']}</td>"
                f"<td>{age}</td>"
                f"<td>{s['threshold_minutes']} min</td>"
                f"<td>{s['latest_ts'] or '-'}</td>"
                f"</tr>"
            )
        return (
            "<p><b>Pipeline staleness alert:</b> "
            f"{len(stale_sources)} stale source(s).</p>"
            "<table border='1' cellpadding='4' cellspacing='0'>"
            "<thead><tr>"
            "<th>Source</th><th>Bucket</th><th>Age</th>"
            "<th>Threshold</th><th>Latest TS</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
        )

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
            ", ".join(f"{s['source']}/{s['bucket']}" for s in stale_sources),
        )

        # Refactor #3 (audit): cooldown + delivery now lives in
        # services.alerts.send_alert (DB-backed via H7). Cooldown key
        # is intentionally coarse ("staleness:any") so a flap of any
        # bucket within a window collapses to one page; per-bucket
        # detail is in the body.
        body_html = self._format_alert_body(stale_sources)
        alerted = await alerts.send_alert(
            subject="[IndexSpreadLab] Pipeline Staleness Alert",
            body_html=body_html,
            cooldown_key="staleness:any",
            cooldown_minutes=int(settings.staleness_cooldown_minutes),
        )

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
