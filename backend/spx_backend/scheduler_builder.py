"""Scheduler construction and job registration extracted from app.py lifespan.

Encapsulates all APScheduler wiring so the lifespan function stays slim
and the scheduling logic is independently testable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger

from spx_backend.config import Settings, settings as default_settings
from spx_backend.ingestion.tradier_client import get_tradier_client
from spx_backend.jobs.cboe_gex_job import build_cboe_gex_job
from spx_backend.jobs.decision_job import DecisionJob
from spx_backend.jobs.eod_events_job import EodEventsJob
from spx_backend.jobs.gex_job import GexJob
from spx_backend.jobs.performance_analytics_job import build_performance_analytics_job
from spx_backend.jobs.quote_job import QuoteJob
from spx_backend.jobs.snapshot_job import build_snapshot_job, build_spy_snapshot_job, build_vix_snapshot_job
from spx_backend.jobs.staleness_monitor_job import build_staleness_monitor_job
from spx_backend.jobs.trade_pnl_job import TradePnlJob
from spx_backend.market_clock import MarketClockCache
from spx_backend.services.sms_notifier import SmsNotifier


_job_failure_last_alert: dict[str, datetime] = {}


def _send_job_failure_email(*, job_id: str, kind: str, detail: str, cfg: Settings | None = None) -> None:
    """Send a job failure/misfire alert via SendGrid with per-job cooldown.

    Reuses the same SendGrid credentials as the staleness monitor.  Skips
    silently when disabled, unconfigured, or within cooldown.

    Parameters
    ----------
    job_id:
        Scheduler job identifier for cooldown tracking.
    kind:
        Event type label (``FAILURE`` or ``MISFIRE``).
    detail:
        Human-readable description of the error or misfire event.
    cfg:
        Settings instance; falls back to the module-level singleton.
    """
    cfg = cfg or default_settings
    if not cfg.job_failure_alert_enabled:
        return
    if not cfg.sendgrid_api_key or not cfg.email_alert_recipient:
        return

    now = datetime.now(tz=ZoneInfo("UTC"))
    last = _job_failure_last_alert.get(job_id)
    if last is not None:
        elapsed = (now - last).total_seconds() / 60.0
        if elapsed < cfg.job_failure_alert_cooldown_minutes:
            logger.debug("scheduler_alert: cooldown active for job_id={}", job_id)
            return

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
    except ImportError:
        logger.error("scheduler_alert: sendgrid package not installed")
        return

    subject = f"[IndexSpreadLab] Job {kind}: {job_id}"
    body = f"Job: {job_id}\nEvent: {kind}\n\n{detail}"
    message = Mail(
        from_email=cfg.email_alert_sender,
        to_emails=cfg.email_alert_recipient,
        subject=subject,
        plain_text_content=body,
    )
    try:
        sg = SendGridAPIClient(cfg.sendgrid_api_key)
        response = sg.send(message)
        if response.status_code in (200, 201, 202):
            _job_failure_last_alert[job_id] = now
            logger.info("scheduler_alert: email sent for job_id={} kind={}", job_id, kind)
        else:
            logger.warning("scheduler_alert: unexpected status={}", response.status_code)
    except Exception as exc:
        logger.error("scheduler_alert: email send failed error={}", exc)


def build_scheduler_event_listener(cfg: Settings | None = None) -> Any:
    """Return an APScheduler event listener that logs and emails on job errors/misfires.

    Parameters
    ----------
    cfg:
        Settings instance used by the email sender.  Falls back to the
        module-level singleton when ``None``.

    Returns
    -------
    Callable
        A listener compatible with ``scheduler.add_listener``.
    """
    def _listener(event: Any) -> None:
        job_id = getattr(event, "job_id", "unknown_job")
        exception = getattr(event, "exception", None)
        traceback_value = getattr(event, "traceback", None)
        if exception is not None:
            logger.error("scheduler: job_id={} failed error={} traceback={}", job_id, exception, traceback_value)
            _send_job_failure_email(
                job_id=job_id, kind="FAILURE",
                detail=f"Error: {exception}\n\nTraceback:\n{traceback_value or 'N/A'}",
                cfg=cfg,
            )
            return
        logger.warning("scheduler: job_id={} missed scheduled run", job_id)
        _send_job_failure_email(job_id=job_id, kind="MISFIRE", detail="Job missed its scheduled execution window.", cfg=cfg)

    return _listener


def normalized_rth_interval_minutes(interval_minutes: int, *, job_id: str) -> int:
    """Normalize configured cadence to values representable in cron minute fields.

    Parameters
    ----------
    interval_minutes:
        Desired scheduling cadence loaded from settings.
    job_id:
        Scheduler job identifier used for actionable warning context.

    Returns
    -------
    int
        A positive minute cadence that evenly divides one hour. Invalid values
        fall back to ``10`` minutes so the RTH schedule remains deterministic.
    """
    if interval_minutes > 0 and (60 % interval_minutes) == 0:
        return interval_minutes
    logger.warning(
        "scheduler: job_id={} invalid_interval_minutes={} -> using fallback=10",
        job_id,
        interval_minutes,
    )
    return 10


def build_rth_regular_trigger(*, interval_minutes: int, timezone: str, job_id: str) -> Any:
    """Build a weekday RTH trigger spanning 09:31 through 15:55 ET.

    Parameters
    ----------
    interval_minutes:
        Desired cadence for regular snapshot/cboe ingestion runs.
    timezone:
        Scheduler timezone string (for example ``America/New_York``).
    job_id:
        Job identifier used to contextualize interval fallback warnings.

    Returns
    -------
    Any
        An APScheduler trigger that fires at 09:31+ cadence in the opening
        hour and then repeats every cadence interval from 10:00 to 15:55.
    """
    from apscheduler.triggers.combining import OrTrigger
    from apscheduler.triggers.cron import CronTrigger

    cadence = normalized_rth_interval_minutes(interval_minutes, job_id=job_id)
    opening_minutes = ",".join(str(minute) for minute in range(31, 60, cadence))
    session_minutes = ",".join(str(minute) for minute in range(0, 60, cadence))
    return OrTrigger(
        [
            CronTrigger(day_of_week="mon-fri", hour=9, minute=opening_minutes, timezone=timezone),
            CronTrigger(day_of_week="mon-fri", hour="10-15", minute=session_minutes, timezone=timezone),
        ]
    )


def build_serialized_run_once_runner(job: Any) -> Any:
    """Wrap ``run_once`` so split scheduler triggers execute serially.

    Parameters
    ----------
    job:
        Job instance exposing ``async run_once(force: bool = False)``.

    Returns
    -------
    Any
        An async callable that forwards ``force`` while preventing overlap
        between regular and forced-close runs.
    """
    run_lock = asyncio.Lock()

    async def _run(*, force: bool = False) -> dict[str, Any]:
        """Execute one serialized scheduler tick for the wrapped job."""
        async with run_lock:
            return await job.run_once(force=force)

    return _run


def build_market_open_guarded_runner(
    run_callable: Any,
    *,
    clock_cache: MarketClockCache,
    timezone: str,
    job_id: str,
    open_trading_days: set[date],
    allow_outside_rth: bool = False,
) -> Any:
    """Guard scheduled runs so holidays skip while preserving close-force behavior.

    Parameters
    ----------
    run_callable:
        Async callable that accepts ``force`` and executes one job tick.
    clock_cache:
        Shared market clock cache used to determine whether the exchange is open.
    timezone:
        Timezone string used to compute the scheduler-local trading date.
    job_id:
        Scheduler job identifier used for structured skip logging.
    open_trading_days:
        Mutable set shared across scheduler wrappers. A date is added when any
        guarded run observes ``is_open=True``, which allows close-force jobs to
        run at 16:00 only on real trading days.
    allow_outside_rth:
        When ``True``, the guard lets the job through even when the market is
        closed.  The job's internal RTH check still applies.

    Returns
    -------
    Any
        Async callable compatible with APScheduler ``add_job``.
    """

    async def _run(*, force: bool = False) -> dict[str, Any]:
        """Execute one guarded scheduler tick with holiday-aware skip semantics."""
        now_et = datetime.now(tz=ZoneInfo(timezone))
        is_open_now = await clock_cache.is_open(now_et)
        if is_open_now:
            open_trading_days.add(now_et.date())
            cutoff = now_et.date() - timedelta(days=14)
            stale_days = [tracked_day for tracked_day in open_trading_days if tracked_day < cutoff]
            for tracked_day in stale_days:
                open_trading_days.discard(tracked_day)

        if force:
            if now_et.date() not in open_trading_days:
                logger.info(
                    "scheduler: job_id={} skipped force run (non_trading_day) now_et={}",
                    job_id,
                    now_et.isoformat(),
                )
                return {"skipped": True, "reason": "non_trading_day", "now_et": now_et.isoformat()}
            return await run_callable(force=True)

        if allow_outside_rth:
            return await run_callable(force=False)

        if not is_open_now:
            logger.info(
                "scheduler: job_id={} skipped run (market_closed_or_holiday) now_et={}",
                job_id,
                now_et.isoformat(),
            )
            return {"skipped": True, "reason": "market_closed_or_holiday", "now_et": now_et.isoformat()}

        return await run_callable(force=False)

    return _run


def _schedule_rth_window_job(
    scheduler: Any,
    *,
    job: Any,
    job_id: str,
    clock_cache: MarketClockCache,
    open_trading_days: set[date],
    interval_minutes: int,
    timezone: str,
    max_job_instances: int,
    misfire_grace_seconds: int,
    allow_outside_rth: bool = False,
) -> None:
    """Schedule one job for RTH cadence with a guaranteed 16:00 ET run.

    The regular trigger runs from 09:31 through 15:55 on weekdays at the
    configured cadence. A separate close trigger runs at 16:00 with
    ``force=True`` and is allowed only if a guarded run observed open market
    status earlier that date, which skips holiday executions.

    When ``allow_outside_rth`` is ``True``, the market-open guard lets the job
    through regardless of exchange status; the job's own RTH check still applies.
    """
    run_callable = build_market_open_guarded_runner(
        build_serialized_run_once_runner(job),
        clock_cache=clock_cache,
        timezone=timezone,
        job_id=job_id,
        open_trading_days=open_trading_days,
        allow_outside_rth=allow_outside_rth,
    )
    scheduler.add_job(
        run_callable,
        trigger=build_rth_regular_trigger(interval_minutes=interval_minutes, timezone=timezone, job_id=job_id),
        id=job_id,
        replace_existing=True,
        max_instances=max_job_instances,
        misfire_grace_time=misfire_grace_seconds,
    )
    scheduler.add_job(
        run_callable,
        "cron",
        day_of_week="mon-fri",
        hour=16,
        minute=0,
        kwargs={"force": True},
        id=f"{job_id}_close",
        replace_existing=True,
        max_instances=max_job_instances,
        misfire_grace_time=misfire_grace_seconds,
    )


@dataclass
class SchedulerContext:
    """Container for all scheduler-related objects created during startup.

    Holds job instances and the scheduler itself so they can be attached
    to ``app.state`` in one sweep without repeating field names.
    """

    scheduler: Any
    tradier: Any
    clock_cache: MarketClockCache
    snapshot_job: Any
    spy_snapshot_job: Any | None
    vix_snapshot_job: Any | None
    quote_job: QuoteJob
    gex_job: GexJob
    cboe_gex_job: Any | None
    decision_job: DecisionJob
    trade_pnl_job: TradePnlJob
    performance_analytics_job: Any | None
    staleness_monitor_job: Any | None

    _warmup_jobs: list[tuple[str, Any]] = field(default_factory=list, repr=False)

    def attach_to_app_state(self, app: Any) -> None:
        """Copy all job references onto ``app.state`` for router access.

        Mirrors the per-job ``app.state.foo = ...`` assignments that the
        lifespan function previously maintained inline.
        """
        for attr in [
            "scheduler", "tradier", "clock_cache",
            "snapshot_job", "spy_snapshot_job", "vix_snapshot_job",
            "quote_job", "gex_job", "cboe_gex_job",
            "decision_job", "trade_pnl_job",
            "performance_analytics_job", "staleness_monitor_job",
        ]:
            setattr(app.state, attr, getattr(self, attr))

    async def run_warmup(self) -> None:
        """Execute startup warmup for ingestion jobs, logging failures without crashing."""
        for job_name, run_once_callable in self._warmup_jobs:
            try:
                await run_once_callable()
                logger.info("startup_warmup: job_id={} status=ok", job_name)
            except Exception as exc:
                logger.exception("startup_warmup: job_id={} status=failed error={}", job_name, exc)


def build_scheduler(cfg: Settings | None = None) -> SchedulerContext:
    """Construct the APScheduler instance with all jobs registered.

    Parameters
    ----------
    cfg:
        Application settings. Falls back to the module-level singleton
        so callers (like tests) can inject overrides.

    Returns
    -------
    SchedulerContext
        Fully wired scheduler with all job instances ready to be attached
        to ``app.state`` and started.
    """
    cfg = cfg or default_settings

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED

    scheduler = AsyncIOScheduler(timezone=cfg.tz)
    scheduler.add_listener(build_scheduler_event_listener(cfg), EVENT_JOB_ERROR | EVENT_JOB_MISSED)

    tradier = get_tradier_client()
    clock_cache = MarketClockCache(tradier=tradier, ttl_seconds=cfg.market_clock_cache_seconds)
    open_trading_days: set[date] = set()
    max_job_instances = 1
    misfire_grace_seconds = 300

    # -- Build job instances --------------------------------------------------
    snapshot_job = build_snapshot_job(tradier=tradier, clock_cache=clock_cache)
    spy_snapshot_job = build_spy_snapshot_job(tradier=tradier, clock_cache=clock_cache) if cfg.spy_snapshot_enabled else None
    vix_snapshot_job = build_vix_snapshot_job(tradier=tradier, clock_cache=clock_cache) if cfg.vix_snapshot_enabled else None
    quote_job = QuoteJob(tradier=tradier, clock_cache=clock_cache)
    gex_job = GexJob(clock_cache=clock_cache)
    cboe_gex_job = build_cboe_gex_job(clock_cache=clock_cache) if cfg.cboe_gex_enabled else None
    sms_notifier = SmsNotifier()
    decision_job = DecisionJob(clock_cache=clock_cache, notifier=sms_notifier)
    trade_pnl_job = TradePnlJob(clock_cache=clock_cache, notifier=sms_notifier)
    performance_analytics_job = build_performance_analytics_job() if cfg.performance_analytics_enabled else None

    # -- RTH-window jobs (snapshot, quote, GEX) --------------------------------
    rth_kwargs = dict(
        clock_cache=clock_cache,
        open_trading_days=open_trading_days,
        timezone=cfg.tz,
        max_job_instances=max_job_instances,
        misfire_grace_seconds=misfire_grace_seconds,
    )

    _schedule_rth_window_job(
        scheduler, job=snapshot_job, job_id="snapshot_job",
        interval_minutes=cfg.snapshot_interval_minutes,
        allow_outside_rth=cfg.allow_snapshot_outside_rth, **rth_kwargs,
    )

    if vix_snapshot_job is not None:
        _schedule_rth_window_job(
            scheduler, job=vix_snapshot_job, job_id="snapshot_job_vix",
            interval_minutes=cfg.vix_snapshot_interval_minutes,
            allow_outside_rth=cfg.vix_allow_snapshot_outside_rth, **rth_kwargs,
        )

    if spy_snapshot_job is not None:
        _schedule_rth_window_job(
            scheduler, job=spy_snapshot_job, job_id="snapshot_job_spy",
            interval_minutes=cfg.spy_snapshot_interval_minutes,
            allow_outside_rth=cfg.spy_allow_snapshot_outside_rth, **rth_kwargs,
        )

    _schedule_rth_window_job(
        scheduler, job=quote_job, job_id="quote_job",
        interval_minutes=cfg.quote_interval_minutes,
        allow_outside_rth=cfg.allow_quotes_outside_rth, **rth_kwargs,
    )
    _schedule_rth_window_job(
        scheduler, job=gex_job, job_id="gex_job",
        interval_minutes=cfg.gex_interval_minutes,
        allow_outside_rth=cfg.gex_allow_outside_rth, **rth_kwargs,
    )

    if cboe_gex_job is not None:
        _schedule_rth_window_job(
            scheduler, job=cboe_gex_job, job_id="cboe_gex_job",
            interval_minutes=cfg.cboe_gex_interval_minutes,
            allow_outside_rth=cfg.cboe_gex_allow_outside_rth, **rth_kwargs,
        )

    # -- Entry-time jobs (decision) --------------------------------------------
    decision_entry_runner = build_market_open_guarded_runner(
        build_serialized_run_once_runner(decision_job),
        clock_cache=clock_cache, timezone=cfg.tz, job_id="decision_job",
        open_trading_days=open_trading_days,
        allow_outside_rth=cfg.decision_allow_outside_rth,
    )
    for hour, minute in cfg.decision_entry_times_list():
        scheduler.add_job(
            decision_entry_runner, "cron",
            day_of_week="mon-fri", hour=hour, minute=minute,
            id=f"decision_job_{hour:02d}{minute:02d}",
            replace_existing=True, max_instances=max_job_instances, misfire_grace_time=misfire_grace_seconds,
        )

    # -- After-close jobs (trade PnL, EOD events) ------------------------------
    if cfg.trade_pnl_enabled:
        trade_pnl_runner = build_serialized_run_once_runner(trade_pnl_job)
        scheduler.add_job(
            trade_pnl_runner, "interval",
            minutes=cfg.trade_pnl_interval_minutes, id="trade_pnl_job",
            replace_existing=True, max_instances=max_job_instances, misfire_grace_time=misfire_grace_seconds,
        )

    if performance_analytics_job is not None:
        _schedule_rth_window_job(
            scheduler, job=performance_analytics_job, job_id="performance_analytics_job",
            interval_minutes=cfg.performance_analytics_interval_minutes, **rth_kwargs,
        )

    if cfg.eod_events_enabled:
        eod_events_job = EodEventsJob()
        eod_events_runner = build_market_open_guarded_runner(
            build_serialized_run_once_runner(eod_events_job),
            clock_cache=clock_cache, timezone=cfg.tz, job_id="eod_events_job", open_trading_days=open_trading_days,
        )
        scheduler.add_job(
            eod_events_runner, "cron",
            day_of_week="mon-fri", hour=cfg.eod_events_hour, minute=cfg.eod_events_minute,
            kwargs={"force": True}, id="eod_events_job",
            replace_existing=True, max_instances=max_job_instances, misfire_grace_time=misfire_grace_seconds,
        )

    # -- Daily retention job (3 AM ET) ------------------------------------------
    if cfg.retention_enabled:
        from spx_backend.jobs import retention_job

        async def _retention_runner():
            await retention_job.run_once()

        scheduler.add_job(
            _retention_runner, "cron",
            hour=3, minute=0,
            id="retention_job", replace_existing=True,
            max_instances=max_job_instances, misfire_grace_time=misfire_grace_seconds,
        )

    # -- Interval jobs (staleness monitor) -------------------------------------
    staleness_monitor_job = build_staleness_monitor_job(clock_cache=clock_cache) if cfg.staleness_alert_enabled else None
    if staleness_monitor_job is not None:
        staleness_runner = build_serialized_run_once_runner(staleness_monitor_job)
        scheduler.add_job(
            staleness_runner, "interval",
            minutes=cfg.staleness_alert_interval_minutes, id="staleness_monitor_job",
            replace_existing=True, max_instances=max_job_instances, misfire_grace_time=misfire_grace_seconds,
        )

    # -- Build warmup list -----------------------------------------------------
    warmup_jobs: list[tuple[str, Any]] = [
        ("quote_job", quote_job.run_once),
        ("snapshot_job", snapshot_job.run_once),
    ]
    if spy_snapshot_job is not None:
        warmup_jobs.append(("snapshot_job_spy", spy_snapshot_job.run_once))
    if vix_snapshot_job is not None:
        warmup_jobs.append(("snapshot_job_vix", vix_snapshot_job.run_once))
    warmup_jobs.append(("gex_job", gex_job.run_once))
    if cboe_gex_job is not None:
        warmup_jobs.append(("cboe_gex_job", cboe_gex_job.run_once))
    if performance_analytics_job is not None:
        warmup_jobs.append(("performance_analytics_job", performance_analytics_job.run_once))

    return SchedulerContext(
        scheduler=scheduler,
        tradier=tradier,
        clock_cache=clock_cache,
        snapshot_job=snapshot_job,
        spy_snapshot_job=spy_snapshot_job,
        vix_snapshot_job=vix_snapshot_job,
        quote_job=quote_job,
        gex_job=gex_job,
        cboe_gex_job=cboe_gex_job,
        decision_job=decision_job,
        trade_pnl_job=trade_pnl_job,
        performance_analytics_job=performance_analytics_job,
        staleness_monitor_job=staleness_monitor_job,
        _warmup_jobs=warmup_jobs,
    )
