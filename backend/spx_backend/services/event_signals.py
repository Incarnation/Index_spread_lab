"""Production event-signal detector for the event-driven trading layer.

Reads recent market context from the database and evaluates configurable
trigger thresholds to produce a list of active event signals for the
current trading day.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database.connection import engine


# Maximum allowed calendar-day span between sorted_spx_days[-3] and
# sorted_spx_days[-1] for a valid 2-day return.  4 covers the normal
# Friday->Monday weekend (Sat/Sun gap) plus one holiday day (e.g.
# Thanksgiving long weekend).  Anything larger (>4) implies a data
# gap big enough that the "2-day return" is actually spanning a much
# longer interval and the signal would mislead the event layer.
#
# Implementation note: we test the *total span* across the 3 trading-day
# window (last - third-to-last) rather than the original spec's pairwise
# gap (max(last - middle, middle - first)).  Total-span is the strictly
# stricter test -- any pairwise gap that fails also makes total-span
# fail, but total-span additionally catches the "two small gaps that
# add up" case (e.g. 3-day weekend + 3-day gap = 6 days total, both
# pairwise gaps within bounds).  Strictly-stricter is the correct side
# of a data-quality gate, so the deviation from the original plan is
# intentional.
_SPX_2D_MAX_CALENDAR_GAP_DAYS = 4


class EventSignalDetector:
    """Evaluate market conditions and return active event signals.

    Parameters
    ----------
    spx_drop_threshold : 1-day SPX return threshold (e.g. -0.01 for -1%).
    spx_drop_2d_threshold : 2-day SPX return threshold.
    vix_spike_threshold : VIX % change threshold (e.g. 0.15 for 15%).
    vix_elevated_threshold : Absolute VIX level threshold.
    term_inversion_threshold : VIX/VIX9D ratio threshold.
    rally_avoidance : Whether to detect rallies for avoidance.
    rally_threshold : 1-day SPX return above this triggers rally signal.
    signal_mode : Filter applied to non-rally signals before returning.
        ``"any"`` keeps every fired signal (legacy live behaviour);
        ``"spx_and_vix"`` requires at least one spx_drop* AND at least
        one vix_* signal else drops all non-rally signals;
        ``"all"`` requires spx_drop*, vix_*, AND term_inversion together.
        Defaults to ``settings.event_signal_mode``.
    """

    def __init__(
        self,
        spx_drop_threshold: float | None = None,
        spx_drop_2d_threshold: float | None = None,
        vix_spike_threshold: float | None = None,
        vix_elevated_threshold: float | None = None,
        term_inversion_threshold: float | None = None,
        rally_avoidance: bool | None = None,
        rally_threshold: float | None = None,
        signal_mode: str | None = None,
    ) -> None:
        self.spx_drop_threshold = spx_drop_threshold if spx_drop_threshold is not None else settings.event_spx_drop_threshold
        self.spx_drop_2d_threshold = spx_drop_2d_threshold if spx_drop_2d_threshold is not None else settings.event_spx_drop_2d_threshold
        self.vix_spike_threshold = vix_spike_threshold if vix_spike_threshold is not None else settings.event_vix_spike_threshold
        self.vix_elevated_threshold = vix_elevated_threshold if vix_elevated_threshold is not None else settings.event_vix_elevated_threshold
        self.term_inversion_threshold = term_inversion_threshold if term_inversion_threshold is not None else settings.event_term_inversion_threshold
        self.rally_avoidance = rally_avoidance if rally_avoidance is not None else settings.event_rally_avoidance
        self.rally_threshold = rally_threshold if rally_threshold is not None else settings.event_rally_threshold
        # Normalize once at construction so _evaluate compares against a
        # known-good token; field_validator on the settings field already
        # rejects bad values, but explicit normalization keeps direct
        # caller-supplied values safe too.
        mode_raw = signal_mode if signal_mode is not None else settings.event_signal_mode
        self.signal_mode = (mode_raw or "any").strip().lower()

    async def detect(self, today: date | None = None) -> list[str]:
        """Return active signal names by reading recent underlying_quotes.

        Parameters
        ----------
        today : The date to evaluate. Defaults to today.

        Returns
        -------
        List of signal names (e.g. ``['spx_drop_1d', 'vix_spike']``).
        """
        if not settings.event_enabled:
            return []

        context = await self._load_context(today or date.today())
        return self._evaluate(context)

    def detect_from_dict(self, context: dict[str, float | None]) -> list[str]:
        """Synchronous evaluation for testing and backtesting.

        Parameters
        ----------
        context : Dict with keys ``prev_spx_return``, ``prev_spx_return_2d``,
            ``prev_vix_pct_change``, ``vix``, ``term_structure``.
        """
        return self._evaluate(context)

    def _evaluate(self, ctx: dict[str, Any]) -> list[str]:
        """Apply all threshold checks against the context dict.

        Parameters
        ----------
        ctx : Market context with lagged returns and current levels.

        Returns
        -------
        List of active signal names, after applying ``signal_mode`` gating.
        ``rally`` (when present) is always preserved because the decision
        path uses it for rally avoidance, not entry triggering.
        """
        signals: list[str] = []

        prev_ret = ctx.get("prev_spx_return")
        prev_ret_2d = ctx.get("prev_spx_return_2d")
        prev_2d_gap = ctx.get("prev_spx_return_2d_gap_days")
        prev_vix_chg = ctx.get("prev_vix_pct_change")
        vix = ctx.get("vix")
        ts = ctx.get("term_structure")

        if prev_ret is not None and prev_ret < self.spx_drop_threshold:
            signals.append("spx_drop_1d")
        # Adjacency guard: a "2-day return" computed across a 5+ calendar
        # day gap (e.g. data outage spanning a week) is not actually a
        # 2-day move and shouldn't fire the 2-day-drop event signal.
        if (
            prev_ret_2d is not None
            and prev_ret_2d < self.spx_drop_2d_threshold
            and (prev_2d_gap is None or prev_2d_gap <= _SPX_2D_MAX_CALENDAR_GAP_DAYS)
        ):
            signals.append("spx_drop_2d")
        elif prev_ret_2d is not None and prev_2d_gap is not None and prev_2d_gap > _SPX_2D_MAX_CALENDAR_GAP_DAYS:
            # Log so the operator can tell the difference between
            # "no signal because flat market" and "no signal because
            # gap was too wide".
            logger.warning(
                "event_signals: spx_drop_2d suppressed (calendar gap={} days exceeds {})",
                prev_2d_gap, _SPX_2D_MAX_CALENDAR_GAP_DAYS,
            )
        if prev_vix_chg is not None and prev_vix_chg > self.vix_spike_threshold:
            signals.append("vix_spike")
        if vix is not None and vix > self.vix_elevated_threshold:
            signals.append("vix_elevated")
        if ts is not None and ts > self.term_inversion_threshold:
            signals.append("term_inversion")
        if self.rally_avoidance and prev_ret is not None and prev_ret > self.rally_threshold:
            signals.append("rally")

        # Apply signal_mode filter.  Rally is always preserved because the
        # decision path uses it for avoidance, not entry.
        gated = self._apply_signal_mode(signals)

        if gated:
            logger.info("event_signals_active signals={} mode={}", gated, self.signal_mode)
        return gated

    def _apply_signal_mode(self, signals: list[str]) -> list[str]:
        """Return the subset of ``signals`` allowed by ``self.signal_mode``.

        ``rally`` is always preserved (governs rally-avoidance, not entries).
        Non-rally signals are kept or dropped as a group:

        * ``"any"`` -- pass through every signal.
        * ``"spx_and_vix"`` -- keep non-rally signals only if at least one
          ``spx_drop*`` AND at least one of ``vix_spike`` / ``vix_elevated``
          fired.  Mirrors the backtester's ``EventConfig.signal_mode``.
        * ``"all"`` -- keep non-rally signals only if all three classes
          (spx_drop, vix, term_inversion) fired.
        """
        rally_signals = [s for s in signals if s == "rally"]
        non_rally = [s for s in signals if s != "rally"]
        if self.signal_mode == "any" or not non_rally:
            return rally_signals + non_rally
        has_spx = any(s.startswith("spx_drop") for s in non_rally)
        has_vix = any(s in ("vix_spike", "vix_elevated") for s in non_rally)
        has_term = "term_inversion" in non_rally
        if self.signal_mode == "spx_and_vix":
            keep = has_spx and has_vix
        elif self.signal_mode == "all":
            keep = has_spx and has_vix and has_term
        else:
            # Defensive: validator should have caught unknown modes at
            # Settings load.  If a direct caller bypassed the validator,
            # be conservative and drop all non-rally signals.
            logger.warning("event_signals: unknown signal_mode={}; dropping non-rally signals", self.signal_mode)
            keep = False
        return rally_signals + (non_rally if keep else [])

    async def _load_context(self, today: date) -> dict[str, float | None]:
        """Read recent quote snapshots to build market context.

        Uses ``underlying_quotes`` rows for SPX and VIX from the last 3 days
        to compute returns and change metrics.

        Parameters
        ----------
        today : Current date.

        Returns
        -------
        Dict suitable for ``_evaluate()``.
        """
        lookback = today - timedelta(days=5)
        upper = today + timedelta(days=1)
        async with engine.connect() as conn:
            rows = await conn.execute(text(
                "SELECT symbol, date_trunc('day', ts)::date AS d, "
                "  (array_agg(last ORDER BY ts DESC))[1] AS close "
                "FROM underlying_quotes "
                "WHERE ts >= :lb AND ts < :upper "
                "  AND symbol IN ('SPX', 'VIX', 'VIX9D') "
                "GROUP BY symbol, d "
                "ORDER BY d"
            ), {"lb": lookback, "upper": upper})
            data = rows.fetchall()

        spx_by_day: dict[date, float] = {}
        vix_by_day: dict[date, float] = {}
        vix9d_by_day: dict[date, float] = {}

        for symbol, d, close in data:
            if symbol == "SPX":
                spx_by_day[d] = close
            elif symbol == "VIX":
                vix_by_day[d] = close
            elif symbol == "VIX9D":
                vix9d_by_day[d] = close

        sorted_spx_days = sorted(spx_by_day.keys())
        sorted_vix_days = sorted(vix_by_day.keys())

        prev_spx_return: float | None = None
        prev_spx_return_2d: float | None = None
        prev_spx_return_2d_gap_days: int | None = None
        prev_vix_pct_change: float | None = None
        vix_now: float | None = None
        term_structure: float | None = None

        if len(sorted_spx_days) >= 2:
            d1, d0 = sorted_spx_days[-2], sorted_spx_days[-1]
            prev_spx_return = (spx_by_day[d0] / spx_by_day[d1]) - 1
        if len(sorted_spx_days) >= 3:
            d2 = sorted_spx_days[-3]
            prev_spx_return_2d = (spx_by_day[sorted_spx_days[-1]] / spx_by_day[d2]) - 1
            # Surface the calendar gap so _evaluate can reject "2-day"
            # returns that actually span much longer windows due to data
            # outages.  Computed in calendar (not trading) days because
            # _SPX_2D_MAX_CALENDAR_GAP_DAYS is also calendar-based.
            prev_spx_return_2d_gap_days = (sorted_spx_days[-1] - d2).days

        if len(sorted_vix_days) >= 2:
            d1, d0 = sorted_vix_days[-2], sorted_vix_days[-1]
            prev_vix_pct_change = (vix_by_day[d0] / vix_by_day[d1]) - 1
        if sorted_vix_days:
            vix_now = vix_by_day[sorted_vix_days[-1]]

        latest_vix9d = vix9d_by_day.get(sorted_vix_days[-1]) if sorted_vix_days and sorted_vix_days[-1] in vix9d_by_day else None
        if vix_now is not None and latest_vix9d is not None and latest_vix9d > 0:
            term_structure = vix_now / latest_vix9d

        return {
            "prev_spx_return": prev_spx_return,
            "prev_spx_return_2d": prev_spx_return_2d,
            "prev_spx_return_2d_gap_days": prev_spx_return_2d_gap_days,
            "prev_vix_pct_change": prev_vix_pct_change,
            "vix": vix_now,
            "term_structure": term_structure,
        }
