"""Production event-signal detector for the event-driven trading layer.

Reads recent market context from the database and evaluates configurable
trigger thresholds to produce a list of active event signals for the
current trading day.

This module also exposes the **shared evaluator** consumed by the
backtest's ``EventSignalDetector`` so live and backtest evaluate
context-dicts through the exact same code path.  See OFFLINE_PIPELINE_AUDIT.md
(C1, H1, M3, M7, M9) for the divergence history that motivated
extracting ``evaluate_event_signals``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping

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


def compute_term_structure(vix9d: float | None, vix: float | None) -> float | None:
    """Return VIX9D / VIX as the canonical term-structure ratio.

    The ratio is > 1.0 when the 9-day implied vol exceeds the 30-day VIX
    (true backwardation / near-term vol stress), and < 1.0 in normal
    contango.  Every other component of the system uses this convention:
    DB writers (``quote_job.compute_market_context`` and
    ``cboe_gex_job._latest_vols``), the offline training pipeline
    (``generate_training_data._build_underlying_quote_frame``), and the
    backtest detector that reads ``term_structure`` directly from the
    candidate row.

    Historically this helper was inlined in ``_load_context`` with the
    operands inverted (``vix / vix9d``); that bug caused ``term_inversion``
    to fire on contango rather than backwardation -- the inverse of what
    every other path treats as "inversion".  Centralising the formula
    makes the convention explicit and gives tests a pure-function target.

    Parameters
    ----------
    vix9d:
        Latest 9-day implied vol (CBOE VIX9D), or ``None`` if unavailable.
    vix:
        Latest 30-day implied vol (CBOE VIX), or ``None`` if unavailable.

    Returns
    -------
    Ratio ``vix9d / vix`` as a float, or ``None`` if either input is
    missing or ``vix`` is non-positive (avoids div-by-zero).
    """
    if vix9d is None or vix is None or vix <= 0:
        return None
    return vix9d / vix


# ---------------------------------------------------------------------------
# Shared evaluator -- consumed by both the live ``EventSignalDetector`` below
# AND the backtest's ``EventSignalDetector`` in
# ``backend/scripts/backtest_strategy.py``.  Keeping the evaluation logic in a
# single pure function permanently fixes the C1+H1+M3+M7 divergence cluster
# documented in OFFLINE_PIPELINE_AUDIT.md.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventThresholds:
    """All threshold + mode parameters needed to evaluate event signals.

    A pure-data carrier so live (driven by ``Settings``) and backtest
    (driven by the optimizer's ``EventConfig``) can both build one and
    feed it to ``evaluate_event_signals`` without sharing a class
    hierarchy.

    Field semantics match the ``Settings.event_*`` fields in
    ``spx_backend/config.py`` and the ``EventConfig`` dataclass in
    ``backend/scripts/backtest_strategy.py``:

    * ``spx_drop_threshold``     -- 1-day SPX return *strictly less than*
                                    this fires ``spx_drop_1d`` (decimal,
                                    e.g. ``-0.01`` = 1% drop).
    * ``spx_drop_2d_threshold``  -- 2-day SPX return strictly less than
                                    this fires ``spx_drop_2d``.
    * ``vix_spike_threshold``    -- prev-day VIX % change strictly greater
                                    than this fires ``vix_spike``.
    * ``vix_elevated_threshold`` -- absolute VIX level strictly greater
                                    than this fires ``vix_elevated``.
    * ``term_inversion_threshold`` -- ``vix9d / vix`` strictly greater
                                      than this fires ``term_inversion``.
    * ``rally_avoidance``        -- toggles the rally signal entirely.
    * ``rally_threshold``        -- prev-day SPX return strictly greater
                                    than this fires ``rally``.
    * ``signal_mode``            -- one of ``"any"``, ``"spx_and_vix"``,
                                    ``"all"``; gates the non-rally
                                    signals as a group.
    * ``spx_drop_min`` / ``spx_drop_max`` (optional) -- M9 magnitude
      window: when set, ``spx_drop_1d`` only fires when the prev-day
      return *also* lies in ``[min, max]``.  ``None`` disables the gate.
      Live's ``Settings`` does not yet expose these fields, so the live
      detector always passes ``None`` here -- the gate is currently a
      backtest-only optimizer knob (see M9 in OFFLINE_PIPELINE_AUDIT.md).
    """

    spx_drop_threshold: float
    spx_drop_2d_threshold: float
    vix_spike_threshold: float
    vix_elevated_threshold: float
    term_inversion_threshold: float
    rally_avoidance: bool
    rally_threshold: float
    signal_mode: str
    spx_drop_min: float | None = None
    spx_drop_max: float | None = None


def evaluate_event_signals(
    ctx: Mapping[str, Any],
    thresholds: EventThresholds,
    *,
    log_warnings: bool = True,
) -> list[str]:
    """Pure-function event-signal evaluator (live + backtest shared).

    Given a market-context dict and a set of thresholds, return the
    list of active signal names (after applying the ``signal_mode``
    gate).  The function is side-effect free except for an optional
    diagnostic ``logger.warning`` when the ``spx_drop_2d`` signal is
    suppressed by the H1 calendar-gap guard.

    Parameters
    ----------
    ctx :
        Mapping with the following optional keys (any may be ``None`` or
        absent; the corresponding signal then never fires):

        * ``prev_spx_return``           -- 1-day SPX return as decimal.
        * ``prev_spx_return_2d``        -- 2-day SPX return as decimal.
        * ``prev_spx_return_2d_gap_days`` -- calendar-day span of the
          2-day window.  When > ``_SPX_2D_MAX_CALENDAR_GAP_DAYS`` the
          ``spx_drop_2d`` signal is suppressed (H1).  ``NaN`` / ``None``
          disables the guard.
        * ``prev_vix_pct_change``       -- prev-day VIX % change as decimal.
        * ``vix``                       -- current VIX level.
        * ``term_structure``            -- ``vix9d / vix`` ratio.

    thresholds :
        ``EventThresholds`` describing the signal-firing thresholds and
        the ``signal_mode`` gate.

    log_warnings :
        When ``True`` (default), emit a ``logger.warning`` if the
        ``spx_drop_2d`` signal is suppressed by the calendar-gap guard.
        The backtest passes ``False`` to keep the per-day loop quiet
        (the diagnostic is more useful in live where the gap is
        unexpected; backtest's gap distribution is dataset-wide and
        shouldn't spam the optimizer log).

    Returns
    -------
    list[str]
        Active signal names after ``signal_mode`` gating.  Order is
        deterministic: ``spx_drop_1d`` -> ``spx_drop_2d`` ->
        ``vix_spike`` -> ``vix_elevated`` -> ``term_inversion`` ->
        ``rally`` (rally is always preserved, even when the gate
        drops the other signals).
    """
    signals: list[str] = []

    prev_ret = ctx.get("prev_spx_return")
    prev_ret_2d = ctx.get("prev_spx_return_2d")
    prev_2d_gap = ctx.get("prev_spx_return_2d_gap_days")
    prev_vix_chg = ctx.get("prev_vix_pct_change")
    vix = ctx.get("vix")
    ts = ctx.get("term_structure")

    # spx_drop_1d -- with optional M9 magnitude-window gate.
    if prev_ret is not None and prev_ret < thresholds.spx_drop_threshold:
        in_range = True
        # When ``spx_drop_min`` is set, the return must also be at or
        # below it; this is the floor (more negative = bigger drop, so
        # ``prev_ret < dmin`` means "drop too small to count").
        if thresholds.spx_drop_min is not None and prev_ret < thresholds.spx_drop_min:
            in_range = False
        # ``spx_drop_max`` is the ceiling (less negative); ``prev_ret > dmax``
        # means "drop too big to count".  Together they bound the
        # fired-on magnitude window.
        if thresholds.spx_drop_max is not None and prev_ret > thresholds.spx_drop_max:
            in_range = False
        if in_range:
            signals.append("spx_drop_1d")

    # spx_drop_2d -- with H1 calendar-gap guard.
    # ``prev_2d_gap`` may be NaN (insufficient history) or None (key
    # not surfaced) -- in either case fall through with no guard,
    # which matches both live (None pre-H1) and backtest (NaN at the
    # head of the daily series).
    gap_ok = (
        prev_2d_gap is None
        or _is_nan_safe(prev_2d_gap)
        or prev_2d_gap <= _SPX_2D_MAX_CALENDAR_GAP_DAYS
    )
    if (
        prev_ret_2d is not None
        and prev_ret_2d < thresholds.spx_drop_2d_threshold
        and gap_ok
    ):
        signals.append("spx_drop_2d")
    elif (
        log_warnings
        and prev_ret_2d is not None
        and prev_ret_2d < thresholds.spx_drop_2d_threshold
        and prev_2d_gap is not None
        and not _is_nan_safe(prev_2d_gap)
        and prev_2d_gap > _SPX_2D_MAX_CALENDAR_GAP_DAYS
    ):
        logger.warning(
            "event_signals: spx_drop_2d suppressed (calendar gap={} days exceeds {})",
            prev_2d_gap, _SPX_2D_MAX_CALENDAR_GAP_DAYS,
        )

    if prev_vix_chg is not None and prev_vix_chg > thresholds.vix_spike_threshold:
        signals.append("vix_spike")
    if vix is not None and vix > thresholds.vix_elevated_threshold:
        signals.append("vix_elevated")
    if ts is not None and ts > thresholds.term_inversion_threshold:
        signals.append("term_inversion")
    if (
        thresholds.rally_avoidance
        and prev_ret is not None
        and prev_ret > thresholds.rally_threshold
    ):
        signals.append("rally")

    return _apply_signal_mode(signals, thresholds.signal_mode)


def _is_nan_safe(value: Any) -> bool:
    """Return True if ``value`` is a float NaN.

    Tolerant of non-float types (dates, strings) so callers don't have
    to guard before calling.  ``math.isnan`` would raise on those;
    ``pandas.isna`` works but we don't want to import pandas in this
    module just for this helper.
    """
    try:
        return value != value  # NaN is the only value that fails self-equality
    except (TypeError, ValueError):
        return False


def _apply_signal_mode(signals: list[str], mode: str) -> list[str]:
    """Return the subset of ``signals`` allowed by ``mode``.

    Pure helper extracted from the original
    ``EventSignalDetector._apply_signal_mode`` so the backtest path can
    reuse the exact same gating logic.

    * ``"any"`` -- pass through every signal.
    * ``"spx_and_vix"`` -- keep non-rally signals only if at least one
      ``spx_drop*`` AND at least one of ``vix_spike`` / ``vix_elevated``
      fired.
    * ``"all"`` -- keep non-rally signals only if all three classes
      (spx_drop, vix, term_inversion) fired.
    * any other value -- defensive fallback: drop all non-rally
      signals.  The settings validator should catch unknown modes at
      load time, but this keeps a direct caller safe.

    ``rally`` is always preserved (it governs rally-avoidance, not entry).
    """
    mode = (mode or "any").strip().lower()

    rally_signals = [s for s in signals if s == "rally"]
    non_rally = [s for s in signals if s != "rally"]

    if mode == "any" or not non_rally:
        return non_rally + rally_signals

    has_spx = any(s.startswith("spx_drop") for s in non_rally)
    has_vix = any(s in ("vix_spike", "vix_elevated") for s in non_rally)
    has_term = "term_inversion" in non_rally
    if mode == "spx_and_vix":
        keep = has_spx and has_vix
    elif mode == "all":
        keep = has_spx and has_vix and has_term
    else:
        logger.warning(
            "event_signals: unknown signal_mode={}; dropping non-rally signals", mode,
        )
        keep = False
    return (non_rally if keep else []) + rally_signals


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

    def _build_thresholds(self) -> EventThresholds:
        """Pack the detector's instance fields into an ``EventThresholds``.

        Centralised here so any future settings additions (e.g. an
        ``event_spx_drop_min`` knob) only need to be wired in one place.
        """
        return EventThresholds(
            spx_drop_threshold=self.spx_drop_threshold,
            spx_drop_2d_threshold=self.spx_drop_2d_threshold,
            vix_spike_threshold=self.vix_spike_threshold,
            vix_elevated_threshold=self.vix_elevated_threshold,
            term_inversion_threshold=self.term_inversion_threshold,
            rally_avoidance=self.rally_avoidance,
            rally_threshold=self.rally_threshold,
            signal_mode=self.signal_mode,
            # Live settings do not yet expose the M9 magnitude window;
            # leave None so the live path matches its pre-refactor
            # behaviour exactly.
            spx_drop_min=None,
            spx_drop_max=None,
        )

    def _evaluate(self, ctx: dict[str, Any]) -> list[str]:
        """Apply all threshold checks against the context dict.

        Thin wrapper that delegates to the module-level
        ``evaluate_event_signals`` so the live and backtest paths share
        a single evaluation implementation (see C1+H1+M3+M7 in
        OFFLINE_PIPELINE_AUDIT.md).
        """
        gated = evaluate_event_signals(ctx, self._build_thresholds())
        if gated:
            logger.info("event_signals_active signals={} mode={}", gated, self.signal_mode)
        return gated

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
        # Canonical orientation is vix9d / vix (see compute_term_structure
        # docstring).  The previous formula was inverted, causing
        # term_inversion to fire on contango.
        term_structure = compute_term_structure(latest_vix9d, vix_now)

        return {
            "prev_spx_return": prev_spx_return,
            "prev_spx_return_2d": prev_spx_return_2d,
            "prev_spx_return_2d_gap_days": prev_spx_return_2d_gap_days,
            "prev_vix_pct_change": prev_vix_pct_change,
            "vix": vix_now,
            "term_structure": term_structure,
        }
