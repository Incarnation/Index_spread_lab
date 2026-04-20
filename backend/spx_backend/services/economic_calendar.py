"""Economic-event calendar source of truth (FOMC / CPI / NFP / OPEX).

Audit Wave 4 / M9
-----------------
The hardcoded calendar tables previously lived only in
``backend/scripts/generate_economic_calendar.py``. The runtime job
``eod_events_job._generate_rows`` reached for them via a
``sys.path.insert`` hack and a relative import of the script module.
That coupled production behavior to a CLI script's import side
effects (logging.basicConfig at import, etc.) and made the data
unimportable from anywhere else without re-doing the path mutation.

This module hosts the canonical calendar generator and re-exports it
for both consumers: ``eod_events_job`` (now imports directly from
``spx_backend.services.economic_calendar``) and the legacy CSV
generator script (which now thinly wraps this module).

Adding a year? Append to ``_FOMC`` / ``_CPI`` / ``_NFP`` here. The
script wrapper picks up the change automatically.
"""
from __future__ import annotations

import calendar
from datetime import date


# FOMC decision dates (announcement day = day 2 of the 2-day meeting).
# Asterisk in the original Federal Reserve calendar means a meeting
# with Summary of Economic Projections (SEP); we track that in
# ``has_projections``.
_FOMC: dict[int, list[tuple[int, int, bool]]] = {
    2021: [
        (1, 27, False), (3, 17, True), (4, 28, False), (6, 16, True),
        (7, 28, False), (9, 22, True), (11, 3, False), (12, 15, True),
    ],
    2022: [
        (1, 26, False), (3, 16, True), (5, 4, False), (6, 15, True),
        (7, 27, False), (9, 21, True), (11, 2, False), (12, 14, True),
    ],
    2023: [
        (2, 1, False), (3, 22, True), (5, 3, False), (6, 14, True),
        (7, 26, False), (9, 20, True), (11, 1, False), (12, 13, True),
    ],
    2024: [
        (1, 31, False), (3, 20, True), (5, 1, False), (6, 12, True),
        (7, 31, False), (9, 18, True), (11, 7, False), (12, 18, True),
    ],
    2025: [
        (1, 29, False), (3, 19, True), (5, 7, False), (6, 18, True),
        (7, 30, False), (9, 17, True), (10, 29, False), (12, 10, True),
    ],
    2026: [
        (1, 28, False), (3, 18, True), (4, 29, False), (6, 17, True),
        (7, 29, False), (9, 16, True), (10, 28, False), (12, 9, True),
    ],
    2027: [
        (1, 27, False), (3, 17, True), (4, 28, False), (6, 16, True),
        (7, 28, False), (9, 22, True), (11, 3, False), (12, 15, True),
    ],
}

# CPI release dates (BLS calendar, month-of-release, release-day).
_CPI: dict[int, list[tuple[int, int]]] = {
    2021: [(1, 13), (2, 10), (3, 10), (4, 13), (5, 12), (6, 10),
           (7, 13), (8, 11), (9, 14), (10, 13), (11, 10), (12, 10)],
    2022: [(1, 12), (2, 10), (3, 10), (4, 12), (5, 11), (6, 10),
           (7, 13), (8, 10), (9, 13), (10, 13), (11, 10), (12, 13)],
    2023: [(1, 12), (2, 14), (3, 14), (4, 12), (5, 10), (6, 13),
           (7, 12), (8, 10), (9, 13), (10, 12), (11, 14), (12, 12)],
    2024: [(1, 11), (2, 13), (3, 12), (4, 10), (5, 15), (6, 12),
           (7, 11), (8, 14), (9, 11), (10, 10), (11, 13), (12, 11)],
    2025: [(1, 15), (2, 12), (3, 12), (4, 10), (5, 13), (6, 11),
           (7, 15), (8, 12), (9, 11), (10, 15), (11, 13), (12, 10)],
    2026: [(1, 14), (2, 11), (3, 11), (4, 14), (5, 13), (6, 10),
           (7, 14), (8, 12), (9, 10), (10, 14), (11, 12), (12, 10)],
    2027: [(1, 13), (2, 10), (3, 10), (4, 13), (5, 12), (6, 10),
           (7, 13), (8, 11), (9, 14), (10, 13), (11, 10), (12, 10)],
}

# NFP / Employment Situation release dates (typically first Friday of month).
_NFP: dict[int, list[tuple[int, int]]] = {
    2021: [(1, 8), (2, 5), (3, 5), (4, 2), (5, 7), (6, 4),
           (7, 2), (8, 6), (9, 3), (10, 8), (11, 5), (12, 3)],
    2022: [(1, 7), (2, 4), (3, 4), (4, 1), (5, 6), (6, 3),
           (7, 8), (8, 5), (9, 2), (10, 7), (11, 4), (12, 2)],
    2023: [(1, 6), (2, 3), (3, 10), (4, 7), (5, 5), (6, 2),
           (7, 7), (8, 4), (9, 1), (10, 6), (11, 3), (12, 8)],
    2024: [(1, 5), (2, 2), (3, 8), (4, 5), (5, 3), (6, 7),
           (7, 5), (8, 2), (9, 6), (10, 4), (11, 1), (12, 6)],
    2025: [(1, 10), (2, 7), (3, 7), (4, 4), (5, 2), (6, 6),
           (7, 3), (8, 1), (9, 5), (10, 3), (11, 7), (12, 5)],
    2026: [(1, 9), (2, 6), (3, 6), (4, 3), (5, 8), (6, 5),
           (7, 2), (8, 7), (9, 4), (10, 2), (11, 6), (12, 4)],
    2027: [(1, 8), (2, 5), (3, 5), (4, 2), (5, 7), (6, 4),
           (7, 2), (8, 6), (9, 3), (10, 8), (11, 5), (12, 3)],
}


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month/year.

    Used as the OPEX (monthly options expiration) anchor; ``calendar``
    enumerates each week as ``[Mon..Sun]`` so Friday is index 4.
    """
    cal = calendar.monthcalendar(year, month)
    fridays = [week[4] for week in cal if week[4] != 0]
    return date(year, month, fridays[2])


def _is_triple_witching(month: int) -> bool:
    """Triple/quadruple witching occurs on the 3rd Friday of Mar/Jun/Sep/Dec."""
    return month in (3, 6, 9, 12)


def generate_rows() -> list[dict]:
    """Build all calendar rows sorted by date.

    Returns
    -------
    list[dict]
        Each dict has keys ``date`` (ISO ``YYYY-MM-DD`` string),
        ``event_type`` (``"FOMC"`` / ``"CPI"`` / ``"NFP"`` / ``"OPEX"``),
        ``has_projections`` (bool, only true for FOMC SEP meetings),
        and ``is_triple_witching`` (bool, true for OPEX in Mar/Jun/Sep/Dec).
        Output is byte-identical to the legacy
        ``backend/scripts/generate_economic_calendar.generate_rows`` so
        downstream CSV consumers see no diff.
    """
    rows: list[dict] = []

    for year, meetings in _FOMC.items():
        for month, day, has_sep in meetings:
            rows.append({
                "date": date(year, month, day).isoformat(),
                "event_type": "FOMC",
                "has_projections": has_sep,
                "is_triple_witching": False,
            })

    for year, releases in _CPI.items():
        for month, day in releases:
            rows.append({
                "date": date(year, month, day).isoformat(),
                "event_type": "CPI",
                "has_projections": False,
                "is_triple_witching": False,
            })

    for year, releases in _NFP.items():
        for month, day in releases:
            rows.append({
                "date": date(year, month, day).isoformat(),
                "event_type": "NFP",
                "has_projections": False,
                "is_triple_witching": False,
            })

    for year in range(2021, 2028):
        for month in range(1, 13):
            opex = _third_friday(year, month)
            rows.append({
                "date": opex.isoformat(),
                "event_type": "OPEX",
                "has_projections": False,
                "is_triple_witching": _is_triple_witching(month),
            })

    rows.sort(key=lambda r: r["date"])
    return rows
