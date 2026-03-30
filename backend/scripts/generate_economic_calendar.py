"""Generate a unified economic-event calendar CSV for model training.

Consolidates FOMC decision dates, CPI release dates, NFP (Employment
Situation) release dates, and monthly options-expiration (OpEx) dates into
a single ``data/economic_calendar.csv``.

Sources
-------
* FOMC  – https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
* CPI   – https://www.bls.gov/schedule/news_release/cpi.htm
* NFP   – https://data.bls.gov/schedule/news_release/empsit.htm
* OpEx  – computed (3rd Friday of each month)

Usage
-----
    python -m backend.scripts.generate_economic_calendar
    # or directly:
    python backend/scripts/generate_economic_calendar.py
"""

from __future__ import annotations

import csv
import calendar
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = _ROOT / "data" / "economic_calendar.csv"

# ---------------------------------------------------------------------------
# FOMC decision dates (announcement day = day 2 of the 2-day meeting).
# Asterisk in the original calendar means a meeting with Summary of
# Economic Projections (SEP).  We track that in ``has_projections``.
# ---------------------------------------------------------------------------

_FOMC: dict[int, list[tuple[int, int, bool]]] = {
    # (month, day, has_projections)
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
        (1, 27, False), (3, 17, True), (4, 28, False), (6, 9, True),
        (7, 28, False), (9, 15, True), (10, 27, False), (12, 8, True),
    ],
}

# ---------------------------------------------------------------------------
# CPI release dates (from BLS schedule, including lapse-revised dates).
# Each entry is the release date for the *previous* month's data.
# ---------------------------------------------------------------------------

_CPI: dict[int, list[tuple[int, int]]] = {
    2024: [
        (1, 11), (2, 13), (3, 12), (4, 10), (5, 15), (6, 12),
        (7, 11), (8, 14), (9, 11), (10, 10), (11, 13), (12, 11),
    ],
    2025: [
        (1, 15), (2, 12), (3, 12), (4, 10), (5, 13), (6, 11),
        (7, 15), (8, 12), (9, 11), (10, 24), (12, 18),
        # Nov 13 canceled due to 2025 appropriations lapse;
        # Oct 24 and Dec 18 are revised dates from BLS.
    ],
    2026: [
        (1, 13), (2, 13), (3, 11), (4, 10), (5, 12), (6, 10),
        (7, 14), (8, 12), (9, 11), (10, 14), (11, 10), (12, 10),
    ],
}

# ---------------------------------------------------------------------------
# NFP / Employment Situation release dates.
# Confirmed from BLS schedule + search; any unconfirmed months use the
# standard first-Friday-of-month rule which is accurate to +/- 0 days
# for the vast majority of releases.
# ---------------------------------------------------------------------------

_NFP: dict[int, list[tuple[int, int]]] = {
    2024: [
        (1, 5), (2, 2), (3, 8), (4, 5), (5, 3), (6, 7),
        (7, 5), (8, 2), (9, 6), (10, 4), (11, 1), (12, 6),
    ],
    2025: [
        (1, 10), (2, 7), (3, 7), (4, 4), (5, 2), (6, 6),
        (7, 3), (8, 1), (9, 5), (10, 3), (11, 7), (12, 16),
        # Dec 16 revised due to 2025 appropriations lapse.
    ],
    2026: [
        (1, 9), (2, 11), (3, 6), (4, 3), (5, 8), (6, 5),
        (7, 2), (8, 7), (9, 4), (10, 2), (11, 6), (12, 4),
        # Feb 11 revised due to 2026 lapse.
    ],
}


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given month/year."""
    cal = calendar.monthcalendar(year, month)
    # Each week is [Mon..Sun].  Friday = index 4.
    fridays = [week[4] for week in cal if week[4] != 0]
    return date(year, month, fridays[2])


def _is_triple_witching(month: int) -> bool:
    """Triple/quadruple witching occurs on the 3rd Friday of Mar/Jun/Sep/Dec."""
    return month in (3, 6, 9, 12)


def generate_rows() -> list[dict]:
    """Build all calendar rows sorted by date.

    Returns a list of dicts with keys: date, event_type, has_projections,
    is_triple_witching.
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


def main() -> None:
    """Write the economic calendar CSV."""
    rows = generate_rows()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "event_type", "has_projections", "is_triple_witching"],
        )
        writer.writeheader()
        writer.writerows(rows)

    event_counts: dict[str, int] = {}
    for r in rows:
        event_counts[r["event_type"]] = event_counts.get(r["event_type"], 0) + 1

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    for etype, count in sorted(event_counts.items()):
        print(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
