"""Generate a unified economic-event calendar CSV for model training.

Audit Wave 4 / M9
-----------------
The hardcoded calendar tables now live in
``spx_backend.services.economic_calendar`` (so ``eod_events_job`` can
import them without a ``sys.path`` mutation). This script is now a
thin CLI wrapper over the service module: pass ``--output PATH`` to
control where the CSV is written.

Sources (originally curated for the calendar tables -- see service docstring):
* FOMC  – https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
* CPI   – https://www.bls.gov/schedule/news_release/cpi.htm
* NFP   – https://data.bls.gov/schedule/news_release/empsit.htm
* OpEx  – computed (3rd Friday of each month)

Usage
-----
    python backend/scripts/generate_economic_calendar.py
    python backend/scripts/generate_economic_calendar.py --output /tmp/cal.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

# CLI bootstrap: this is a standalone offline regeneration tool, not
# part of the runtime backend.  The runtime job
# (``spx_backend.jobs.eod_events_job``) imports ``generate_rows``
# directly from the service module without any ``sys.path`` games.
# This script, however, can be invoked as ``python backend/scripts/...``
# from the repo root before ``spx_backend`` is installed as a package
# (e.g. on a fresh checkout / CI worker), so we need the explicit
# insert here to make ``spx_backend`` importable in that context.
# Audit M9 / Refactor: the goal of removing ``sys.path`` mutations
# was achieved for the *runtime* code path; CLI wrappers like this
# one keep the bootstrap by necessity.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from spx_backend.services.economic_calendar import generate_rows  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)

DEFAULT_OUTPUT_CSV = _REPO_ROOT / "data" / "economic_calendar.csv"


def main() -> None:
    """Parse CLI arguments and write the economic calendar CSV."""
    parser = argparse.ArgumentParser(
        description="Generate a unified economic-event calendar CSV.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    args = parser.parse_args()
    output_csv: Path = args.output

    rows = generate_rows()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "event_type", "has_projections", "is_triple_witching"],
        )
        writer.writeheader()
        writer.writerows(rows)

    event_counts: dict[str, int] = {}
    for r in rows:
        event_counts[r["event_type"]] = event_counts.get(r["event_type"], 0) + 1

    logger.info("Wrote %d rows to %s", len(rows), output_csv)
    logger.info("Event counts: %s", event_counts)


if __name__ == "__main__":
    main()
