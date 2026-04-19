"""Download historical options data from Databento for SPX and SPY.

Modes
-----

``--phase sample``
    Download 1 day of every configured job via the **streaming** API to
    verify schema/columns (fast, ~1 GB).  Streaming returns a Pandas
    DataFrame in-memory, so the output is **always Parquet**, regardless
    of dataset (OPRA, DBEQ, etc.).

``--phase full``
    Download the full date range.  OPRA jobs go through the **batch**
    API (split-by-day ``.dbn.zst`` files, ~10 GB total); the small
    supporting DBEQ job (SPY equity OHLCV) uses streaming and writes a
    single Parquet covering the whole range.

``--verify``
    Load every Parquet file under ``data/databento/`` and print summary
    stats (record counts, column names, date ranges).  Useful after a
    sample run to sanity-check the layout before kicking off ``full``.

``--verify-dbn``
    Scan ``.dbn.zst`` files and check for date gaps versus the trading
    calendar (L1 holiday list).  Only relevant after ``--phase full``.

On-disk layout
--------------

The exact file format and naming **depends on which mode produced the
data**, because the streaming and batch APIs serialize differently.
``--verify`` glob-walks both, but operators inspecting a subdirectory by
hand should expect the following:

* ``--phase full`` (OPRA via batch, M12 audit fix) ::

      data/databento/{spx,spy}/{cbbo-1m,definition,statistics}/YYYYMMDD.dbn.zst
      data/databento/{spx,spy}/{...}/_jobs/<job_id>.json   # batch metadata

  One ``.dbn.zst`` per *trading day*; ``YYYYMMDD`` is the session date.

* ``--phase sample`` (any dataset via streaming, M12 audit fix) ::

      data/databento/{spx,spy}/{cbbo-1m,definition,statistics}/{start}_{end}.parquet

  A **single Parquet** spanning the sample window (``SAMPLE_START`` to
  ``SAMPLE_END`` -- a one-trading-day range by default).  The filename
  uses ``YYYY-MM-DD_YYYY-MM-DD.parquet``, *not* the per-day
  ``YYYYMMDD.dbn.zst`` layout that ``--phase full`` produces.  This is
  intentional: streaming returns DataFrames, so writing Parquet here
  avoids a needless ``.dbn.zst`` round-trip.

* DBEQ underlying (SPY equity OHLCV) -- always streaming, in **either**
  mode ::

      data/databento/underlying/spy_equity_1m.parquet

  A single Parquet spanning the full requested range; the filename comes
  from the job's ``filename`` override rather than the ``start_end``
  template.

Operators inspecting ``data/databento/spy/cbbo-1m/`` after a ``--phase
sample`` run will therefore see a Parquet (e.g.
``2025-01-15_2025-01-16.parquet``), **not** a ``.dbn.zst``.  Both
layouts are intentional and supported by ``--verify``.

Usage
-----

::

    python -m backend.scripts.download_databento --phase sample
    python -m backend.scripts.download_databento --phase full
    python -m backend.scripts.download_databento --verify
    python -m backend.scripts.download_databento --verify-dbn

See ``OFFLINE_PIPELINE_AUDIT.md`` (M12) for the rationale behind the
per-mode layout and why we did not unify on ``.dbn.zst`` for sample
mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


def require_databento_api_key() -> str:
    """Return the Databento API key from the environment.

    Keys must never be hardcoded or committed. Set ``DATABENTO_API_KEY`` in
    the environment or in a local ``.env`` file (gitignored).

    Returns
    -------
    str
        Non-empty API key string.

    Raises
    ------
    SystemExit
        If the variable is missing or empty.
    """
    key = (os.getenv("DATABENTO_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "DATABENTO_API_KEY is not set. Export it in your shell or add it to "
            ".env (see .env.example). Never commit API keys to the repository."
        )
    return key


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_databento_dir() -> Path:
    """Return the Databento data directory, env-driven when available.

    L5 fix: prefer ``settings.databento_dir`` so operators can relocate
    the 120 GB tree to dedicated storage without code changes.  Falls
    back to the historical ``data/databento`` location when the
    spx_backend package can't be imported (offline environments) or
    the setting is left at its default.

    Relative paths are anchored to the repo root; absolute paths are
    used as-is so an external SSD / NFS mount can be specified directly
    via env (``DATABENTO_DIR=/mnt/ssd/databento``).
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        from spx_backend.config import settings as _live  # type: ignore
        configured = Path(_live.databento_dir)
    except Exception:  # pragma: no cover -- import optional offline
        configured = Path("data/databento")
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured


DATA_DIR = _resolve_databento_dir()
META_DIR = DATA_DIR / "_meta"

# Static NYSE holiday calendar covering 2020–2030.  Sourced from the
# official NYSE holiday schedule and used both for forward downloads
# (2026 production) and backfills (2020–2024 historical reruns).  See
# L1 in OFFLINE_PIPELINE_AUDIT.md — the previous 2026-only set silently
# treated all other-year holidays as trading days.
#
# Half-day early closes (day-after-Thanksgiving, Christmas Eve when
# market opens, July 3rd in some years) are NOT included here because
# Databento still has data for those days; the function only excludes
# full-closure days.
US_EQUITY_HOLIDAYS = frozenset({
    # 2020
    date(2020, 1, 1),  date(2020, 1, 20), date(2020, 2, 17),
    date(2020, 4, 10), date(2020, 5, 25), date(2020, 7, 3),
    date(2020, 9, 7),  date(2020, 11, 26), date(2020, 12, 25),
    # 2021
    date(2021, 1, 1),  date(2021, 1, 18), date(2021, 2, 15),
    date(2021, 4, 2),  date(2021, 5, 31), date(2021, 7, 5),
    date(2021, 9, 6),  date(2021, 11, 25), date(2021, 12, 24),
    # 2022
    date(2022, 1, 17), date(2022, 2, 21), date(2022, 4, 15),
    date(2022, 5, 30), date(2022, 6, 20), date(2022, 7, 4),
    date(2022, 9, 5),  date(2022, 11, 24), date(2022, 12, 26),
    # 2023
    date(2023, 1, 2),  date(2023, 1, 16), date(2023, 2, 20),
    date(2023, 4, 7),  date(2023, 5, 29), date(2023, 6, 19),
    date(2023, 7, 4),  date(2023, 9, 4),  date(2023, 11, 23),
    date(2023, 12, 25),
    # 2024
    date(2024, 1, 1),  date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4),  date(2024, 9, 2),  date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1),  date(2025, 1, 9),  date(2025, 1, 20),
    date(2025, 2, 17), date(2025, 4, 18), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4),  date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3),  date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3),  date(2026, 9, 7),  date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1),  date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5),  date(2027, 9, 6),  date(2027, 11, 25),
    date(2027, 12, 24),
    # 2028
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14),
    date(2028, 5, 29), date(2028, 6, 19), date(2028, 7, 4),
    date(2028, 9, 4),  date(2028, 11, 23), date(2028, 12, 25),
    # 2029
    date(2029, 1, 1),  date(2029, 1, 15), date(2029, 2, 19),
    date(2029, 3, 30), date(2029, 5, 28), date(2029, 6, 19),
    date(2029, 7, 4),  date(2029, 9, 3),  date(2029, 11, 22),
    date(2029, 12, 25),
    # 2030
    date(2030, 1, 1),  date(2030, 1, 21), date(2030, 2, 18),
    date(2030, 4, 19), date(2030, 5, 27), date(2030, 6, 19),
    date(2030, 7, 4),  date(2030, 9, 2),  date(2030, 11, 28),
    date(2030, 12, 25),
})

# Backwards-compat alias preserved so any external scripts still
# importing the old name continue to work; new code should use the
# year-agnostic ``US_EQUITY_HOLIDAYS`` constant.
US_EQUITY_HOLIDAYS_2026 = US_EQUITY_HOLIDAYS

DOWNLOAD_JOBS: list[dict] = [
    {
        "label": "SPX cbbo-1m",
        "group": "spx",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPX.OPT",
        "stype_in": "parent",
        "schema": "cbbo-1m",
        "subdir": "spx/cbbo-1m",
    },
    {
        "label": "SPX definition",
        "group": "spx",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPX.OPT",
        "stype_in": "parent",
        "schema": "definition",
        "subdir": "spx/definition",
    },
    {
        "label": "SPX statistics",
        "group": "spx",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPX.OPT",
        "stype_in": "parent",
        "schema": "statistics",
        "subdir": "spx/statistics",
    },
    {
        "label": "SPXW cbbo-1m",
        "group": "spxw",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPXW.OPT",
        "stype_in": "parent",
        "schema": "cbbo-1m",
        "subdir": "spxw/cbbo-1m",
    },
    {
        "label": "SPXW definition",
        "group": "spxw",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPXW.OPT",
        "stype_in": "parent",
        "schema": "definition",
        "subdir": "spxw/definition",
    },
    {
        "label": "SPXW statistics",
        "group": "spxw",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPXW.OPT",
        "stype_in": "parent",
        "schema": "statistics",
        "subdir": "spxw/statistics",
    },
    {
        "label": "SPY cbbo-1m",
        "group": "spy_options",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPY.OPT",
        "stype_in": "parent",
        "schema": "cbbo-1m",
        "subdir": "spy/cbbo-1m",
    },
    {
        "label": "SPY definition",
        "group": "spy_options",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPY.OPT",
        "stype_in": "parent",
        "schema": "definition",
        "subdir": "spy/definition",
    },
    {
        "label": "SPY statistics",
        "group": "spy_options",
        "dataset": "OPRA.PILLAR",
        "symbols": "SPY.OPT",
        "stype_in": "parent",
        "schema": "statistics",
        "subdir": "spy/statistics",
    },
    {
        "label": "SPY equity 1m",
        "group": "spy_equity",
        "dataset": "DBEQ.BASIC",
        "symbols": "SPY",
        "stype_in": "raw_symbol",
        "schema": "ohlcv-1m",
        "subdir": "underlying",
        "filename": "spy_equity_1m",
    },
]

DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-03-10"
SAMPLE_START = "2026-01-02"
SAMPLE_END = "2026-01-03"


def _ensure_dir(path: Path) -> Path:
    """Create directory and parents if they don't exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_trading_days(start_str: str, end_str: str) -> list[date]:
    """Return a sorted list of US equity trading days in [start, end).

    Excludes weekends and the static NYSE holiday set in
    ``US_EQUITY_HOLIDAYS`` (covers 2020–2030; see L1 in
    OFFLINE_PIPELINE_AUDIT.md).  The end date is exclusive, matching
    Databento's query convention.

    Half-day early closes are treated as trading days because
    Databento publishes data for the partial session.  Years outside
    2020–2030 will only excludes weekends — extend
    ``US_EQUITY_HOLIDAYS`` if you backfill earlier or later than that
    window.

    Args:
        start_str: Start date as YYYY-MM-DD (inclusive).
        end_str: End date as YYYY-MM-DD (exclusive).

    Returns:
        Sorted list of date objects representing trading days.
    """
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    if start.year < 2020 or end.year > 2031:
        # Surface a warning so an operator backfilling 2018 or
        # forward-projecting 2032 doesn't silently get rows for
        # holiday closures the static set doesn't cover yet.
        print(f"WARNING: get_trading_days({start_str}, {end_str}) extends "
              "outside the 2020-2030 holiday calendar — extend "
              "US_EQUITY_HOLIDAYS in download_databento.py to cover "
              "the requested range or expect spurious 'trading days' "
              "on actual closures.", flush=True)
    days = []
    d = start
    while d < end:
        if d.weekday() < 5 and d not in US_EQUITY_HOLIDAYS:
            days.append(d)
        d += timedelta(days=1)
    return days


def get_existing_dates(directory: Path) -> set[str]:
    """Scan a directory for YYYYMMDD.dbn.zst files and return the date strings.

    Args:
        directory: Path to scan for .dbn.zst files.

    Returns:
        Set of date strings like {'20260102', '20260103', ...}.
    """
    dates = set()
    if not directory.exists():
        return dates
    for f in directory.iterdir():
        m = re.match(r"^(\d{8})\.dbn\.zst$", f.name)
        if m:
            dates.add(m.group(1))
    return dates


def _download_streaming(
    client: db.Historical,
    job: dict,
    start: str,
    end: str,
) -> Path:
    """Download via streaming API and save as Parquet.

    Best for small requests like SPY equity OHLCV or single-day samples.

    Args:
        client: Authenticated Databento Historical client.
        job: Download job config dict with dataset, symbols, schema, subdir keys.
        start: Start date YYYY-MM-DD.
        end: End date YYYY-MM-DD.

    Returns:
        Path to the output Parquet file.
    """
    out_dir = _ensure_dir(DATA_DIR / job["subdir"])
    fname = job.get("filename", f"{start}_{end}")
    out_path = out_dir / f"{fname}.parquet"

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"    [skip] {out_path.name} already exists ({size_mb:.1f} MB)")
        return out_path

    print(f"    Downloading via streaming API...", flush=True)
    data = client.timeseries.get_range(
        dataset=job["dataset"],
        symbols=job["symbols"],
        stype_in=job["stype_in"],
        schema=job["schema"],
        start=start,
        end=end,
    )
    df = data.to_df()

    if len(df) == 0:
        logger.warning("No records returned for %s (%s to %s)", job["label"], start, end)
        return out_path

    df.to_parquet(out_path, engine="pyarrow")
    size_mb = out_path.stat().st_size / 1e6
    print(f"    [done] {out_path.name}: {len(df):,} records, {size_mb:.1f} MB")
    return out_path


def _download_batch(
    client: db.Historical,
    job: dict,
    start: str,
    end: str,
) -> Path:
    """Download via batch API, keeping per-day .dbn.zst files with YYYYMMDD naming.

    Submits a batch job to Databento, polls until complete, downloads the
    split-by-day .dbn.zst files into the organized directory, renames them to
    the simplified YYYYMMDD.dbn.zst format, and archives job metadata.

    Existing dates are skipped: if all expected dates already have files locally,
    the batch submission is skipped entirely.

    Args:
        client: Authenticated Databento Historical client.
        job: Download job config dict.
        start: Start date YYYY-MM-DD.
        end: End date YYYY-MM-DD.

    Returns:
        Path to the output directory.
    """
    out_dir = _ensure_dir(DATA_DIR / job["subdir"])
    existing = get_existing_dates(out_dir)
    expected = {d.strftime("%Y%m%d") for d in get_trading_days(start, end)}
    missing = expected - existing

    if not missing:
        print(f"    [skip] All {len(expected)} dates already present in {job['subdir']}")
        return out_dir

    print(
        f"    {len(existing)} dates cached, {len(missing)} missing. "
        f"Submitting batch job...",
        flush=True,
    )

    batch_result = client.batch.submit_job(
        dataset=job["dataset"],
        symbols=job["symbols"],
        stype_in=job["stype_in"],
        schema=job["schema"],
        encoding="dbn",
        compression="zstd",
        start=start,
        end=end,
        split_duration="day",
    )

    if isinstance(batch_result, dict):
        job_id = batch_result["id"]
    else:
        job_id = batch_result.id
    print(f"    Job ID: {job_id}", flush=True)

    while True:
        all_jobs = client.batch.list_jobs()
        matched = [
            j
            for j in all_jobs
            if (j["id"] if isinstance(j, dict) else j.id) == job_id
        ]
        if not matched:
            logger.error("Job %s not found in list_jobs", job_id)
            return out_dir
        current = matched[0]
        state = current["state"] if isinstance(current, dict) else current.state
        if state == "done":
            print(f"    Job complete. Downloading files...", flush=True)
            break
        elif state in ("expired", "error"):
            logger.error("Batch job %s", state)
            return out_dir
        else:
            print(f"    Status: {state} ... waiting 30s", flush=True)
            time.sleep(30)

    # Download into a temporary staging directory to avoid polluting the organized dir
    staging_dir = DATA_DIR / "_staging" / job_id
    _ensure_dir(staging_dir)
    filenames = client.batch.download(job_id, output_dir=str(staging_dir))
    print(f"    Downloaded {len(filenames)} file(s)", flush=True)

    # Move .dbn.zst files to organized dir with YYYYMMDD.dbn.zst naming
    moved = 0
    schema = job["schema"]
    for fpath_str in filenames:
        fpath = Path(fpath_str)
        if not fpath.name.endswith(".dbn.zst"):
            continue
        m = re.search(r"(\d{8})\." + re.escape(schema) + r"\.dbn\.zst$", fpath.name)
        if not m:
            m = re.search(r"(\d{8})\.\w[\w-]*\.dbn\.zst$", fpath.name)
        if m:
            date_str = m.group(1)
            dest = out_dir / f"{date_str}.dbn.zst"
            if not dest.exists():
                shutil.move(str(fpath), str(dest))
                moved += 1
            else:
                fpath.unlink()
        else:
            fpath.unlink()

    print(f"    [done] {moved} new date files added to {job['subdir']}", flush=True)

    _save_batch_metadata(job_id, job, staging_dir)

    # Clean up staging
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_parent = DATA_DIR / "_staging"
    if staging_parent.exists() and not any(staging_parent.iterdir()):
        staging_parent.rmdir()

    return out_dir


def _save_batch_metadata(job_id: str, job: dict, staging_dir: Path) -> None:
    """Archive batch job metadata JSON files into _meta/batch_jobs.json.

    Reads condition.json, manifest.json, metadata.json from the staging
    directory (if present) and appends the entry to the consolidated metadata
    file.

    Args:
        job_id: Databento batch job ID string.
        job: Download job config dict.
        staging_dir: Path to the staging directory with JSON sidecar files.
    """
    meta_dir = _ensure_dir(META_DIR)
    meta_file = meta_dir / "batch_jobs.json"

    existing_meta = []
    if meta_file.exists():
        with open(meta_file) as f:
            existing_meta = json.load(f)

    entry = {"job_id": job_id, "target_directory": job["subdir"]}
    for json_name in ["metadata.json", "condition.json", "manifest.json"]:
        json_path = staging_dir / json_name
        if json_path.exists():
            with open(json_path) as f:
                entry[json_name.replace(".json", "")] = json.load(f)

    existing_meta.append(entry)
    with open(meta_file, "w") as f:
        json.dump(existing_meta, f, indent=2)


def _filter_jobs(skip_groups: set[str] | None = None) -> list[dict]:
    """Return download jobs after removing any whose group is in *skip_groups*.

    Parameters
    ----------
    skip_groups:
        Set of group names to exclude (e.g. ``{"spy_options"}``).

    Returns
    -------
    list[dict]
        Filtered list of download job dicts.
    """
    if not skip_groups:
        return list(DOWNLOAD_JOBS)
    skipped = [j for j in DOWNLOAD_JOBS if j.get("group") in skip_groups]
    if skipped:
        print(f"  Skipping groups: {', '.join(sorted(skip_groups))}")
        for j in skipped:
            print(f"    - {j['label']}")
        print()
    return [j for j in DOWNLOAD_JOBS if j.get("group") not in skip_groups]


def run_sample(client: db.Historical, *, skip_groups: set[str] | None = None) -> None:
    """Phase 1: download 1 day of each item via streaming to verify format.

    Args:
        client: Authenticated Databento Historical client.
        skip_groups: Optional set of group names to exclude from download.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Sample Download ({SAMPLE_START} to {SAMPLE_END})")
    print(f"{'='*60}\n")

    for job in _filter_jobs(skip_groups):
        print(f"  [{job['label']}]")
        try:
            _download_streaming(client, job, SAMPLE_START, SAMPLE_END)
        except Exception as e:
            logger.error("%s: %s", type(e).__name__, e)
        print()


def run_full(client: db.Historical, start: str, end: str, *, skip_groups: set[str] | None = None) -> None:
    """Phase 2: download the full date range.

    Uses batch API for OPRA option data (large, stored as per-day .dbn.zst)
    and streaming for small supporting datasets like SPY equity OHLCV.

    Args:
        client: Authenticated Databento Historical client.
        start: Start date YYYY-MM-DD.
        end: End date YYYY-MM-DD.
        skip_groups: Optional set of group names to exclude from download.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full Download ({start} to {end})")
    print(f"{'='*60}\n", flush=True)

    for job in _filter_jobs(skip_groups):
        print(f"  [{job['label']}]", flush=True)
        try:
            if job["dataset"] == "DBEQ.BASIC":
                _download_streaming(client, job, start, end)
            else:
                _download_batch(client, job, start, end)
        except Exception as e:
            logger.error("%s: %s", type(e).__name__, e)
        print(flush=True)


def verify() -> None:
    """Load each Parquet in data/databento/ and print summary stats."""
    print(f"\n{'='*60}")
    print(f"VERIFY: Checking downloaded Parquet files")
    print(f"{'='*60}\n")

    if not DATA_DIR.exists():
        print(f"  No data directory found at {DATA_DIR}")
        return

    parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
    if not parquet_files:
        print(f"  No Parquet files found in {DATA_DIR}")
        return

    for pf in parquet_files:
        rel = pf.relative_to(DATA_DIR)
        try:
            df = pd.read_parquet(pf)
            size_mb = pf.stat().st_size / 1e6
            print(f"  {rel}")
            print(f"    Rows: {len(df):,} | Cols: {len(df.columns)} | Size: {size_mb:.1f} MB")
            print(f"    Columns: {list(df.columns)}")

            if "bid_px_00" in df.columns:
                valid_bids = df["bid_px_00"].dropna()
                if len(valid_bids) > 0:
                    print(f"    Bid range: {valid_bids.min():.2f} - {valid_bids.max():.2f}")
                symbols = df["symbol"].nunique() if "symbol" in df.columns else "N/A"
                print(f"    Unique symbols: {symbols}")

            if "strike_price" in df.columns:
                print(f"    Strike range: {df['strike_price'].min():.0f} - {df['strike_price'].max():.0f}")
                print(f"    Expirations: {df['expiration'].nunique()}")
                classes = (
                    df["instrument_class"].value_counts().to_dict()
                    if "instrument_class" in df.columns
                    else {}
                )
                print(f"    C/P split: {classes}")

            if "stat_type" in df.columns:
                oi = df[df["quantity"] > 0]
                print(f"    Records with OI > 0: {len(oi):,} / {len(df):,}")
                if len(oi) > 0:
                    print(f"    OI range: {oi['quantity'].min():,} - {oi['quantity'].max():,}")

            if "open" in df.columns and "close" in df.columns:
                print(f"    Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
                print(f"    Volume range: {df['volume'].min():,} - {df['volume'].max():,}")

            print()
        except Exception as e:
            logger.error("%s: %s", rel, e)


def verify_dbn(start: str, end: str) -> dict[str, dict]:
    """Scan .dbn.zst directories and report date coverage vs trading calendar.

    Checks each {spx,spxw,spy}/{cbbo-1m,definition,statistics} directory
    for YYYYMMDD.dbn.zst files and compares against the expected trading
    days in [start, end).

    Args:
        start: Start date YYYY-MM-DD (inclusive).
        end: End date YYYY-MM-DD (exclusive).

    Returns:
        Dict mapping subdir path strings to dicts with keys:
          expected (int), present (int), missing (list[str]), extra (list[str]),
          total_size_mb (float).
    """
    print(f"\n{'='*60}")
    print(f"VERIFY-DBN: Checking .dbn.zst date coverage ({start} to {end})")
    print(f"{'='*60}\n")

    expected_days = get_trading_days(start, end)
    expected_strs = {d.strftime("%Y%m%d") for d in expected_days}

    subdirs = [
        "spx/cbbo-1m", "spx/definition", "spx/statistics",
        "spxw/cbbo-1m", "spxw/definition", "spxw/statistics",
        "spy/cbbo-1m", "spy/definition", "spy/statistics",
    ]

    results: dict[str, dict] = {}
    all_ok = True

    for subdir in subdirs:
        dir_path = DATA_DIR / subdir
        actual = get_existing_dates(dir_path)
        missing = sorted(expected_strs - actual)
        extra = sorted(actual - expected_strs)

        total_bytes = sum(
            f.stat().st_size for f in dir_path.iterdir() if f.suffix == ".zst"
        ) if dir_path.exists() else 0
        total_mb = total_bytes / 1e6

        results[subdir] = {
            "expected": len(expected_strs),
            "present": len(actual),
            "missing": missing,
            "extra": extra,
            "total_size_mb": total_mb,
        }

        status = "OK" if not missing else f"MISSING {len(missing)}"
        print(f"  {subdir}: {len(actual)}/{len(expected_strs)} days [{status}] ({total_mb:.1f} MB)")

        if missing:
            all_ok = False
            for m in missing:
                print(f"    MISSING: {m}")
        if extra:
            for e in extra:
                print(f"    EXTRA:   {e}")

    # Also check underlying
    underlying_path = DATA_DIR / "underlying" / "spy_equity_1m.parquet"
    if underlying_path.exists():
        size_mb = underlying_path.stat().st_size / 1e6
        print(f"\n  underlying/spy_equity_1m.parquet: {size_mb:.1f} MB")
    else:
        print(f"\n  underlying/spy_equity_1m.parquet: NOT FOUND")
        all_ok = False

    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'ISSUES FOUND'}")
    return results


def main() -> None:
    """Parse CLI arguments and dispatch to the requested download/verify mode."""
    all_groups = sorted({j.get("group", "") for j in DOWNLOAD_JOBS if j.get("group")})
    parser = argparse.ArgumentParser(description="Download Databento historical data")
    parser.add_argument(
        "--phase",
        choices=["sample", "full"],
        help="Download phase: sample (1 day) or full (complete range)",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded Parquet files")
    parser.add_argument(
        "--verify-dbn",
        action="store_true",
        help="Check .dbn.zst date coverage against trading calendar",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=all_groups,
        default=[],
        help=f"Skip download groups: {', '.join(all_groups)}",
    )

    args = parser.parse_args()
    skip_groups = set(args.skip) if args.skip else None

    if args.verify:
        verify()
        return

    if args.verify_dbn:
        verify_dbn(args.start, args.end)
        return

    if not args.phase:
        parser.error("Either --phase, --verify, or --verify-dbn is required")

    import databento as db  # lazy: not needed for verify/verify-dbn paths

    client = db.Historical(require_databento_api_key())
    print(f"Databento client initialized. Output dir: {DATA_DIR}")

    if args.phase == "sample":
        run_sample(client, skip_groups=skip_groups)
        print("Sample download complete. Run with --verify to inspect the data.")
    elif args.phase == "full":
        run_full(client, args.start, args.end, skip_groups=skip_groups)
        print("Full download complete. Run with --verify-dbn to check coverage.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)
