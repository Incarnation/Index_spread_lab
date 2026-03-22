"""Download historical options data from Databento for SPX and SPY.

Supports multiple modes:
  --phase sample   Download 1 day to verify schema/columns (fast, ~1 GB)
  --phase full     Download the full date range via batch API (~10 GB)
  --verify         Load Parquet files and print summary stats
  --verify-dbn     Scan .dbn.zst files and check for date gaps vs trading calendar

Data is stored as per-day .dbn.zst files under:
    data/databento/{spx,spy}/{cbbo-1m,definition,statistics}/YYYYMMDD.dbn.zst
    data/databento/underlying/spy_equity_1m.parquet

Usage:
    python -m backend.scripts.download_databento --phase sample
    python -m backend.scripts.download_databento --phase full
    python -m backend.scripts.download_databento --verify
    python -m backend.scripts.download_databento --verify-dbn
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATABENTO_API_KEY = os.getenv(
    "DATABENTO_API_KEY", "db-7wQqjpM5bNu9crhPVL3GctAKanPPd"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "databento"
META_DIR = DATA_DIR / "_meta"

US_EQUITY_HOLIDAYS_2026 = {
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
}

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

    Excludes weekends and known 2026 NYSE holidays. The end date is exclusive,
    matching Databento's query convention.

    Args:
        start_str: Start date as YYYY-MM-DD (inclusive).
        end_str: End date as YYYY-MM-DD (exclusive).

    Returns:
        Sorted list of date objects representing trading days.
    """
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    days = []
    d = start
    while d < end:
        if d.weekday() < 5 and d not in US_EQUITY_HOLIDAYS_2026:
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
        print(f"    [warn] No records returned for {job['label']} ({start} to {end})")
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
            print(f"    [ERROR] Job {job_id} not found in list_jobs")
            return out_dir
        current = matched[0]
        state = current["state"] if isinstance(current, dict) else current.state
        if state == "done":
            print(f"    Job complete. Downloading files...", flush=True)
            break
        elif state in ("expired", "error"):
            print(f"    [ERROR] Batch job {state}")
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
            print(f"    [ERROR] {type(e).__name__}: {e}")
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
            print(f"    [ERROR] {type(e).__name__}: {e}")
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
            print(f"  {rel}: ERROR - {e}\n")


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

    client = db.Historical(DATABENTO_API_KEY)
    print(f"Databento client initialized. Output dir: {DATA_DIR}")

    if args.phase == "sample":
        run_sample(client, skip_groups=skip_groups)
        print("Sample download complete. Run with --verify to inspect the data.")
    elif args.phase == "full":
        run_full(client, args.start, args.end, skip_groups=skip_groups)
        print("Full download complete. Run with --verify-dbn to check coverage.")


if __name__ == "__main__":
    main()
