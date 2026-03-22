"""Tests for download_databento.py: trading day calendar and date-gap verification."""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pytest

from backend.scripts.download_databento import (
    get_existing_dates,
    get_trading_days,
    verify_dbn,
)


# ---------------------------------------------------------------------------
# get_trading_days
# ---------------------------------------------------------------------------


class TestGetTradingDays:
    def test_skips_weekends(self):
        """Jan 5 2026 is Monday, Jan 3-4 are Sat/Sun -- should be excluded."""
        days = get_trading_days("2026-01-02", "2026-01-06")
        date_strs = [d.strftime("%A") for d in days]
        assert "Saturday" not in date_strs
        assert "Sunday" not in date_strs

    def test_skips_new_years(self):
        """Jan 1 2026 is a holiday."""
        days = get_trading_days("2026-01-01", "2026-01-03")
        assert date(2026, 1, 1) not in days
        assert date(2026, 1, 2) in days

    def test_skips_mlk_day(self):
        """Jan 19 2026 (MLK Day) should be excluded."""
        days = get_trading_days("2026-01-19", "2026-01-20")
        assert days == []

    def test_skips_presidents_day(self):
        """Feb 16 2026 (Presidents' Day) should be excluded."""
        days = get_trading_days("2026-02-16", "2026-02-17")
        assert days == []

    def test_full_jan_to_mar(self):
        """Jan 1 to Mar 10 should yield exactly 45 trading days."""
        days = get_trading_days("2026-01-01", "2026-03-10")
        assert len(days) == 45

    def test_end_exclusive(self):
        """End date is exclusive: asking for Jan 2 to Jan 2 returns nothing."""
        days = get_trading_days("2026-01-02", "2026-01-02")
        assert days == []

    def test_single_day(self):
        """Requesting exactly one trading day."""
        days = get_trading_days("2026-01-02", "2026-01-03")
        assert len(days) == 1
        assert days[0] == date(2026, 1, 2)

    def test_returns_sorted(self):
        """Output should be sorted chronologically."""
        days = get_trading_days("2026-01-01", "2026-02-01")
        assert days == sorted(days)


# ---------------------------------------------------------------------------
# get_existing_dates
# ---------------------------------------------------------------------------


class TestGetExistingDates:
    def test_reads_dbn_zst_names(self, tmp_path: Path):
        """Correctly parses YYYYMMDD.dbn.zst filenames."""
        (tmp_path / "20260102.dbn.zst").touch()
        (tmp_path / "20260105.dbn.zst").touch()
        (tmp_path / "readme.txt").touch()
        result = get_existing_dates(tmp_path)
        assert result == {"20260102", "20260105"}

    def test_ignores_non_matching(self, tmp_path: Path):
        """Files not matching YYYYMMDD.dbn.zst are ignored."""
        (tmp_path / "opra-pillar-20260102.cbbo-1m.dbn.zst").touch()
        (tmp_path / "data.parquet").touch()
        result = get_existing_dates(tmp_path)
        assert result == set()

    def test_nonexistent_dir(self, tmp_path: Path):
        """Returns empty set for a directory that doesn't exist."""
        result = get_existing_dates(tmp_path / "nonexistent")
        assert result == set()

    def test_empty_dir(self, tmp_path: Path):
        """Returns empty set for an empty directory."""
        result = get_existing_dates(tmp_path)
        assert result == set()


# ---------------------------------------------------------------------------
# verify_dbn (integration-style with temp dirs)
# ---------------------------------------------------------------------------


class TestVerifyDbn:
    def _setup_dirs(self, base: Path, dates: list[str]) -> None:
        """Create the standard directory structure with the given date files."""
        for subdir in [
            "spx/cbbo-1m", "spx/definition", "spx/statistics",
            "spxw/cbbo-1m", "spxw/definition", "spxw/statistics",
            "spy/cbbo-1m", "spy/definition", "spy/statistics",
        ]:
            d = base / subdir
            d.mkdir(parents=True, exist_ok=True)
            for dt in dates:
                (d / f"{dt}.dbn.zst").touch()

        (base / "underlying").mkdir(parents=True, exist_ok=True)
        (base / "underlying" / "spy_equity_1m.parquet").touch()

    def test_all_dates_present(self, tmp_path: Path, monkeypatch):
        """When all trading days are present, results show no missing dates."""
        import backend.scripts.download_databento as mod

        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)

        all_dates = [d.strftime("%Y%m%d") for d in get_trading_days("2026-01-01", "2026-01-10")]
        self._setup_dirs(tmp_path, all_dates)

        results = verify_dbn("2026-01-01", "2026-01-10")
        for subdir, info in results.items():
            assert info["missing"] == [], f"{subdir} has unexpected missing dates"
            assert info["present"] == info["expected"]

    def test_detects_missing_dates(self, tmp_path: Path, monkeypatch):
        """When a date is removed, verify_dbn reports it as missing."""
        import backend.scripts.download_databento as mod

        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)

        all_dates = [d.strftime("%Y%m%d") for d in get_trading_days("2026-01-01", "2026-01-10")]
        self._setup_dirs(tmp_path, all_dates)

        # Remove one date from spx/cbbo-1m
        removed = all_dates[2]
        (tmp_path / "spx" / "cbbo-1m" / f"{removed}.dbn.zst").unlink()

        results = verify_dbn("2026-01-01", "2026-01-10")
        assert removed in results["spx/cbbo-1m"]["missing"]
        assert results["spx/cbbo-1m"]["present"] == len(all_dates) - 1

    def test_detects_extra_dates(self, tmp_path: Path, monkeypatch):
        """Extra files outside the expected range are reported."""
        import backend.scripts.download_databento as mod

        monkeypatch.setattr(mod, "DATA_DIR", tmp_path)

        all_dates = [d.strftime("%Y%m%d") for d in get_trading_days("2026-01-01", "2026-01-10")]
        extra_date = "20251231"
        self._setup_dirs(tmp_path, all_dates + [extra_date])

        results = verify_dbn("2026-01-01", "2026-01-10")
        assert extra_date in results["spx/cbbo-1m"]["extra"]
