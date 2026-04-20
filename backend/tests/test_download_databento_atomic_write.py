"""Wave 6 (audit) H9 contract test: atomic Parquet write + corrupt-detection.

Pre-H9 behavior: ``_download_streaming`` called ``df.to_parquet(out_path)``
directly. A crash mid-flush (KeyboardInterrupt, OOM kill, disk full)
left a partial Parquet at the canonical path. The next run saw
``out_path.exists()`` and skipped; offline training silently consumed
the truncated file days later when something tried to decode it.

Post-H9 behavior:

1. Streaming writes go through ``<out>.tmp`` first; only ``os.replace``
   moves the finished bytes to the canonical name. A mid-flush crash
   leaves the ``.tmp`` orphan, never the canonical file.
2. On retry, the canonical path is decode-probed via
   ``_is_parquet_decodable`` before the cheap ``out_path.exists()``
   skip. A previously-corrupt file is unlinked and re-downloaded on
   the same run.

These tests exercise both branches with monkeypatching: no real
Databento client and no real Parquet library calls are made.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_databento as dd  # noqa: E402


def _make_fake_client(rows: int = 5):
    """Build a stand-in for ``databento.Historical`` with a 5-row DataFrame."""
    fake_df = MagicMock()
    fake_df.__len__ = MagicMock(return_value=rows)
    fake_df.to_parquet = MagicMock()

    fake_data = MagicMock()
    fake_data.to_df = MagicMock(return_value=fake_df)

    client = MagicMock()
    client.timeseries.get_range = MagicMock(return_value=fake_data)
    return client, fake_df


# ---------------------------------------------------------------------------
# 1. Mid-flush abort leaves the canonical path absent and tmp cleaned up
# ---------------------------------------------------------------------------


class TestAtomicWriteOnAbort:
    """A KeyboardInterrupt mid-`to_parquet` must NOT leave a half-file."""

    def test_aborted_to_parquet_leaves_no_canonical_file(self, tmp_path, monkeypatch):
        """``df.to_parquet`` raises -> tmp deleted; out_path never created."""
        # H9 (audit): redirect DATA_DIR so the test stays sandboxed.
        monkeypatch.setattr(dd, "DATA_DIR", tmp_path)

        client, fake_df = _make_fake_client(rows=5)
        # Simulate KeyboardInterrupt mid-write.
        fake_df.to_parquet.side_effect = KeyboardInterrupt("simulated mid-flush abort")

        job = {
            "label": "SPY equity 1m",
            "subdir": "underlying",
            "filename": "spy_equity_1m",
            "dataset": "DBEQ.BASIC",
            "symbols": "SPY",
            "stype_in": "raw_symbol",
            "schema": "ohlcv-1m",
        }

        with pytest.raises(KeyboardInterrupt):
            dd._download_streaming(client, job, "2026-01-02", "2026-01-03")

        out_dir = tmp_path / "underlying"
        canonical = out_dir / "spy_equity_1m.parquet"
        tmp_file = out_dir / "spy_equity_1m.parquet.tmp"

        assert not canonical.exists(), "canonical Parquet must NOT be created on abort"
        assert not tmp_file.exists(), "tmp Parquet must be cleaned up on abort"

    def test_successful_write_promotes_tmp_to_canonical(self, tmp_path, monkeypatch):
        """Happy path: tmp is renamed to canonical; only the canonical exists."""
        monkeypatch.setattr(dd, "DATA_DIR", tmp_path)

        captured_tmp_paths: list[Path] = []

        def fake_to_parquet(path, engine):
            # Record the path the writer was handed so we can assert it
            # was the .tmp variant rather than the canonical name.
            captured_tmp_paths.append(Path(path))
            Path(path).write_bytes(b"PAR1\x00fakeparquet")

        client, fake_df = _make_fake_client(rows=5)
        fake_df.to_parquet.side_effect = fake_to_parquet

        job = {
            "label": "SPY equity 1m",
            "subdir": "underlying",
            "filename": "spy_equity_1m",
            "dataset": "DBEQ.BASIC",
            "symbols": "SPY",
            "stype_in": "raw_symbol",
            "schema": "ohlcv-1m",
        }

        result = dd._download_streaming(client, job, "2026-01-02", "2026-01-03")

        canonical = tmp_path / "underlying" / "spy_equity_1m.parquet"
        tmp_file = tmp_path / "underlying" / "spy_equity_1m.parquet.tmp"

        assert result == canonical
        assert canonical.exists(), "canonical Parquet must exist after successful write"
        assert not tmp_file.exists(), "tmp Parquet must be renamed away"
        assert all(p.suffix == ".tmp" for p in captured_tmp_paths), (
            "writer must always be handed the .tmp path, never the canonical"
        )


# ---------------------------------------------------------------------------
# 2. Corrupt cached file is detected + deleted + re-downloaded
# ---------------------------------------------------------------------------


class TestCorruptCacheRedownload:
    """An existing-but-corrupt Parquet must be re-fetched, not silently skipped."""

    def test_corrupt_existing_parquet_is_redownloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "DATA_DIR", tmp_path)

        # Plant a "corrupt" file at the canonical path.
        out_dir = tmp_path / "underlying"
        out_dir.mkdir(parents=True, exist_ok=True)
        canonical = out_dir / "spy_equity_1m.parquet"
        canonical.write_bytes(b"NOT A PARQUET FILE")

        # Force the decode probe to report corrupt without doing any
        # real pyarrow I/O.
        monkeypatch.setattr(dd, "_is_parquet_decodable", lambda p: False)

        # Successful re-download writes a 5-row Parquet.
        def fake_to_parquet(path, engine):
            Path(path).write_bytes(b"PAR1\x00freshparquet")

        client, fake_df = _make_fake_client(rows=5)
        fake_df.to_parquet.side_effect = fake_to_parquet

        job = {
            "label": "SPY equity 1m",
            "subdir": "underlying",
            "filename": "spy_equity_1m",
            "dataset": "DBEQ.BASIC",
            "symbols": "SPY",
            "stype_in": "raw_symbol",
            "schema": "ohlcv-1m",
        }

        result = dd._download_streaming(client, job, "2026-01-02", "2026-01-03")

        assert result == canonical
        assert canonical.read_bytes() == b"PAR1\x00freshparquet", (
            "corrupt cached Parquet must be replaced by the freshly-downloaded one"
        )
        # Sanity: the streaming API was actually called this time.
        client.timeseries.get_range.assert_called_once()

    def test_decodable_existing_parquet_is_skipped(self, tmp_path, monkeypatch):
        """Decode-probe-passing cached file shorts out without re-downloading."""
        monkeypatch.setattr(dd, "DATA_DIR", tmp_path)
        monkeypatch.setattr(dd, "_is_parquet_decodable", lambda p: True)

        out_dir = tmp_path / "underlying"
        out_dir.mkdir(parents=True, exist_ok=True)
        canonical = out_dir / "spy_equity_1m.parquet"
        canonical.write_bytes(b"PAR1\x00existingparquet")

        client, _ = _make_fake_client(rows=0)

        job = {
            "label": "SPY equity 1m",
            "subdir": "underlying",
            "filename": "spy_equity_1m",
            "dataset": "DBEQ.BASIC",
            "symbols": "SPY",
            "stype_in": "raw_symbol",
            "schema": "ohlcv-1m",
        }

        result = dd._download_streaming(client, job, "2026-01-02", "2026-01-03")

        assert result == canonical
        client.timeseries.get_range.assert_not_called()


# ---------------------------------------------------------------------------
# 3. get_existing_dates removes corrupt .dbn.zst on H9 path
# ---------------------------------------------------------------------------


class TestGetExistingDatesProbe:
    """Corrupt .dbn.zst files are removed and excluded from the returned set."""

    def test_corrupt_dbn_is_unlinked_and_excluded(self, tmp_path, monkeypatch):
        # Two files: one "good", one "corrupt" by name pattern.
        good = tmp_path / "20260102.dbn.zst"
        corrupt = tmp_path / "20260103.dbn.zst"
        good.write_bytes(b"\x28\xb5\x2f\xfd\x00")  # zstd magic prefix
        corrupt.write_bytes(b"NOT ZSTD")

        # Decode probe: True for the good file, False for the corrupt.
        def fake_probe(path: Path) -> bool:
            return path.name == "20260102.dbn.zst"

        monkeypatch.setattr(dd, "_is_dbn_decodable", fake_probe)

        result = dd.get_existing_dates(tmp_path, decode_probe=True)

        assert result == {"20260102"}, (
            "decode-probed scan must drop the corrupt file from the result set"
        )
        assert good.exists(), "decodable file must remain on disk"
        assert not corrupt.exists(), "corrupt file must be unlinked"

    def test_decode_probe_disabled_keeps_corrupt(self, tmp_path, monkeypatch):
        """Presence-only mode (--verify-dbn) does NOT delete files as a side effect."""
        good = tmp_path / "20260102.dbn.zst"
        corrupt = tmp_path / "20260103.dbn.zst"
        good.write_bytes(b"\x28\xb5\x2f\xfd\x00")
        corrupt.write_bytes(b"NOT ZSTD")

        # Decode probe MUST NOT be invoked when decode_probe=False.
        def fake_probe(path: Path) -> bool:
            raise AssertionError("decode probe should not run with decode_probe=False")

        monkeypatch.setattr(dd, "_is_dbn_decodable", fake_probe)

        result = dd.get_existing_dates(tmp_path, decode_probe=False)

        assert result == {"20260102", "20260103"}
        assert corrupt.exists(), "presence-only scan must NOT unlink files"


# ---------------------------------------------------------------------------
# 4. M17 argparse-time validation rejects bad ranges
# ---------------------------------------------------------------------------


class TestM17ArgumentValidation:
    """``main()`` should hard-fail at parse time on bad date ranges."""

    def test_start_after_end_exits(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "download_databento.py",
                "--phase", "sample",
                "--start", "2026-03-10",
                "--end", "2026-01-01",
            ],
        )
        with pytest.raises(SystemExit):
            dd.main()
        err = capsys.readouterr().err
        assert "M17" in err

    def test_end_in_future_exits(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            [
                "download_databento.py",
                "--phase", "sample",
                "--start", "2026-01-02",
                "--end", "2099-01-01",
            ],
        )
        with pytest.raises(SystemExit):
            dd.main()
        err = capsys.readouterr().err
        assert "M17" in err
