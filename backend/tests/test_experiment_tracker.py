"""Tests for the experiment tracking module."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_tracker import CsvExperimentTracker


class TestCsvExperimentTracker:
    """Verify CsvExperimentTracker run lifecycle."""

    def test_start_and_end_run(self, tmp_path: Path) -> None:
        """Start/end creates a run directory with metadata."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)
        run_id = tracker.start_run("test-run", {"mode": "staged"})

        assert run_id.startswith("run_")
        assert tracker.current_run_dir is not None
        assert tracker.current_run_dir.exists()

        tracker.end_run()
        assert tracker.current_run_id is None

        meta_path = tmp_path / run_id / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["run_name"] == "test-run"
        assert meta["status"] == "completed"
        assert "elapsed_seconds" in meta
        assert meta["params"]["mode"] == "staged"

    def test_log_metric(self, tmp_path: Path) -> None:
        """Metrics are stored in metadata after end_run."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)
        run_id = tracker.start_run("metric-test", {})
        tracker.log_metric("best_sharpe", 2.34)
        tracker.log_metric("num_configs", 1000)
        tracker.end_run()

        meta = json.loads((tmp_path / run_id / "metadata.json").read_text())
        assert meta["metrics"]["best_sharpe"] == 2.34
        assert meta["metrics"]["num_configs"] == 1000

    def test_log_artifact(self, tmp_path: Path) -> None:
        """Artifacts are copied into the run directory."""
        src = tmp_path / "source.csv"
        src.write_text("a,b\n1,2\n")

        tracker = CsvExperimentTracker(experiments_dir=tmp_path / "experiments")
        run_id = tracker.start_run("artifact-test", {})
        tracker.log_artifact(src, "results.csv")
        tracker.end_run()

        artifact = tmp_path / "experiments" / run_id / "results.csv"
        assert artifact.exists()
        assert artifact.read_text() == "a,b\n1,2\n"

    def test_log_artifact_missing_file(self, tmp_path: Path) -> None:
        """Missing artifact file is skipped without error."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)
        tracker.start_run("missing-artifact", {})
        tracker.log_artifact(Path("/nonexistent/file.csv"), "results.csv")
        tracker.end_run()

    def test_log_summary(self, tmp_path: Path) -> None:
        """Summary JSON is written to the run directory."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)
        run_id = tracker.start_run("summary-test", {})
        tracker.log_summary({"top_5": [{"sharpe": 2.0}]})
        tracker.end_run()

        summary = json.loads((tmp_path / run_id / "summary.json").read_text())
        assert summary["top_5"][0]["sharpe"] == 2.0

    def test_list_runs(self, tmp_path: Path) -> None:
        """list_runs returns all recorded runs."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)

        tracker.start_run("run-1", {"idx": 1})
        tracker.end_run()
        tracker.start_run("run-2", {"idx": 2})
        tracker.end_run()

        runs = tracker.list_runs()
        assert len(runs) == 2
        names = {r["run_name"] for r in runs}
        assert names == {"run-1", "run-2"}

    def test_list_runs_empty(self, tmp_path: Path) -> None:
        """Empty experiments directory returns empty list."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)
        assert tracker.list_runs() == []

    def test_end_run_with_failure_status(self, tmp_path: Path) -> None:
        """end_run with status='failed' records the failure."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)
        run_id = tracker.start_run("fail-test", {})
        tracker.end_run(status="failed")

        meta = json.loads((tmp_path / run_id / "metadata.json").read_text())
        assert meta["status"] == "failed"

    def test_end_run_without_start(self, tmp_path: Path) -> None:
        """end_run without start_run is a no-op."""
        tracker = CsvExperimentTracker(experiments_dir=tmp_path)
        tracker.end_run()
