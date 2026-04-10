"""Experiment tracking for optimizer runs.

Provides an abstract ``ExperimentTracker`` interface with a file-based
``CsvExperimentTracker`` implementation.  Each optimizer run is recorded
in a unique directory under ``data/experiments/`` with metadata, a summary
of top configs, and an optional copy/symlink to the results CSV.

Designed for future MLflow compatibility: a ``MlflowExperimentTracker``
subclass can implement the same interface using ``mlflow.*`` calls.

Usage::

    tracker = CsvExperimentTracker()
    run_id = tracker.start_run("staged-v2", {"mode": "staged", "workers": 8})
    tracker.log_metric("num_configs", 276480)
    tracker.log_metric("best_sharpe", 2.34)
    tracker.log_artifact(Path("data/backtest_results.csv"), "results.csv")
    tracker.log_summary({"top_5": [...]})
    tracker.end_run()
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_EXPERIMENTS_DIR = _DATA_DIR / "experiments"


class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking.

    Subclass this to implement file-based, MLflow, or other backends.
    The lifecycle is: ``start_run`` -> ``log_*`` calls -> ``end_run``.
    """

    @abstractmethod
    def start_run(self, run_name: str, params: dict[str, Any]) -> str:
        """Begin a new experiment run. Returns a unique run ID."""
        ...

    @abstractmethod
    def log_metric(self, key: str, value: float) -> None:
        """Record a scalar metric for the current run."""
        ...

    @abstractmethod
    def log_artifact(self, local_path: Path, artifact_name: str) -> None:
        """Attach a file artifact (e.g. results CSV) to the current run."""
        ...

    @abstractmethod
    def log_summary(self, summary: dict[str, Any]) -> None:
        """Record a structured summary (top-N configs, Pareto stats, etc.)."""
        ...

    @abstractmethod
    def end_run(self, status: str = "completed") -> None:
        """Finalize the current run with the given status."""
        ...

    @abstractmethod
    def list_runs(self) -> list[dict[str, Any]]:
        """List all recorded experiment runs with metadata."""
        ...


def _git_short_hash() -> str:
    """Return the short git hash of HEAD, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _git_is_dirty() -> bool:
    """Check if the git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class CsvExperimentTracker(ExperimentTracker):
    """File-based experiment tracker using ``data/experiments/run_<id>/``.

    Each run gets its own directory containing:
    - ``metadata.json``: git hash, config, timestamps, CLI params
    - ``summary.json``: top-N configs, Pareto stats, metrics
    - Artifact copies (results CSV, config YAML, etc.)
    """

    def __init__(self, experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR) -> None:
        self._experiments_dir = experiments_dir
        self._current_run_id: str | None = None
        self._current_run_dir: Path | None = None
        self._metadata: dict[str, Any] = {}
        self._metrics: dict[str, float] = {}
        self._start_time: float = 0.0

    @property
    def current_run_id(self) -> str | None:
        """The ID of the active run, or None if no run is active."""
        return self._current_run_id

    @property
    def current_run_dir(self) -> Path | None:
        """The directory of the active run, or None."""
        return self._current_run_dir

    def start_run(self, run_name: str, params: dict[str, Any]) -> str:
        """Create a new run directory and write initial metadata.

        Parameters
        ----------
        run_name : Human-readable name for the run.
        params : Dict of parameters (CLI args, config file, etc.).

        Returns
        -------
        Unique run ID string.
        """
        git_hash = _git_short_hash()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        run_id = f"run_{ts}_{git_hash}_{short_uuid}"

        run_dir = self._experiments_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._current_run_id = run_id
        self._current_run_dir = run_dir
        self._start_time = time.time()
        self._metrics = {}

        self._metadata = {
            "run_id": run_id,
            "run_name": run_name,
            "git_hash": git_hash,
            "git_dirty": _git_is_dirty(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "params": params,
        }
        self._write_metadata()

        logger.info("Experiment run started: %s (%s)", run_name, run_id)
        return run_id

    def log_metric(self, key: str, value: float) -> None:
        """Record a metric. Overwrites if key already exists."""
        self._metrics[key] = value

    def log_artifact(self, local_path: Path, artifact_name: str) -> None:
        """Copy a file into the run directory under the given name."""
        if self._current_run_dir is None:
            raise RuntimeError("No active run -- call start_run() first")
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning("Artifact not found, skipping: %s", local_path)
            return
        dest = self._current_run_dir / artifact_name
        shutil.copy2(local_path, dest)
        logger.info("Artifact saved: %s -> %s", local_path.name, dest)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Write a summary JSON file in the run directory."""
        if self._current_run_dir is None:
            raise RuntimeError("No active run -- call start_run() first")
        summary_path = self._current_run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    def end_run(self, status: str = "completed") -> None:
        """Finalize the run, writing final metadata and metrics."""
        if self._current_run_dir is None:
            return

        elapsed = time.time() - self._start_time
        self._metadata["status"] = status
        self._metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._metadata["elapsed_seconds"] = round(elapsed, 1)
        self._metadata["metrics"] = self._metrics
        self._write_metadata()

        logger.info("Experiment run %s: %s (%.1fs)",
                     status, self._current_run_id, elapsed)

        self._current_run_id = None
        self._current_run_dir = None

    def list_runs(self) -> list[dict[str, Any]]:
        """List all experiment runs, sorted newest first by started_at.

        Returns a list of metadata dicts, one per run directory.
        """
        runs = []
        if not self._experiments_dir.exists():
            return runs

        for run_dir in self._experiments_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    meta["run_dir"] = str(run_dir)
                    runs.append(meta)
                except (json.JSONDecodeError, OSError):
                    continue

        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs

    def create_git_tag(self, tag_name: str | None = None) -> bool:
        """Create a lightweight git tag for the current run.

        Parameters
        ----------
        tag_name : Custom tag name; defaults to ``exp/<run_id>``.

        Returns
        -------
        True if the tag was created successfully.
        """
        if self._current_run_id is None:
            return False
        tag = tag_name or f"exp/{self._current_run_id}"
        try:
            result = subprocess.run(
                ["git", "tag", tag],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info("Git tag created: %s", tag)
                self._metadata["git_tag"] = tag
                self._write_metadata()
                return True
            logger.warning("Git tag failed: %s", result.stderr.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def _write_metadata(self) -> None:
        """Persist the current metadata dict to disk."""
        if self._current_run_dir is None:
            return
        meta_path = self._current_run_dir / "metadata.json"
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._metadata, indent=2, default=str))
        tmp.rename(meta_path)
