"""Tests for the end-to-end pipeline orchestrator (``scripts/run_pipeline.py``).

These tests monkeypatch ``_run_phase`` so the orchestrator's subprocess
call is captured as a plain ``list[str]`` without actually running any
downstream scripts.  That lets us assert on the exact argv the
orchestrator would have handed to ``backtest_strategy.py`` in both the
``optimize`` and ``walkforward`` phases, without paying any backtest
runtime.

The specific regression guard here covers Track B of the Tier-2 follow-
up: the orchestrator must forward a ``--holdout-min-pass-windows``
override to the child CLI iff the operator set one, and must NOT forward
it (preserving ``cli.py``'s phase-aware default of 2-for-walkforward /
1-for-optimize) when the operator leaves it unset.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[1]
_SCRIPTS = _BACKEND / "scripts"


@pytest.fixture
def run_pipeline_module(monkeypatch):
    """Import ``scripts/run_pipeline.py`` as a module for in-process
    testing.  ``importlib.util`` keeps this test file independent of
    whether ``scripts/`` is on ``sys.path`` globally."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "run_pipeline_under_test", _SCRIPTS / "run_pipeline.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _invoke_and_capture(
    run_pipeline_module, monkeypatch, tmp_path, argv: list[str],
) -> list[list[str]]:
    """Run ``run_pipeline.main()`` with the given argv and return every
    ``cmd`` list the orchestrator would have dispatched to a subprocess.

    ``_run_phase`` is replaced with a recorder that returns True so the
    orchestrator continues past each phase.  The ``expected_output``
    existence check is also neutralised so the test doesn't have to
    fabricate CSV files on disk.
    """
    captured: list[list[str]] = []

    def _fake_run_phase(name, cmd, expected_output=None, log=None):
        captured.append(list(cmd))
        return True

    monkeypatch.setattr(run_pipeline_module, "_run_phase", _fake_run_phase)
    # Redirect the data dir so the orchestrator doesn't write a
    # pipeline_log_*.json into the real data/ directory during tests.
    monkeypatch.setattr(run_pipeline_module, "_DATA_DIR", tmp_path)

    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", *argv])
    run_pipeline_module.main()
    return captured


class TestHoldoutMinPassWindowsPassthrough:
    """Track B: --holdout-min-pass-windows flows through run_pipeline
    into the child backtest_strategy.py invocation exactly as typed."""

    def test_walkforward_forwards_override_when_set(
        self, run_pipeline_module, monkeypatch, tmp_path,
    ):
        """When the operator passes --holdout-min-pass-windows 1 at the
        orchestrator level, the walkforward phase's child cmdline must
        contain that same flag with the same value, so cli.py's
        phase-aware default of 2 is overridden.

        This is the D.b finding codified: operators experimenting with
        tight pre-filtered grids need a way to dial the cross-window
        regularizer down to 1 without editing source."""
        argv = [
            "--run-name", "wf-mpw1",
            "--phase", "walkforward",
            "--optimizer-config", "configs/optimizer/event_only_v2.yaml",
            "--workers", "2",
            "--holdout-months", "1",
            "--holdout-min-pass-windows", "1",
        ]
        captured = _invoke_and_capture(
            run_pipeline_module, monkeypatch, tmp_path, argv,
        )

        assert len(captured) == 1, (
            "Walkforward phase should dispatch exactly one subprocess"
        )
        cmd = captured[0]
        assert "--walkforward" in cmd
        assert "--holdout-min-pass-windows" in cmd
        idx = cmd.index("--holdout-min-pass-windows")
        assert cmd[idx + 1] == "1"

    def test_walkforward_omits_override_when_unset(
        self, run_pipeline_module, monkeypatch, tmp_path,
    ):
        """When the operator does NOT set --holdout-min-pass-windows, the
        orchestrator must NOT pass it through.  Forwarding a sentinel
        would overwrite cli.py's phase-aware default (2 for walk-
        forward); the whole point of the conditional forwarding is
        to preserve that default."""
        argv = [
            "--run-name", "wf-default",
            "--phase", "walkforward",
            "--optimizer-config", "configs/optimizer/event_only_v2.yaml",
            "--workers", "2",
            "--holdout-months", "1",
        ]
        captured = _invoke_and_capture(
            run_pipeline_module, monkeypatch, tmp_path, argv,
        )
        cmd = captured[0]
        assert "--walkforward" in cmd
        assert "--holdout-min-pass-windows" not in cmd, (
            f"Orchestrator should let cli.py use its own default, "
            f"but forwarded the flag anyway: {cmd}"
        )

    def test_optimize_forwards_override_when_set(
        self, run_pipeline_module, monkeypatch, tmp_path,
    ):
        """Same contract on the plain --optimize path: when the
        operator sets --holdout-min-pass-windows 2, the optimize
        phase's cmdline must carry it so cli.py's optimize-side
        default of 1 can be overridden (e.g. to A/B the picker on
        plain-optimize runs)."""
        argv = [
            "--run-name", "opt-mpw2",
            "--phase", "optimize",
            "--optimizer-config", "configs/optimizer/event_only_v2.yaml",
            "--workers", "2",
            "--holdout-months", "1",
            "--holdout-min-pass-windows", "2",
        ]
        captured = _invoke_and_capture(
            run_pipeline_module, monkeypatch, tmp_path, argv,
        )
        cmd = captured[0]
        assert "--optimize" in cmd
        assert "--holdout-min-pass-windows" in cmd
        idx = cmd.index("--holdout-min-pass-windows")
        assert cmd[idx + 1] == "2"

    def test_optimize_omits_override_when_unset(
        self, run_pipeline_module, monkeypatch, tmp_path,
    ):
        """Mirror of the walkforward test for the --optimize path."""
        argv = [
            "--run-name", "opt-default",
            "--phase", "optimize",
            "--optimizer-config", "configs/optimizer/event_only_v2.yaml",
            "--workers", "2",
            "--holdout-months", "1",
        ]
        captured = _invoke_and_capture(
            run_pipeline_module, monkeypatch, tmp_path, argv,
        )
        cmd = captured[0]
        assert "--optimize" in cmd
        assert "--holdout-min-pass-windows" not in cmd

    def test_override_ignored_when_holdout_disabled(
        self, run_pipeline_module, monkeypatch, tmp_path,
    ):
        """Edge case: --holdout-min-pass-windows is only meaningful when
        --holdout-months > 0.  The current orchestrator nests the
        forward inside ``if args.holdout_months > 0``, so we guard
        against a future refactor that accidentally lifts the forward
        outside that block and ends up passing an override to a
        cli.py invocation that has no holdout data slice."""
        argv = [
            "--run-name", "wf-no-holdout",
            "--phase", "walkforward",
            "--optimizer-config", "configs/optimizer/event_only_v2.yaml",
            "--workers", "2",
            # note: --holdout-months defaults to 0 here
            "--holdout-min-pass-windows", "3",
        ]
        captured = _invoke_and_capture(
            run_pipeline_module, monkeypatch, tmp_path, argv,
        )
        cmd = captured[0]
        assert "--holdout-min-pass-windows" not in cmd, (
            f"Override must not be forwarded when holdout is disabled: {cmd}"
        )
