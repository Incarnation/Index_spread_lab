"""Tests for ingest_optimizer_results.py.

Wave 2 / H2 regression: optimizer_runs INSERT, optimizer_results
to_sql, and optimizer_walkforward to_sql must all share a SINGLE
transaction so we never end up with a parent row whose children are
missing.  The legacy implementation opened three separate
transactions; if the second or third write failed, the first commit
left a "zombie" run row that subsequent retries refused to repair
because the duplicate-detection guard at the top of `ingest_results`
would short-circuit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ingest_optimizer_results as ior  # noqa: E402


class _FakeEngine:
    """Minimal engine stub that hands out a tracked Connection from
    ``begin()`` and a no-op connection from ``connect()`` so the
    duplicate-detection guard returns no row."""

    def __init__(self) -> None:
        self.begin_calls = 0
        self.connect_calls = 0
        self.connection = MagicMock(name="conn")
        # `engine.connect()` is used only for the dup-detection SELECT.
        # Make .first() return None so we proceed to the writes.
        self.dup_conn = MagicMock(name="dup_conn")
        self.dup_conn.execute.return_value.first.return_value = None
        self.dup_conn.__enter__ = MagicMock(return_value=self.dup_conn)
        self.dup_conn.__exit__ = MagicMock(return_value=False)

    def connect(self):  # type: ignore[no-untyped-def]
        self.connect_calls += 1
        return self.dup_conn

    def begin(self):  # type: ignore[no-untyped-def]
        self.begin_calls += 1
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=self.connection)
        cm.__exit__ = MagicMock(return_value=False)
        return cm


def _write_results_csv(tmp_path: Path) -> Path:
    """Write a minimal optimizer-results CSV that ingest_results can parse.

    Must include `sharpe` and `max_dd_pct` so the Pareto computation runs.
    """
    df = pd.DataFrame({
        "p_max_trades_per_day": [1, 2],
        "total_trades": [10, 20],
        "return_pct": [10.0, 20.0],
        "sharpe": [1.0, 1.5],
        "max_dd_pct": [-5.0, -3.0],
    })
    p = tmp_path / "results.csv"
    df.to_csv(p, index=False)
    return p


def _write_wf_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "window": ["w1", "w2"],
        "p_max_trades_per_day": [1, 2],
        "train_sharpe": [1.0, 1.5],
        "test_sharpe": [0.8, 1.0],
    })
    p = tmp_path / "wf.csv"
    df.to_csv(p, index=False)
    return p


class TestIngestResultsAtomicity:
    """H2: the three writes must use a SINGLE engine.begin() block, and
    pandas.to_sql must be invoked with the SAME connection object that
    was used for the optimizer_runs INSERT.  This guarantees a partial
    failure rolls back the parent row alongside the children."""

    def test_writes_share_a_single_transaction(self, tmp_path: Path) -> None:
        engine = _FakeEngine()
        results_csv = _write_results_csv(tmp_path)
        wf_csv = _write_wf_csv(tmp_path)

        with patch.object(ior, "create_engine", return_value=engine), \
             patch("pandas.DataFrame.to_sql") as mock_to_sql:
            ior.ingest_results(
                results_csv=results_csv,
                run_name="atomicity-test",
                walkforward_csv=wf_csv,
                database_url="postgresql://stub",
            )

        # Exactly ONE engine.begin() block (the legacy bug used 1+1+1=3).
        assert engine.begin_calls == 1, (
            f"expected a single engine.begin() block but got {engine.begin_calls}"
        )

        # Both to_sql calls must target the SAME connection object that
        # the optimizer_runs INSERT used inside engine.begin().
        assert mock_to_sql.call_count == 2  # results + walkforward
        for call in mock_to_sql.call_args_list:
            con_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("con")
            assert con_arg is engine.connection, (
                "to_sql must use the transactional connection from "
                "engine.begin(), not the engine itself; otherwise the "
                "child writes commit independently and a partial failure "
                "leaves a zombie optimizer_runs row."
            )

    def test_skips_walkforward_block_when_csv_missing(self, tmp_path: Path) -> None:
        engine = _FakeEngine()
        results_csv = _write_results_csv(tmp_path)

        with patch.object(ior, "create_engine", return_value=engine), \
             patch("pandas.DataFrame.to_sql") as mock_to_sql:
            ior.ingest_results(
                results_csv=results_csv,
                run_name="no-wf",
                walkforward_csv=None,
                database_url="postgresql://stub",
            )

        # Still a single transaction, with only optimizer_results inside.
        assert engine.begin_calls == 1
        assert mock_to_sql.call_count == 1
        target_table = mock_to_sql.call_args_list[0].args[0]
        assert target_table == "optimizer_results"
