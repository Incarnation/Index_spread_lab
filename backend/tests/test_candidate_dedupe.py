"""Tests for ``backend/spx_backend/services/candidate_dedupe.py``.

Covers the audit M3 fix: live and backtest paths must produce the
SAME dedup key for the same underlying spread legs, regardless of
which input shape (nested ``chosen_legs_json`` vs flat row columns)
the caller happens to be using.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from spx_backend.services.candidate_dedupe import (
    CandidateKey,
    candidate_dedupe_key,
)


# ---------------------------------------------------------------------------
# Live shape (nested ``chosen_legs_json``)
# ---------------------------------------------------------------------------


class TestLiveShape:
    """Mimics the dict shape ``DecisionJob`` builds before persisting."""

    def test_returns_4_tuple(self) -> None:
        candidate = {
            "expiration": "2026-04-18",
            "chosen_legs_json": {
                "spread_side": "put",
                "short": {"symbol": "SPXW  260418P05000000"},
                "long":  {"symbol": "SPXW  260418P04990000"},
            },
        }
        key = candidate_dedupe_key(candidate)
        assert isinstance(key, tuple)
        assert len(key) == 4
        assert key == (
            "put",
            "2026-04-18",
            "SPXW  260418P05000000",
            "SPXW  260418P04990000",
        )

    def test_expiration_falls_back_to_chosen_legs(self) -> None:
        # When the top-level expiration is missing, the nested one wins.
        candidate = {
            "chosen_legs_json": {
                "spread_side": "call",
                "short": {"symbol": "AAA"},
                "long":  {"symbol": "BBB"},
                "expiration": "2026-05-01",
            },
        }
        assert candidate_dedupe_key(candidate) == (
            "call", "2026-05-01", "AAA", "BBB",
        )

    def test_missing_legs_collapse_to_empty_strings(self) -> None:
        candidate = {"chosen_legs_json": {}}
        assert candidate_dedupe_key(candidate) == ("", "", "", "")


# ---------------------------------------------------------------------------
# Flat row shape (backtest training_candidates row)
# ---------------------------------------------------------------------------


class TestFlatShape:
    """Mimics the columns present in ``training_candidates.csv`` rows."""

    def test_returns_4_tuple_for_flat_row(self) -> None:
        row = {
            "spread_side": "put",
            "expiration": "2026-04-18",
            "short_symbol": "SPXW  260418P05000000",
            "long_symbol":  "SPXW  260418P04990000",
        }
        assert candidate_dedupe_key(row) == (
            "put",
            "2026-04-18",
            "SPXW  260418P05000000",
            "SPXW  260418P04990000",
        )

    def test_pandas_series_works(self) -> None:
        # ``pd.Series`` exposes ``.get()`` so it should drop in cleanly.
        row = pd.Series({
            "spread_side": "call",
            "expiration": "2026-04-18",
            "short_symbol": "SHORT",
            "long_symbol":  "LONG",
        })
        assert candidate_dedupe_key(row) == (
            "call", "2026-04-18", "SHORT", "LONG",
        )

    def test_nan_collapses_to_empty_string(self) -> None:
        # The ``training_candidates.csv`` reader leaves missing cells as
        # ``float('nan')``, which would otherwise break tuple equality
        # because ``nan != nan``.
        row = {
            "spread_side": "put",
            "expiration": float("nan"),
            "short_symbol": "SHORT",
            "long_symbol": float("nan"),
        }
        key = candidate_dedupe_key(row)
        assert key == ("put", "", "SHORT", "")


# ---------------------------------------------------------------------------
# Cross-shape parity: this is the actual M3 fix
# ---------------------------------------------------------------------------


class TestLiveBacktestParity:
    """The same underlying legs MUST produce the same key in both shapes."""

    def test_same_legs_same_key(self) -> None:
        live = {
            "expiration": "2026-04-18",
            "chosen_legs_json": {
                "spread_side": "put",
                "short": {"symbol": "SHORT_SYM"},
                "long":  {"symbol": "LONG_SYM"},
            },
        }
        backtest_row = {
            "spread_side": "put",
            "expiration": "2026-04-18",
            "short_symbol": "SHORT_SYM",
            "long_symbol": "LONG_SYM",
        }
        assert candidate_dedupe_key(live) == candidate_dedupe_key(backtest_row)

    def test_different_short_symbol_yields_different_key(self) -> None:
        a = {
            "spread_side": "put",
            "expiration": "2026-04-18",
            "short_symbol": "SHORT_A",
            "long_symbol": "LONG",
        }
        b = {**a, "short_symbol": "SHORT_B"}
        assert candidate_dedupe_key(a) != candidate_dedupe_key(b)

    def test_different_index_same_legs_same_key(self) -> None:
        # The audit's M3 motivation: two backtest rows with different
        # DataFrame indices but identical legs MUST be deduped.
        df = pd.DataFrame([
            {
                "spread_side": "put",
                "expiration": "2026-04-18",
                "short_symbol": "SHORT",
                "long_symbol":  "LONG",
                "credit_to_width": 0.30,
            },
            {
                "spread_side": "put",
                "expiration": "2026-04-18",
                "short_symbol": "SHORT",
                "long_symbol":  "LONG",
                "credit_to_width": 0.25,  # different ranking, same legs
            },
        ], index=[42, 99])
        keys = {candidate_dedupe_key(row) for _, row in df.iterrows()}
        assert len(keys) == 1, (
            "M3 regression: two rows with identical legs must dedup to "
            "ONE key regardless of DataFrame index"
        )


# ---------------------------------------------------------------------------
# Defensive coverage
# ---------------------------------------------------------------------------


class TestDefensive:
    @pytest.mark.parametrize("nullish", [None, "", 0])
    def test_empty_chosen_legs_falls_back_to_flat(self, nullish: Any) -> None:
        # When ``chosen_legs_json`` is None / empty / zero, the flat
        # columns should still be picked up.
        row = {
            "chosen_legs_json": nullish,
            "spread_side": "put",
            "expiration": "2026-04-18",
            "short_symbol": "X",
            "long_symbol": "Y",
        }
        # Note: ``0`` is falsy so ``or {}`` triggers; the behavior is
        # documented in the helper.
        assert candidate_dedupe_key(row) == ("put", "2026-04-18", "X", "Y")

    def test_keys_are_hashable(self) -> None:
        # The whole point: keys go into a ``set``.
        candidate_a = {
            "spread_side": "put", "expiration": "2026-04-18",
            "short_symbol": "A", "long_symbol": "B",
        }
        candidate_b = {**candidate_a, "short_symbol": "C"}
        seen: set[CandidateKey] = set()
        seen.add(candidate_dedupe_key(candidate_a))
        seen.add(candidate_dedupe_key(candidate_b))
        seen.add(candidate_dedupe_key(candidate_a))  # duplicate
        assert len(seen) == 2

    def test_decision_job_wrapper_matches_shared_helper(self) -> None:
        # Sanity check: the DecisionJob wrapper must produce the same
        # value as the shared helper (otherwise the live path could
        # drift again post-M3).
        from spx_backend.jobs.decision_job import DecisionJob

        candidate = {
            "expiration": "2026-04-18",
            "chosen_legs_json": {
                "spread_side": "call",
                "short": {"symbol": "SHORT_X"},
                "long":  {"symbol": "LONG_Y"},
            },
        }
        # We don't need a fully-wired DecisionJob to call the bound
        # method -- the wrapper is a pure function in disguise.
        wrapper_key = DecisionJob._candidate_dedupe_key(
            DecisionJob.__new__(DecisionJob), candidate,
        )
        assert wrapper_key == candidate_dedupe_key(candidate)
