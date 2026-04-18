"""Unit tests for spx_backend.utils.options.resolve_option_right.

Resolution priority is short_leg -> long_leg -> strategy_type substring;
mismatched legs return None with a warning rather than picking one.
"""
from __future__ import annotations

import pytest
from loguru import logger

from spx_backend.utils.options import resolve_option_right


class TestResolveOptionRightFromLegs:
    """Leg-level option_right wins over strategy_type substring."""

    def test_short_leg_put_single_letter(self) -> None:
        # Schema stores 'C' or 'P' in trade_legs.option_right.
        result = resolve_option_right(None, {"option_right": "P"}, None)
        assert result == "put"

    def test_short_leg_call_single_letter(self) -> None:
        result = resolve_option_right(None, {"option_right": "C"}, None)
        assert result == "call"

    def test_short_leg_word_form(self) -> None:
        # Some upstream callers pass full words for clarity.
        assert resolve_option_right(None, {"option_right": "Put"}, None) == "put"
        assert resolve_option_right(None, {"option_right": "CALL"}, None) == "call"

    def test_short_leg_lowercase(self) -> None:
        assert resolve_option_right(None, {"option_right": "p"}, None) == "put"
        assert resolve_option_right(None, {"option_right": "c"}, None) == "call"

    def test_long_leg_used_when_short_missing(self) -> None:
        # Short leg has no option_right set; fall through to long leg.
        result = resolve_option_right(
            None,
            {"strike": 4500.0},
            {"option_right": "P"},
        )
        assert result == "put"

    def test_short_leg_wins_over_long_leg_when_both_present(self) -> None:
        # Both legs agree (puts), so either source would give the same
        # answer; this test pins the resolution-order docstring.
        result = resolve_option_right(
            None,
            {"option_right": "P"},
            {"option_right": "P"},
        )
        assert result == "put"

    def test_short_leg_wins_over_strategy_type(self) -> None:
        # Even if strategy_type says "call", the leg-level data is
        # authoritative.
        result = resolve_option_right(
            "credit_vertical_call",
            {"option_right": "P"},
            None,
        )
        assert result == "put"


class TestResolveOptionRightMismatch:
    """Disagreeing legs must return None and emit a warning."""

    def test_legs_disagree_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        # loguru routes through stderr by default; redirect into caplog
        # so we can assert the warning landed.
        sink_id = logger.add(caplog.handler, format="{message}")
        try:
            result = resolve_option_right(
                "credit_vertical_put",
                {"option_right": "P"},
                {"option_right": "C"},
            )
        finally:
            logger.remove(sink_id)
        assert result is None
        assert any("resolve_option_right_mismatch" in r.message for r in caplog.records)


class TestResolveOptionRightStrategyTypeFallback:
    """strategy_type substring is only consulted when legs are silent."""

    def test_strategy_type_put_used_when_legs_null(self) -> None:
        # Legacy row: trade_legs.option_right is NULL on both legs but
        # strategy_type carries the right.
        result = resolve_option_right(
            "credit_vertical_put",
            {"option_right": None},
            {"option_right": None},
        )
        assert result == "put"

    def test_strategy_type_call_used_when_legs_missing(self) -> None:
        # Both leg dicts entirely missing the option_right key.
        result = resolve_option_right("credit_vertical_call", {}, {})
        assert result == "call"

    def test_strategy_type_with_no_keyword_returns_none(self) -> None:
        # "credit_spread" lacks both substrings; can't decide.
        result = resolve_option_right("credit_spread", {}, {})
        assert result is None

    def test_all_sources_absent(self) -> None:
        assert resolve_option_right(None, None, None) is None

    def test_strategy_type_empty_string(self) -> None:
        assert resolve_option_right("", {}, {}) is None


class TestResolveOptionRightUnknownTokens:
    """Garbage option_right values fall through to next source."""

    def test_unknown_short_leg_falls_through_to_long(self) -> None:
        # Short leg has an unrecognized token; long leg has a valid
        # one; we should use the long leg.
        result = resolve_option_right(
            None,
            {"option_right": "FOO"},
            {"option_right": "C"},
        )
        assert result == "call"

    def test_unknown_short_leg_falls_through_to_strategy_type(self) -> None:
        result = resolve_option_right(
            "credit_vertical_put",
            {"option_right": "X"},
            {"option_right": None},
        )
        assert result == "put"
