"""Unit tests for ``services.option_row_sanitizer.is_quote_valid``.

Audit Refactor #4 follow-up: validator extracted from
``decision_job._get_option_rows`` so the read-path filter and the ingest
binder share one definition of "usable two-sided quote with a known
delta + strike". These tests pin the contract.
"""
from __future__ import annotations

import pytest

from spx_backend.services.option_row_sanitizer import is_quote_valid


class TestNullRejections:
    """Any None-valued required field must reject the row."""

    @pytest.mark.parametrize(
        "bid, ask, delta, strike",
        [
            (None, 1.0, 0.3, 100.0),  # null bid
            (0.5, None, 0.3, 100.0),  # null ask
            (0.5, 1.0, None, 100.0),  # null delta
            (0.5, 1.0, 0.3, None),    # null strike
            (None, None, None, None), # all null
        ],
    )
    def test_rejects_any_null_required_field(self, bid, ask, delta, strike):
        """Confirms each individual NULL trips the reject path."""
        assert is_quote_valid(bid, ask, delta, strike) is False


class TestSignGuards:
    """Sign-violation cases the ingest sanitizer also rejects."""

    def test_rejects_zero_ask(self):
        """``ask == 0`` means no live offer; not a tradable two-sided market."""
        assert is_quote_valid(0.5, 0.0, 0.3, 100.0) is False

    def test_rejects_negative_ask(self):
        """Negative ask is impossible (vendor data corruption)."""
        assert is_quote_valid(0.5, -0.1, 0.3, 100.0) is False

    def test_rejects_negative_bid(self):
        """Negative bid is impossible (vendor data corruption)."""
        assert is_quote_valid(-0.1, 1.0, 0.3, 100.0) is False

    def test_accepts_zero_bid(self):
        """``bid == 0`` is legitimate for far-OTM options awaiting a buyer."""
        assert is_quote_valid(0.0, 0.05, 0.01, 100.0) is True

    def test_accepts_locked_market(self):
        """``bid == ask`` (locked / crossed) is rare but not invalid here."""
        assert is_quote_valid(0.5, 0.5, 0.3, 100.0) is True


class TestHappyPath:
    """Realistic SPX/SPY-shaped valid quotes."""

    def test_valid_short_strike_quote(self):
        """Typical short-leg credit-spread quote shape."""
        assert is_quote_valid(bid=2.10, ask=2.30, delta=-0.30, strike=4500.0) is True

    def test_valid_long_strike_quote(self):
        """Typical long-leg quote at lower delta."""
        assert is_quote_valid(bid=0.85, ask=1.05, delta=-0.18, strike=4475.0) is True

    def test_accepts_zero_delta(self):
        """Deep-OTM quote with delta rounded to 0.0 is still usable."""
        assert is_quote_valid(bid=0.05, ask=0.10, delta=0.0, strike=4000.0) is True

    def test_accepts_negative_delta(self):
        """Delta sign is the caller's concern (puts have negative delta)."""
        assert is_quote_valid(bid=1.0, ask=1.2, delta=-0.4, strike=4500.0) is True
