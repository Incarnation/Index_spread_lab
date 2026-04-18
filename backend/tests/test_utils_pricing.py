"""Unit tests for spx_backend.utils.pricing.mid_price.

The strict validation policy is the whole point of this helper, so the
test matrix exhaustively covers every reject condition documented in
the module docstring plus a few canonical accept cases.
"""
from __future__ import annotations

import math

import pytest

from spx_backend.utils.pricing import mid_price


class TestMidPriceAccepts:
    """Cases where mid_price MUST return the canonical (bid+ask)/2."""

    def test_typical_two_sided_quote(self) -> None:
        # Plain happy path: tight bid/ask returns midpoint.
        assert mid_price(4.5, 5.5) == pytest.approx(5.0)

    def test_integer_inputs(self) -> None:
        # Integer inputs coerce cleanly through float().
        assert mid_price(1, 3) == pytest.approx(2.0)

    def test_string_inputs_coerce(self) -> None:
        # Numeric strings are tolerated because tradier feed sometimes
        # delivers stringified floats; only non-numeric strings fail.
        assert mid_price("1.0", "3.0") == pytest.approx(2.0)

    def test_equal_bid_ask(self) -> None:
        # Locked market (bid == ask) is unusual but valid; returns the
        # locked price.
        assert mid_price(5.0, 5.0) == pytest.approx(5.0)

    def test_small_positive_values(self) -> None:
        # Sub-penny ticks (some products quote in 0.01 increments).
        assert mid_price(0.05, 0.10) == pytest.approx(0.075)


class TestMidPriceRejectsMissing:
    """None inputs must always return None."""

    def test_both_none(self) -> None:
        assert mid_price(None, None) is None

    def test_bid_none(self) -> None:
        assert mid_price(None, 5.0) is None

    def test_ask_none(self) -> None:
        assert mid_price(5.0, None) is None


class TestMidPriceRejectsNonNumeric:
    """Inputs that float() cannot coerce must return None."""

    def test_dict_input(self) -> None:
        assert mid_price({"price": 5}, 5.0) is None

    def test_list_input(self) -> None:
        assert mid_price(5.0, [5.0]) is None

    def test_garbage_string(self) -> None:
        assert mid_price("not-a-number", 5.0) is None


class TestMidPriceRejectsNonFinite:
    """NaN and +/-Inf must return None to prevent silent propagation."""

    def test_nan_bid(self) -> None:
        assert mid_price(float("nan"), 5.0) is None

    def test_nan_ask(self) -> None:
        assert mid_price(5.0, float("nan")) is None

    def test_pos_inf(self) -> None:
        assert mid_price(5.0, math.inf) is None

    def test_neg_inf(self) -> None:
        assert mid_price(-math.inf, 5.0) is None


class TestMidPriceRejectsZeroOrNegative:
    """Strict positivity: any zero or negative input rejects."""

    def test_dead_quote_both_zero(self) -> None:
        # The classic "free money" bug: feed dropped both sides; previous
        # _mid copies returned 0 here and the credit-to-width gate
        # silently passed.
        assert mid_price(0.0, 0.0) is None

    def test_zero_bid_positive_ask(self) -> None:
        # Deep-OTM no-bid quote: legitimate market state but we choose
        # to drop these because credit math on such candidates is too
        # noisy to be safe.  Documented in module docstring.
        assert mid_price(0.0, 5.0) is None

    def test_positive_bid_zero_ask(self) -> None:
        # Symmetric case; ask=0 indicates a feed bug far more often
        # than a real market.
        assert mid_price(5.0, 0.0) is None

    def test_negative_bid(self) -> None:
        assert mid_price(-1.0, 5.0) is None

    def test_negative_ask(self) -> None:
        assert mid_price(5.0, -1.0) is None


class TestMidPriceRejectsCrossed:
    """bid > ask is a market-data error, never a tradeable quote."""

    def test_clearly_crossed(self) -> None:
        assert mid_price(5.5, 4.5) is None

    def test_just_crossed(self) -> None:
        # 1-cent crossed; still rejected because we never want to
        # average an impossible quote.
        assert mid_price(5.01, 5.0) is None
