"""Unit tests for :mod:`spx_backend.services.gex_math`.

Locks the SqueezeMetrics convention against hand-computed numbers so
future changes to either writer can't silently re-introduce the C1
magnitude divergence (audit Wave 1).
"""
from __future__ import annotations

import math

import pytest

from spx_backend.services.gex_math import (
    apply_vendor_units,
    compute_gex_per_strike,
)


class TestComputeGexPerStrike:
    """Direct algebraic checks against the SqueezeMetrics formula."""

    def test_call_returns_positive_signed_value(self) -> None:
        """Calls contribute the canonical formula with a + sign."""
        # Hand-computed: 1000 * 0.005 * 100 * 5500^2 * 0.01
        # = 1000 * 0.005 * 100 * 30_250_000 * 0.01
        # = 1000 * 0.005 * 100 * 302_500
        # = 151_250_000
        expected = 1000 * 0.005 * 100 * (5500.0 ** 2) * 0.01
        actual = compute_gex_per_strike(
            oi=1000, gamma_per_share=0.005, spot=5500.0, right="C",
        )
        assert math.isclose(actual, expected, rel_tol=1e-12)
        assert actual > 0
        assert math.isclose(actual, 151_250_000.0, rel_tol=1e-12)

    def test_put_returns_negative_signed_value(self) -> None:
        """Puts contribute the canonical formula with a - sign."""
        expected = -(1000 * 0.005 * 100 * (5500.0 ** 2) * 0.01)
        actual = compute_gex_per_strike(
            oi=1000, gamma_per_share=0.005, spot=5500.0, right="P",
        )
        assert math.isclose(actual, expected, rel_tol=1e-12)
        assert actual < 0
        assert math.isclose(actual, -151_250_000.0, rel_tol=1e-12)

    def test_long_form_right_tokens_accepted(self) -> None:
        """Both 'C'/'P' and 'call'/'put' yield the same magnitudes."""
        short_call = compute_gex_per_strike(
            oi=500, gamma_per_share=0.001, spot=4500.0, right="C",
        )
        long_call = compute_gex_per_strike(
            oi=500, gamma_per_share=0.001, spot=4500.0, right="call",
        )
        assert math.isclose(short_call, long_call, rel_tol=1e-12)

        short_put = compute_gex_per_strike(
            oi=500, gamma_per_share=0.001, spot=4500.0, right="P",
        )
        long_put = compute_gex_per_strike(
            oi=500, gamma_per_share=0.001, spot=4500.0, right="put",
        )
        assert math.isclose(short_put, long_put, rel_tol=1e-12)

    def test_unknown_right_raises_value_error(self) -> None:
        """Unrecognized option-right tokens are rejected, not silently mis-signed."""
        with pytest.raises(ValueError, match="unsupported option right"):
            compute_gex_per_strike(
                oi=100, gamma_per_share=0.001, spot=5000.0, right="X",  # type: ignore[arg-type]
            )

    def test_zero_oi_returns_zero(self) -> None:
        """No open interest at a strike contributes nothing to GEX."""
        assert compute_gex_per_strike(
            oi=0, gamma_per_share=0.005, spot=5500.0, right="C",
        ) == 0.0

    def test_zero_gamma_returns_zero(self) -> None:
        """Deep ITM/OTM strikes with zero gamma contribute nothing."""
        assert compute_gex_per_strike(
            oi=10_000, gamma_per_share=0.0, spot=5500.0, right="C",
        ) == 0.0
        assert compute_gex_per_strike(
            oi=10_000, gamma_per_share=0.0, spot=5500.0, right="P",
        ) == 0.0

    def test_call_plus_put_at_same_strike_with_same_oi_and_gamma_nets_to_zero(self) -> None:
        """Equal OI / equal gamma at a strike yields zero net GEX (sign cancellation)."""
        call_gex = compute_gex_per_strike(
            oi=2_500, gamma_per_share=0.003, spot=5500.0, right="C",
        )
        put_gex = compute_gex_per_strike(
            oi=2_500, gamma_per_share=0.003, spot=5500.0, right="P",
        )
        assert math.isclose(call_gex + put_gex, 0.0, abs_tol=1e-9)

    def test_custom_contract_multiplier_scales_linearly(self) -> None:
        """Non-default multipliers (e.g. mini-options) scale the result linearly."""
        baseline = compute_gex_per_strike(
            oi=100, gamma_per_share=0.002, spot=4000.0, right="C",
            contract_multiplier=100,
        )
        mini = compute_gex_per_strike(
            oi=100, gamma_per_share=0.002, spot=4000.0, right="C",
            contract_multiplier=10,
        )
        assert math.isclose(baseline / mini, 10.0, rel_tol=1e-12)

    def test_dimensional_check_spx_atm_5_dte(self) -> None:
        """SPX ATM with 5-DTE-typical inputs lands in the canonical 1e8-1e10 range.

        Sanity-checks the formula against expected industry magnitudes
        (SqueezeMetrics SPX GEX is reported in $1B-$30B for the full
        chain; one ATM strike contributes ~$1e8). If this assertion
        ever fails, the formula has drifted from convention.
        """
        # Realistic 5-DTE SPX ATM inputs.
        gex = compute_gex_per_strike(
            oi=20_000, gamma_per_share=0.002, spot=5500.0, right="C",
        )
        # Expected: 20_000 * 0.002 * 100 * 30_250_000 * 0.01
        # = 20_000 * 0.002 * 100 * 302_500 = 1.21e9
        assert 1e8 < gex < 1e10, f"out-of-canon magnitude: {gex:.3e}"


class TestApplyVendorUnits:
    """Adapter conversion from MZData / CBOE units to SqueezeMetrics dollars.

    The empirical 100x scalar was derived in audit Wave 1 Phase 1 by
    comparing one TRADIER snapshot's stored gex_net against the same-date
    CBOE snapshot for SPX 2026-04-17 (TRADIER 5.96e10 vs CBOE 5.879e8;
    ratio 101.4x ~= 100x contract multiplier). See gex_math.py for the
    full evidence trail.
    """

    def test_applies_100x_contract_multiplier_correction(self) -> None:
        """Vendor value times 100 yields SqueezeMetrics canonical units.

        Pinned to the live-DB observation: CBOE per-snapshot gex_net of
        ~5.879e8 for SPX corresponds to canonical ~5.879e10.
        """
        assert apply_vendor_units(5.879e8, spot=7126.0) == pytest.approx(5.879e10, rel=1e-12)
        assert apply_vendor_units(-2_166_000.0, spot=7126.0) == pytest.approx(-216_600_000.0, rel=1e-12)

    def test_zero_vendor_value_returns_zero(self) -> None:
        """No vendor exposure yields zero canonical exposure."""
        assert apply_vendor_units(0.0, spot=5500.0) == 0.0

    def test_negative_vendor_value_preserves_sign(self) -> None:
        """Sign convention pass-through: negative vendor exposure stays negative."""
        result = apply_vendor_units(-1_000_000.0, spot=5500.0)
        assert result == -100_000_000.0
        assert result < 0

    def test_spot_argument_is_accepted_but_unused_in_constant_scalar(self) -> None:
        """Spot kept in signature so a future spot-dependent correction is one edit.

        Verifies the function handles a wide spot range without raising
        and that the result is invariant in spot for the constant-scalar
        adapter.
        """
        for spot in (100.0, 1_000.0, 5_500.0, 20_000.0):
            assert apply_vendor_units(123.45, spot=spot) == pytest.approx(12_345.0, rel=1e-12)
