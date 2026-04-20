"""Wave 5 (audit) C1 parity gate: gex_job vs cboe_gex_job ≤ 1% on a synthetic chain.

The audit's C1 finding required that the two GEX writers produce
directly-comparable per-strike dollar GEX numbers when fed the same
underlying physical state. After Wave 1's vendor-units correction
(``apply_vendor_units`` in ``services/gex_math.py``) the two paths
should agree to within float-rounding tolerance on a synthetic 1-call
+ 1-put chain.

This test pins that contract by:

1. Constructing a single ``(strike=spot, OI, gamma)`` row for both a
   call and a put.
2. Computing the expected per-strike dollar GEX via the canonical
   ``compute_gex_per_strike`` (the TRADIER path).
3. Computing the same per-strike dollar GEX via the CBOE adapter
   ``apply_vendor_units`` against the vendor's "dollars per share per
   1% move" units (i.e. the canonical formula divided by the 100x
   contract multiplier so the adapter restores it).
4. Asserting the two agree within 1% (the C1 gate).

If this test ever fails, EITHER the canonical formula has drifted OR
the vendor-units adapter has drifted; in both cases the audit
demands a code-level fix, not a tolerance bump.
"""
from __future__ import annotations

import pytest

from spx_backend.services.gex_math import (
    apply_vendor_units,
    compute_gex_per_strike,
)


def _vendor_units(*, oi: int, gamma: float, spot: float, sign: float) -> float:
    """Build the vendor's dollars-per-share-per-1%-move per-strike value.

    The vendor publishes ``netGamma`` already pre-multiplied by spot^2
    and the per-1%-move scalar (0.01), but WITHOUT the contract
    multiplier. ``apply_vendor_units`` reapplies the * 100 to bring
    the value into SqueezeMetrics dollars; here we synthesize the
    vendor's pre-multiplier value directly so we can drive
    ``apply_vendor_units`` and compare.
    """
    return sign * float(oi) * float(gamma) * (float(spot) ** 2) * 0.01


@pytest.mark.parametrize(
    "oi,gamma,spot",
    [
        (1_000, 0.005, 5500.0),       # ATM SPX
        (250, 0.0008, 6100.0),        # ITM SPX
        (50_000, 0.020, 500.0),       # ATM SPY
    ],
)
def test_call_per_strike_parity_within_one_percent(oi: int, gamma: float, spot: float) -> None:
    """A 1-call chain agrees between TRADIER and CBOE paths to <1%."""
    tradier_call = compute_gex_per_strike(
        oi=oi, gamma_per_share=gamma, spot=spot,
        right="C", contract_multiplier=100,
    )

    vendor_call = _vendor_units(oi=oi, gamma=gamma, spot=spot, sign=1.0)
    cboe_call = apply_vendor_units(vendor_call, spot)

    assert tradier_call != 0
    rel_diff = abs(tradier_call - cboe_call) / abs(tradier_call)
    assert rel_diff < 0.01, (
        f"call parity > 1%: tradier={tradier_call:.6e} cboe={cboe_call:.6e} "
        f"rel_diff={rel_diff:.6%}"
    )


@pytest.mark.parametrize(
    "oi,gamma,spot",
    [
        (1_000, 0.005, 5500.0),       # ATM SPX
        (200, 0.0006, 5200.0),
        (40_000, 0.018, 500.0),       # ATM SPY
    ],
)
def test_put_per_strike_parity_within_one_percent(oi: int, gamma: float, spot: float) -> None:
    """A 1-put chain agrees between TRADIER and CBOE paths to <1%."""
    tradier_put = compute_gex_per_strike(
        oi=oi, gamma_per_share=gamma, spot=spot,
        right="P", contract_multiplier=100,
    )

    vendor_put = _vendor_units(oi=oi, gamma=gamma, spot=spot, sign=-1.0)
    cboe_put = apply_vendor_units(vendor_put, spot)

    assert tradier_put != 0
    rel_diff = abs(tradier_put - cboe_put) / abs(tradier_put)
    assert rel_diff < 0.01, (
        f"put parity > 1%: tradier={tradier_put:.6e} cboe={cboe_put:.6e} "
        f"rel_diff={rel_diff:.6%}"
    )


def test_one_call_plus_one_put_chain_parity() -> None:
    """A 2-row (1C + 1P) synthetic chain agrees within 1% across the writers.

    Mirrors the audit's recommended C1 gate exactly: a synthetic
    minimal chain that exercises both signs and aggregates them.
    """
    spot = 5500.0
    multiplier = 100

    call_oi, call_gamma = 1_000, 0.005
    put_oi, put_gamma = 750, 0.004

    # Tradier path -- per-strike sum.
    tradier_total = (
        compute_gex_per_strike(
            oi=call_oi, gamma_per_share=call_gamma, spot=spot,
            right="C", contract_multiplier=multiplier,
        )
        + compute_gex_per_strike(
            oi=put_oi, gamma_per_share=put_gamma, spot=spot,
            right="P", contract_multiplier=multiplier,
        )
    )

    # CBOE path -- vendor's pre-aggregated netGamma, run through the
    # units adapter.
    vendor_total = (
        _vendor_units(oi=call_oi, gamma=call_gamma, spot=spot, sign=1.0)
        + _vendor_units(oi=put_oi, gamma=put_gamma, spot=spot, sign=-1.0)
    )
    cboe_total = apply_vendor_units(vendor_total, spot)

    assert tradier_total != 0
    rel_diff = abs(tradier_total - cboe_total) / abs(tradier_total)
    assert rel_diff < 0.01, (
        f"chain parity > 1%: tradier={tradier_total:.6e} cboe={cboe_total:.6e} "
        f"rel_diff={rel_diff:.6%}"
    )
