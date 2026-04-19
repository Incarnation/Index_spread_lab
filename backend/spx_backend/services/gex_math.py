"""Canonical SqueezeMetrics gamma-exposure (GEX) math.

One source of truth for the dollar-GEX formula consumed by both writers
in the ingestion path:

* :mod:`spx_backend.jobs.gex_job` (TRADIER): per-strike accumulation
  from raw ``option_chain_rows`` (``oi``, ``gamma_per_share``).
* :mod:`spx_backend.jobs.cboe_gex_job` (CBOE): per-strike vendor
  ``netGamma`` series from MZData / CBOE; vendor units are converted to
  the SqueezeMetrics convention through :func:`apply_vendor_units`.

The SqueezeMetrics convention (white paper) defines per-strike dollar
GEX as ``OI * gamma * multiplier * S^2 * 0.01`` with calls signed
``+`` and puts signed ``-``. Aggregating across strikes yields the
canonical net dealer GEX in dollars per 1% move in spot. Putting the
formula in one place makes the C1 magnitude correction (audit Wave 1)
auditable in one diff and prevents the two writers from drifting again.

See :doc:`backend/spx_backend/jobs/INGEST_AUDIT.md` (finding C1) for the
divergence history that motivated this extraction.
"""
from __future__ import annotations

from typing import Final, Literal

# SqueezeMetrics GEX convention scales gamma to a 1% move in spot. The
# constant lives at module scope so a one-line edit in this file changes
# it everywhere both writers consume it.
_PCT_MOVE_SCALAR: Final[float] = 0.01

# Standard equity-options contract size (one contract = 100 shares).
# Both SPX and SPY trade on this multiplier. Stays a named constant so a
# future weeklys / mini-options change is one edit.
_DEFAULT_CONTRACT_MULTIPLIER: Final[int] = 100

# Sign convention: dealers are typically short calls (collect premium)
# and short puts. SqueezeMetrics convention writes calls as positive and
# puts as negative so the signed sum yields net dealer gamma directly.
_CALL_SIGN: Final[float] = 1.0
_PUT_SIGN: Final[float] = -1.0

OptionRight = Literal["C", "P", "call", "put", "CALL", "PUT"]


def _resolve_sign(right: OptionRight) -> float:
    """Map an option-right token to the SqueezeMetrics sign (calls + / puts -).

    Accepts the existing ``option_chain_rows.option_right`` payloads
    (``"C"`` / ``"P"``) used by ``gex_job`` and the longer-form
    ``"call"`` / ``"put"`` strings used by some test fixtures and the
    CBOE adapter.
    """
    normalized = right.strip().upper()[:1] if isinstance(right, str) else ""
    if normalized == "C":
        return _CALL_SIGN
    if normalized == "P":
        return _PUT_SIGN
    raise ValueError(f"unsupported option right: {right!r}; expected 'C'/'P' or 'call'/'put'")


def compute_gex_per_strike(
    *,
    oi: int,
    gamma_per_share: float,
    spot: float,
    right: OptionRight,
    contract_multiplier: int = _DEFAULT_CONTRACT_MULTIPLIER,
) -> float:
    """Return signed dollar GEX for a single ``(strike, right)`` row.

    Implements the SqueezeMetrics convention exactly:
    ``sign(right) * OI * gamma_per_share * contract_multiplier * spot^2 * 0.01``.
    Caller aggregates across strikes (and across rights) by summation;
    the signed contributions handle the calls-vs-puts arithmetic.

    Parameters
    ----------
    oi:
        Open interest in **contracts** (not shares). One contract is
        ``contract_multiplier`` shares.
    gamma_per_share:
        Black-Scholes gamma per share, the second derivative of option
        price with respect to spot. Tradier's ``greeks.gamma`` is in
        this convention; values typically in ``0.0001 .. 0.05`` for SPX.
    spot:
        Underlying spot price at snapshot time.
    right:
        ``"C"``/``"call"`` for calls, ``"P"``/``"put"`` for puts.
    contract_multiplier:
        Shares per contract; defaults to 100 (standard equity options).

    Returns
    -------
    float
        Signed dollar GEX contribution for this ``(strike, right)``.
        Zero when ``oi`` or ``gamma_per_share`` is zero.

    Raises
    ------
    ValueError
        When ``right`` is not a recognized option-side token.
    """
    sign = _resolve_sign(right)
    return sign * float(oi) * float(gamma_per_share) * float(contract_multiplier) * (float(spot) ** 2) * _PCT_MOVE_SCALAR


# Vendor unit correction determined empirically (audit Wave 1 Phase 1).
# A live-DB probe compared one TRADIER snapshot's stored gex_net against
# the canonical SqueezeMetrics recomputation (SUM over option_chain_rows
# of `sign * oi * gamma * 100 * spot^2 * 0.01`) and they matched
# exactly: TRADIER's writer is dimensionally correct.
#
# The same probe compared CBOE's per-snapshot gex_net (5.879e8 for SPX
# 2026-04-17) against TRADIER's (5.96e10 for the same date). The ratio
# is 101.4x, matching the standard equity-options contract multiplier
# (100 shares per contract) within rounding. Per-strike cross-checks
# confirmed mzdata publishes its `netGamma` series in dollars-per-share
# (i.e. omits the * 100 contract multiplier) while every other factor
# (per-1%-move scalar, S^2) is already applied vendor-side.
#
# Multiplying by 100 in this adapter brings CBOE into the SqueezeMetrics
# convention so both writers populate `gex_snapshots.gex_net` in
# directly comparable units. The constant lives here so a future
# vendor-units change is one edit.
_MZDATA_NET_GAMMA_SCALAR: Final[float] = 100.0


def apply_vendor_units(vendor_value: float, spot: float) -> float:
    """Convert one mzdata / CBOE per-strike exposure value to SqueezeMetrics dollars.

    The vendor publishes pre-aggregated dealer gamma exposure per strike
    (`callAbsGamma`, `putAbsGamma`, `netGamma`) in dollars-per-share-per-1%-move.
    Multiplying by the contract multiplier (100) yields dollars-per-1%-move
    in the SqueezeMetrics convention used by :func:`compute_gex_per_strike`.

    Parameters
    ----------
    vendor_value:
        Raw per-strike value from ``CboeExposureItem.call_abs_gamma`` /
        ``put_abs_gamma`` / ``net_gamma``.
    spot:
        Underlying spot price at snapshot time. Reserved so a future
        spot-dependent correction (e.g. an additional S-factor in a
        future vendor schema change) is a one-line edit in this file.

    Returns
    -------
    float
        Dollar GEX contribution in SqueezeMetrics units (per 1% move).
    """
    # Spot is unused in the constant-scalar mapping but kept in the
    # signature so callers don't need to change shape if a future
    # vendor-schema migration introduces a spot-dependent correction.
    del spot
    return float(vendor_value) * _MZDATA_NET_GAMMA_SCALAR
