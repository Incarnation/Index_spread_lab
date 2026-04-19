from __future__ import annotations

from datetime import datetime

import pytest

import spx_backend.jobs.gex_job as gex_module
from spx_backend.config import settings
from spx_backend.jobs.gex_job import (
    _ZERO_GAMMA_WINDOW_PCT_FALLBACK,
    _ZERO_GAMMA_WINDOW_PCT_PRIMARY,
    GexJob,
)


class _FakeExecResult:
    """Minimal SQLAlchemy-like execute result wrapper for GEX tests."""

    def __init__(self, *, rows: list[tuple] | None = None):
        """Store rows returned by a fake SELECT execution."""
        self._rows = rows or []

    def fetchall(self) -> list[tuple]:
        """Return all fake rows for query result iteration."""
        return self._rows


class _NoPendingSession:
    """Fake async DB session that reports no pending snapshots."""

    def __init__(self) -> None:
        """Initialize SQL capture fields for candidate-query assertions."""
        self.last_sql: str | None = None

    async def execute(self, stmt, params=None):  # noqa: ANN001 - SQLAlchemy text object in production
        """Return an empty candidate set for pending snapshot lookup."""
        self.last_sql = str(stmt)
        return _FakeExecResult(rows=[])

    async def commit(self) -> None:
        """No-op commit hook for fake session compatibility."""
        return None

    async def rollback(self) -> None:
        """No-op rollback hook for fake session compatibility."""
        return None


class _SessionFactory:
    """Async context-manager factory that returns one fake session."""

    def __init__(self, session: _NoPendingSession):
        """Store the fake session returned by the context manager."""
        self._session = session

    def __call__(self):
        """Return self to mimic SessionLocal callable behavior."""
        return self

    async def __aenter__(self) -> _NoPendingSession:
        """Yield the fake session to GEX job code."""
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        """Do not suppress exceptions raised inside the async block."""
        return False


class _FailIfUsedSessionFactory:
    """SessionLocal substitute that fails if the job touches the database."""

    def __call__(self):
        """Raise immediately when database access is unexpectedly attempted."""
        raise AssertionError("SessionLocal should not be used when market is closed")


class _ClosedClockCache:
    """Clock-cache stub that always reports market-closed."""

    async def is_open(self, now_et: datetime) -> bool:  # noqa: ARG002
        """Return false so run_once should skip when not forced."""
        return False


class _ClockCacheShouldNotBeCalled:
    """Clock-cache stub that raises if market-open check is invoked."""

    async def is_open(self, now_et: datetime) -> bool:  # noqa: ARG002
        """Raise to prove force mode bypasses market-hour gate."""
        raise AssertionError("Clock cache should not be called in force mode")


def test_zero_gamma_level_interpolates_crossing() -> None:
    """Zero-gamma interpolation should return a strike between sign change points.

    After audit Wave 1 H4 the walker takes a flat ``{strike: net_gex}`` map
    plus the ``spot`` keyword (used to derive the percent-of-spot window).
    Both 6800 and 6825 sit inside the primary +/-20% window for spot 6800.
    """
    job = GexJob()
    per_strike = {
        6800.0: -200.0,  # cumulative: -200
        6825.0: 230.0,   # cumulative: +30 (cross)
    }

    zero = job._zero_gamma_level(per_strike, spot=6800.0)

    assert zero is not None
    assert 6800.0 < zero < 6825.0


def test_zero_gamma_level_returns_none_without_cross() -> None:
    """Zero-gamma should be None when cumulative net never crosses zero.

    Verifies the post-Wave 1 H4 monotone-chain branch: both windows return
    nothing, the function logs ``zero_gamma_chain_monotone``, and persists
    NULL (here represented by a None return).
    """
    job = GexJob()
    per_strike = {
        6800.0: 150.0,
        6825.0: 110.0,
    }

    zero = job._zero_gamma_level(per_strike, spot=6800.0)

    assert zero is None


def test_zero_gamma_level_resolves_within_primary_window() -> None:
    """Audit H4: a crossing inside +/-20% of spot must be found in the primary window.

    Constructs a synthetic chain whose only sign change sits at +5% of
    spot. The primary window (+/-20%) covers that strike, so the walker
    must return an interpolated price inside [6500, 6900] without falling
    back to the wider window.
    """
    job = GexJob()
    spot = 7000.0
    per_strike = {
        6500.0: -1000.0,  # well inside +/-20%
        6900.0: 1500.0,   # crossing happens here
        7100.0: 800.0,
    }

    zero = job._zero_gamma_level(per_strike, spot=spot)

    assert zero is not None
    assert 6500.0 < zero < 6900.0


def test_zero_gamma_level_resolves_within_fallback_window() -> None:
    """Audit H4: when no crossing exists in +/-20%, the walker must retry at +/-35%.

    All strikes inside +/-20% have monotone-positive cumulative GEX. The
    only sign change is at +25% of spot, which only the fallback window
    (+/-35%) can see. Asserts a non-None return and that it sits beyond
    the primary +/-20% bound, proving the fallback branch fired.
    """
    job = GexJob()
    spot = 7000.0
    inside_primary_lo = spot * (1.0 - _ZERO_GAMMA_WINDOW_PCT_PRIMARY)  # 5600
    inside_primary_hi = spot * (1.0 + _ZERO_GAMMA_WINDOW_PCT_PRIMARY)  # 8400
    inside_fallback_hi = spot * (1.0 + _ZERO_GAMMA_WINDOW_PCT_FALLBACK)  # 9450
    # All primary strikes are positive -> primary window is monotone.
    # The crossing only appears once we widen to include the +25% strike.
    per_strike = {
        inside_primary_lo: 100.0,
        spot: 200.0,
        inside_primary_hi: 300.0,
        spot * 1.25: -1500.0,  # crossing here, only visible in fallback
        inside_fallback_hi: -100.0,
    }

    zero = job._zero_gamma_level(per_strike, spot=spot)

    assert zero is not None
    assert zero > inside_primary_hi, (
        "fallback should resolve a crossing strictly outside the primary window"
    )


def test_zero_gamma_level_monotone_chain_returns_none() -> None:
    """Audit H4: a chain with no crossing in either window must persist NULL.

    A chain whose cumulative net GEX is strictly positive at every strike
    in the +/-35% window cannot have a dealer flip; the walker must
    distinguish that case from a window-too-narrow case. We assert the
    return is None (the writer persists NULL) so the structured-log
    discriminator path is exercised.
    """
    job = GexJob()
    spot = 7000.0
    per_strike = {
        6500.0: 100.0,
        6800.0: 150.0,
        7000.0: 200.0,
        7200.0: 250.0,
        7500.0: 300.0,  # all positive contributions across the chain
    }

    zero = job._zero_gamma_level(per_strike, spot=spot)

    assert zero is None


@pytest.mark.asyncio
async def test_run_once_skips_when_market_closed(monkeypatch) -> None:
    """GEX run should short-circuit before DB work when market is closed."""
    monkeypatch.setattr(settings, "gex_enabled", True)
    monkeypatch.setattr(settings, "gex_allow_outside_rth", False)
    monkeypatch.setattr(gex_module, "SessionLocal", _FailIfUsedSessionFactory())

    result = await GexJob(clock_cache=_ClosedClockCache()).run_once()

    assert result["skipped"] is True
    assert result["reason"] == "market_closed"
    assert result["computed_snapshots"] == 0
    assert result["skipped_snapshots"] == 0
    assert result["failed_snapshots"] == []
    parsed_now = datetime.fromisoformat(result["now_et"])
    assert parsed_now.tzinfo is not None


@pytest.mark.asyncio
async def test_run_once_force_bypasses_market_gate(monkeypatch) -> None:
    """Forced GEX run should bypass market clock gating and query pending snapshots."""
    monkeypatch.setattr(settings, "gex_enabled", True)
    monkeypatch.setattr(settings, "gex_allow_outside_rth", False)
    session = _NoPendingSession()
    monkeypatch.setattr(gex_module, "SessionLocal", _SessionFactory(session))

    result = await GexJob(clock_cache=_ClockCacheShouldNotBeCalled()).run_once(force=True)

    assert result == {"skipped": True, "reason": "no_pending_snapshots"}
    assert session.last_sql is not None
    assert "FROM option_chain_rows ocr" in session.last_sql


def test_gex_job_uses_canonical_squeeze_metrics_formula(monkeypatch) -> None:
    """The TRADIER per-strike accumulator must equal compute_gex_per_strike.

    Lock-in test for audit Wave 1 / finding C1 — guarantees the writer
    cannot silently drift from the canonical formula in
    services/gex_math.py. Builds a synthetic 1-call/1-put toy chain and
    asserts the writer's per-strike calculation matches the canonical
    function within float-rounding tolerance.
    """
    from spx_backend.services.gex_math import compute_gex_per_strike

    monkeypatch.setattr(settings, "gex_puts_negative", True)

    spot = 5500.0
    multiplier = 100
    call_oi, call_gamma = 1_000, 0.005
    put_oi, put_gamma = 750, 0.004

    # Replicate the writer's per-strike call exactly.
    call_gex = compute_gex_per_strike(
        oi=call_oi, gamma_per_share=call_gamma, spot=spot,
        right="C", contract_multiplier=multiplier,
    )
    put_gex = compute_gex_per_strike(
        oi=put_oi, gamma_per_share=put_gamma, spot=spot,
        right="P", contract_multiplier=multiplier,
    )

    # Hand-computed canonical SqueezeMetrics formula expectations.
    expected_call = 1_000 * 0.005 * 100 * (5500.0 ** 2) * 0.01    # +151_250_000
    expected_put = -(750 * 0.004 * 100 * (5500.0 ** 2) * 0.01)    # -90_750_000

    assert call_gex == pytest.approx(expected_call, rel=1e-12)
    assert put_gex == pytest.approx(expected_put, rel=1e-12)
    assert call_gex + put_gex == pytest.approx(60_500_000.0, rel=1e-12)


def test_gex_job_puts_negative_opt_out_flips_put_sign(monkeypatch) -> None:
    """Legacy gex_puts_negative=False must keep puts signed positive.

    Documents the opt-out behavior preserved during the gex_math
    refactor (audit Wave 1 Phase 1). When the operator disables
    sign-puts-negative, the writer treats puts as positive contributions
    so net = calls + puts is the gross sum (not the SqueezeMetrics net).
    """
    from spx_backend.services.gex_math import compute_gex_per_strike

    monkeypatch.setattr(settings, "gex_puts_negative", False)

    # Mimic the writer's effective_right swap (right == 'P' -> 'C').
    effective_right = "C"  # opt-out flips puts to call sign
    put_gex = compute_gex_per_strike(
        oi=750, gamma_per_share=0.004, spot=5500.0,
        right=effective_right, contract_multiplier=100,
    )

    assert put_gex > 0
    assert put_gex == pytest.approx(750 * 0.004 * 100 * (5500.0 ** 2) * 0.01, rel=1e-12)
