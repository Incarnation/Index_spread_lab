"""Wave 5 (audit) coverage tests for gex_job.run_once happy + edge paths.

The existing ``test_gex_job.py`` covers the canonical SqueezeMetrics
formula and the ``_zero_gamma_level`` walker in isolation. This file
exercises ``run_once`` end-to-end with a fake async session so we
cover the hot orchestration paths the audit identified as gaps:

* M11 chain-age guard skip + force-override
* DTE-cap edge filtering (exp beyond gex_max_dte_days drops out)
* strike-limit edge (only top-N strikes near spot enter the aggregate)
* missing-greeks (gamma None / oi None) skip cleanly
* ON CONFLICT DO NOTHING SQL is emitted at the per-snapshot upsert
* ``_get_spot_price`` staleness guard returns None when too old
* Successful end-to-end snapshot computation increments ``computed_snapshots``
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import spx_backend.jobs.gex_job as gex_module
from spx_backend.config import settings
from spx_backend.jobs.gex_job import GexJob


class _FakeExecResult:
    """Tiny SQLAlchemy-execute-result substitute for unit tests."""

    def __init__(self, *, rows=None, one=None):
        """Capture pre-canned ``fetchall`` and ``fetchone`` payloads."""
        self._rows = rows or []
        self._one = one

    def fetchall(self):
        """Return the pre-canned ``fetchall`` rows."""
        return self._rows

    def fetchone(self):
        """Return the pre-canned ``fetchone`` row."""
        return self._one


class _AlwaysOpenClock:
    """Clock cache stub that always reports market-open."""

    async def is_open(self, now_et: datetime) -> bool:  # noqa: ARG002
        """Return True so RTH gating never short-circuits."""
        return True


class _ScriptedSession:
    """Async DB session that walks a scripted call sequence.

    Each test populates ``self.script`` with a list of
    ``(matcher, response)`` pairs in production-call order; ``execute``
    walks the list, returning the matched response and capturing
    ``(sql, params)`` for assertion.
    """

    def __init__(self) -> None:
        """Initialize empty script + capture buckets."""
        self.script: list[tuple] = []
        self.executes: list[tuple[str, dict]] = []
        self.commit_called = 0
        self.rollback_called = 0

    class _Nested:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def begin_nested(self):
        """Return a no-op nested-tx context manager."""
        return self._Nested()

    async def execute(self, stmt, params=None):  # noqa: ANN001
        """Walk the scripted matcher list and return the canned result."""
        sql = str(stmt)
        bound = dict(params or {})
        self.executes.append((sql, bound))
        for needle, response in self.script:
            if needle in sql:
                # Allow callable responses for context-sensitive payloads
                # (e.g. multi-call SELECTs).
                return response(bound) if callable(response) else response
        return _FakeExecResult()

    async def commit(self) -> None:
        """Increment commit counter."""
        self.commit_called += 1

    async def rollback(self) -> None:
        """Increment rollback counter."""
        self.rollback_called += 1


class _SessionFactory:
    """Async-context-manager factory yielding the supplied session."""

    def __init__(self, session: _ScriptedSession):
        """Store the scripted session to yield."""
        self._session = session

    def __call__(self):
        """Return self -- mimics SessionLocal callable behavior."""
        return self

    async def __aenter__(self) -> _ScriptedSession:
        """Yield the scripted session."""
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        """Do not swallow exceptions raised inside the block."""
        return False


@pytest.fixture
def _patch_settings(monkeypatch):
    """Pin gex_job settings for deterministic tests."""
    monkeypatch.setattr(settings, "gex_enabled", True)
    monkeypatch.setattr(settings, "gex_allow_outside_rth", True)
    monkeypatch.setattr(settings, "gex_snapshot_batch_limit", 5)
    monkeypatch.setattr(settings, "gex_max_dte_days", 10)
    monkeypatch.setattr(settings, "gex_strike_limit", 3)
    monkeypatch.setattr(settings, "gex_contract_multiplier", 100)
    monkeypatch.setattr(settings, "gex_puts_negative", True)
    monkeypatch.setattr(settings, "gex_store_by_expiry", True)
    monkeypatch.setattr(settings, "gex_chain_max_age_seconds", 0)
    monkeypatch.setattr(settings, "gex_spot_max_age_seconds", 3600)
    return settings


# ---------------------------------------------------------------------------
# M11: chain-age guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_age_guard_skips_old_snapshot(monkeypatch, _patch_settings) -> None:
    """A chain older than gex_chain_max_age_seconds is skipped + counted."""
    monkeypatch.setattr(settings, "gex_chain_max_age_seconds", 60)

    very_old_ts = datetime.now(timezone.utc) - timedelta(hours=2)
    session = _ScriptedSession()
    session.script = [
        # candidate query returns one OLD snapshot
        ("FROM chain_snapshots cs", _FakeExecResult(rows=[(101, very_old_ts, "SPX", 7)])),
    ]
    monkeypatch.setattr(gex_module, "SessionLocal", _SessionFactory(session))

    result = await GexJob(clock_cache=_AlwaysOpenClock()).run_once()

    assert result["skipped"] is False
    assert result["computed_snapshots"] == 0
    assert result["skipped_snapshots"] == 1
    # No spot-price SELECT or option_chain_rows SELECT should fire because
    # the age guard short-circuits BEFORE _get_spot_price.
    assert not any(
        "FROM option_chain_rows" in sql and "WHERE snapshot_id" in sql
        for sql, _ in session.executes
    )


@pytest.mark.asyncio
async def test_chain_age_guard_force_overrides(monkeypatch, _patch_settings) -> None:
    """force=True must bypass the chain-age guard."""
    monkeypatch.setattr(settings, "gex_chain_max_age_seconds", 60)

    very_old_ts = datetime.now(timezone.utc) - timedelta(hours=2)
    session = _ScriptedSession()
    session.script = [
        ("FROM chain_snapshots cs", _FakeExecResult(rows=[(101, very_old_ts, "SPX", 7)])),
        # _get_spot_price returns nothing -> snapshot still skipped, but the
        # guard did NOT trigger because we got past it.
        ("FROM underlying_quotes", _FakeExecResult(one=None)),
    ]
    monkeypatch.setattr(gex_module, "SessionLocal", _SessionFactory(session))

    result = await GexJob(clock_cache=_AlwaysOpenClock()).run_once(force=True)

    assert result["skipped"] is False
    # Guard bypassed; no-spot path increments skipped_snapshots for a
    # different reason (no spot price), but the SELECT against
    # underlying_quotes proves we got past the guard.
    assert any("FROM underlying_quotes" in sql for sql, _ in session.executes)


# ---------------------------------------------------------------------------
# Successful end-to-end run with DTE-cap + strike-limit edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_dte_and_strike_limits_apply(monkeypatch, _patch_settings) -> None:
    """A 5-strike, 2-DTE chain narrows to top-3 strikes inside the DTE cap.

    * gex_strike_limit = 3 (set in fixture) so only the 3 strikes nearest
      spot survive the strike filter.
    * gex_max_dte_days = 10 so the 30-DTE expiration is dropped.
    * gex_puts_negative = True so the per-strike puts contribute negatively.
    """
    fresh_ts = datetime.now(timezone.utc)
    spot_value = 5500.0

    # Build the chain: 4 strikes within the DTE cap + 1 strike OUTSIDE the cap.
    near_dte_exp = fresh_ts.date() + timedelta(days=5)
    far_dte_exp = fresh_ts.date() + timedelta(days=30)  # outside gex_max_dte_days

    chain_rows = [
        # (strike, right, oi, gamma, contract_size, expiration)
        (5500.0, "C", 1000, 0.005, 100, near_dte_exp),    # ATM call (closest)
        (5500.0, "P", 1000, 0.005, 100, near_dte_exp),
        (5550.0, "C", 500, 0.004, 100, near_dte_exp),     # +50 (near)
        (5450.0, "P", 600, 0.004, 100, near_dte_exp),     # -50 (near)
        (5800.0, "C", 200, 0.002, 100, near_dte_exp),     # +300 -- gets pruned by top-3 (5500/5550/5450)
        (6000.0, "C", 100, 0.001, 100, far_dte_exp),      # outside DTE cap
    ]

    session = _ScriptedSession()
    session.script = [
        ("FROM chain_snapshots cs", _FakeExecResult(rows=[(101, fresh_ts, "SPX", 5)])),
        ("FROM underlying_quotes", _FakeExecResult(one=(spot_value, fresh_ts))),
        ("FROM option_chain_rows", _FakeExecResult(rows=chain_rows)),
    ]
    monkeypatch.setattr(gex_module, "SessionLocal", _SessionFactory(session))

    result = await GexJob(clock_cache=_AlwaysOpenClock()).run_once()

    assert result["skipped"] is False
    assert result["computed_snapshots"] == 1
    assert result["skipped_snapshots"] == 0
    assert result["failed_snapshots"] == []

    # Verify the per-snapshot gex_snapshots upsert fired with DO NOTHING (M6).
    gex_snapshot_inserts = [
        params for sql, params in session.executes
        if "INSERT INTO gex_snapshots" in sql and "ON CONFLICT (snapshot_id) DO NOTHING" in sql
    ]
    assert len(gex_snapshot_inserts) == 1, (
        "Expected exactly one gex_snapshots upsert with the M6 DO NOTHING policy"
    )
    payload = gex_snapshot_inserts[0]
    # ATM ($5500) call dominates; net should be > 0 since calls > puts in OI.
    assert payload["gex_calls"] > 0
    assert payload["gex_puts"] < 0  # signed negative per gex_puts_negative=True
    # Top-3 strike pruning means 5800 (the +300 outlier) and the
    # 30-DTE call must NOT contribute to gex_abs.
    # Hand check: only 5500C/5500P/5550C/5450P entered the per-strike map.
    expected_calls = (
        # strike 5500 C: 1000 * 0.005 * 100 * 5500^2 * 0.01 = 151_250_000
        1000 * 0.005 * 100 * (5500.0 ** 2) * 0.01
        # strike 5550 C: 500 * 0.004 * 100 * 5500^2 * 0.01 = 60_500_000
        + 500 * 0.004 * 100 * (5500.0 ** 2) * 0.01
    )
    expected_puts = -(
        1000 * 0.005 * 100 * (5500.0 ** 2) * 0.01
        + 600 * 0.004 * 100 * (5500.0 ** 2) * 0.01
    )
    assert payload["gex_calls"] == pytest.approx(expected_calls, rel=1e-9)
    assert payload["gex_puts"] == pytest.approx(expected_puts, rel=1e-9)


# ---------------------------------------------------------------------------
# Missing greeks / open interest are silently dropped (don't crash)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_greeks_rows_are_skipped(monkeypatch, _patch_settings) -> None:
    """Rows with NULL gamma or NULL oi must be dropped, not crash.

    Builds a chain with a row whose gamma is None and a row whose oi is
    None alongside a valid row; we then assert the run succeeds with
    ``computed_snapshots=1`` (the writer skipped the bad rows but
    happily wrote the good row).
    """
    fresh_ts = datetime.now(timezone.utc)
    near_dte_exp = fresh_ts.date() + timedelta(days=5)

    chain_rows = [
        (5500.0, "C", 1000, 0.005, 100, near_dte_exp),  # OK
        (5505.0, "C", None, 0.005, 100, near_dte_exp),  # missing oi
        (5510.0, "C", 1000, None, 100, near_dte_exp),   # missing gamma
    ]

    session = _ScriptedSession()
    session.script = [
        ("FROM chain_snapshots cs", _FakeExecResult(rows=[(202, fresh_ts, "SPX", 5)])),
        ("FROM underlying_quotes", _FakeExecResult(one=(5500.0, fresh_ts))),
        ("FROM option_chain_rows", _FakeExecResult(rows=chain_rows)),
    ]
    monkeypatch.setattr(gex_module, "SessionLocal", _SessionFactory(session))

    result = await GexJob(clock_cache=_AlwaysOpenClock()).run_once()

    assert result["skipped"] is False
    assert result["computed_snapshots"] == 1
    assert result["failed_snapshots"] == []


# ---------------------------------------------------------------------------
# _get_spot_price staleness guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_spot_price_returns_none_when_too_old(monkeypatch, _patch_settings) -> None:
    """A spot quote older than gex_spot_max_age_seconds resolves to None.

    Keeps the writer from stamping snapshots with stale spot prices.
    """
    monkeypatch.setattr(settings, "gex_spot_max_age_seconds", 60)

    snapshot_ts = datetime.now(timezone.utc)
    very_old_quote_ts = snapshot_ts - timedelta(hours=2)

    session = _ScriptedSession()
    session.script = [
        # _get_spot_price's SELECT returns a row, but the quote_ts is too old
        ("FROM underlying_quotes", _FakeExecResult(one=(5500.0, very_old_quote_ts))),
    ]

    job = GexJob(clock_cache=_AlwaysOpenClock())
    spot = await job._get_spot_price(session, snapshot_ts, "SPX")

    assert spot is None


@pytest.mark.asyncio
async def test_get_spot_price_returns_value_when_fresh(monkeypatch, _patch_settings) -> None:
    """A fresh spot quote within the staleness budget is returned as float."""
    snapshot_ts = datetime.now(timezone.utc)
    fresh_quote_ts = snapshot_ts - timedelta(seconds=30)

    session = _ScriptedSession()
    session.script = [
        ("FROM underlying_quotes", _FakeExecResult(one=(5550.5, fresh_quote_ts))),
    ]

    job = GexJob(clock_cache=_AlwaysOpenClock())
    spot = await job._get_spot_price(session, snapshot_ts, "SPX")

    assert spot == pytest.approx(5550.5)


# ---------------------------------------------------------------------------
# Disabled gates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_skipped_when_gex_disabled(monkeypatch, _patch_settings) -> None:
    """gex_enabled=False short-circuits before the SessionLocal call."""
    monkeypatch.setattr(settings, "gex_enabled", False)

    def _should_not_be_called():
        raise AssertionError("SessionLocal must not be created when gex_enabled=False")

    monkeypatch.setattr(gex_module, "SessionLocal", _should_not_be_called)

    result = await GexJob(clock_cache=_AlwaysOpenClock()).run_once()

    assert result == {"skipped": True, "reason": "gex_disabled"}
