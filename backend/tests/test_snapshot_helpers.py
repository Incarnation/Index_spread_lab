from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import spx_backend.jobs.snapshot_job as snapshot_module
from spx_backend.config import settings
from spx_backend.jobs.snapshot_job import (
    SnapshotJob,
    SnapshotJobConfig,
    _parse_chain_options,
    _select_strikes_near_spot,
)


def test_select_strikes_near_spot_balances_below_and_above() -> None:
    options = [{"strike": s} for s in [6800, 6810, 6820, 6830, 6840, 6850, 6860]]
    selected = _select_strikes_near_spot(options, spot=6835.0, each_side=2)

    # Expect two nearest below and two nearest above-or-at insertion point.
    assert selected == {6820.0, 6830.0, 6840.0, 6850.0}


def test_select_strikes_near_spot_returns_empty_when_no_strikes() -> None:
    selected = _select_strikes_near_spot(options=[{"strike": None}], spot=6835.0, each_side=2)

    assert selected == set()


def test_parse_chain_options_returns_empty_when_options_container_is_none() -> None:
    """Tradier edge payload ``{\"options\": None}`` should not raise."""
    parsed = _parse_chain_options({"options": None})

    assert parsed == []


def test_parse_chain_options_filters_non_dict_entries() -> None:
    """Parser keeps only dict rows when payload list contains malformed entries."""
    parsed = _parse_chain_options({"options": {"option": [{"symbol": "A"}, None, "bad", {"symbol": "B"}]}})

    assert parsed == [{"symbol": "A"}, {"symbol": "B"}]


class _FakeExecResult:
    """Minimal async execute result wrapper for snapshot job tests."""

    def __init__(self, *, fetchone_result=None, scalar_result=None):
        """Store row-like and scalar return values for SQL branches."""
        self._fetchone_result = fetchone_result
        self._scalar_result = scalar_result

    def fetchone(self):
        """Return one row-like object for select queries."""
        return self._fetchone_result

    def scalar_one(self):
        """Return one scalar value for insert-returning queries."""
        return self._scalar_result


class _CaptureSession:
    """Fake session that records SQL inserts and returns canned results."""

    def __init__(self):
        """Initialize empty SQL capture lists."""
        self.chain_insert_params: list[dict] = []
        self.option_insert_params: list[dict] = []

    class _NestedTx:
        """No-op async savepoint context manager for test session stubs."""

        async def __aenter__(self):
            """Enter nested transaction without side effects."""
            return None

        async def __aexit__(self, exc_type, exc, tb):
            """Do not suppress exceptions raised inside nested blocks."""
            return False

    async def execute(self, stmt, params=None):
        """Capture SQL params and provide minimal results by query type."""
        sql = str(stmt)
        if "FROM underlying_quotes" in sql:
            # No spot quote -> store full chain without strike trimming.
            return _FakeExecResult(fetchone_result=None)
        if "INSERT INTO chain_snapshots" in sql:
            self.chain_insert_params.append(dict(params or {}))
            return _FakeExecResult(scalar_result=501)
        if "INSERT INTO option_chain_rows" in sql:
            self.option_insert_params.append(dict(params or {}))
            return _FakeExecResult(scalar_result=None)
        raise AssertionError(f"Unexpected SQL in test: {sql}")

    async def commit(self):
        """No-op commit for fake session."""
        return None

    async def rollback(self):
        """No-op rollback for fake session."""
        return None

    def begin_nested(self):
        """Return no-op nested transaction context manager."""
        return self._NestedTx()


class _SessionFactory:
    """Async context manager factory that returns one fake session."""

    def __init__(self, session: _CaptureSession):
        """Store the fake session used by __aenter__."""
        self._session = session

    def __call__(self):
        """Return self to mimic SessionLocal callable behavior."""
        return self

    async def __aenter__(self):
        """Yield fake session to the job code."""
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        """Do not suppress exceptions in tests."""
        return False


class _FakeTradier:
    """Small Tradier stub for one expiration and one option row."""

    async def get_option_expirations(self, underlying: str) -> dict:
        """Return a single future expiration for deterministic selection."""
        exp = (datetime.now(tz=ZoneInfo(settings.tz)).date() + timedelta(days=30)).isoformat()
        return {"expirations": {"date": [exp]}}

    async def get_option_chain(self, *, underlying: str, expiration: str, greeks: bool) -> dict:
        """Return one chain option without an explicit underlying field."""
        return {
            "options": {
                "option": {
                    "symbol": f"{underlying}_{expiration}_C20",
                    "expiration_date": expiration,
                    "strike": 20.0,
                    "option_type": "call",
                    "bid": 1.1,
                    "ask": 1.3,
                    "last": 1.2,
                    "volume": 10,
                    "open_interest": 250,
                    "greeks": {"delta": 0.35},
                }
            }
        }


@pytest.mark.parametrize(
    ("underlying", "job_name", "targets"),
    [
        ("VIX", "snapshot_job_vix_test", [14, 21]),
        ("SPY", "snapshot_job_spy_test", [3, 5]),
    ],
)
@pytest.mark.asyncio
async def test_snapshot_job_uses_configured_underlying_for_chain_and_rows(
    monkeypatch,
    underlying: str,
    job_name: str,
    targets: list[int],
) -> None:
    """Ensure per-job underlying config is persisted on snapshot + option rows."""
    capture_session = _CaptureSession()
    monkeypatch.setattr(snapshot_module, "SessionLocal", _SessionFactory(capture_session))

    job = SnapshotJob(
        tradier=_FakeTradier(),
        config=SnapshotJobConfig(
            job_name=job_name,
            underlying=underlying,
            dte_mode="range",
            dte_targets=targets,
            dte_min_days=0,
            dte_max_days=365,
            range_fallback_enabled=False,
            range_fallback_count=1,
            dte_tolerance_days=2,
            strikes_each_side=0,
            allow_outside_rth=True,
        ),
    )

    result = await job.run_once(force=True)

    assert result["skipped"] is False
    assert capture_session.chain_insert_params[0]["underlying"] == underlying
    assert capture_session.option_insert_params[0]["underlying"] == underlying
