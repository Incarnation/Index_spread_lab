from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

import spx_backend.jobs.cboe_gex_job as cboe_module
from spx_backend.config import settings
from spx_backend.jobs.cboe_gex_job import CboeGexJob


class _FakeExecResult:
    """Small SQLAlchemy-like result wrapper for async unit tests."""

    def __init__(self, *, fetchone_result=None, fetchall_result=None, scalar_result=None):
        """Store one-row, many-row, and scalar responses for fake queries."""
        self._fetchone_result = fetchone_result
        self._fetchall_result = fetchall_result or []
        self._scalar_result = scalar_result

    def fetchone(self):
        """Return one row-like object."""
        return self._fetchone_result

    def fetchall(self):
        """Return all row-like objects."""
        return self._fetchall_result

    def scalar_one(self):
        """Return scalar value for INSERT .. RETURNING calls."""
        return self._scalar_result


class _RecordingSession:
    """Fake async DB session that records write parameters."""

    def __init__(self) -> None:
        """Initialize SQL capture buckets."""
        self.chain_snapshot_inserts: list[dict] = []
        self.gex_snapshot_upserts: list[dict] = []
        self.gex_by_strike_upserts: list[dict] = []
        self.gex_by_expiry_strike_upserts: list[dict] = []

    class _NestedTx:
        """No-op async savepoint context manager."""

        async def __aenter__(self):
            """Enter savepoint block without side effects."""
            return None

        async def __aexit__(self, exc_type, exc, tb):
            """Do not suppress nested exceptions."""
            return False

    def begin_nested(self):
        """Return no-op savepoint manager for session compatibility."""
        return self._NestedTx()

    async def execute(self, stmt, params=None):  # noqa: ANN001
        """Capture SQL writes and return canned results for reads."""
        sql = str(stmt)
        payload = dict(params or {})
        if "SELECT snapshot_id" in sql and "FROM chain_snapshots" in sql:
            return _FakeExecResult(fetchone_result=None)
        if "INSERT INTO chain_snapshots" in sql:
            self.chain_snapshot_inserts.append(payload)
            return _FakeExecResult(scalar_result=501)
        if "INSERT INTO gex_snapshots" in sql:
            self.gex_snapshot_upserts.append(payload)
            return _FakeExecResult()
        if "INSERT INTO gex_by_strike" in sql:
            self.gex_by_strike_upserts.append(payload)
            return _FakeExecResult()
        if "INSERT INTO gex_by_expiry_strike" in sql:
            self.gex_by_expiry_strike_upserts.append(payload)
            return _FakeExecResult()
        raise AssertionError(f"Unexpected SQL in test: {sql}")

    async def commit(self) -> None:
        """No-op commit."""
        return None

    async def rollback(self) -> None:
        """No-op rollback."""
        return None


class _SessionFactory:
    """Async context manager factory returning one fake session."""

    def __init__(self, session: _RecordingSession):
        """Store fake session yielded by context manager."""
        self._session = session

    def __call__(self):
        """Return self to mimic SessionLocal callable behavior."""
        return self

    async def __aenter__(self):
        """Yield configured fake session."""
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        """Do not suppress exceptions."""
        return False


class _ClosedClockCache:
    """Clock-cache stub that always reports market closed."""

    async def is_open(self, now_et: datetime) -> bool:  # noqa: ARG002
        """Return false so non-force runs skip outside RTH."""
        return False


class _FakeMzDataClient:
    """Minimal MZData stub returning one valid exposure payload."""

    async def get_live_option_exposure(self, symbol: str) -> dict:
        """Return one precomputed CBOE exposure payload for tests."""
        return {
            "spotPrice": 6015.0,
            "timestamp": "2026-02-16T15:00:00Z",
            "data": [
                {
                    "expiration": "2026-02-20",
                    "dte": 4,
                    "strikes": ["6000", "6010"],
                    "netGamma": [10.0, -5.0],
                    "call": {"absGamma": [12.0, 3.0], "openInterest": [100, 50]},
                    "put": {"absGamma": [2.0, 8.0], "openInterest": [80, 120]},
                }
            ],
        }


class _FakeMzDataClientWithWideDte:
    """MZData stub that returns expirations spanning beyond 10 trading slots."""

    async def get_live_option_exposure(self, symbol: str) -> dict:
        """Return one payload that includes trading-slot DTE values 0..12."""
        as_of = date(2026, 2, 16)
        data: list[dict] = []
        for offset in range(13):
            expiration = as_of + timedelta(days=offset)
            strike = 6000 + offset
            data.append(
                {
                    "expiration": expiration.isoformat(),
                    "dte": 999,
                    "strikes": [str(strike)],
                    "netGamma": [10.0 + float(offset)],
                    "call": {"absGamma": [5.0], "openInterest": [100]},
                    "put": {"absGamma": [1.0], "openInterest": [40]},
                }
            )
        return {
            "spotPrice": 6015.0,
            "timestamp": "2026-02-16T15:00:00Z",
            "data": data,
        }


@pytest.mark.asyncio
async def test_cboe_gex_job_skips_when_market_closed(monkeypatch) -> None:
    """CBOE job should short-circuit when market is closed and force is false."""
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)

    result = await CboeGexJob(mzdata=_FakeMzDataClient(), clock_cache=_ClosedClockCache()).run_once()

    assert result["skipped"] is True
    assert result["reason"] == "market_closed"


@pytest.mark.asyncio
async def test_cboe_gex_job_persists_source_tagged_rows(monkeypatch) -> None:
    """CBOE job should persist source tags and trading-slot DTE fields."""
    capture_session = _RecordingSession()
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlying", "SPX")
    monkeypatch.setattr(settings, "gex_max_dte_days", 10)
    monkeypatch.setattr(cboe_module, "SessionLocal", _SessionFactory(capture_session))

    result = await CboeGexJob(mzdata=_FakeMzDataClient()).run_once(force=True)

    assert result["skipped"] is False
    assert result["inserted_snapshots"] == 1
    assert result["gex_snapshots_upserted"] == 1
    assert result["gex_by_strike_upserted"] == 2
    assert result["gex_by_expiry_strike_upserted"] == 2
    assert result["skipped_items"] == 0
    assert result["skipped_items_by_reason"] == {}
    assert capture_session.chain_snapshot_inserts[0]["source"] == "CBOE"
    assert capture_session.chain_snapshot_inserts[0]["target_dte"] == 1
    assert capture_session.gex_snapshot_upserts[0]["source"] == "CBOE"
    assert all(row["dte_days"] == 1 for row in capture_session.gex_by_expiry_strike_upserts)


@pytest.mark.asyncio
async def test_cboe_gex_job_skips_items_outside_max_trading_slot_dte(monkeypatch) -> None:
    """CBOE job should skip expirations beyond configured trading-slot DTE cap."""
    capture_session = _RecordingSession()
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlying", "SPX")
    monkeypatch.setattr(settings, "gex_max_dte_days", 10)
    monkeypatch.setattr(cboe_module, "SessionLocal", _SessionFactory(capture_session))

    result = await CboeGexJob(mzdata=_FakeMzDataClientWithWideDte()).run_once(force=True)

    assert result["skipped"] is False
    assert result["inserted_snapshots"] == 11
    assert result["gex_snapshots_upserted"] == 11
    assert result["gex_by_strike_upserted"] == 11
    assert result["gex_by_expiry_strike_upserted"] == 11
    assert result["skipped_items"] == 2
    assert result["skipped_items_by_reason"] == {"dte_out_of_range": 2}
    assert len(capture_session.chain_snapshot_inserts) == 11
    assert all(0 <= row["target_dte"] <= 10 for row in capture_session.chain_snapshot_inserts)
