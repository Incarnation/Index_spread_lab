from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

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
        """Capture SQL writes and return canned results for reads.

        ``params`` is either a ``dict`` (single-row execute) or a
        ``list[dict]`` (executemany). L9 (audit) batches per-strike
        writes to gex_by_strike / gex_by_expiry_strike via executemany,
        so we normalize both shapes into the per-row capture buckets so
        downstream assertions on row counts continue to work
        unchanged.
        """
        sql = str(stmt)
        if "SELECT snapshot_id" in sql and "FROM chain_snapshots" in sql:
            return _FakeExecResult(
                fetchone_result=SimpleNamespace(snapshot_id=501)
            )
        if "INSERT INTO chain_snapshots" in sql:
            payload = dict(params or {})
            self.chain_snapshot_inserts.append(payload)
            return _FakeExecResult(
                fetchone_result=SimpleNamespace(snapshot_id=501),
                scalar_result=501,
            )
        if "INSERT INTO gex_snapshots" in sql:
            payload = dict(params or {})
            self.gex_snapshot_upserts.append(payload)
            return _FakeExecResult()
        if "INSERT INTO gex_by_strike" in sql:
            if isinstance(params, list):
                self.gex_by_strike_upserts.extend(dict(row) for row in params)
            else:
                self.gex_by_strike_upserts.append(dict(params or {}))
            return _FakeExecResult()
        if "INSERT INTO gex_by_expiry_strike" in sql:
            if isinstance(params, list):
                self.gex_by_expiry_strike_upserts.extend(dict(row) for row in params)
            else:
                self.gex_by_expiry_strike_upserts.append(dict(params or {}))
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
        """Return one precomputed CBOE exposure payload for the requested symbol."""
        normalized_symbol = symbol.strip().upper()
        strike_base_by_symbol = {"SPX": 6000.0, "SPY": 500.0, "VIX": 15.0}
        spot_by_symbol = {"SPX": 6015.0, "SPY": 506.5, "VIX": 17.2}
        strike_base = strike_base_by_symbol.get(normalized_symbol, 100.0)
        spot_price = spot_by_symbol.get(normalized_symbol, 100.0)
        return {
            "spotPrice": spot_price,
            "timestamp": "2026-02-16T15:00:00Z",
            "data": [
                {
                    "expiration": "2026-02-20",
                    "dte": 4,
                    "strikes": [str(strike_base), str(strike_base + 10.0)],
                    "netGamma": [10.0, -5.0],
                    "call": {"absGamma": [12.0, 3.0], "openInterest": [100, 50]},
                    "put": {"absGamma": [2.0, 8.0], "openInterest": [80, 120]},
                }
            ],
        }


class _FakeMzDataClientWithSelectiveFailure:
    """MZData stub that fails selected symbols while serving others."""

    def __init__(self, failing_symbols: set[str]) -> None:
        """Store normalized symbol set that should raise fetch failures."""
        self._failing_symbols = {symbol.strip().upper() for symbol in failing_symbols}
        self._fallback = _FakeMzDataClient()

    async def get_live_option_exposure(self, symbol: str) -> dict:
        """Raise for configured symbols and return normal payload otherwise."""
        normalized_symbol = symbol.strip().upper()
        if normalized_symbol in self._failing_symbols:
            raise RuntimeError(f"forced_fetch_failure:{normalized_symbol}")
        return await self._fallback.get_live_option_exposure(normalized_symbol)


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
    """CBOE job should ingest configured symbols and persist source tags.

    VIX is requested in the underlyings list but is rejected by the
    defense-in-depth guard added in audit Wave 1 (finding H3); only
    SPX and SPY produce snapshots.
    """
    capture_session = _RecordingSession()
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "SPX,SPY,VIX")
    monkeypatch.setattr(settings, "cboe_gex_underlying", "SPX")
    monkeypatch.setattr(settings, "gex_max_dte_days", 10)
    monkeypatch.setattr(cboe_module, "SessionLocal", _SessionFactory(capture_session))

    result = await CboeGexJob(mzdata=_FakeMzDataClient()).run_once(force=True)

    assert result["skipped"] is False
    assert result["underlyings"] == ["SPX", "SPY", "VIX"]
    assert set(result["processed_underlyings"]) == {"SPX", "SPY"}
    assert result["skipped_underlyings"] == [{"underlying": "VIX", "reason": "vix_excluded"}]
    assert len(result["underlying_results"]) == 3
    assert result["inserted_snapshots"] == 2
    assert result["gex_snapshots_upserted"] == 2
    assert result["gex_by_strike_upserted"] == 4
    assert result["gex_by_expiry_strike_upserted"] == 4
    assert result["skipped_items"] == 0
    assert result["skipped_items_by_reason"] == {}
    assert {row["underlying"] for row in capture_session.chain_snapshot_inserts} == {"SPX", "SPY"}
    assert all(row["source"] == "CBOE" for row in capture_session.chain_snapshot_inserts)
    assert all(row["target_dte"] == 1 for row in capture_session.chain_snapshot_inserts)
    assert all(row["source"] == "CBOE" for row in capture_session.gex_snapshot_upserts)
    assert all(row["dte_days"] == 1 for row in capture_session.gex_by_expiry_strike_upserts)


@pytest.mark.asyncio
async def test_cboe_gex_job_skips_items_outside_max_trading_slot_dte(monkeypatch) -> None:
    """CBOE job should skip expirations beyond configured trading-slot DTE cap."""
    capture_session = _RecordingSession()
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "SPX")
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


@pytest.mark.asyncio
async def test_cboe_gex_job_continues_when_one_symbol_fetch_fails(monkeypatch) -> None:
    """CBOE job should continue processing remaining symbols after one fetch failure.

    With VIX rejected by the defense-in-depth guard (audit Wave 1, H3)
    and SPY's mzdata fetch forced to fail, only SPX produces snapshots.
    """
    capture_session = _RecordingSession()
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "SPX,SPY,VIX")
    monkeypatch.setattr(settings, "cboe_gex_underlying", "SPX")
    monkeypatch.setattr(settings, "gex_max_dte_days", 10)
    monkeypatch.setattr(cboe_module, "SessionLocal", _SessionFactory(capture_session))

    result = await CboeGexJob(mzdata=_FakeMzDataClientWithSelectiveFailure({"SPY"})).run_once(force=True)

    assert result["skipped"] is False
    assert result["underlyings"] == ["SPX", "SPY", "VIX"]
    assert set(result["processed_underlyings"]) == {"SPX"}
    assert result["inserted_snapshots"] == 1
    assert result["gex_snapshots_upserted"] == 1
    assert result["gex_by_strike_upserted"] == 2
    assert result["gex_by_expiry_strike_upserted"] == 2
    skipped_by_underlying = {item["underlying"]: item["reason"] for item in result["skipped_underlyings"]}
    assert skipped_by_underlying == {"SPY": "exposure_fetch_failed", "VIX": "vix_excluded"}
    assert {row["underlying"] for row in capture_session.chain_snapshot_inserts} == {"SPX"}
    assert any("forced_fetch_failure:SPY" in str(item.get("error", "")) for item in result["failed_items"])


def test_zero_gamma_level_uses_gex_net_when_present() -> None:
    """_zero_gamma_level should prefer gex_net over gex_calls+gex_puts when available.

    This ensures the zero-gamma crossing is consistent with the aggregate
    gex_net stored in context_snapshots (H3 fix).
    """
    job = CboeGexJob(mzdata=object(), clock_cache=None)  # type: ignore[arg-type]

    # gex_net differs from gex_calls + gex_puts to prove the method uses gex_net
    per_strike = {
        5000.0: {"gex_calls": 100.0, "gex_puts": -50.0, "gex_net": -200.0},
        5050.0: {"gex_calls": 80.0,  "gex_puts": -30.0, "gex_net": 150.0},
    }

    result = job._zero_gamma_level(per_strike)

    # With gex_net: cumulative goes -200 -> (-200 + 150) = -50 — no crossing
    # With gex_calls+gex_puts: cumulative goes 50 -> (50 + 50) = 100 — also no crossing
    # Let's craft a case that actually crosses using gex_net but NOT gex_calls+gex_puts
    per_strike_crossing = {
        5000.0: {"gex_calls": 10.0, "gex_puts": -5.0, "gex_net": -100.0},
        5050.0: {"gex_calls": 10.0, "gex_puts": -5.0, "gex_net": 200.0},
    }
    result_crossing = job._zero_gamma_level(per_strike_crossing)
    assert result_crossing is not None

    # prev_cum=-100, cum=(-100+200)=100 => weight = 100/(100+100) = 0.5
    # expected = 5000 + (5050 - 5000) * 0.5 = 5025.0
    assert result_crossing == 5025.0


def test_zero_gamma_level_fallback_without_gex_net() -> None:
    """When gex_net is absent, _zero_gamma_level falls back to gex_calls + gex_puts."""
    job = CboeGexJob(mzdata=object(), clock_cache=None)  # type: ignore[arg-type]

    per_strike = {
        5000.0: {"gex_calls": -80.0, "gex_puts": -20.0},
        5050.0: {"gex_calls": 120.0, "gex_puts": -10.0},
    }

    result = job._zero_gamma_level(per_strike)

    # Without gex_net: cumulative goes -100 -> (-100 + 110) = 10 — crossing!
    assert result is not None
    # Interpolation: prev_cum=-100, cum=10 => weight = 100/110
    expected = 5000.0 + (5050.0 - 5000.0) * (100.0 / 110.0)
    assert abs(result - expected) < 0.1


@pytest.mark.asyncio
async def test_cboe_gex_job_vix_skip_guard_short_circuits_before_fetch(monkeypatch) -> None:
    """The defense-in-depth VIX guard should skip BEFORE calling mzdata.

    Audit Wave 1 / finding H3 — VIX exposure snapshots were dropped
    because mzdata coverage was 0%. This guard ensures a config-only
    re-introduction (e.g. someone setting CBOE_GEX_UNDERLYINGS=VIX)
    cannot cause writes; the mzdata client is never even invoked.
    """
    capture_session = _RecordingSession()

    class _NeverCalledMzDataClient:
        """MZData stub that raises if called — proves the guard short-circuits."""

        async def get_live_option_exposure(self, symbol: str) -> dict:
            """Fail loudly so the test catches any code path that reaches mzdata for VIX."""
            raise AssertionError(f"mzdata client must not be called for VIX: {symbol}")

    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "VIX")
    monkeypatch.setattr(settings, "cboe_gex_underlying", "VIX")
    monkeypatch.setattr(cboe_module, "SessionLocal", _SessionFactory(capture_session))

    result = await CboeGexJob(mzdata=_NeverCalledMzDataClient()).run_once(force=True)

    assert result["skipped"] is True
    assert result["reason"] == "vix_excluded"
    assert result["inserted_snapshots"] == 0
    assert result["gex_snapshots_upserted"] == 0
    assert result["gex_by_strike_upserted"] == 0
    assert result["gex_by_expiry_strike_upserted"] == 0
    assert result["skipped_underlyings"] == [{"underlying": "VIX", "reason": "vix_excluded"}]
    assert capture_session.chain_snapshot_inserts == []
    assert capture_session.gex_snapshot_upserts == []
    assert capture_session.gex_by_strike_upserts == []


@pytest.mark.asyncio
async def test_cboe_gex_job_applies_100x_vendor_units_scalar(monkeypatch) -> None:
    """Persisted CBOE gex_net must equal vendor netGamma * 100 (audit Wave 1, C1).

    Pinned to the live-DB observation that mzdata publishes per-strike
    exposures in dollars-per-share-per-1%-move (omits the 100-share
    contract multiplier). The fix routes every per-strike value through
    services.gex_math.apply_vendor_units which multiplies by 100; this
    test asserts the scaling is observable in the persisted gex_snapshot
    payload.
    """
    capture_session = _RecordingSession()
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "SPX")
    monkeypatch.setattr(settings, "cboe_gex_underlying", "SPX")
    monkeypatch.setattr(settings, "gex_max_dte_days", 10)
    monkeypatch.setattr(cboe_module, "SessionLocal", _SessionFactory(capture_session))

    result = await CboeGexJob(mzdata=_FakeMzDataClient()).run_once(force=True)

    # Fixture netGamma = [10.0, -5.0]; after the 100x adapter the
    # per-snapshot gex_net should be 100*(10.0 + -5.0) = 500.0.
    assert result["skipped"] is False
    assert len(capture_session.gex_snapshot_upserts) == 1
    spx_snapshot = capture_session.gex_snapshot_upserts[0]
    assert spx_snapshot["underlying"] == "SPX"
    assert spx_snapshot["source"] == "CBOE"
    assert spx_snapshot["gex_net"] == pytest.approx(500.0, rel=1e-12)
    # Fixture call.absGamma=[12,3] and put.absGamma=[2,8] (negated):
    # gex_calls = 100*(12+3) = 1500.0; gex_puts = 100*(-2 + -8) = -1000.0.
    assert spx_snapshot["gex_calls"] == pytest.approx(1500.0, rel=1e-12)
    assert spx_snapshot["gex_puts"] == pytest.approx(-1000.0, rel=1e-12)
    # gex_abs sums |call| + |put| = (1200+300) + (200+800) = 2500.0.
    assert spx_snapshot["gex_abs"] == pytest.approx(2500.0, rel=1e-12)

    # Per-strike persistence carries the same 100x scalar.
    assert len(capture_session.gex_by_strike_upserts) == 2
    by_strike = {row["strike"]: row for row in capture_session.gex_by_strike_upserts}
    assert by_strike[6000.0]["gex_net"] == pytest.approx(1000.0, rel=1e-12)   # 100 * 10.0
    assert by_strike[6010.0]["gex_net"] == pytest.approx(-500.0, rel=1e-12)   # 100 * -5.0
    assert by_strike[6000.0]["gex_calls"] == pytest.approx(1200.0, rel=1e-12)  # 100 * 12.0
    assert by_strike[6000.0]["gex_puts"] == pytest.approx(-200.0, rel=1e-12)   # 100 * -2.0


@pytest.mark.asyncio
async def test_cboe_gex_job_falls_back_to_canonical_sum_when_vendor_net_gamma_missing(monkeypatch) -> None:
    """When mzdata netGamma series is missing, fall back to call+put sum without double-scaling.

    Lock-in test for the Phase 2 refactor: avoid double-applying the
    100x adapter when computing the default value for a missing
    netGamma index.
    """
    capture_session = _RecordingSession()
    monkeypatch.setattr(settings, "cboe_gex_enabled", True)
    monkeypatch.setattr(settings, "cboe_gex_allow_outside_rth", False)
    monkeypatch.setattr(settings, "cboe_gex_underlyings", "SPX")
    monkeypatch.setattr(settings, "cboe_gex_underlying", "SPX")
    monkeypatch.setattr(settings, "gex_max_dte_days", 10)
    monkeypatch.setattr(cboe_module, "SessionLocal", _SessionFactory(capture_session))

    class _MzDataNoNetGamma:
        """MZData stub that omits netGamma; writer must derive from call+put."""

        async def get_live_option_exposure(self, symbol: str) -> dict:
            """Return a payload with absent netGamma for the requested symbol."""
            return {
                "spotPrice": 6015.0,
                "timestamp": "2026-02-16T15:00:00Z",
                "data": [
                    {
                        "expiration": "2026-02-20",
                        "dte": 4,
                        "strikes": ["6000.0"],
                        # No netGamma key on purpose.
                        "call": {"absGamma": [12.0], "openInterest": [100]},
                        "put": {"absGamma": [2.0], "openInterest": [80]},
                    }
                ],
            }

    result = await CboeGexJob(mzdata=_MzDataNoNetGamma()).run_once(force=True)

    assert result["skipped"] is False
    assert len(capture_session.gex_snapshot_upserts) == 1
    spx_snapshot = capture_session.gex_snapshot_upserts[0]
    # Without vendor netGamma, derive from already-scaled call_gamma + put_gamma:
    # call = 100 * 12 = 1200, put = -(100 * 2) = -200, net = 1000.
    assert spx_snapshot["gex_net"] == pytest.approx(1000.0, rel=1e-12)
    assert spx_snapshot["gex_calls"] == pytest.approx(1200.0, rel=1e-12)
    assert spx_snapshot["gex_puts"] == pytest.approx(-200.0, rel=1e-12)
