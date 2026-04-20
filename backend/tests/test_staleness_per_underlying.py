"""Wave 5 (audit) H8 contract test: per-underlying GROUP BY in staleness_monitor.

The pre-H8 implementation issued a single global ``MAX(ts)`` per table,
which masked per-underlying staleness whenever any other underlying was
fresh (e.g. SPX flowing while VIX was silent). The H8 fix groups by the
natural partitioning key (``underlying`` for chain_snapshots /
gex_snapshots, ``symbol`` for underlying_quotes) so a single stale
bucket fires.

This test pins that contract in three independent ways so a future
"refactor that drops the GROUP BY" can never regress it silently:

1. SQL-string contract: every per-bucket check must include a
   ``GROUP BY`` clause and bind the right grouping column.
2. Behavior contract: a fixture where SPX is fresh but VIX is stale
   must surface a stale row for VIX -- not collapse to a single
   global "fresh" answer.
3. Source-filter contract (M12): the CBOE-only variant must add a
   ``WHERE source = 'CBOE'`` clause so a Tradier-only outage does
   NOT fire the CBOE-specific alert.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from spx_backend.jobs.staleness_monitor_job import StalenessMonitorJob


UTC = ZoneInfo("UTC")


class _CapturingSession:
    """Fake async session that captures SQL + returns canned per-bucket rows.

    ``per_bucket`` is keyed by table name and yields ``(bucket, latest)``
    tuples for ``GROUP BY`` queries. ``global_max`` is keyed by table
    name and yields a single ``latest`` for global ``MAX(ts)`` queries.
    Every ``execute`` call records ``(sql, params)`` to ``self.executed``
    so the contract tests can introspect.
    """

    def __init__(
        self,
        *,
        per_bucket: dict[str, list[tuple[str, datetime | None]]] | None = None,
        global_max: dict[str, datetime | None] | None = None,
    ):
        self._per_bucket = per_bucket or {}
        self._global = global_max or {}
        self.executed: list[tuple[str, dict]] = []

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        bound = dict(params or {})
        self.executed.append((sql, bound))
        # Route GROUP BY queries against the per-bucket fixture; route
        # global MAX(ts) queries against the global fixture.
        if "GROUP BY" in sql:
            is_cboe_filtered = bound.get("src") == "CBOE"
            for table, rows in self._per_bucket.items():
                if table not in sql:
                    continue
                if is_cboe_filtered and table == "gex_snapshots":
                    rows = self._per_bucket.get("gex_snapshots_cboe", rows)
                fetchall = [
                    SimpleNamespace(bucket=b, latest=ts) for b, ts in rows
                ]
                return MagicMock(fetchall=MagicMock(return_value=fetchall))
            return MagicMock(fetchall=MagicMock(return_value=[]))
        for table, ts in self._global.items():
            if table in sql:
                return MagicMock(
                    fetchone=MagicMock(return_value=SimpleNamespace(latest=ts))
                )
        return MagicMock(fetchone=MagicMock(return_value=SimpleNamespace(latest=None)))


@pytest.fixture(autouse=True)
def _patch_settings():
    """Pin staleness thresholds + expected-bucket lists for tests."""
    with patch("spx_backend.jobs.staleness_monitor_job.settings") as s:
        s.tz = "America/New_York"
        s.staleness_alert_enabled = True
        s.staleness_cooldown_minutes = 360
        s.staleness_quotes_max_minutes = 120
        s.staleness_snapshots_max_minutes = 120
        s.staleness_gex_max_minutes = 120
        s.staleness_cboe_gex_max_minutes = 120
        s.staleness_decisions_max_minutes = 480
        s.snapshot_underlying = "SPX"
        s.cboe_gex_underlyings = "SPX,SPY"
        s.quote_symbols = "SPX,VIX"
        yield s


# ---------------------------------------------------------------------------
# 1. SQL-string contract: GROUP BY clause must be present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_bucket_queries_emit_group_by_clause():
    """Every per-bucket check must issue a GROUP BY against its grouping column.

    Pins the H8 contract at the SQL level: a future refactor that
    accidentally collapses back to a global MAX(ts) breaks this test
    immediately.
    """
    now = datetime.now(tz=UTC)
    fresh = now - timedelta(minutes=5)
    session = _CapturingSession(
        per_bucket={
            "underlying_quotes": [("SPX", fresh), ("VIX", fresh)],
            "chain_snapshots": [("SPX", fresh)],
            "gex_snapshots": [("SPX", fresh), ("SPY", fresh)],
            "gex_snapshots_cboe": [("SPX", fresh), ("SPY", fresh)],
        },
        global_max={"trade_decisions": fresh},
    )

    job = StalenessMonitorJob()
    await job._check_freshness(session)

    underlying_quotes_calls = [
        sql for sql, _ in session.executed if "underlying_quotes" in sql
    ]
    chain_calls = [sql for sql, _ in session.executed if "FROM chain_snapshots" in sql]
    gex_calls = [sql for sql, _ in session.executed if "FROM gex_snapshots" in sql]
    decisions_calls = [
        sql for sql, _ in session.executed if "FROM trade_decisions" in sql
    ]

    # Per-bucket tables must group by their natural partitioning key.
    assert all("GROUP BY symbol" in sql for sql in underlying_quotes_calls), (
        f"underlying_quotes must group by symbol; got: {underlying_quotes_calls}"
    )
    assert all("GROUP BY underlying" in sql for sql in chain_calls), (
        f"chain_snapshots must group by underlying; got: {chain_calls}"
    )
    assert all("GROUP BY underlying" in sql for sql in gex_calls), (
        f"gex_snapshots must group by underlying; got: {gex_calls}"
    )

    # Global checks must NOT include a GROUP BY clause.
    assert decisions_calls, "expected at least one trade_decisions check"
    assert all("GROUP BY" not in sql for sql in decisions_calls)


# ---------------------------------------------------------------------------
# 2. Behavior contract: per-underlying staleness fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_stale_underlying_fires_with_other_fresh():
    """SPX fresh + VIX stale must produce a stale entry for VIX."""
    now = datetime.now(tz=UTC)
    fresh = now - timedelta(minutes=5)
    very_stale = now - timedelta(minutes=600)

    session = _CapturingSession(
        per_bucket={
            "underlying_quotes": [
                ("SPX", fresh),       # fresh
                ("VIX", very_stale),  # stale
            ],
            "chain_snapshots": [("SPX", fresh)],
            "gex_snapshots": [("SPX", fresh), ("SPY", fresh)],
            "gex_snapshots_cboe": [("SPX", fresh), ("SPY", fresh)],
        },
        global_max={"trade_decisions": fresh},
    )

    result = await StalenessMonitorJob()._check_freshness(session)

    stale_quote_buckets = [
        s for s in result if s["source"] == "underlying_quotes"
    ]
    assert any(
        s["bucket"] == "VIX" and s["age_minutes"] is not None and s["age_minutes"] > 120
        for s in stale_quote_buckets
    ), f"expected stale VIX entry; got: {stale_quote_buckets}"
    # SPX should NOT appear in the stale list.
    assert not any(
        s["bucket"] == "SPX" for s in stale_quote_buckets
    ), "SPX is fresh; must not surface as stale"


# ---------------------------------------------------------------------------
# 3. Source-filter contract (M12): CBOE variant binds the right WHERE clause
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cboe_specific_check_binds_source_filter():
    """The CBOE-source variant must add `WHERE source = :src` and bind 'CBOE'."""
    now = datetime.now(tz=UTC)
    fresh = now - timedelta(minutes=5)

    session = _CapturingSession(
        per_bucket={
            "underlying_quotes": [("SPX", fresh), ("VIX", fresh)],
            "chain_snapshots": [("SPX", fresh)],
            "gex_snapshots": [("SPX", fresh), ("SPY", fresh)],
            "gex_snapshots_cboe": [("SPX", fresh), ("SPY", fresh)],
        },
        global_max={"trade_decisions": fresh},
    )

    job = StalenessMonitorJob()
    await job._check_freshness(session)

    cboe_filtered_calls = [
        (sql, params)
        for sql, params in session.executed
        if "FROM gex_snapshots" in sql and params.get("src") == "CBOE"
    ]

    assert cboe_filtered_calls, (
        "expected at least one gex_snapshots SELECT bound with src='CBOE'"
    )
    sql, _ = cboe_filtered_calls[0]
    assert "source = :src" in sql or "source=:src" in sql, (
        f"CBOE-specific check must filter by source; got SQL: {sql}"
    )
