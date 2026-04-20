"""DAO for ``chain_snapshots`` get-or-insert with ON CONFLICT DO NOTHING.

Audit Refactor #2 (folds in M4 + M1)
------------------------------------
``snapshot_job`` (Tradier ingest) and ``cboe_gex_job`` (mzdata GEX
anchors) both need a row in ``chain_snapshots`` keyed by
``(ts, underlying, expiration, source)``. Before this DAO each call site
rolled its own variant, with three problems:

1. **M4 race**: ``cboe_gex_job._get_existing_snapshot_id`` did a
   SELECT-then-INSERT pair without a transactional gate. Two
   concurrent CBOE runs (or a CBOE run racing the Tradier writer on
   the same key) would race past the SELECT and hit
   ``IntegrityError`` from the
   ``uq_chain_snapshots_ts_und_exp_src`` UNIQUE index on the second
   INSERT.
2. **M1 missing payload-kind**: ``snapshot_job`` writes full chains;
   ``cboe_gex_job`` writes FK-anchor-only rows whose
   ``option_chain_rows`` table is intentionally empty. There was no
   way to discriminate them after the fact, so backfill / retention
   sweeps could not safely treat the CBOE anchors differently.
3. **Duplication**: each writer hand-built the same INSERT/SELECT SQL.

This module centralises the pattern as ``get_or_insert_anchor``. The
``payload_kind`` argument is mandatory and always set, fulfilling M1.
The INSERT uses ``ON CONFLICT (ts, underlying, expiration, source) DO
NOTHING RETURNING snapshot_id``; on a suppressed conflict we run a
single SELECT for the existing row, fulfilling M4. Both writes happen
in the caller's outer transaction (no internal commit), so the caller
keeps full control over the surrounding ``begin_nested()``.

ON CONFLICT policy choice (audit M6 background)
-----------------------------------------------
This DAO is *anchor-only*: it reserves the FK target ``snapshot_id``
for downstream child rows (``option_chain_rows``, ``gex_snapshots``,
``gex_by_strike``, ``gex_by_expiry_strike``). The conflict policy is
``DO NOTHING`` because re-running ``snapshot_job`` or ``cboe_gex_job``
for the same ``(ts, underlying, expiration, source)`` key must not
mutate the anchor row -- if the checksum or payload_kind differs the
caller wants to know (we return ``inserted=False`` so the caller can
log + telemetry-emit), not have us silently overwrite it.

Downstream child writers pick their own policy:

* ``gex_snapshots`` from ``gex_job`` uses ``DO NOTHING`` (Tradier
  GEX is a function of an immutable chain, so re-runs are no-ops);
* ``gex_snapshots`` from ``cboe_gex_job`` uses ``DO UPDATE`` (mzdata
  may republish revised exposures for the same anchor and we want
  the latest revision to win);
* ``option_chain_rows`` uses no ON CONFLICT clause -- the snapshot
  has a fresh ``snapshot_id`` so the rows are always inserts.

These policy comments are duplicated at each call site by audit M6
so a future reader doesn't have to chase the policy across files.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import text


# Allowed values for the ``payload_kind`` discriminator. Mirrored from
# the CHECK constraint in migration 023; kept in Python so writers fail
# fast in tests instead of round-tripping to Postgres for a CHECK
# violation. Update both this set and the migration if you add a kind.
PAYLOAD_KIND_OPTIONS_CHAIN = "options_chain"
PAYLOAD_KIND_GEX_ANCHOR = "gex_anchor"
_ALLOWED_PAYLOAD_KINDS: frozenset[str] = frozenset(
    {PAYLOAD_KIND_OPTIONS_CHAIN, PAYLOAD_KIND_GEX_ANCHOR}
)


async def get_or_insert_anchor(
    *,
    session,
    ts: datetime,
    underlying: str,
    source: str,
    target_dte: int,
    expiration: date,
    checksum: str,
    payload_kind: str,
) -> tuple[int, bool]:
    """Insert (or fetch existing) chain_snapshots anchor row.

    Parameters
    ----------
    session:
        Async SQLAlchemy session. Caller owns the surrounding
        transaction (typically ``async with session.begin_nested()``).
    ts, underlying, source, expiration:
        Components of the natural key
        ``uq_chain_snapshots_ts_und_exp_src``.
    target_dte:
        Snapshot writer's intended DTE bucket; used downstream for
        DTE-target filtering. Not part of the unique key.
    checksum:
        Caller-computed payload checksum, written on insert only. On
        conflict the existing row's checksum is left untouched (the
        caller's payload is identical at the natural-key level by
        construction).
    payload_kind:
        Either ``PAYLOAD_KIND_OPTIONS_CHAIN`` (snapshot_job, full chain
        rows in option_chain_rows follow) or ``PAYLOAD_KIND_GEX_ANCHOR``
        (cboe_gex_job, no option_chain_rows follow). Required by audit
        M1; raises ``ValueError`` for any other value so a typo is
        caught at the call site rather than via the migration 023
        CHECK constraint.

    Returns
    -------
    tuple[int, bool]
        ``(snapshot_id, inserted)`` -- ``inserted=True`` when this call
        created a new row, ``False`` when an existing row was reused
        (race or replay).

    Raises
    ------
    ValueError
        If ``payload_kind`` is not in the allowed set.
    RuntimeError
        If neither the INSERT nor the fallback SELECT return a row
        (should be unreachable; indicates schema drift or a deleted
        row racing the writer).
    """
    if payload_kind not in _ALLOWED_PAYLOAD_KINDS:
        raise ValueError(
            f"chain_snapshot_dao: invalid payload_kind={payload_kind!r}; "
            f"allowed={sorted(_ALLOWED_PAYLOAD_KINDS)}"
        )

    insert_params = {
        "ts": ts,
        "underlying": underlying,
        "source": source,
        "target_dte": target_dte,
        "expiration": expiration,
        "checksum": checksum,
        "payload_kind": payload_kind,
    }
    insert_result = await session.execute(
        text(
            """
            INSERT INTO chain_snapshots
                (ts, underlying, source, target_dte, expiration, checksum, payload_kind)
            VALUES
                (:ts, :underlying, :source, :target_dte, :expiration, :checksum, :payload_kind)
            ON CONFLICT (ts, underlying, expiration, source) DO NOTHING
            RETURNING snapshot_id
            """
        ),
        insert_params,
    )
    inserted_row = insert_result.fetchone()
    if inserted_row is not None:
        return int(inserted_row.snapshot_id), True

    select_result = await session.execute(
        text(
            """
            SELECT snapshot_id
            FROM chain_snapshots
            WHERE ts = :ts
              AND underlying = :underlying
              AND source = :source
              AND expiration = :expiration
            ORDER BY snapshot_id DESC
            LIMIT 1
            """
        ),
        {
            "ts": ts,
            "underlying": underlying,
            "source": source,
            "expiration": expiration,
        },
    )
    existing = select_result.fetchone()
    if existing is None:
        # Defensive: ON CONFLICT DO NOTHING suppressed the INSERT but
        # the row vanished by the time we SELECT. Could happen if a
        # retention sweep raced between INSERT and SELECT, but the
        # caller has no recovery path; surfacing as RuntimeError is
        # correct.
        raise RuntimeError(
            f"chain_snapshot_dao: no row returned by INSERT (suppressed conflict) "
            f"or fallback SELECT for "
            f"(ts={ts}, underlying={underlying}, source={source}, "
            f"expiration={expiration}); see audit M4."
        )
    return int(existing.snapshot_id), False
