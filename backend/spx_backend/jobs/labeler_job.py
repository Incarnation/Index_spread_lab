from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.db import SessionLocal


def _mid(bid: float | None, ask: float | None) -> float | None:
    """Return mid quote when both bid and ask are available."""
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


def _to_json(value: dict | None) -> str | None:
    """Serialize dict payloads for JSONB SQL inserts."""
    if value is None:
        return None
    return json.dumps(value, default=str)


@dataclass(frozen=True)
class LabelMark:
    """Single spread mark for candidate outcome evaluation."""

    ts: datetime
    short_bid: float | None
    short_ask: float | None
    long_bid: float | None
    long_ask: float | None


def evaluate_candidate_outcome(
    *,
    entry_credit: float,
    marks: list[LabelMark],
    contracts: int,
    take_profit_pct: float,
    contract_multiplier: int,
) -> dict | None:
    """Evaluate TP50 live outcome while also tracking TP100 at expiry mark."""
    if not marks:
        return None

    tp_threshold = max(entry_credit, 0.0) * contract_multiplier * max(contracts, 1) * max(take_profit_pct, 0.0)
    tp100_threshold = max(entry_credit, 0.0) * contract_multiplier * max(contracts, 1)
    first_tp50_ts: datetime | None = None
    first_tp50_pnl: float | None = None
    first_tp50_exit_cost: float | None = None
    last_pnl: float | None = None
    last_exit_cost: float | None = None
    last_ts: datetime | None = None

    for mark in marks:
        short_mid = _mid(mark.short_bid, mark.short_ask)
        long_mid = _mid(mark.long_bid, mark.long_ask)
        if short_mid is None or long_mid is None:
            continue
        exit_cost = short_mid - long_mid
        pnl = (entry_credit - exit_cost) * contract_multiplier * max(contracts, 1)
        last_pnl = pnl
        last_exit_cost = exit_cost
        last_ts = mark.ts

        # Capture first TP50 hit for live-style realized outcome.
        if first_tp50_ts is None and pnl >= tp_threshold:
            first_tp50_ts = mark.ts
            first_tp50_pnl = pnl
            first_tp50_exit_cost = exit_cost

    if last_ts is None:
        return None

    hit_tp100_at_expiry = bool(last_pnl is not None and last_pnl >= tp100_threshold)

    if first_tp50_ts is not None:
        return {
            "resolved": True,
            "hit_tp50_before_sl_or_expiry": True,
            "hit_tp100_at_expiry": hit_tp100_at_expiry,
            "realized_pnl": first_tp50_pnl,
            "exit_cost": first_tp50_exit_cost,
            "expiry_pnl": last_pnl,
            "expiry_exit_cost": last_exit_cost,
            "expiry_ts": last_ts,
            "exit_reason": "TAKE_PROFIT_50",
            "resolved_ts": first_tp50_ts,
        }

    return {
        "resolved": True,
        "hit_tp50_before_sl_or_expiry": False,
        "hit_tp100_at_expiry": hit_tp100_at_expiry,
        "realized_pnl": last_pnl,
        "exit_cost": last_exit_cost,
        "expiry_pnl": last_pnl,
        "expiry_exit_cost": last_exit_cost,
        "expiry_ts": last_ts,
        "exit_reason": "EXPIRY_OR_LAST_MARK",
        "resolved_ts": last_ts,
    }


@dataclass(frozen=True)
class LabelerJob:
    """Resolve pending candidate outcomes into label columns/json."""

    async def _has_hit_tp100_column(self, *, session) -> bool:
        """Check if optional trade_candidates.hit_tp100_at_expiry column exists."""
        row = (
            await session.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'trade_candidates'
                      AND column_name = 'hit_tp100_at_expiry'
                    LIMIT 1
                    """
                )
            )
        ).fetchone()
        return row is not None

    async def _load_forward_marks(
        self,
        *,
        session,
        candidate_ts: datetime,
        underlying: str,
        expiration: date,
        short_symbol: str,
        long_symbol: str,
    ) -> list[LabelMark]:
        """Load forward spread marks from snapshots for the same expiration."""
        rows = await session.execute(
            text(
                """
                SELECT cs.ts, s.bid AS short_bid, s.ask AS short_ask, l.bid AS long_bid, l.ask AS long_ask
                FROM chain_snapshots cs
                JOIN option_chain_rows s ON s.snapshot_id = cs.snapshot_id AND s.option_symbol = :short_symbol
                JOIN option_chain_rows l ON l.snapshot_id = cs.snapshot_id AND l.option_symbol = :long_symbol
                WHERE cs.underlying = :underlying
                  AND cs.expiration = :expiration
                  AND cs.ts >= :candidate_ts
                ORDER BY cs.ts ASC
                """
            ),
            {
                "short_symbol": short_symbol,
                "long_symbol": long_symbol,
                "underlying": underlying,
                "expiration": expiration,
                "candidate_ts": candidate_ts,
            },
        )
        out: list[LabelMark] = []
        for row in rows.fetchall():
            out.append(
                LabelMark(
                    ts=row.ts,
                    short_bid=row.short_bid,
                    short_ask=row.short_ask,
                    long_bid=row.long_bid,
                    long_ask=row.long_ask,
                )
            )
        return out

    def _extract_candidate_fields(self, candidate_json: dict) -> dict | None:
        """Extract required symbols/metadata from stored candidate payload."""
        legs = candidate_json.get("legs") or {}
        short_leg = legs.get("short") or {}
        long_leg = legs.get("long") or {}

        short_symbol = short_leg.get("symbol")
        long_symbol = long_leg.get("symbol")
        underlying = candidate_json.get("underlying")
        expiration_raw = candidate_json.get("expiration")
        contracts = candidate_json.get("contracts") or settings.decision_contracts

        if not isinstance(short_symbol, str) or not isinstance(long_symbol, str):
            return None
        if not isinstance(underlying, str):
            return None
        if isinstance(expiration_raw, date):
            expiration = expiration_raw
        elif isinstance(expiration_raw, str):
            try:
                expiration = date.fromisoformat(expiration_raw)
            except ValueError:
                return None
        else:
            return None
        return {
            "short_symbol": short_symbol,
            "long_symbol": long_symbol,
            "underlying": underlying,
            "expiration": expiration,
            "contracts": int(contracts),
        }

    async def run_once(self, *, force: bool = False) -> dict:
        """Resolve pending candidate labels in bounded batches."""
        tz = ZoneInfo(settings.tz)
        now_et = datetime.now(tz=tz)
        now_utc = now_et.astimezone(ZoneInfo("UTC"))
        if not settings.labeler_enabled:
            return {"skipped": True, "reason": "labeler_disabled", "now_et": now_et.isoformat()}

        resolved_count = 0
        pending_count = 0
        error_count = 0
        skipped_fresh = 0

        async with SessionLocal() as session:
            has_hit_tp100_column = await self._has_hit_tp100_column(session=session)
            pending_rows = await session.execute(
                text(
                    """
                    SELECT candidate_id, ts, candidate_json, entry_credit
                    FROM trade_candidates
                    WHERE label_status = 'pending'
                    ORDER BY ts ASC
                    LIMIT :limit
                    """
                ),
                {"limit": settings.labeler_batch_limit},
            )
            rows = pending_rows.fetchall()
            pending_count = len(rows)

            min_age = timedelta(minutes=max(settings.labeler_min_age_minutes, 0))
            max_wait = timedelta(hours=max(settings.labeler_max_wait_hours, 1))
            for row in rows:
                candidate_id = row.candidate_id
                candidate_ts = row.ts
                entry_credit = float(row.entry_credit or 0.0)
                candidate_json = row.candidate_json if isinstance(row.candidate_json, dict) else {}

                age = now_utc - candidate_ts
                if (not force) and age < min_age:
                    skipped_fresh += 1
                    continue

                fields = self._extract_candidate_fields(candidate_json)
                if fields is None:
                    await session.execute(
                        text(
                            """
                            UPDATE trade_candidates
                            SET label_status = 'error',
                                label_schema_version = :label_schema_version,
                                resolved_at = :resolved_at,
                                label_error = :label_error
                            WHERE candidate_id = :candidate_id
                            """
                        ),
                        {
                            "candidate_id": candidate_id,
                            "label_schema_version": settings.label_schema_version,
                            "resolved_at": now_utc,
                            "label_error": "invalid_candidate_payload",
                        },
                    )
                    error_count += 1
                    continue

                marks = await self._load_forward_marks(
                    session=session,
                    candidate_ts=candidate_ts,
                    underlying=fields["underlying"],
                    expiration=fields["expiration"],
                    short_symbol=fields["short_symbol"],
                    long_symbol=fields["long_symbol"],
                )
                result = evaluate_candidate_outcome(
                    entry_credit=entry_credit,
                    marks=marks,
                    contracts=fields["contracts"],
                    take_profit_pct=settings.labeler_take_profit_pct,
                    contract_multiplier=settings.label_contract_multiplier,
                )

                if result is None:
                    # Keep candidates pending while data is still expected to arrive.
                    if age < max_wait:
                        continue
                    await session.execute(
                        text(
                            """
                            UPDATE trade_candidates
                            SET label_status = 'error',
                                label_schema_version = :label_schema_version,
                                resolved_at = :resolved_at,
                                label_error = :label_error
                            WHERE candidate_id = :candidate_id
                            """
                        ),
                        {
                            "candidate_id": candidate_id,
                            "label_schema_version": settings.label_schema_version,
                            "resolved_at": now_utc,
                            "label_error": "no_forward_marks",
                        },
                    )
                    error_count += 1
                    continue

                label_json = {
                    "schema_version": settings.label_schema_version,
                    "resolved_at_utc": result["resolved_ts"].isoformat(),
                    "horizon": "to_expiration",
                    "exit_reason": result["exit_reason"],
                    "entry_credit": entry_credit,
                    "exit_cost": result["exit_cost"],
                    "realized_pnl": result["realized_pnl"],
                    "hit_tp50_before_sl_or_expiry": result["hit_tp50_before_sl_or_expiry"],
                    "hit_tp100_at_expiry": result["hit_tp100_at_expiry"],
                    "expiry_pnl": result.get("expiry_pnl"),
                    "expiry_exit_cost": result.get("expiry_exit_cost"),
                    "expiry_ts_utc": (
                        result["expiry_ts"].isoformat()
                        if isinstance(result.get("expiry_ts"), datetime)
                        else None
                    ),
                }
                await session.execute(
                    text(
                        f"""
                        UPDATE trade_candidates
                        SET label_json = CAST(:label_json AS jsonb),
                            label_schema_version = :label_schema_version,
                            label_status = 'resolved',
                            label_horizon = 'to_expiration',
                            resolved_at = :resolved_at,
                            realized_pnl = :realized_pnl,
                            hit_tp50_before_sl_or_expiry = :hit_tp50_before_sl_or_expiry
                            {", hit_tp100_at_expiry = :hit_tp100_at_expiry" if has_hit_tp100_column else ""}
                            ,
                            label_error = NULL
                        WHERE candidate_id = :candidate_id
                        """
                    ),
                    (
                        {
                            "candidate_id": candidate_id,
                            "label_json": _to_json(label_json),
                            "label_schema_version": settings.label_schema_version,
                            "resolved_at": result["resolved_ts"],
                            "realized_pnl": result["realized_pnl"],
                            "hit_tp50_before_sl_or_expiry": result["hit_tp50_before_sl_or_expiry"],
                            "hit_tp100_at_expiry": result["hit_tp100_at_expiry"],
                        }
                        if has_hit_tp100_column
                        else {
                            "candidate_id": candidate_id,
                            "label_json": _to_json(label_json),
                            "label_schema_version": settings.label_schema_version,
                            "resolved_at": result["resolved_ts"],
                            "realized_pnl": result["realized_pnl"],
                            "hit_tp50_before_sl_or_expiry": result["hit_tp50_before_sl_or_expiry"],
                        }
                    ),
                )
                resolved_count += 1

            await session.commit()

        logger.info(
            "labeler_job: pending={} resolved={} errors={} skipped_fresh={}",
            pending_count,
            resolved_count,
            error_count,
            skipped_fresh,
        )
        return {
            "skipped": False,
            "reason": None,
            "now_et": now_et.isoformat(),
            "pending": pending_count,
            "resolved": resolved_count,
            "errors": error_count,
            "skipped_fresh": skipped_fresh,
        }


def build_labeler_job() -> LabelerJob:
    """Factory helper for LabelerJob."""
    return LabelerJob()
