from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from spx_backend.config import settings
from spx_backend.database import SessionLocal


@dataclass(frozen=True)
class TradeAnalyticsRow:
    """Normalized trade row used by the performance aggregation job.

    Parameters
    ----------
    trade_id:
        Unique trade identifier.
    status:
        Trade lifecycle status (`OPEN`, `CLOSED`, `ROLLED`).
    trade_source:
        Trade origin (`live`, `paper`, `backtest`).
    strategy_type:
        Strategy descriptor used to infer side buckets.
    entry_time:
        Trade entry timestamp in UTC.
    exit_time:
        Trade exit timestamp when closed, otherwise None.
    target_dte:
        Optional target DTE chosen at entry.
    delta_target:
        Optional decision delta associated with the trade.
    current_pnl:
        Mark-to-market PnL for open positions.
    realized_pnl:
        Final realized PnL for closed positions.
    """

    trade_id: int
    status: str
    trade_source: str
    strategy_type: str
    entry_time: datetime
    exit_time: datetime | None
    target_dte: int | None
    delta_target: float | None
    current_pnl: float | None
    realized_pnl: float | None


def _as_float(value: Any) -> float | None:
    """Convert arbitrary values to float where possible."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    """Convert arbitrary values to integer where possible."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_trade_row(row: Any) -> TradeAnalyticsRow | None:
    """Normalize one SQL row into a stable dataclass payload.

    Returns
    -------
    TradeAnalyticsRow | None
        Parsed dataclass when required fields are valid, otherwise None to
        safely skip malformed rows without failing the full analytics run.
    """
    entry_time = row.entry_time if isinstance(row.entry_time, datetime) else None
    if entry_time is None:
        return None
    return TradeAnalyticsRow(
        trade_id=int(row.trade_id),
        status=str(row.status or "UNKNOWN").upper(),
        trade_source=str(row.trade_source or "unknown").lower(),
        strategy_type=str(row.strategy_type or ""),
        entry_time=entry_time,
        exit_time=(row.exit_time if isinstance(row.exit_time, datetime) else None),
        target_dte=_as_int(row.target_dte),
        delta_target=_as_float(row.delta_target),
        current_pnl=_as_float(row.current_pnl),
        realized_pnl=_as_float(row.realized_pnl),
    )


def _derive_spread_side(strategy_type: str) -> str:
    """Infer spread side label from strategy type text."""
    normalized = strategy_type.strip().lower()
    if "put" in normalized:
        return "put"
    if "call" in normalized:
        return "call"
    return "unknown"


def _bucket_dte(target_dte: int | None) -> str:
    """Bucket target DTE values into stable dashboard groups."""
    if target_dte is None:
        return "unknown"
    if target_dte <= 0:
        return "0"
    if target_dte <= 3:
        return "1-3"
    if target_dte <= 7:
        return "4-7"
    if target_dte <= 14:
        return "8-14"
    return "15+"


def _bucket_delta(delta_target: float | None) -> str:
    """Bucket absolute delta targets into broad risk bands."""
    if delta_target is None:
        return "unknown"
    value = abs(float(delta_target))
    if value <= 0.10:
        return "0.00-0.10"
    if value <= 0.20:
        return "0.11-0.20"
    if value <= 0.30:
        return "0.21-0.30"
    if value <= 0.40:
        return "0.31-0.40"
    return "0.41+"


def _mode_pnl_points(record: TradeAnalyticsRow, *, as_of_date: date) -> list[tuple[str, float, date]]:
    """Return realized/combined mode PnL points for one trade.

    Returns
    -------
    list[tuple[str, float, date]]
        Tuples of `(mode, pnl, equity_bucket_date)` for every valid mode the
        record contributes to.
    """
    points: list[tuple[str, float, date]] = []
    if record.status == "CLOSED" and record.realized_pnl is not None:
        realized_date = (record.exit_time.date() if record.exit_time is not None else record.entry_time.date())
        points.append(("realized", float(record.realized_pnl), realized_date))
        points.append(("combined", float(record.realized_pnl), realized_date))
        return points
    if record.current_pnl is not None:
        points.append(("combined", float(record.current_pnl), as_of_date))
    return points


def _dimension_values(record: TradeAnalyticsRow) -> dict[str, str]:
    """Build the dimension value map used for breakdown aggregates."""
    entry_utc = record.entry_time.astimezone(ZoneInfo("UTC"))
    return {
        "side": _derive_spread_side(record.strategy_type),
        "dte_bucket": _bucket_dte(record.target_dte),
        "delta_bucket": _bucket_delta(record.delta_target),
        "weekday": entry_utc.strftime("%a"),
        "hour": f"{entry_utc.hour:02d}",
        "source": record.trade_source,
    }


def _new_stat_bucket() -> dict[str, float]:
    """Create a mutable metric bucket used by aggregators."""
    return {
        "trade_count": 0.0,
        "win_count": 0.0,
        "loss_count": 0.0,
        "pnl_sum": 0.0,
        "win_pnl_sum": 0.0,
        "loss_pnl_sum": 0.0,
    }


def _accumulate(stats: dict[str, float], pnl: float) -> None:
    """Accumulate one trade outcome into an aggregate bucket."""
    stats["trade_count"] += 1.0
    stats["pnl_sum"] += pnl
    if pnl > 0:
        stats["win_count"] += 1.0
        stats["win_pnl_sum"] += pnl
    elif pnl < 0:
        stats["loss_count"] += 1.0
        stats["loss_pnl_sum"] += pnl


@dataclass(frozen=True)
class PerformanceAnalyticsJob:
    """Build aggregate PnL analytics tables from normalized trade records."""

    async def run_once(self, *, force: bool = False) -> dict[str, Any]:
        """Compute and persist one aggregate analytics snapshot.

        Parameters
        ----------
        force:
            Included for scheduler/admin API parity. This job currently has no
            gating and always runs when enabled.

        Returns
        -------
        dict[str, Any]
            Run status, table-row counts, and source trade cardinalities.
        """
        # Allow explicit admin/manual force runs even when scheduled mode is disabled.
        if (not settings.performance_analytics_enabled) and (not force):
            return {"skipped": True, "reason": "performance_analytics_disabled"}

        now_utc = datetime.now(tz=ZoneInfo("UTC"))
        started_at = now_utc
        breakdown_stats: dict[tuple[str, date, str, str], dict[str, float]] = {}
        equity_stats: dict[tuple[str, date], dict[str, float]] = {}
        source_trade_count = 0
        source_closed_count = 0
        source_open_count = 0
        skipped_rows = 0

        async with SessionLocal() as session:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT
                          t.trade_id,
                          t.status,
                          t.trade_source,
                          t.strategy_type,
                          t.entry_time,
                          t.exit_time,
                          t.target_dte,
                          t.current_pnl,
                          t.realized_pnl,
                          td.delta_target
                        FROM trades t
                        LEFT JOIN trade_decisions td ON td.decision_id = t.decision_id
                        ORDER BY t.entry_time ASC, t.trade_id ASC
                        """
                    )
                )
            ).fetchall()

            source_trade_count = len(rows)
            for raw_row in rows:
                record = _normalize_trade_row(raw_row)
                if record is None:
                    skipped_rows += 1
                    continue
                if record.status == "CLOSED":
                    source_closed_count += 1
                elif record.status == "OPEN":
                    source_open_count += 1

                # Breakdown dimensions use entry-time bucketing to preserve the
                # original decision context (side/delta/dte/session entry hour).
                dimensions = _dimension_values(record)
                entry_date = record.entry_time.astimezone(ZoneInfo("UTC")).date()
                for mode, pnl_value, equity_bucket_date in _mode_pnl_points(record, as_of_date=now_utc.date()):
                    equity_key = (mode, equity_bucket_date)
                    equity_bucket = equity_stats.setdefault(equity_key, _new_stat_bucket())
                    _accumulate(equity_bucket, pnl_value)

                    for dimension_type, dimension_value in dimensions.items():
                        breakdown_key = (mode, entry_date, dimension_type, dimension_value)
                        breakdown_bucket = breakdown_stats.setdefault(breakdown_key, _new_stat_bucket())
                        _accumulate(breakdown_bucket, pnl_value)

            # Keep only the latest analytics snapshot to avoid unbounded table
            # growth while still preserving transactional dashboard consistency.
            await session.execute(text("DELETE FROM trade_performance_snapshots"))
            snapshot_row = (
                await session.execute(
                    text(
                        """
                        INSERT INTO trade_performance_snapshots (
                          as_of_ts,
                          job_started_at,
                          job_finished_at,
                          source_trade_count,
                          source_closed_count,
                          source_open_count
                        )
                        VALUES (
                          :as_of_ts,
                          :job_started_at,
                          :job_finished_at,
                          :source_trade_count,
                          :source_closed_count,
                          :source_open_count
                        )
                        RETURNING analytics_snapshot_id
                        """
                    ),
                    {
                        "as_of_ts": now_utc,
                        "job_started_at": started_at,
                        "job_finished_at": datetime.now(tz=ZoneInfo("UTC")),
                        "source_trade_count": source_trade_count,
                        "source_closed_count": source_closed_count,
                        "source_open_count": source_open_count,
                    },
                )
            ).fetchone()
            analytics_snapshot_id = int(snapshot_row.analytics_snapshot_id)

            breakdown_rows = []
            for (mode, bucket_date, dimension_type, dimension_value), stats in breakdown_stats.items():
                breakdown_rows.append(
                    {
                        "analytics_snapshot_id": analytics_snapshot_id,
                        "mode": mode,
                        "bucket_date": bucket_date,
                        "dimension_type": dimension_type,
                        "dimension_value": dimension_value,
                        "trade_count": int(stats["trade_count"]),
                        "win_count": int(stats["win_count"]),
                        "loss_count": int(stats["loss_count"]),
                        "pnl_sum": float(stats["pnl_sum"]),
                        "win_pnl_sum": float(stats["win_pnl_sum"]),
                        "loss_pnl_sum": float(stats["loss_pnl_sum"]),
                    }
                )
            if breakdown_rows:
                await session.execute(
                    text(
                        """
                        INSERT INTO trade_performance_breakdowns (
                          analytics_snapshot_id, mode, bucket_date, dimension_type, dimension_value,
                          trade_count, win_count, loss_count, pnl_sum, win_pnl_sum, loss_pnl_sum
                        )
                        VALUES (
                          :analytics_snapshot_id, :mode, :bucket_date, :dimension_type, :dimension_value,
                          :trade_count, :win_count, :loss_count, :pnl_sum, :win_pnl_sum, :loss_pnl_sum
                        )
                        """
                    ),
                    breakdown_rows,
                )

            equity_rows: list[dict[str, Any]] = []
            for mode in ("realized", "combined"):
                cumulative = 0.0
                peak = 0.0
                dates_for_mode = sorted(d for (row_mode, d) in equity_stats.keys() if row_mode == mode)
                for bucket_date in dates_for_mode:
                    stats = equity_stats[(mode, bucket_date)]
                    cumulative += float(stats["pnl_sum"])
                    peak = max(peak, cumulative)
                    equity_rows.append(
                        {
                            "analytics_snapshot_id": analytics_snapshot_id,
                            "mode": mode,
                            "bucket_date": bucket_date,
                            "trade_count": int(stats["trade_count"]),
                            "win_count": int(stats["win_count"]),
                            "loss_count": int(stats["loss_count"]),
                            "pnl_sum": float(stats["pnl_sum"]),
                            "win_pnl_sum": float(stats["win_pnl_sum"]),
                            "loss_pnl_sum": float(stats["loss_pnl_sum"]),
                            "cumulative_pnl": cumulative,
                            "peak_pnl": peak,
                            "drawdown": peak - cumulative,
                        }
                    )
            if equity_rows:
                await session.execute(
                    text(
                        """
                        INSERT INTO trade_performance_equity_curve (
                          analytics_snapshot_id, mode, bucket_date, trade_count, win_count, loss_count,
                          pnl_sum, win_pnl_sum, loss_pnl_sum, cumulative_pnl, peak_pnl, drawdown
                        )
                        VALUES (
                          :analytics_snapshot_id, :mode, :bucket_date, :trade_count, :win_count, :loss_count,
                          :pnl_sum, :win_pnl_sum, :loss_pnl_sum, :cumulative_pnl, :peak_pnl, :drawdown
                        )
                        """
                    ),
                    equity_rows,
                )

            await session.commit()

        logger.info(
            "performance_analytics_job: force={} source_trades={} skipped_rows={} breakdown_rows={} equity_rows={}",
            force,
            source_trade_count,
            skipped_rows,
            len(breakdown_rows),
            len(equity_rows),
        )
        return {
            "skipped": False,
            "reason": None,
            "as_of_utc": now_utc.isoformat(),
            "analytics_snapshot_id": analytics_snapshot_id,
            "source_trade_count": source_trade_count,
            "source_closed_count": source_closed_count,
            "source_open_count": source_open_count,
            "skipped_rows": skipped_rows,
            "breakdown_rows": len(breakdown_rows),
            "equity_rows": len(equity_rows),
        }


def build_performance_analytics_job() -> PerformanceAnalyticsJob:
    """Factory helper for the performance analytics aggregation job."""
    return PerformanceAnalyticsJob()
