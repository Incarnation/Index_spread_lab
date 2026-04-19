"""Pure label-evaluation helpers for offline backtesting and ML training.

This module hosts the spread-mark TP/SL/expiry simulation helpers that
power offline tooling.

Why ``backend/scripts`` and not ``spx_backend/services``?
    The only consumers of these helpers are now offline -- the spread-audit
    test suite (``test_credit_spread_audit.py``) and any future offline
    backtest / ML training pipeline that wants to recompute outcomes from a
    cached series of marks.  Keeping them next to ``backtest_strategy.py`` /
    ``generate_training_data.py`` reflects that scope and avoids re-introducing
    an import dependency that the live trading service has no reason to pull
    in.

Re-import path for tests::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from _label_helpers import LabelMark, evaluate_candidate_outcome

The helpers are unchanged behaviourally from the original ``labeler_job``
implementation; this is a relocation, not a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from spx_backend.utils.pricing import mid_price


@dataclass(frozen=True)
class LabelMark:
    """Single spread mark used by ``evaluate_candidate_outcome``.

    A mark is a snapshot of both legs' bid/ask quotes at one instant.  The
    evaluator walks a chronologically-sorted list of these and replays the
    TP/SL/expiry decision logic that the live system would have applied.

    Attributes
    ----------
    ts : Timestamp of the snapshot.  Required.  Must be timezone-aware in
        practice; the evaluator does not enforce tzinfo because some legacy
        offline pipelines pass naive UTC datetimes.
    short_bid, short_ask : Quote on the short leg.  ``None`` permitted for
        missing data; the mark will be skipped if either side cannot produce
        a valid mid.
    long_bid, long_ask : Quote on the long leg.  Same null-handling.
    """

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
    stop_loss_pct: float | None = None,
    stop_loss_basis: str = "max_profit",
    max_loss_points: float | None = None,
) -> dict | None:
    """Evaluate TP/SL/expiry outcome from a chronological series of spread marks.

    Simulates the live TP/SL logic in chronological order:
    - First event wins (TP hit checked before SL within each mark).
    - Also tracks TP100 at expiry for the ML label.

    Parameters
    ----------
    entry_credit : Per-contract credit received at entry.
    marks : Chronologically ordered spread marks.
    contracts : Number of contracts.
    take_profit_pct : TP threshold as a fraction of max_profit.
    contract_multiplier : Dollars per point (typically 100 for SPX).
    stop_loss_pct : SL multiplier.  None disables SL simulation.
    stop_loss_basis : ``"max_profit"`` or ``"max_loss"``.
    max_loss_points : Spread width minus entry credit (in points), needed
        when ``stop_loss_basis="max_loss"``.

    Returns
    -------
    dict or None
        Outcome record with keys ``resolved``, ``hit_tp50_before_sl_or_expiry``,
        ``hit_sl_before_tp_or_expiry``, ``hit_tp100_at_expiry``, ``realized_pnl``,
        ``exit_cost``, ``sl_pnl``, ``sl_ts``, ``expiry_pnl``, ``expiry_exit_cost``,
        ``expiry_ts``, ``exit_reason``, ``resolved_ts``.  Returns ``None`` when
        the mark series is empty or all marks fail the mid-price gate.
    """
    if not marks:
        return None

    # n_contracts coerced to >= 1 so that a misconfigured 0/negative contract
    # count never silently zeroes out PnL.  Matches the live job's defensive
    # behaviour that this helper was extracted from.
    n_contracts = max(contracts, 1)
    max_profit = max(entry_credit, 0.0) * contract_multiplier * n_contracts
    tp_threshold = max_profit * max(take_profit_pct, 0.0)
    tp100_threshold = max_profit

    # SL threshold can be expressed as a fraction of either max_profit (the
    # default) or max_loss in dollars; the latter requires the spread width
    # in points to be supplied.
    sl_threshold: float | None = None
    if stop_loss_pct is not None:
        if stop_loss_basis == "max_loss" and max_loss_points is not None:
            sl_threshold = max(max_loss_points, 0.0) * contract_multiplier * n_contracts * stop_loss_pct
        else:
            sl_threshold = max_profit * stop_loss_pct

    first_tp50_ts: datetime | None = None
    first_tp50_pnl: float | None = None
    first_tp50_exit_cost: float | None = None
    first_sl_ts: datetime | None = None
    first_sl_pnl: float | None = None
    first_sl_exit_cost: float | None = None
    last_pnl: float | None = None
    last_exit_cost: float | None = None
    last_ts: datetime | None = None

    for mark in marks:
        # mid_price returns None for any non-finite, non-positive, or crossed
        # quote -- those marks are silently dropped from the outcome walk.
        short_mid = mid_price(mark.short_bid, mark.short_ask)
        long_mid = mid_price(mark.long_bid, mark.long_ask)
        if short_mid is None or long_mid is None:
            continue
        exit_cost = short_mid - long_mid
        pnl = (entry_credit - exit_cost) * contract_multiplier * n_contracts
        last_pnl = pnl
        last_exit_cost = exit_cost
        last_ts = mark.ts

        if first_tp50_ts is None and pnl >= tp_threshold:
            first_tp50_ts = mark.ts
            first_tp50_pnl = pnl
            first_tp50_exit_cost = exit_cost

        if first_sl_ts is None and sl_threshold is not None and pnl <= -abs(sl_threshold):
            first_sl_ts = mark.ts
            first_sl_pnl = pnl
            first_sl_exit_cost = exit_cost

    if last_ts is None:
        return None

    hit_tp100_at_expiry = bool(last_pnl is not None and last_pnl >= tp100_threshold)

    # Determine which event happened first (TP or SL).  Ties go to TP --
    # matches the live job's optimistic resolution policy.
    hit_sl = first_sl_ts is not None
    hit_tp = first_tp50_ts is not None
    tp_first = hit_tp and (not hit_sl or first_tp50_ts <= first_sl_ts)  # type: ignore[operator]
    sl_first = hit_sl and (not hit_tp or first_sl_ts < first_tp50_ts)  # type: ignore[operator]

    if tp_first:
        return {
            "resolved": True,
            "hit_tp50_before_sl_or_expiry": True,
            "hit_sl_before_tp_or_expiry": False,
            "hit_tp100_at_expiry": hit_tp100_at_expiry,
            "realized_pnl": first_tp50_pnl,
            "exit_cost": first_tp50_exit_cost,
            "sl_pnl": first_sl_pnl,
            "sl_ts": first_sl_ts.isoformat() if first_sl_ts else None,
            "expiry_pnl": last_pnl,
            "expiry_exit_cost": last_exit_cost,
            "expiry_ts": last_ts,
            "exit_reason": "TAKE_PROFIT_50",
            "resolved_ts": first_tp50_ts,
        }

    if sl_first:
        return {
            "resolved": True,
            "hit_tp50_before_sl_or_expiry": False,
            "hit_sl_before_tp_or_expiry": True,
            "hit_tp100_at_expiry": hit_tp100_at_expiry,
            "realized_pnl": first_sl_pnl,
            "exit_cost": first_sl_exit_cost,
            "sl_pnl": first_sl_pnl,
            "sl_ts": first_sl_ts.isoformat() if first_sl_ts else None,
            "expiry_pnl": last_pnl,
            "expiry_exit_cost": last_exit_cost,
            "expiry_ts": last_ts,
            "exit_reason": "STOP_LOSS",
            "resolved_ts": first_sl_ts,
        }

    return {
        "resolved": True,
        "hit_tp50_before_sl_or_expiry": False,
        "hit_sl_before_tp_or_expiry": False,
        "hit_tp100_at_expiry": hit_tp100_at_expiry,
        "realized_pnl": last_pnl,
        "exit_cost": last_exit_cost,
        "sl_pnl": None,
        "sl_ts": None,
        "expiry_pnl": last_pnl,
        "expiry_exit_cost": last_exit_cost,
        "expiry_ts": last_ts,
        "exit_reason": "EXPIRY_OR_LAST_MARK",
        "resolved_ts": last_ts,
    }
