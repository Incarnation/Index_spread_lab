from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from spx_backend.jobs.labeler_job import LabelMark, evaluate_candidate_outcome


def test_evaluate_candidate_outcome_hits_take_profit_first() -> None:
    marks = [
        LabelMark(
            ts=datetime(2026, 2, 13, 15, 0, tzinfo=ZoneInfo("UTC")),
            short_bid=1.20,
            short_ask=1.30,
            long_bid=0.40,
            long_ask=0.50,
        ),
        LabelMark(
            ts=datetime(2026, 2, 13, 15, 5, tzinfo=ZoneInfo("UTC")),
            short_bid=0.75,
            short_ask=0.85,
            long_bid=0.25,
            long_ask=0.35,
        ),
    ]
    # entry_credit = 1.00 -> TP50 threshold = 50 dollars (1 contract, 100 multiplier)
    result = evaluate_candidate_outcome(
        entry_credit=1.0,
        marks=marks,
        contracts=1,
        take_profit_pct=0.50,
        contract_multiplier=100,
    )
    assert result is not None
    assert result["hit_tp50_before_sl_or_expiry"] is True
    assert result["exit_reason"] == "TAKE_PROFIT_50"


def test_evaluate_candidate_outcome_resolves_on_last_mark_when_no_tp() -> None:
    marks = [
        LabelMark(
            ts=datetime(2026, 2, 13, 15, 0, tzinfo=ZoneInfo("UTC")),
            short_bid=1.35,
            short_ask=1.45,
            long_bid=0.35,
            long_ask=0.45,
        ),
        LabelMark(
            ts=datetime(2026, 2, 13, 15, 5, tzinfo=ZoneInfo("UTC")),
            short_bid=1.30,
            short_ask=1.40,
            long_bid=0.50,
            long_ask=0.60,
        ),
    ]
    result = evaluate_candidate_outcome(
        entry_credit=1.0,
        marks=marks,
        contracts=1,
        take_profit_pct=0.50,
        contract_multiplier=100,
    )
    assert result is not None
    assert result["hit_tp50_before_sl_or_expiry"] is False
    assert result["exit_reason"] == "EXPIRY_OR_LAST_MARK"
    assert result["realized_pnl"] < 50.0


def test_evaluate_candidate_outcome_none_for_empty_marks() -> None:
    result = evaluate_candidate_outcome(
        entry_credit=1.0,
        marks=[],
        contracts=1,
        take_profit_pct=0.50,
        contract_multiplier=100,
    )
    assert result is None
