from __future__ import annotations

from spx_backend.jobs.feature_builder_job import build_candidate_hash


def _candidate_json(short_symbol: str = "P_SHORT") -> dict:
    return {
        "underlying": "SPX",
        "snapshot_id": 101,
        "expiration": "2026-02-20",
        "target_dte": 3,
        "delta_target": 0.1,
        "spread_side": "put",
        "width_points": 25.0,
        "contracts": 1,
        "legs": {
            "short": {"symbol": short_symbol, "strike": 6800.0},
            "long": {"symbol": "P_LONG", "strike": 6775.0},
        },
    }


def test_candidate_hash_is_stable_for_equivalent_payloads() -> None:
    a = _candidate_json()
    b = {
        # same values, different key order
        "target_dte": 3,
        "underlying": "SPX",
        "expiration": "2026-02-20",
        "snapshot_id": 101,
        "spread_side": "put",
        "delta_target": 0.1,
        "contracts": 1,
        "width_points": 25.0,
        "legs": {
            "long": {"strike": 6775.0, "symbol": "P_LONG"},
            "short": {"strike": 6800.0, "symbol": "P_SHORT"},
        },
    }
    assert build_candidate_hash(a) == build_candidate_hash(b)


def test_candidate_hash_changes_when_legs_change() -> None:
    base_hash = build_candidate_hash(_candidate_json(short_symbol="P_SHORT_A"))
    changed_hash = build_candidate_hash(_candidate_json(short_symbol="P_SHORT_B"))
    assert base_hash != changed_hash
