from __future__ import annotations

from spx_backend.jobs.feature_builder_job import FeatureBuilderJob
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


def test_build_candidate_json_includes_cboe_context() -> None:
    job = FeatureBuilderJob()
    candidate = {
        "snapshot_id": 101,
        "expiration": "2026-02-20",
        "target_dte": 3,
        "delta_target": 0.1,
        "credit": 1.2,
        "score": 2.5,
        "delta_diff": 0.01,
        "chosen_legs_json": {
            "spread_side": "put",
            "width_points": 10.0,
            "short": {"symbol": "P_SHORT", "strike": 6800.0, "qty": 1},
            "long": {"symbol": "P_LONG", "strike": 6790.0, "qty": 1},
            "context": {"vix": 18.0, "term_structure": 1.01, "spy_price": 600.0, "spx_price": 6000.0},
        },
    }
    cboe_context = {"expiry_gex_net": 1234.5, "gamma_wall_distance_ratio": 0.004}

    payload = job._build_candidate_json(candidate, cboe_context=cboe_context)

    assert payload["cboe_context"] == cboe_context
