from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import statistics
from typing import Any


def _as_float(value: Any) -> float | None:
    """Best-effort float coercion for JSON-derived payloads."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    """Best-effort int coercion for JSON-derived payloads."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bucket(value: float | None, *, step: float) -> float | None:
    """Bucket a value to a fixed step for stable grouping."""
    if value is None:
        return None
    if step <= 0:
        return value
    bucketed = math.floor(value / step) * step
    return round(bucketed, 6)


def _classify_vix_regime(vix: float | None) -> str:
    """Classify VIX level into low/normal/high regimes."""
    if vix is None:
        return "unknown"
    if vix < 15.0:
        return "low"
    if vix <= 25.0:
        return "normal"
    return "high"


def _classify_term_structure_regime(term_structure: float | None) -> str:
    """Classify VIX9D/VIX ratio into contango/flat/backwardation regimes."""
    if term_structure is None:
        return "unknown"
    if term_structure < 0.95:
        return "contango"
    if term_structure <= 1.05:
        return "flat"
    return "backwardation"


def compute_max_drawdown(pnls: list[float]) -> float:
    """Compute max drawdown from a sequence of incremental PnLs."""
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def compute_tail_loss_proxy(pnls: list[float], *, tail_fraction: float = 0.10) -> float | None:
    """Compute average of worst-tail outcomes as a tail-loss proxy."""
    if not pnls:
        return None
    if tail_fraction <= 0:
        tail_fraction = 0.10
    tail_n = max(1, int(math.ceil(len(pnls) * tail_fraction)))
    worst = sorted(pnls)[:tail_n]
    return sum(worst) / float(tail_n)


def compute_margin_usage_dollars(max_loss_points: float | None, contracts: int, contract_multiplier: int) -> float:
    """Convert spread max-loss points to approximate dollar margin usage."""
    if max_loss_points is None:
        return 0.0
    return max(float(max_loss_points), 0.0) * max(contracts, 1) * max(contract_multiplier, 1)


def summarize_strategy_quality(
    *,
    realized_pnls: list[float],
    margin_usages: list[float],
    hit_tp50_count: int,
    hit_tp100_count: int,
) -> dict[str, float | int | None]:
    """Compute strategy quality and risk metrics used by gates/dashboard."""
    resolved = len(realized_pnls)
    if resolved <= 0:
        return {
            "resolved": 0,
            "tp50": 0,
            "tp100_at_expiry": 0,
            "tp50_rate": None,
            "tp100_at_expiry_rate": None,
            "expectancy": None,
            "avg_realized_pnl": None,
            "max_drawdown": None,
            "tail_loss_proxy": None,
            "avg_margin_usage": None,
        }

    expectancy = sum(realized_pnls) / float(resolved)
    max_drawdown = compute_max_drawdown(realized_pnls)
    tail_loss_proxy = compute_tail_loss_proxy(realized_pnls, tail_fraction=0.10)
    avg_margin = sum(margin_usages) / float(len(margin_usages)) if margin_usages else None

    return {
        "resolved": resolved,
        "tp50": hit_tp50_count,
        "tp100_at_expiry": hit_tp100_count,
        "tp50_rate": hit_tp50_count / float(resolved) if resolved > 0 else None,
        "tp100_at_expiry_rate": hit_tp100_count / float(resolved) if resolved > 0 else None,
        "expectancy": expectancy,
        "avg_realized_pnl": expectancy,
        "max_drawdown": max_drawdown,
        "tail_loss_proxy": tail_loss_proxy,
        "avg_margin_usage": avg_margin,
    }


def extract_candidate_features(
    *,
    candidate_json: dict[str, Any],
    max_loss_points: float | None,
    contract_multiplier: int = 100,
) -> dict[str, Any]:
    """Extract normalized model features from candidate payload."""
    spread_side = str(candidate_json.get("spread_side") or "unknown").lower()
    target_dte = _as_int(candidate_json.get("target_dte"))
    delta_target = _as_float(candidate_json.get("delta_target"))
    credit_to_width = _as_float(candidate_json.get("credit_to_width"))
    if credit_to_width is None:
        entry_credit = _as_float(candidate_json.get("entry_credit"))
        width_points = _as_float(candidate_json.get("width_points"))
        if entry_credit is not None and width_points is not None and width_points > 0:
            credit_to_width = entry_credit / width_points

    context_flags_raw = candidate_json.get("context_flags") or []
    context_flags = [str(v) for v in context_flags_raw] if isinstance(context_flags_raw, list) else []
    if "gex_support" in context_flags:
        context_regime = "support"
    elif "gex_headwind" in context_flags:
        context_regime = "headwind"
    else:
        context_regime = "neutral"
    context_payload = candidate_json.get("context")
    context = context_payload if isinstance(context_payload, dict) else {}
    vix = _as_float(candidate_json.get("vix"))
    if vix is None:
        vix = _as_float(context.get("vix")) if isinstance(context, dict) else None
    term_structure = _as_float(candidate_json.get("term_structure"))
    if term_structure is None:
        term_structure = _as_float(context.get("term_structure")) if isinstance(context, dict) else None
    raw_vix_regime = candidate_json.get("vix_regime")
    if isinstance(raw_vix_regime, str) and raw_vix_regime.strip():
        vix_regime = raw_vix_regime.strip().lower()
    else:
        vix_regime = _classify_vix_regime(vix)
    raw_term_regime = candidate_json.get("term_structure_regime")
    if isinstance(raw_term_regime, str) and raw_term_regime.strip():
        term_structure_regime = raw_term_regime.strip().lower()
    else:
        term_structure_regime = _classify_term_structure_regime(term_structure)

    contracts = _as_int(candidate_json.get("contracts")) or 1
    margin_usage = compute_margin_usage_dollars(
        max_loss_points=max_loss_points,
        contracts=contracts,
        contract_multiplier=contract_multiplier,
    )
    delta_bucket = _bucket(delta_target, step=0.05)
    credit_bucket = _bucket(credit_to_width, step=0.05)

    return {
        "spread_side": spread_side,
        "target_dte": target_dte,
        "delta_target": delta_target,
        "credit_to_width": credit_to_width,
        "context_regime": context_regime,
        "vix_regime": vix_regime,
        "term_structure_regime": term_structure_regime,
        "contracts": contracts,
        "margin_usage": margin_usage,
        "delta_bucket": delta_bucket,
        "credit_bucket": credit_bucket,
    }


def build_bucket_key(features: dict[str, Any]) -> str:
    """Build deterministic bucket key for empirical model stats."""
    return "|".join(
        [
            str(features.get("spread_side") or "unknown"),
            str(features.get("target_dte") if features.get("target_dte") is not None else "na"),
            str(features.get("delta_bucket") if features.get("delta_bucket") is not None else "na"),
            str(features.get("credit_bucket") if features.get("credit_bucket") is not None else "na"),
            str(features.get("context_regime") or "neutral"),
            str(features.get("vix_regime") or "unknown"),
            str(features.get("term_structure_regime") or "unknown"),
        ]
    )


def _stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Compute empirical stats for a set of labeled candidate rows."""
    if not rows:
        return {
            "count": 0,
            "wins": 0,
            "prob_tp50": None,
            "expected_pnl": None,
            "pnl_std": None,
            "tail_loss_proxy": None,
            "avg_margin_usage": None,
        }
    pnls = [float(r["realized_pnl"]) for r in rows]
    wins = sum(1 for r in rows if bool(r["hit_tp50"]))
    margins = [float(r.get("margin_usage", 0.0)) for r in rows]
    return {
        "count": len(rows),
        "wins": wins,
        "prob_tp50": wins / float(len(rows)),
        "expected_pnl": sum(pnls) / float(len(pnls)),
        "pnl_std": (statistics.pstdev(pnls) if len(pnls) > 1 else 0.0),
        "tail_loss_proxy": compute_tail_loss_proxy(pnls, tail_fraction=0.10),
        "avg_margin_usage": (sum(margins) / float(len(margins)) if margins else None),
    }


def train_bucket_model(
    *,
    rows: list[dict[str, Any]],
    min_bucket_size: int = 12,
    prior_strength: float = 8.0,
    utility_prob_weight: float = 0.35,
    utility_tail_penalty: float = 0.20,
    utility_margin_penalty: float = 0.02,
) -> dict[str, Any]:
    """Train a lightweight empirical bucket model for TP50 + expected PnL."""
    usable = [r for r in rows if r.get("realized_pnl") is not None and r.get("features")]
    global_stats = _stats_for_rows(usable)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        key = build_bucket_key(row["features"])
        groups[key].append(row)

    bucket_stats: dict[str, dict[str, Any]] = {}
    global_prob = float(global_stats["prob_tp50"] or 0.0)
    global_exp = float(global_stats["expected_pnl"] or 0.0)
    global_tail = float(global_stats["tail_loss_proxy"] or 0.0)
    global_margin = float(global_stats["avg_margin_usage"] or 0.0)
    global_std = float(global_stats["pnl_std"] or 0.0)

    for key, bucket_rows in groups.items():
        raw = _stats_for_rows(bucket_rows)
        count = int(raw["count"] or 0)
        if count <= 0:
            continue
        wins = int(raw["wins"] or 0)
        pnl_sum = sum(float(r["realized_pnl"]) for r in bucket_rows)
        margin_sum = sum(float(r.get("margin_usage", 0.0)) for r in bucket_rows)

        smooth_denom = count + max(prior_strength, 0.0)
        prob = ((wins + global_prob * prior_strength) / smooth_denom) if smooth_denom > 0 else global_prob
        exp = ((pnl_sum + global_exp * prior_strength) / smooth_denom) if smooth_denom > 0 else global_exp
        tail = (
            ((float(raw["tail_loss_proxy"] or 0.0) * count) + (global_tail * prior_strength)) / smooth_denom
            if smooth_denom > 0
            else global_tail
        )
        std = (
            ((float(raw["pnl_std"] or 0.0) * count) + (global_std * prior_strength)) / smooth_denom
            if smooth_denom > 0
            else global_std
        )
        avg_margin = ((margin_sum + global_margin * prior_strength) / smooth_denom) if smooth_denom > 0 else global_margin

        bucket_stats[key] = {
            "count": count,
            "prob_tp50": prob,
            "expected_pnl": exp,
            "tail_loss_proxy": tail,
            "pnl_std": std,
            "avg_margin_usage": avg_margin,
            "source": ("bucket" if count >= min_bucket_size else "bucket_low_sample"),
        }

    return {
        "model_type": "bucket_empirical_v1",
        "trained_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "min_bucket_size": min_bucket_size,
        "prior_strength": prior_strength,
        "utility_weights": {
            "prob_weight": utility_prob_weight,
            "tail_penalty": utility_tail_penalty,
            "margin_penalty": utility_margin_penalty,
        },
        "global": global_stats,
        "buckets": bucket_stats,
    }


def compute_utility_score(
    *,
    probability_win: float,
    expected_pnl: float,
    tail_loss_proxy: float,
    margin_usage: float,
    prob_weight: float,
    tail_penalty: float,
    margin_penalty: float,
) -> float:
    """Combine expected return and risk proxies into one utility score."""
    ev_component = expected_pnl
    prob_component = (probability_win - 0.5) * max(abs(expected_pnl), 1.0) * prob_weight
    tail_component = max(0.0, -tail_loss_proxy) * tail_penalty
    margin_component = (max(margin_usage, 0.0) / 1000.0) * margin_penalty
    return ev_component + prob_component - tail_component - margin_component


def predict_with_bucket_model(*, model_payload: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Score one candidate using the trained empirical bucket model."""
    bucket_key = build_bucket_key(features)
    global_stats = model_payload.get("global") or {}
    buckets = model_payload.get("buckets") or {}
    min_bucket_size = int(model_payload.get("min_bucket_size") or 0)

    stats = buckets.get(bucket_key)
    if stats is None or int(stats.get("count") or 0) < min_bucket_size:
        stats = {
            "count": int((stats or {}).get("count") or 0),
            "prob_tp50": float(global_stats.get("prob_tp50") or 0.0),
            "expected_pnl": float(global_stats.get("expected_pnl") or 0.0),
            "tail_loss_proxy": float(global_stats.get("tail_loss_proxy") or 0.0),
            "pnl_std": float(global_stats.get("pnl_std") or 0.0),
            "avg_margin_usage": float(global_stats.get("avg_margin_usage") or 0.0),
            "source": "global_fallback",
        }

    probability_win = float(stats.get("prob_tp50") or 0.0)
    expected_pnl = float(stats.get("expected_pnl") or 0.0)
    tail_loss_proxy = float(stats.get("tail_loss_proxy") or 0.0)
    pnl_std = float(stats.get("pnl_std") or 0.0)
    margin_usage = float(features.get("margin_usage") or stats.get("avg_margin_usage") or 0.0)

    weights = model_payload.get("utility_weights") or {}
    prob_weight = float(weights.get("prob_weight") or 0.35)
    tail_penalty = float(weights.get("tail_penalty") or 0.20)
    margin_penalty = float(weights.get("margin_penalty") or 0.02)

    utility_score = compute_utility_score(
        probability_win=probability_win,
        expected_pnl=expected_pnl,
        tail_loss_proxy=tail_loss_proxy,
        margin_usage=margin_usage,
        prob_weight=prob_weight,
        tail_penalty=tail_penalty,
        margin_penalty=margin_penalty,
    )
    return {
        "bucket_key": bucket_key,
        "bucket_count": int(stats.get("count") or 0),
        "source": stats.get("source") or "unknown",
        "probability_win": probability_win,
        "expected_pnl": expected_pnl,
        "tail_loss_proxy": tail_loss_proxy,
        "pnl_std": pnl_std,
        "margin_usage": margin_usage,
        "utility_score": utility_score,
    }

