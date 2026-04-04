from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import statistics
from typing import Any
from zoneinfo import ZoneInfo


FULL_BUCKET_DIMENSIONS: tuple[str, ...] = (
    "spread_side",
    "target_dte",
    "delta_bucket",
    "credit_bucket",
    "context_regime",
    "vix_regime",
    "term_structure_regime",
    "spy_spx_ratio_regime",
    "cboe_regime",
    "cboe_wall_proximity",
    "vix_delta_interaction_bucket",
    "dte_credit_interaction_bucket",
)
LEGACY_BUCKET_DIMENSIONS: tuple[str, ...] = (
    "spread_side",
    "target_dte",
    "delta_bucket",
    "credit_bucket",
    "context_regime",
    "vix_regime",
    "term_structure_regime",
    "spy_spx_ratio_regime",
)
RELAXED_MARKET_BUCKET_DIMENSIONS: tuple[str, ...] = (
    "spread_side",
    "target_dte",
    "delta_bucket",
    "credit_bucket",
    "context_regime",
    "vix_regime",
    "cboe_regime",
    "cboe_wall_proximity",
    "vix_delta_interaction_bucket",
    "dte_credit_interaction_bucket",
)
CORE_BUCKET_DIMENSIONS: tuple[str, ...] = (
    "spread_side",
    "target_dte",
    "delta_bucket",
    "credit_bucket",
    "cboe_regime",
    "vix_delta_interaction_bucket",
    "dte_credit_interaction_bucket",
)
BUCKET_HIERARCHY_ORDER: tuple[str, ...] = ("full", "relaxed_market", "core", "global")


def _bucket_token(features: dict[str, Any], key: str) -> str:
    """Format one bucket dimension value into a deterministic string token.

    Parameters
    ----------
    features:
        Feature payload used by training or inference.
    key:
        Feature key to serialize into a stable bucket token.

    Returns
    -------
    str
        Stable token representation where missing values become `"na"`.
    """
    value = features.get(key)
    if value is None:
        return "na"
    return str(value)


def _build_bucket_key_for_dimensions(features: dict[str, Any], dimensions: tuple[str, ...]) -> str:
    """Build a deterministic key from an explicit ordered dimension list.

    Parameters
    ----------
    features:
        Candidate feature dictionary.
    dimensions:
        Ordered feature keys to include in the serialized key.

    Returns
    -------
    str
        Pipe-delimited bucket key used for grouped empirical statistics.
    """
    return "|".join(_bucket_token(features, key) for key in dimensions)


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


def _classify_spy_spx_ratio_regime(spy_spx_ratio: float | None) -> str:
    """Classify SPY/SPX ratio into discount/parity/premium regimes."""
    if spy_spx_ratio is None:
        return "unknown"
    if spy_spx_ratio < 0.099:
        return "discount"
    if spy_spx_ratio <= 0.101:
        return "parity"
    return "premium"


def _classify_cboe_regime(expiry_gex_net: float | None) -> str:
    """Classify CBOE expiry net-gamma into support/headwind/neutral."""
    if expiry_gex_net is None:
        return "unknown"
    if expiry_gex_net > 0:
        return "support"
    if expiry_gex_net < 0:
        return "headwind"
    return "neutral"


def _classify_cboe_wall_proximity(distance_ratio: float | None) -> str:
    """Bucket CBOE wall-distance ratio for robust sparse-data grouping."""
    if distance_ratio is None:
        return "unknown"
    if distance_ratio <= 0.003:
        return "near"
    if distance_ratio <= 0.01:
        return "mid"
    return "far"


def _classify_skew_regime(skew: float | None) -> str:
    """Classify CBOE SKEW index into low/normal/elevated regimes.

    SKEW typically ranges from ~100 to ~170. Below 120 indicates low
    tail-risk pricing; above 145 reflects elevated perceived tail risk.
    """
    if skew is None:
        return "unknown"
    if skew < 120.0:
        return "low"
    if skew <= 145.0:
        return "normal"
    return "elevated"


def _classify_vvix_regime(vvix: float | None) -> str:
    """Classify VVIX (vol-of-VIX) into low/normal/elevated regimes.

    VVIX typically ranges from ~60 to ~150+. Below 80 signals calm
    vol expectations; above 105 indicates heightened vol-of-vol.
    """
    if vvix is None:
        return "unknown"
    if vvix < 80.0:
        return "low"
    if vvix <= 105.0:
        return "normal"
    return "elevated"


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
    """Extract normalized model features from a candidate payload.

    Parameters
    ----------
    candidate_json:
        Raw candidate payload persisted by feature-builder/decision steps.
    max_loss_points:
        Candidate max loss in spread points, used to derive margin usage.
    contract_multiplier:
        Contract multiplier used to convert points into dollar risk.

    Returns
    -------
    dict[str, Any]
        Normalized feature dictionary with base dimensions and sparse-data
        interaction buckets for model training and inference.
    """
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
    spy_spx_ratio = _as_float(candidate_json.get("spy_spx_ratio"))
    if spy_spx_ratio is None:
        spy_price = _as_float(candidate_json.get("spy_price"))
        if spy_price is None:
            spy_price = _as_float(context.get("spy_price")) if isinstance(context, dict) else None
        spx_price = _as_float(candidate_json.get("spx_price"))
        if spx_price is None:
            spx_price = _as_float(context.get("spx_price")) if isinstance(context, dict) else None
        if spy_price is not None and spx_price is not None and spx_price > 0:
            spy_spx_ratio = spy_price / spx_price
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
    raw_spy_ratio_regime = candidate_json.get("spy_spx_ratio_regime")
    if isinstance(raw_spy_ratio_regime, str) and raw_spy_ratio_regime.strip():
        spy_spx_ratio_regime = raw_spy_ratio_regime.strip().lower()
    else:
        spy_spx_ratio_regime = _classify_spy_spx_ratio_regime(spy_spx_ratio)
    cboe_payload = candidate_json.get("cboe_context")
    cboe_context = cboe_payload if isinstance(cboe_payload, dict) else {}
    expiry_gex_net = _as_float(candidate_json.get("expiry_gex_net"))
    if expiry_gex_net is None:
        expiry_gex_net = _as_float(cboe_context.get("expiry_gex_net")) if isinstance(cboe_context, dict) else None
    gamma_wall_distance_ratio = _as_float(candidate_json.get("gamma_wall_distance_ratio"))
    if gamma_wall_distance_ratio is None:
        gamma_wall_distance_ratio = (
            _as_float(cboe_context.get("gamma_wall_distance_ratio")) if isinstance(cboe_context, dict) else None
        )
    if gamma_wall_distance_ratio is None:
        gamma_wall_distance_ratio = _as_float(cboe_context.get("call_wall_distance_ratio")) if isinstance(cboe_context, dict) else None
    if gamma_wall_distance_ratio is None:
        gamma_wall_distance_ratio = _as_float(cboe_context.get("put_wall_distance_ratio")) if isinstance(cboe_context, dict) else None
    cboe_regime = _classify_cboe_regime(expiry_gex_net)
    cboe_wall_proximity = _classify_cboe_wall_proximity(gamma_wall_distance_ratio)

    skew = _as_float(candidate_json.get("skew"))
    skew_regime = _classify_skew_regime(skew)
    vvix = _as_float(candidate_json.get("vvix"))
    vvix_regime = _classify_vvix_regime(vvix)

    is_opex_day = bool(candidate_json.get("is_opex_day", False))
    is_fomc_day = bool(candidate_json.get("is_fomc_day", False))
    is_triple_witching = bool(candidate_json.get("is_triple_witching", False))
    is_cpi_day = bool(candidate_json.get("is_cpi_day", False))
    is_nfp_day = bool(candidate_json.get("is_nfp_day", False))

    contracts = _as_int(candidate_json.get("contracts")) or 1
    margin_usage = compute_margin_usage_dollars(
        max_loss_points=max_loss_points,
        contracts=contracts,
        contract_multiplier=contract_multiplier,
    )
    delta_bucket = _bucket(delta_target, step=0.05)
    credit_bucket = _bucket(credit_to_width, step=0.05)
    vix_delta_interaction_bucket = (
        None if delta_bucket is None else f"{vix_regime}:{delta_bucket:.2f}"
    )
    dte_credit_interaction_bucket = (
        None if target_dte is None or credit_bucket is None else f"{target_dte}:{credit_bucket:.2f}"
    )

    return {
        "spread_side": spread_side,
        "target_dte": target_dte,
        "delta_target": delta_target,
        "credit_to_width": credit_to_width,
        "context_regime": context_regime,
        "vix_regime": vix_regime,
        "term_structure_regime": term_structure_regime,
        "spy_spx_ratio_regime": spy_spx_ratio_regime,
        "cboe_regime": cboe_regime,
        "cboe_wall_proximity": cboe_wall_proximity,
        "skew_regime": skew_regime,
        "vvix_regime": vvix_regime,
        "is_opex_day": is_opex_day,
        "is_fomc_day": is_fomc_day,
        "is_triple_witching": is_triple_witching,
        "is_cpi_day": is_cpi_day,
        "is_nfp_day": is_nfp_day,
        "contracts": contracts,
        "margin_usage": margin_usage,
        "delta_bucket": delta_bucket,
        "credit_bucket": credit_bucket,
        "vix_delta_interaction_bucket": vix_delta_interaction_bucket,
        "dte_credit_interaction_bucket": dte_credit_interaction_bucket,
    }


def build_bucket_key(features: dict[str, Any]) -> str:
    """Build the full deterministic bucket key used by the latest model.

    Parameters
    ----------
    features:
        Candidate feature dictionary generated by `extract_candidate_features`.

    Returns
    -------
    str
        Full bucket key including interaction dimensions.
    """
    return _build_bucket_key_for_dimensions(features, FULL_BUCKET_DIMENSIONS)


def build_legacy_bucket_key(features: dict[str, Any]) -> str:
    """Build the pre-interaction bucket key for backward compatibility.

    Parameters
    ----------
    features:
        Candidate feature dictionary generated by `extract_candidate_features`.

    Returns
    -------
    str
        Legacy bucket key matching model payloads trained before interaction
        dimensions were introduced.
    """
    return _build_bucket_key_for_dimensions(features, LEGACY_BUCKET_DIMENSIONS)


def build_bucket_key_levels(features: dict[str, Any]) -> dict[str, str]:
    """Build all hierarchy bucket keys for staged fallback lookups.

    Parameters
    ----------
    features:
        Candidate feature dictionary generated by `extract_candidate_features`.

    Returns
    -------
    dict[str, str]
        Mapping from hierarchy level (`full`, `relaxed_market`, `core`) to
        deterministic key strings.
    """
    return {
        "full": _build_bucket_key_for_dimensions(features, FULL_BUCKET_DIMENSIONS),
        "relaxed_market": _build_bucket_key_for_dimensions(features, RELAXED_MARKET_BUCKET_DIMENSIONS),
        "core": _build_bucket_key_for_dimensions(features, CORE_BUCKET_DIMENSIONS),
    }


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
    pnls = [float(r["realized_pnl"]) for r in rows
            if r.get("realized_pnl") is not None and not math.isnan(float(r["realized_pnl"]))]
    if not pnls:
        return {
            "count": 0, "wins": 0, "prob_tp50": None,
            "expected_pnl": None, "pnl_std": None,
            "tail_loss_proxy": None, "avg_margin_usage": None,
        }
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


def _adaptive_prior_strength(
    *,
    base_prior_strength: float,
    total_rows: int,
    reference_rows: int,
    min_prior_strength: float,
    max_prior_strength: float,
) -> float:
    """Scale prior strength based on sample size for sparse-data robustness.

    Parameters
    ----------
    base_prior_strength:
        User-configured baseline Bayesian prior strength.
    total_rows:
        Number of usable labeled rows available for training.
    reference_rows:
        Row-count reference where prior remains near the base value.
    min_prior_strength:
        Lower clamp to avoid near-zero smoothing on large samples.
    max_prior_strength:
        Upper clamp to avoid over-smoothing on tiny samples.

    Returns
    -------
    float
        Effective prior strength used for bucket smoothing in this run.
    """
    safe_base = max(float(base_prior_strength), 0.0)
    safe_reference = max(int(reference_rows), 1)
    safe_total = max(int(total_rows), 1)
    scaled = safe_base * (safe_reference / float(safe_total))
    lower = max(float(min_prior_strength), 0.0)
    upper = max(float(max_prior_strength), lower)
    return min(max(scaled, lower), upper)


def _build_hierarchical_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group rows by each hierarchy level key for fallback-friendly training.

    Parameters
    ----------
    rows:
        Training rows containing `features`, realized PnL, and hit flags.

    Returns
    -------
    dict[str, dict[str, list[dict[str, Any]]]]
        Nested map keyed by hierarchy level, then by deterministic bucket key.
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "full": defaultdict(list),
        "relaxed_market": defaultdict(list),
        "core": defaultdict(list),
    }
    for row in rows:
        level_keys = build_bucket_key_levels(row["features"])
        for level_name, key in level_keys.items():
            grouped[level_name][key].append(row)
    return grouped


def _smoothed_bucket_stats(
    *,
    bucket_rows: list[dict[str, Any]],
    global_stats: dict[str, float | int | None],
    prior_strength: float,
    min_bucket_size: int,
    level_name: str,
) -> dict[str, Any]:
    """Compute smoothed bucket stats for one hierarchy level group.

    Parameters
    ----------
    bucket_rows:
        Rows assigned to this grouped key.
    global_stats:
        Global baseline stats used as Bayesian prior anchor.
    prior_strength:
        Effective prior mass used for smoothing.
    min_bucket_size:
        Minimum count treated as well-supported.
    level_name:
        Hierarchy level name (`full`, `relaxed_market`, or `core`).

    Returns
    -------
    dict[str, Any]
        Smoothed bucket statistics with support-aware source labeling.
    """
    raw = _stats_for_rows(bucket_rows)
    count = int(raw["count"] or 0)
    wins = int(raw["wins"] or 0)
    pnl_sum = sum(float(r["realized_pnl"]) for r in bucket_rows)
    margin_sum = sum(float(r.get("margin_usage", 0.0)) for r in bucket_rows)

    global_prob = float(global_stats["prob_tp50"] or 0.0)
    global_exp = float(global_stats["expected_pnl"] or 0.0)
    global_tail = float(global_stats["tail_loss_proxy"] or 0.0)
    global_margin = float(global_stats["avg_margin_usage"] or 0.0)
    global_std = float(global_stats["pnl_std"] or 0.0)
    smooth_denom = count + max(prior_strength, 0.0)
    if smooth_denom <= 0:
        smooth_denom = 1.0

    prob = (wins + global_prob * prior_strength) / smooth_denom
    exp = (pnl_sum + global_exp * prior_strength) / smooth_denom
    tail = ((float(raw["tail_loss_proxy"] or 0.0) * count) + (global_tail * prior_strength)) / smooth_denom
    std = ((float(raw["pnl_std"] or 0.0) * count) + (global_std * prior_strength)) / smooth_denom
    avg_margin = (margin_sum + global_margin * prior_strength) / smooth_denom
    support_state = "bucket" if count >= min_bucket_size else "low_sample"
    return {
        "count": count,
        "prob_tp50": prob,
        "expected_pnl": exp,
        "tail_loss_proxy": tail,
        "pnl_std": std,
        "avg_margin_usage": avg_margin,
        "level": level_name,
        "source": f"{level_name}_{support_state}",
    }


def train_bucket_model(
    *,
    rows: list[dict[str, Any]],
    min_bucket_size: int = 12,
    prior_strength: float = 8.0,
    adaptive_prior_enabled: bool = True,
    adaptive_prior_reference_rows: int = 200,
    adaptive_prior_min: float = 2.0,
    adaptive_prior_max: float = 24.0,
    utility_prob_weight: float = 0.35,
    utility_tail_penalty: float = 0.20,
    utility_margin_penalty: float = 0.02,
) -> dict[str, Any]:
    """Train a smoothed empirical bucket model with hierarchical fallback stats.

    Parameters
    ----------
    rows:
        Labeled candidate rows with extracted features and realized outcomes.
    min_bucket_size:
        Minimum bucket support treated as high-confidence.
    prior_strength:
        Baseline Bayesian prior mass for bucket smoothing.
    adaptive_prior_enabled:
        Enables sample-size aware scaling of prior strength.
    adaptive_prior_reference_rows:
        Reference row count for adaptive prior scaling.
    adaptive_prior_min:
        Lower clamp for effective adaptive prior.
    adaptive_prior_max:
        Upper clamp for effective adaptive prior.
    utility_prob_weight:
        Weight applied to probability component in utility score.
    utility_tail_penalty:
        Penalty weight for adverse tail-risk proxy.
    utility_margin_penalty:
        Penalty weight for margin usage.

    Returns
    -------
    dict[str, Any]
        Model payload containing global stats, hierarchy bucket stats, utility
        weights, and fallback order metadata.
    """
    usable = [r for r in rows if r.get("realized_pnl") is not None and r.get("features")]
    global_stats = _stats_for_rows(usable)
    effective_prior = float(prior_strength)
    if adaptive_prior_enabled:
        effective_prior = _adaptive_prior_strength(
            base_prior_strength=prior_strength,
            total_rows=len(usable),
            reference_rows=adaptive_prior_reference_rows,
            min_prior_strength=adaptive_prior_min,
            max_prior_strength=adaptive_prior_max,
        )

    grouped = _build_hierarchical_groups(usable)
    bucket_hierarchy: dict[str, dict[str, dict[str, Any]]] = {}
    for level_name in ("full", "relaxed_market", "core"):
        level_groups = grouped[level_name]
        level_stats: dict[str, dict[str, Any]] = {}
        for key, bucket_rows in level_groups.items():
            if not bucket_rows:
                continue
            level_stats[key] = _smoothed_bucket_stats(
                bucket_rows=bucket_rows,
                global_stats=global_stats,
                prior_strength=effective_prior,
                min_bucket_size=min_bucket_size,
                level_name=level_name,
            )
        bucket_hierarchy[level_name] = level_stats

    return {
        "model_type": "bucket_empirical_v1",
        "trained_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "min_bucket_size": min_bucket_size,
        "prior_strength": effective_prior,
        "prior_strength_base": float(prior_strength),
        "adaptive_prior_enabled": adaptive_prior_enabled,
        "adaptive_prior_reference_rows": adaptive_prior_reference_rows,
        "adaptive_prior_min": adaptive_prior_min,
        "adaptive_prior_max": adaptive_prior_max,
        "utility_weights": {
            "prob_weight": utility_prob_weight,
            "tail_penalty": utility_tail_penalty,
            "margin_penalty": utility_margin_penalty,
        },
        "global": global_stats,
        "buckets": bucket_hierarchy.get("full", {}),
        "bucket_hierarchy": bucket_hierarchy,
        "fallback_order": list(BUCKET_HIERARCHY_ORDER),
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


def _resolve_bucket_stats(model_payload: dict[str, Any], features: dict[str, Any], min_bucket_size: int) -> tuple[dict[str, Any], str, str]:
    """Resolve prediction stats via hierarchy fallbacks, then global fallback.

    Parameters
    ----------
    model_payload:
        Trained model payload returned by `train_bucket_model`.
    features:
        Candidate feature dictionary for scoring.
    min_bucket_size:
        Minimum support threshold used to classify low-sample buckets.

    Returns
    -------
    tuple[dict[str, Any], str, str]
        Selected stats payload, selected hierarchy level, and selected key.
    """
    fallback_order_raw = model_payload.get("fallback_order")
    fallback_order = (
        [str(level) for level in fallback_order_raw]
        if isinstance(fallback_order_raw, list) and fallback_order_raw
        else list(BUCKET_HIERARCHY_ORDER)
    )
    hierarchy = model_payload.get("bucket_hierarchy")
    level_keys = build_bucket_key_levels(features)
    low_sample_candidate: tuple[dict[str, Any], str, str] | None = None
    if isinstance(hierarchy, dict):
        for level_name in fallback_order:
            if level_name == "global":
                continue
            level_key = level_keys.get(level_name)
            if not level_key:
                continue
            level_map = hierarchy.get(level_name)
            if not isinstance(level_map, dict):
                continue
            stats = level_map.get(level_key)
            if not isinstance(stats, dict):
                continue
            count = int(stats.get("count") or 0)
            if count >= min_bucket_size:
                return stats, level_name, level_key
            if count > 0 and low_sample_candidate is None:
                low_sample_candidate = (stats, level_name, level_key)
        if low_sample_candidate is not None:
            return low_sample_candidate

    buckets = model_payload.get("buckets")
    if isinstance(buckets, dict):
        for key, level_name in (
            (build_bucket_key(features), "full"),
            (build_legacy_bucket_key(features), "legacy_full"),
        ):
            stats = buckets.get(key)
            if not isinstance(stats, dict):
                continue
            count = int(stats.get("count") or 0)
            if count >= min_bucket_size:
                return stats, level_name, key
            if count > 0 and low_sample_candidate is None:
                low_sample_candidate = (stats, level_name, key)
        if low_sample_candidate is not None:
            return low_sample_candidate

    global_stats = model_payload.get("global") or {}
    return (
        {
            "count": 0,
            "prob_tp50": float(global_stats.get("prob_tp50") or 0.0),
            "expected_pnl": float(global_stats.get("expected_pnl") or 0.0),
            "tail_loss_proxy": float(global_stats.get("tail_loss_proxy") or 0.0),
            "pnl_std": float(global_stats.get("pnl_std") or 0.0),
            "avg_margin_usage": float(global_stats.get("avg_margin_usage") or 0.0),
            "source": "global_fallback",
            "level": "global",
        },
        "global",
        "global",
    )


def predict_with_bucket_model(*, model_payload: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Score one candidate with hierarchical bucket fallback.

    Parameters
    ----------
    model_payload:
        Trained payload containing global stats and bucket hierarchy maps.
    features:
        Candidate features produced by `extract_candidate_features`.

    Returns
    -------
    dict[str, Any]
        Prediction payload with probabilities, expected PnL, utility score,
        and selected fallback level metadata.
    """
    min_bucket_size = int(model_payload.get("min_bucket_size") or 0)
    stats, bucket_level, bucket_key = _resolve_bucket_stats(model_payload, features, min_bucket_size)

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
        "bucket_level": bucket_level,
        "bucket_count": int(stats.get("count") or 0),
        "source": stats.get("source") or "unknown",
        "probability_win": probability_win,
        "expected_pnl": expected_pnl,
        "tail_loss_proxy": tail_loss_proxy,
        "pnl_std": pnl_std,
        "margin_usage": margin_usage,
        "utility_score": utility_score,
    }


# ===================================================================
# XGBoost entry model helpers (live inference)
# ===================================================================

_XGB_CONTINUOUS = [
    "vix", "vix9d", "term_structure", "vvix", "skew",
    "delta_target", "credit_to_width", "entry_credit", "width_points",
    "spot", "spy_price",
    "short_iv", "long_iv", "short_delta", "long_delta",
    "offline_gex_net", "offline_zero_gamma",
    "max_loss",
]

_XGB_ORDINAL = ["dte_target", "entry_hour"]
_XGB_BINARY = ["is_opex_day", "is_fomc_day", "is_triple_witching", "is_cpi_day", "is_nfp_day"]


def extract_xgb_features(
    candidate_json: dict[str, Any],
    *,
    max_loss_points: float | None = None,
    contract_multiplier: int = 100,
    candidate_ts: datetime | None = None,
) -> dict[str, Any]:
    """Extract continuous features for the XGBoost entry model.

    Maps the same ``candidate_json`` payload used by
    ``extract_candidate_features`` into the flat continuous/ordinal/binary
    columns that the offline ``xgb_model.py`` expects, plus engineered
    interaction features.

    Parameters
    ----------
    candidate_json:
        Raw candidate payload from trade_candidates.
    max_loss_points:
        Max loss in spread points (width - credit).  Stored directly as
        the ``max_loss`` feature to match the offline training CSV scale.
    contract_multiplier:
        Retained for API compatibility; no longer used for max_loss.
    candidate_ts:
        Candidate timestamp; used to derive ``entry_hour`` (ET).
        Falls back to ``candidate_json["entry_dt"]`` if not provided.

    Returns
    -------
    dict with feature names as keys and float/int values.
    """
    ctx = candidate_json.get("context") or {}
    cboe = candidate_json.get("cboe_context") or {}

    def _f(key: str, *fallback_dicts: dict) -> float | None:
        v = _as_float(candidate_json.get(key))
        if v is not None:
            return v
        for d in fallback_dicts:
            v = _as_float(d.get(key))
            if v is not None:
                return v
        return None

    vix = _f("vix", ctx)
    vix9d = _f("vix9d", ctx)
    term_structure = _f("term_structure", ctx)
    vvix = _f("vvix", ctx)
    skew = _f("skew", ctx)
    delta_target = _f("delta_target")
    entry_credit = _f("entry_credit")
    width_points = _f("width_points")
    credit_to_width = _f("credit_to_width")
    if credit_to_width is None and entry_credit and width_points and width_points > 0:
        credit_to_width = entry_credit / width_points

    spot = _f("spot", ctx)
    if spot is None:
        spot = _f("spx_price", ctx)
    spy_price = _f("spy_price", ctx)

    # Production candidate_json nests leg fields under "legs.short" / "legs.long";
    # offline training CSVs have flat "short_iv" etc.  Try top-level first, then
    # fall back to the nested leg structure.
    legs = candidate_json.get("legs") or {}
    short_leg = legs.get("short") or {}
    long_leg = legs.get("long") or {}

    short_iv = _f("short_iv")
    if short_iv is None:
        short_iv = _as_float(short_leg.get("iv"))
    long_iv = _f("long_iv")
    if long_iv is None:
        long_iv = _as_float(long_leg.get("iv"))
    short_delta = _f("short_delta")
    if short_delta is None:
        short_delta = _as_float(short_leg.get("delta"))
    long_delta = _f("long_delta")
    if long_delta is None:
        long_delta = _as_float(long_leg.get("delta"))
    gex_net = _f("offline_gex_net", cboe)
    if gex_net is None:
        gex_net = _f("expiry_gex_net", cboe)
    zero_gamma = _f("offline_zero_gamma", cboe)

    ml_points = max_loss_points
    if ml_points is None:
        ml_points = _as_float(candidate_json.get("max_loss"))
    # Keep max_loss in points (width - credit) to match the offline training
    # CSV.  Previous formula multiplied by contracts * contract_multiplier,
    # producing dollar values ~100x larger than training splits.
    max_loss = ml_points

    target_dte = _as_int(candidate_json.get("target_dte"))
    spread_side = str(candidate_json.get("spread_side") or "unknown").lower()
    is_put = 1 if spread_side == "put" else 0

    is_opex = int(bool(candidate_json.get("is_opex_day", False)))
    is_fomc = int(bool(candidate_json.get("is_fomc_day", False)))
    is_tw = int(bool(candidate_json.get("is_triple_witching", False)))
    is_cpi = int(bool(candidate_json.get("is_cpi_day", False)))
    is_nfp = int(bool(candidate_json.get("is_nfp_day", False)))

    # Entry hour (Eastern Time)
    entry_hour: int | None = None
    if candidate_ts is not None:
        et = candidate_ts.astimezone(ZoneInfo("America/New_York"))
        entry_hour = et.hour
    else:
        raw_dt = candidate_json.get("entry_dt") or candidate_json.get("ts")
        if raw_dt:
            try:
                parsed = datetime.fromisoformat(str(raw_dt))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                entry_hour = parsed.astimezone(ZoneInfo("America/New_York")).hour
            except (ValueError, TypeError):
                pass

    features = {
        "vix": vix, "vix9d": vix9d, "term_structure": term_structure,
        "vvix": vvix, "skew": skew, "delta_target": delta_target,
        "credit_to_width": credit_to_width, "entry_credit": entry_credit,
        "width_points": width_points, "spot": spot, "spy_price": spy_price,
        "short_iv": short_iv, "long_iv": long_iv,
        "short_delta": short_delta, "long_delta": long_delta,
        "offline_gex_net": gex_net, "offline_zero_gamma": zero_gamma,
        "max_loss": max_loss,
        "dte_target": target_dte, "entry_hour": entry_hour,
        "is_opex_day": is_opex, "is_fomc_day": is_fomc, "is_triple_witching": is_tw,
        "is_cpi_day": is_cpi, "is_nfp_day": is_nfp,
        "is_put": is_put,
        # Engineered
        "vix_x_delta": (vix or 0) * (delta_target or 0),
        "dte_x_credit": (target_dte or 0) * (credit_to_width or 0),
        "gex_sign": (1.0 if (gex_net or 0) > 0 else (-1.0 if (gex_net or 0) < 0 else 0.0)),
    }
    return features


def predict_xgb_entry(
    model_payload: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    """Score a candidate using an XGBoost entry model stored in model_payload.

    Loads XGBoost Booster objects from the JSON strings stored in the
    payload, builds a DMatrix from the feature dict, and returns
    probability_win / expected_pnl / utility_score matching the
    interface of ``predict_with_bucket_model``.

    Parameters
    ----------
    model_payload:
        Must contain ``classifier_json``, ``regressor_json``, and
        ``feature_names``.  ``model_type`` should be ``"xgb_entry_v1"``.
    features:
        Dict from ``extract_xgb_features``.

    Returns
    -------
    dict with probability_win, expected_pnl, utility_score, and source.
    """
    import numpy as np
    import xgboost as xgb

    feature_names: list[str] = model_payload["feature_names"]
    row = [float(features.get(fn) or 0) if features.get(fn) is not None else float("nan")
           for fn in feature_names]
    dmat = xgb.DMatrix(np.array([row]), feature_names=feature_names)

    cls_booster = xgb.Booster()
    cls_booster.load_model(bytearray(model_payload["classifier_json"].encode("utf-8")))
    reg_booster = xgb.Booster()
    reg_booster.load_model(bytearray(model_payload["regressor_json"].encode("utf-8")))

    prob_win = float(cls_booster.predict(dmat)[0])
    expected_pnl = float(reg_booster.predict(dmat)[0])
    utility_score = prob_win * max(expected_pnl, 0.0)

    return {
        "source": "xgb_entry_v1",
        "probability_win": prob_win,
        "expected_pnl": expected_pnl,
        "utility_score": utility_score,
        "bucket_key": None,
        "bucket_level": None,
        "bucket_count": 0,
        "tail_loss_proxy": 0.0,
        "pnl_std": 0.0,
        "margin_usage": 0.0,
    }

