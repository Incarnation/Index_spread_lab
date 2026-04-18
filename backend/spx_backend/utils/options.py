"""Canonical put/call resolution for trade legs.

Background
----------
``trade_pnl_job._intrinsic_exit_cost`` historically branched on the
substring ``"put"`` / ``"call"`` inside ``trades.strategy_type``.  That
coupled correctness to a free-text column: any strategy_type that did
not contain ``"put"`` or ``"call"`` (e.g. ``"credit_spread"``) silently
dropped tier-2 intrinsic settlement and forced the trade to tier-3 even
when leg strikes were perfectly valid.

The right authoritative source is ``trade_legs.option_right``
(``"P"`` or ``"C"`` per the schema docstring), which is captured at
trade-creation time from the actual option symbol.  Strategy-type
substring checks remain only as a backward-compatible fallback for
legacy rows whose ``option_right`` column happens to be ``NULL``.

Returning ``None`` is a legitimate outcome: callers (currently only
``_intrinsic_exit_cost``) already treat ``None`` as "I cannot resolve
the option right; fall through to the next valuation tier".
"""
from __future__ import annotations

from typing import Any, Literal

from loguru import logger

OptionRight = Literal["put", "call"]


def _normalize_right(value: Any) -> OptionRight | None:
    """Normalize a raw option-right token to ``"put"`` or ``"call"``.

    Accepts ``"P"``, ``"p"``, ``"PUT"``, ``"put"`` and similar variants
    for puts; accepts ``"C"``, ``"c"``, ``"CALL"``, ``"call"`` and
    similar variants for calls.  Returns ``None`` for anything else,
    including ``None`` itself, empty strings, and unknown tokens.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    # Single-letter form (matches db_schema.sql comment "'C' or 'P'").
    if text in {"p", "put"}:
        return "put"
    if text in {"c", "call"}:
        return "call"
    return None


def resolve_option_right(
    strategy_type: str | None,
    short_leg: dict | None,
    long_leg: dict | None,
) -> OptionRight | None:
    """Resolve the put/call right for a vertical-spread trade.

    Resolution order (first non-``None`` wins):

    1. ``short_leg["option_right"]`` after normalization.
    2. ``long_leg["option_right"]`` after normalization.
    3. Substring scan of ``strategy_type`` for ``"put"`` / ``"call"``.

    If ``short_leg`` and ``long_leg`` both carry an ``option_right``
    that disagree (one says put, the other says call), this is treated
    as a malformed spread: the function logs a warning and returns
    ``None`` rather than silently picking one.

    Parameters
    ----------
    strategy_type:
        Free-text label from ``trades.strategy_type`` (e.g.
        ``"credit_vertical_put"``).  May be ``None``.
    short_leg, long_leg:
        Leg dictionaries.  Expected to carry an ``"option_right"`` key
        whose value is a single-letter or word-form right token.  May
        themselves be ``None`` when called from contexts that lack leg
        data.

    Returns
    -------
    Literal["put", "call"] | None
        The resolved right when any source can decide; ``None`` when
        no source decides or when legs disagree.
    """
    short_right = _normalize_right((short_leg or {}).get("option_right"))
    long_right = _normalize_right((long_leg or {}).get("option_right"))

    # Leg-level disagreement is a real data integrity issue: a credit
    # vertical must be either both puts or both calls.  Returning None
    # forces the caller to use a fallback tier instead of silently
    # picking a wrong intrinsic formula.
    if short_right is not None and long_right is not None and short_right != long_right:
        logger.warning(
            "resolve_option_right_mismatch: short={} long={} strategy_type={}",
            short_right, long_right, strategy_type,
        )
        return None

    # Prefer the short leg first because it is the one whose intrinsic
    # value drives the spread's exit cost; long-leg falls in next.
    if short_right is not None:
        return short_right
    if long_right is not None:
        return long_right

    # Strategy-type substring fallback: only for legacy rows whose legs
    # have NULL option_right.  We tolerate generic naming like
    # "credit_vertical_put" or "vertical_put_spread".
    if strategy_type:
        text = strategy_type.lower()
        if "put" in text:
            return "put"
        if "call" in text:
            return "call"

    return None
