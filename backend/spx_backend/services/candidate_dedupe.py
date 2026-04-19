"""Shared candidate-dedupe key generation for live and backtest paths.

Both the live :class:`DecisionJob` and the backtest engine in
``backend/scripts/backtest_strategy.py`` need a *stable identity* for a
spread candidate so the same exact leg pair is never executed twice in
the same decision pass / day.

Historically these two paths used different dedup strategies:

- Live ``decision_job._candidate_dedupe_key`` keyed on
  ``(spread_side, expiration, short_symbol, long_symbol)`` extracted
  from ``chosen_legs_json``.
- Backtest used ``DataFrame.index``-based dedup between event and
  scheduled candidate slates within a single day, which means the same
  leg pair sourced from two different rows could be picked twice.

Audit finding **M3** in
``backend/scripts/OFFLINE_PIPELINE_AUDIT.md`` flagged this divergence.
This module is the single canonical implementation; both consumers are
thin wrappers around :func:`candidate_dedupe_key`.

Input shapes accepted
---------------------

The helper accepts any mapping-like object that exposes either:

1. Live shape: ``chosen_legs_json={"short": {"symbol": ...},
   "long": {"symbol": ...}, "spread_side": ...}``
   plus a top-level ``expiration``.
2. Flat shape: top-level ``short_symbol``, ``long_symbol``,
   ``spread_side``, ``expiration`` (this is the
   ``training_candidates.csv`` row layout used by the backtest).

If both shapes are present (very rare during transitions), the live
nested shape wins. Missing fields normalize to empty strings so the key
is always a 4-tuple of ``str`` and is safe to use in a ``set`` or as a
``dict`` key.
"""
from __future__ import annotations

from typing import Any, Mapping


CandidateKey = tuple[str, str, str, str]
"""Shape of the dedupe key: ``(spread_side, expiration, short_symbol, long_symbol)``."""


def _stringify(value: Any) -> str:
    """Coerce *value* to ``str`` while normalizing ``None`` and pandas-NaN.

    The backtest's ``training_candidates.csv`` is read via pandas so
    missing cells can be ``float('nan')`` rather than ``None``. Both
    must collapse to the empty string so two rows with the same
    "missing-X" semantics produce the same key.
    """
    if value is None:
        return ""
    # NaN is the only float that fails self-equality.
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def candidate_dedupe_key(candidate: Mapping[str, Any]) -> CandidateKey:
    """Return a stable identity tuple for *candidate*.

    See module docstring for accepted input shapes. The returned
    4-tuple is suitable for use in a ``set`` or as a ``dict`` key, and
    is identical between live and backtest as long as the underlying
    leg identity is the same.

    Parameters
    ----------
    candidate
        A mapping (``dict``, ``pandas.Series`` row, or any object
        supporting ``__getitem__`` and ``.get(key, default)``) that
        carries the spread legs in either the live nested shape or the
        flat backtest row shape.

    Returns
    -------
    Tuple of ``(spread_side, expiration, short_symbol, long_symbol)``,
    each element guaranteed to be a ``str`` (never ``None`` or NaN).
    """
    chosen_legs = candidate.get("chosen_legs_json") or {}

    # Prefer nested live shape when present.
    short_obj = chosen_legs.get("short") or {}
    long_obj = chosen_legs.get("long") or {}
    short_symbol = _stringify(short_obj.get("symbol"))
    long_symbol = _stringify(long_obj.get("symbol"))
    spread_side = _stringify(chosen_legs.get("spread_side"))
    # ``chosen_legs`` may itself carry an ``expiration`` (live path) but
    # the top-level ``expiration`` field is also valid (backtest +
    # legacy live rows).
    expiration = _stringify(
        candidate.get("expiration") or chosen_legs.get("expiration")
    )

    # Fallback to flat shape when the nested keys were absent.
    if not short_symbol:
        short_symbol = _stringify(candidate.get("short_symbol"))
    if not long_symbol:
        long_symbol = _stringify(candidate.get("long_symbol"))
    if not spread_side:
        spread_side = _stringify(candidate.get("spread_side"))

    return (spread_side, expiration, short_symbol, long_symbol)
