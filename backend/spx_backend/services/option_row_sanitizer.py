"""Option chain row sanitizer + binder.

Audit Refactor #4 (folds in M18)
--------------------------------
Two consumers parse / validate Tradier option chain rows:

* ``snapshot_job`` ingests them into ``option_chain_rows``.
* ``decision_job`` reads them back with ``WHERE delta IS NOT NULL`` /
  ``WHERE option_right IS NOT NULL`` style filters on the read path.

Before this module each consumer rolled its own filtering:
``snapshot_job`` skipped non-dict / no-symbol entries inline but
silently inserted rows with ``option_right=NULL`` (audit M18 found
~2.4% of rows orphaned this way -- effectively dead weight that
``decision_job`` later excluded with no log line). This module
centralises the rules, names them, and emits one structured log line
per skipped row so the operator sees vendor drift instead of a quiet
NULL accumulation.

The single public entry point is ``sanitize_chain_options`` which
returns a tuple ``(rows, skip_counts)``:

* ``rows`` -- list of fully-typed dicts ready for binding by
  ``session.execute(..., rows)`` (M3 executemany).
* ``skip_counts`` -- dict[reason -> count] for run-level reporting; the
  caller logs the totals once per expiration to keep per-row log
  volume manageable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from loguru import logger


SKIP_REASON_NOT_DICT = "not_dict"
SKIP_REASON_NO_SYMBOL = "no_symbol"
SKIP_REASON_NULL_RIGHT = "null_option_right"  # audit M18
SKIP_REASON_NULL_STRIKE = "null_strike"
SKIP_REASON_STRIKE_FILTERED = "strike_filtered"


@dataclass(frozen=True)
class SanitizedRows:
    """Result bundle returned by ``sanitize_chain_options``.

    Attributes
    ----------
    rows:
        List of dicts ready for ``session.execute`` binding.
    skip_counts:
        Per-reason skip tally used for structured run-level logging.
    """

    rows: list[dict]
    skip_counts: dict[str, int]


def _to_int(value: object) -> int | None:
    """Best-effort int coercion; None for non-numeric / None inputs."""
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _to_float(value: object) -> float | None:
    """Best-effort float coercion; None for non-numeric / None inputs."""
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_date(value: object) -> date | None:
    """Coerce ISO date string / date object to a ``date``; None otherwise."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except Exception:
            return None
    return None


def is_quote_valid(
    bid: float | None,
    ask: float | None,
    delta: float | None,
    strike: float | None,
) -> bool:
    """Read-side gate: rejects rows the ingest sanitizer would also reject.

    Audit Refactor #4 originally centralised the *binding* rules in
    :func:`sanitize_chain_options` (used by ``snapshot_job`` at ingest).
    The read path in ``decision_job._get_option_rows`` independently
    re-implemented the same null/sign guards inline. This helper closes
    that gap so the two consumers share one source of truth for "what
    counts as a usable two-sided quote with a known greek".

    Returns True iff every required field is present (no NULLs) and the
    quote is at least nominally two-sided (``ask > 0``, ``bid >= 0``).
    Permissively accepts ``ask == bid`` (locked market) and any
    ``delta`` value including zero (deep-OTM short-dated options); the
    sign of delta is the caller's concern, not the validator's.

    Parameters
    ----------
    bid, ask:
        Quote prices. ``ask <= 0`` (no offer) and ``bid < 0`` (impossible)
        are rejected. ``ask == 0`` means "no live offer", which the
        decision pipeline treats as unusable.
    delta:
        Option delta from the greeks. Required so the candidate ranker
        can score the row by absolute delta distance to the target.
    strike:
        Option strike. Required so spread legs can be paired by width.
    """
    if bid is None or ask is None or delta is None or strike is None:
        return False
    if ask <= 0 or bid < 0:
        return False
    return True


def normalize_option_right(opt: dict) -> str | None:
    """Normalize option right to ``'C'`` or ``'P'``; None if unrecognized.

    Tradier sends one of ``option_type``, ``put_call``, or ``right`` and
    the field name shifts between feed versions. This helper checks all
    three with the same precedence the legacy snapshot_job used so the
    refactor is bytecode-identical for any payload that previously
    parsed.
    """
    val = opt.get("option_type") or opt.get("put_call") or opt.get("right")
    if isinstance(val, str):
        v = val.strip().upper()
        if v in {"CALL", "C"}:
            return "C"
        if v in {"PUT", "P"}:
            return "P"
    return None


def sanitize_chain_options(
    options: list[Any],
    *,
    snapshot_id: int,
    underlying: str,
    fallback_expiration: date,
    selected_strikes: set[float] | None = None,
    job_name: str = "snapshot_job",
) -> SanitizedRows:
    """Filter and bind option chain rows for a single expiration.

    Parameters
    ----------
    options:
        Raw list returned by ``_parse_chain_options``. May contain
        non-dict junk; the sanitizer is defensive.
    snapshot_id:
        FK into ``chain_snapshots``; passed through to every output row.
    underlying:
        Underlying ticker (e.g. ``"SPX"``); copied onto every row.
    fallback_expiration:
        Expiration date used when ``opt['expiration_date']`` is missing
        or unparseable. Snapshot_job already knows the expiration from
        the loop variable so a simple fallback is correct.
    selected_strikes:
        Optional strike-filter set. When provided, rows whose strike is
        not in the set are skipped (counted under ``strike_filtered``).
        ``None`` disables the filter (full chain ingest).
    job_name:
        Used in log lines to disambiguate multiple snapshot streams.

    Returns
    -------
    SanitizedRows
        ``rows`` for binding by ``executemany``; ``skip_counts`` for
        structured run-level reporting by the caller.
    """
    rows: list[dict] = []
    skip_counts: dict[str, int] = {}
    for opt in options:
        if not isinstance(opt, dict):
            skip_counts[SKIP_REASON_NOT_DICT] = skip_counts.get(SKIP_REASON_NOT_DICT, 0) + 1
            continue
        symbol = opt.get("symbol")
        if not symbol:
            skip_counts[SKIP_REASON_NO_SYMBOL] = skip_counts.get(SKIP_REASON_NO_SYMBOL, 0) + 1
            continue
        strike_val = _to_float(opt.get("strike"))
        if strike_val is None:
            skip_counts[SKIP_REASON_NULL_STRIKE] = skip_counts.get(SKIP_REASON_NULL_STRIKE, 0) + 1
            continue
        if selected_strikes is not None and strike_val not in selected_strikes:
            skip_counts[SKIP_REASON_STRIKE_FILTERED] = (
                skip_counts.get(SKIP_REASON_STRIKE_FILTERED, 0) + 1
            )
            continue
        # M18 (audit): historically rows with NULL option_right got
        # inserted and then ignored downstream. Skip-and-log at ingest
        # so vendor drift surfaces in the run summary instead of
        # silently bloating the table.
        opt_right = normalize_option_right(opt)
        if opt_right is None:
            skip_counts[SKIP_REASON_NULL_RIGHT] = (
                skip_counts.get(SKIP_REASON_NULL_RIGHT, 0) + 1
            )
            continue
        greeks = opt.get("greeks") or {}
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "option_symbol": symbol,
                "underlying": underlying,
                "expiration": (_to_date(opt.get("expiration_date")) or fallback_expiration),
                "strike": strike_val,
                "option_right": opt_right,
                "bid": _to_float(opt.get("bid")),
                "ask": _to_float(opt.get("ask")),
                "last": _to_float(opt.get("last")),
                "volume": _to_int(opt.get("volume")),
                "open_interest": _to_int(opt.get("open_interest")),
                "contract_size": _to_int(opt.get("contract_size")),
                "delta": _to_float(greeks.get("delta")),
                "gamma": _to_float(greeks.get("gamma")),
                "theta": _to_float(greeks.get("theta")),
                "vega": _to_float(greeks.get("vega")),
                "rho": _to_float(greeks.get("rho")),
                "bid_iv": _to_float(greeks.get("bid_iv")),
                "mid_iv": _to_float(greeks.get("mid_iv")),
                "ask_iv": _to_float(greeks.get("ask_iv")),
                "greeks_updated_at": greeks.get("updated_at"),
                "raw_json": json.dumps(opt, default=str),
            }
        )

    if skip_counts:
        # Single structured log line per (job, snapshot) -- avoids
        # per-row log volume while still surfacing M18 drift to
        # whatever scrapes the structured log stream.
        logger.info(
            "{}: option_row_sanitizer snapshot_id={} kept={} skipped={}",
            job_name,
            snapshot_id,
            len(rows),
            skip_counts,
        )
    return SanitizedRows(rows=rows, skip_counts=skip_counts)
