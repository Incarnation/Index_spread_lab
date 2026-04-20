"""Pure parsers shared by ingestion jobs.

Audit Wave 4 / Refactor #1
--------------------------
``cboe_gex_job`` historically embedded ~150 lines of pure
type-coercion and JSON-shape normalisation helpers (``_to_float``,
``_to_int``, ``_parse_iso_date``, ``_parse_payload_timestamp``,
``CboeExposureItem``, ``_normalize_exposure_items``). They had no
DB / network / config dependencies but lived next to the orchestration
code, which (a) made the orchestration harder to read and (b) made
the parsers untestable in isolation without importing the full
SQLAlchemy + market-clock + alerts stack.

This module hosts the canonical parsers. ``cboe_gex_job`` re-exports
them under their original private underscored names so existing imports
(internal call sites + tests) keep working bytecode-identically; new
call sites should prefer the public names exposed here.

The functions are intentionally pure:
* No I/O (no DB, no HTTP, no logger -- callers do the logging on top).
* No mutable global state.
* Defensive against malformed vendor payloads (return ``None`` /
  empty list rather than raising on bad input).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo


def to_float(value: Any) -> float | None:
    """Convert an arbitrary value to float when possible.

    Returns
    -------
    float | None
        ``None`` for ``None`` input or any value that fails ``float()``.
    """
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    """Convert an arbitrary value to int when possible.

    Returns
    -------
    int | None
        ``None`` for ``None`` input or any value that fails ``int()``.
    """
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def to_float_list(value: Any) -> list[float | None]:
    """Normalize JSON arrays into a float-or-None list.

    Returns an empty list when the input is not a list (the most common
    vendor-drift failure mode). Each element is coerced via
    ``to_float`` so a partially-bad list still yields a usable shape.
    """
    if not isinstance(value, list):
        return []
    return [to_float(item) for item in value]


def to_int_list(value: Any) -> list[int | None]:
    """Normalize JSON arrays into an int-or-None list.

    See ``to_float_list`` for the defensive shape contract.
    """
    if not isinstance(value, list):
        return []
    return [to_int(item) for item in value]


def parse_iso_date(value: Any) -> date | None:
    """Parse ISO date text (or pass-through a ``date``) into a ``date``.

    Returns ``None`` on any non-date-shaped input so call sites can
    skip the offending row without try/except scaffolding.
    """
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except Exception:
        return None


def parse_payload_timestamp(value: Any, *, fallback: datetime) -> datetime:
    """Parse a vendor exposure timestamp into a tz-aware UTC datetime.

    Accepts ``datetime`` (naive promoted to UTC by L3 audit fix to
    *America/New_York* -- mzdata anchors its CBOE-mode exposures to ET,
    not UTC; promoting a naive value to UTC would shift it forward by
    4-5 hours), ISO 8601 strings (with or without ``Z`` suffix), and
    falls back to ``fallback`` (typically the writer's ``now_utc``)
    for unparseable input. The fallback is required so callers always
    get a usable timestamp even when the vendor omits the field
    entirely.
    """
    et_zone = ZoneInfo("America/New_York")
    utc_zone = ZoneInfo("UTC")
    if isinstance(value, datetime):
        # L3 (audit): naive vendor datetimes are ET, not UTC.
        if value.tzinfo:
            return value.astimezone(utc_zone)
        return value.replace(tzinfo=et_zone).astimezone(utc_zone)
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            try:
                normalized = raw.replace("Z", "+00:00")
                parsed = datetime.fromisoformat(normalized)
                # L3 (audit): same naive→ET promotion as above.
                if parsed.tzinfo:
                    return parsed.astimezone(utc_zone)
                return parsed.replace(tzinfo=et_zone).astimezone(utc_zone)
            except Exception:
                pass
    return fallback


def series_float(values: list[float | None], index: int, *, default: float = 0.0) -> float:
    """Return one float series value by index with a safe default.

    Used by aggregation paths that walk parallel ``strikes`` and
    ``netGamma`` arrays; out-of-range / ``None`` entries collapse to
    ``default`` so the aggregation never short-circuits on a single
    bad cell.
    """
    if index < 0 or index >= len(values):
        return default
    value = values[index]
    return default if value is None else float(value)


def series_int(values: list[int | None], index: int, *, default: int = 0) -> int:
    """Return one integer series value by index with a safe default.

    Mirror of ``series_float`` for ``openInterest`` arrays. The default
    is ``0`` (not ``None``) so downstream sums treat missing OI as
    "no contribution" rather than NaN-poisoning the running totals.
    """
    if index < 0 or index >= len(values):
        return default
    value = values[index]
    return default if value is None else int(value)


@dataclass(frozen=True)
class CboeExposureItem:
    """Normalized one-expiration exposure row from MZData.

    Field semantics
    ---------------
    expiration:
        The expiration date this row represents.
    dte_days:
        Vendor-reported days-to-expiration; ``None`` when MZData
        omits the field.
    strikes:
        Sorted strike series. Index ``i`` aligns with the same index
        in every parallel array below.
    net_gamma:
        Per-strike net dealer gamma exposure (calls minus puts in
        MZData's sign convention).
    call_abs_gamma / put_abs_gamma:
        Per-strike absolute gamma exposure for each side; used for
        side-decomposed aggregates.
    call_open_interest / put_open_interest:
        Per-strike open interest for each side; used to weight
        zero-gamma estimation and for diagnostics.
    raw_item:
        The original vendor dict, retained so checksumming / replay /
        forensic dumps can reconstruct the unfiltered payload.
    """

    expiration: date
    dte_days: int | None
    strikes: list[float | None]
    net_gamma: list[float | None]
    call_abs_gamma: list[float | None]
    put_abs_gamma: list[float | None]
    call_open_interest: list[int | None]
    put_open_interest: list[int | None]
    raw_item: dict[str, Any]


def normalize_cboe_exposure_items(payload: dict[str, Any]) -> list[CboeExposureItem]:
    """Convert one exposure payload into normalized per-expiration items.

    Returns an empty list when the payload's ``data`` field is missing
    or is not a list. ``cboe_gex_job._run_once_for_underlying`` checks
    this *and* the additional empty-list case (audit M10) before this
    function is called; here we only handle the malformed-shape cases.
    """
    items_raw = payload.get("data")
    if not isinstance(items_raw, list):
        return []

    normalized: list[CboeExposureItem] = []
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        expiration = parse_iso_date(item.get("expiration"))
        if expiration is None:
            continue
        call_payload = item.get("call") if isinstance(item.get("call"), dict) else {}
        put_payload = item.get("put") if isinstance(item.get("put"), dict) else {}
        strikes = to_float_list(item.get("strikes"))
        normalized.append(
            CboeExposureItem(
                expiration=expiration,
                dte_days=to_int(item.get("dte")),
                strikes=strikes,
                net_gamma=to_float_list(item.get("netGamma")),
                call_abs_gamma=to_float_list(call_payload.get("absGamma")),
                put_abs_gamma=to_float_list(put_payload.get("absGamma")),
                call_open_interest=to_int_list(call_payload.get("openInterest")),
                put_open_interest=to_int_list(put_payload.get("openInterest")),
                raw_item=item,
            )
        )
    return normalized
