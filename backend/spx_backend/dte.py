from __future__ import annotations

from datetime import date


def trading_dte_lookup(expirations: list[date], as_of: date) -> dict[date, int]:
    """Map expiration date -> trading-day DTE.

    DTE semantics:
    - 0DTE when expiration is the same trading date as `as_of`
    - otherwise 1DTE is the next available expiration date returned by Tradier
    """
    future = sorted({exp for exp in expirations if exp >= as_of})
    if not future:
        return {}
    base = 0 if as_of in future else 1
    return {exp: idx + base for idx, exp in enumerate(future)}


def choose_expiration_for_trading_dte(
    expirations: list[date],
    target_dte: int,
    as_of: date,
    tolerance: int = 1,
) -> date | None:
    """Pick the closest expiration within trading-day DTE tolerance."""
    lookup = trading_dte_lookup(expirations, as_of)
    if not lookup:
        return None
    candidates = [exp for exp, dte in lookup.items() if abs(dte - target_dte) <= tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda exp: (abs(lookup[exp] - target_dte), exp))


def closest_expiration_for_trading_dte(expirations: list[date], target_dte: int, as_of: date) -> date | None:
    """Pick the closest expiration to a trading-day DTE target."""
    lookup = trading_dte_lookup(expirations, as_of)
    if not lookup:
        return None
    return min(lookup.keys(), key=lambda exp: (abs(lookup[exp] - target_dte), exp))
