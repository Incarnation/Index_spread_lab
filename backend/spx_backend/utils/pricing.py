"""Canonical bid/ask midpoint helper for the whole backend.

Background
----------
Several jobs (``decision_job``, ``trade_pnl_job``, ``labeler_job``)
historically defined their own ``_mid(bid, ask)`` helper.  Each copy
quietly accepted dead quotes such as ``(0, 0)`` or crossed books
(``bid > ask``), producing a midpoint of ``0`` that flowed straight
into entry-credit and exit-cost math.  In practice that allowed
"free money" entries (credit-to-width gates trivially passing on a
zero exit cost) and spurious take-profit fires on closed trades whose
chain rows had been dropped by the data feed.

This module provides the single canonical ``mid_price`` helper.
Callers should ``from spx_backend.utils.pricing import mid_price`` and
treat ``None`` as "no usable mid; skip this candidate / fall through
to the next valuation tier".

Validation policy (strict)
--------------------------
``mid_price(bid, ask)`` returns ``None`` when ANY of the following hold:

* ``bid`` or ``ask`` is ``None``.
* ``bid`` or ``ask`` cannot be coerced to ``float`` (non-numeric input).
* ``bid`` or ``ask`` is non-finite (``NaN`` / ``+Inf`` / ``-Inf``).
* ``bid <= 0`` or ``ask <= 0`` (rejects dead quotes and one-sided zeros).
* ``bid > ask`` (rejects crossed books).

Otherwise returns ``(bid + ask) / 2.0``.
"""
from __future__ import annotations

import math
from typing import Any


def mid_price(bid: Any, ask: Any) -> float | None:
    """Return the midpoint of a strictly valid bid/ask pair, else ``None``.

    Parameters
    ----------
    bid:
        Raw bid price.  Accepted types: ``int``, ``float``, or anything
        that ``float()`` can coerce.  ``None`` and non-numeric inputs
        return ``None``.
    ask:
        Raw ask price, same rules as ``bid``.

    Returns
    -------
    float | None
        ``(bid + ask) / 2`` when both legs satisfy the strict validation
        policy described in the module docstring.  ``None`` otherwise.

    Notes
    -----
    Callers MUST treat ``None`` as "no usable mid"; they should not fall
    back to ``0`` or any other sentinel because that would re-introduce
    the bug this helper exists to prevent.
    """
    # Reject missing inputs up-front so the float() coercion below never
    # has to deal with None.
    if bid is None or ask is None:
        return None

    # Coerce to float; non-numeric types (e.g. dict, list, malformed
    # strings) cause TypeError / ValueError which we treat as "invalid".
    try:
        bid_f = float(bid)
        ask_f = float(ask)
    except (TypeError, ValueError):
        return None

    # NaN and +/-Inf would silently propagate through downstream credit
    # math and tank PnL accounting; reject explicitly.
    if not math.isfinite(bid_f) or not math.isfinite(ask_f):
        return None

    # Strict positivity: dead quotes (0, 0), one-sided zeros (0, ask) and
    # (bid, 0), and any negative price are not legitimate two-sided
    # quotes for the credit-spread universe we trade.
    if bid_f <= 0.0 or ask_f <= 0.0:
        return None

    # Crossed book: bid > ask is a market data error (or stale quote
    # caught mid-update).  Treat as unusable rather than averaging an
    # impossible quote.
    if bid_f > ask_f:
        return None

    return (bid_f + ask_f) / 2.0
