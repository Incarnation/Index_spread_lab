"""Shared low-level helpers reused across jobs and services.

This package hosts pure-function utilities (no DB / IO side effects)
that should not duplicate across job/service modules.  See
``pricing.py`` for canonical bid/ask midpoint computation and
``options.py`` for canonical put/call resolution from heterogeneous
trade-leg shapes.
"""
