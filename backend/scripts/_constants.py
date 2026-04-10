"""Shared constants for offline analysis scripts.

Centralises values that were previously duplicated across
backtest_strategy, generate_training_data, sl_recovery_analysis, and
portfolio_manager.  Import from here rather than redefining locally.
"""

from __future__ import annotations

# SPX credit-spread contract economics
MARGIN_PER_LOT: int = 1000   # 10-pt wide SPX spread × $100/pt multiplier
CONTRACT_MULT: int = 100     # options contract multiplier ($/point)
CONTRACTS: int = 1           # default lot size for single-leg analysis
