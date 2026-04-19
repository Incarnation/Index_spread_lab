"""Operator-driven CSV regression scripts for the three monolith splits.

Each module under this package re-runs the same pipeline twice (via the
back-compat shim and via the new package path) and asserts byte-identical
output. Used as a pre-merge step on real data; not part of the pytest
suite (those would need Databento + production-DB access).

See ``tools/csv_regression/README.md`` for operator instructions.
"""
