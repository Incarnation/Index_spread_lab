"""Unit tests for ``backend/scripts/export_production_data.py`` helpers.

We don't connect to a real PostgreSQL database here.  These tests cover
the pure helpers added in Wave 3 (H7 + L9):

* ``_redact_order_payload``  -- PII allowlist filter for orders.*_json
* ``_atomic_write_csv``      -- temp+rename CSV writes
* ``_atomic_write_parquet``  -- temp+rename Parquet writes

Behavioural tests for each ``export_*`` function would require a live
psycopg2 connection (or a fully mocked SQLAlchemy stack), which is out
of scope for the unit-test layer; they are exercised by the operational
``--start/--end`` smoke runs documented in
``backend/scripts/OFFLINE_PIPELINE_AUDIT.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the scripts/ folder importable.  ``export_production_data`` is a
# top-level module under ``backend/scripts/`` rather than a package.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from export_production_data import (  # noqa: E402  -- after sys.path edit
    _ORDER_REQUEST_KEEP_KEYS,
    _ORDER_RESPONSE_KEEP_KEYS,
    _atomic_write_csv,
    _atomic_write_parquet,
    _redact_order_payload,
)


# ---------------------------------------------------------------------------
# PII redaction (H7)
# ---------------------------------------------------------------------------


class TestRedactOrderPayload:
    """Behavioural tests for ``_redact_order_payload``.

    The function must:
      1. Drop every key not in the allowlist.
      2. Keep allowlisted keys verbatim, including their nested values.
      3. Pass through primitives unchanged (int/float/str/bool/None).
      4. Accept either a JSON string or an already-parsed dict.
      5. Recurse into lists.
      6. Never raise on malformed JSON — return ``None`` instead.
    """

    def test_filters_out_non_allowlisted_keys(self):
        # ``account_id`` and ``api_key`` represent broker PII that must
        # never reach an offline CSV.  Allowlisted fields survive.
        payload = {
            "class": "option",
            "symbol": "SPX",
            "side": "sell_to_open",
            "quantity": 1,
            "account_id": "VA1234567",
            "api_key": "sk-secret-xyz",
            "client_ip": "192.168.0.1",
        }
        out = _redact_order_payload(payload, _ORDER_REQUEST_KEEP_KEYS)
        assert "account_id" not in out
        assert "api_key" not in out
        assert "client_ip" not in out
        assert out["class"] == "option"
        assert out["symbol"] == "SPX"
        assert out["side"] == "sell_to_open"
        assert out["quantity"] == 1

    def test_accepts_json_string_input(self):
        # The DB column is JSONB but pandas often surfaces it as a string
        # depending on the driver — the helper must handle both shapes.
        payload_str = json.dumps({
            "class": "option",
            "symbol": "SPX",
            "secret_token": "leak-me-not",
        })
        out = _redact_order_payload(payload_str, _ORDER_REQUEST_KEEP_KEYS)
        assert isinstance(out, dict)
        assert "secret_token" not in out
        assert out["class"] == "option"

    def test_returns_none_for_invalid_json_string(self):
        # A non-JSON string (e.g. raw HTTP error body) must be dropped
        # entirely so a stack trace doesn't accidentally leak.
        out = _redact_order_payload(
            "not-actually-json {[", _ORDER_REQUEST_KEEP_KEYS
        )
        assert out is None

    def test_returns_none_for_none_input(self):
        # NULL columns from the DB show up as None; we just propagate.
        assert _redact_order_payload(None, _ORDER_REQUEST_KEEP_KEYS) is None

    def test_recurses_into_lists(self):
        # Multi-leg order requests come through as a top-level list of
        # leg dicts under ``preview`` etc.  Each list element must be
        # filtered, not passed through whole.
        payload = [
            {"symbol": "SPX", "account_id": "VA1"},
            {"symbol": "SPY", "account_id": "VA2"},
        ]
        out = _redact_order_payload(payload, _ORDER_REQUEST_KEEP_KEYS)
        assert isinstance(out, list)
        assert len(out) == 2
        for entry in out:
            assert "account_id" not in entry
            assert entry["symbol"] in {"SPX", "SPY"}

    def test_response_allowlist_keeps_status_and_id(self):
        # The response allowlist is intentionally broader than the
        # request allowlist (status, fill price, error messages are all
        # forensics-relevant); broker partner_id is allowed because it's
        # a tracking ID and not user-identifying.
        payload = {
            "id": "ord-1234",
            "status": "filled",
            "avg_fill_price": 1.23,
            "exec_quantity": 1,
            "errors": None,
            # PII to drop:
            "account_number": "VA1234567",
            "client_ip": "10.0.0.1",
        }
        out = _redact_order_payload(payload, _ORDER_RESPONSE_KEEP_KEYS)
        assert out["id"] == "ord-1234"
        assert out["status"] == "filled"
        assert out["avg_fill_price"] == 1.23
        assert "account_number" not in out
        assert "client_ip" not in out

    def test_empty_dict_after_redaction_is_preserved(self):
        # If the entire payload is non-allowlisted, the helper returns
        # ``{}`` rather than ``None``.  This signals to a forensics
        # consumer that a payload existed but contained nothing safe to
        # share, distinct from "no payload was ever written".
        payload = {"only_secret": "xyz", "another_secret": 42}
        out = _redact_order_payload(payload, _ORDER_REQUEST_KEEP_KEYS)
        assert out == {}

    def test_primitive_passthrough(self):
        # Top-level primitives (the column technically permits a JSON
        # number/string) come back unchanged.
        assert _redact_order_payload(42, _ORDER_REQUEST_KEEP_KEYS) == 42
        # Strings are first JSON-parsed; "42" parses to int 42.
        assert _redact_order_payload("42", _ORDER_REQUEST_KEEP_KEYS) == 42


# ---------------------------------------------------------------------------
# Atomic writers (L9)
# ---------------------------------------------------------------------------


class TestAtomicWriters:
    """Verify temp+rename semantics for CSV and Parquet writes.

    Functional checks — we can't easily assert atomicity without a
    crash injection harness, but we can assert that:
      1. The destination file is created with the expected contents.
      2. No leftover ``*.tmp`` file lingers after a successful write.
      3. The destination directory is auto-created if missing.
    """

    def test_atomic_write_csv_creates_destination(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        dest = tmp_path / "subdir" / "out.csv"
        # Subdir does not pre-exist; helper must mkdir parents.
        assert not dest.parent.exists()

        _atomic_write_csv(df, dest)

        assert dest.exists()
        assert not dest.with_suffix(".csv.tmp").exists()
        round_tripped = pd.read_csv(dest)
        pd.testing.assert_frame_equal(round_tripped, df)

    def test_atomic_write_csv_overwrites_existing(self, tmp_path):
        # Re-writing an existing file should produce the new content
        # (regression: ``os.replace`` on POSIX/macOS overwrites the
        # destination atomically).
        dest = tmp_path / "out.csv"
        _atomic_write_csv(pd.DataFrame({"a": [1]}), dest)
        _atomic_write_csv(pd.DataFrame({"a": [2, 3]}), dest)

        out = pd.read_csv(dest)
        assert list(out["a"]) == [2, 3]
        assert not dest.with_suffix(".csv.tmp").exists()

    def test_atomic_write_parquet_creates_destination(self, tmp_path):
        # Skip if pyarrow isn't available in the CI runner.
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        dest = tmp_path / "subdir" / "out.parquet"

        _atomic_write_parquet(df, dest)

        assert dest.exists()
        assert not dest.with_suffix(".parquet.tmp").exists()
        round_tripped = pd.read_parquet(dest)
        pd.testing.assert_frame_equal(
            round_tripped.reset_index(drop=True),
            df.reset_index(drop=True),
        )
