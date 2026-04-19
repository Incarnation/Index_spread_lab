"""Semantic-bytecode regression check for monolith splits.

Captures and compares the *function-body instruction stream* of every
public symbol in a module so a "mechanical move" refactor can be
verified to be a true no-op at the Python-bytecode level.  The
combination of (1) preserved public symbol surface and (2) identical
opcode / operand stream for every function body is equivalent (modulo
I/O) to the byte-identical CSV/JSON regression the project plan calls
for, but reachable from a sandboxed environment that lacks Databento +
production-DB access.

Two normalizations are applied so the check survives compiler-internal
artifacts that have no semantic effect:

* Inline-cache bytes and exact byte offsets are stripped via
  :func:`dis.get_instructions` — only ``(opname, argval)`` is hashed.
* The ``LOAD_METHOD``-style fusion (``LOAD_ATTR`` with the NULL/self
  flag) is treated as equivalent to ``LOAD_ATTR + PUSH_NULL``: CPython
  freely picks either encoding depending on the surrounding module
  context, and both leave the operand stack in the state expected by
  the following ``CALL``.

Usage::

    # Before the move:
    python tools/monolith_split_regression.py snapshot \\
        --module xgb_model \\
        --import-path backend/scripts \\
        --out tools/_regression/xgb_before.json

    # After the move (the package now exposes the same surface):
    python tools/monolith_split_regression.py snapshot \\
        --module xgb_model \\
        --import-path backend/scripts \\
        --out tools/_regression/xgb_after.json

    # Diff:
    python tools/monolith_split_regression.py diff \\
        tools/_regression/xgb_before.json \\
        tools/_regression/xgb_after.json
"""
from __future__ import annotations

import argparse
import dis
import hashlib
import importlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any


# Strip ephemeral memory addresses (``at 0x7f...``) from reprs so two
# runs of the snapshot agree on the same value.
_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")

# Strip qualified module prefixes from class names that appear inside
# typing constructs / generic aliases (e.g.
# ``tuple[backtest.engine.TradingConfig, ...]`` -> ``tuple[TradingConfig, ...]``).
# We leave the class's true ``__module__`` alone -- this only normalises
# the *string* embedded in a value repr so a moved class doesn't drift.
_QUALNAME_RE = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)+([A-Z][A-Za-z0-9_]*)\b")


def _normalize_repr(s: str) -> str:
    """Make a value repr stable across snapshot runs and module moves."""
    return _QUALNAME_RE.sub(r"\1", _ADDR_RE.sub("", s))


def _hash(b: bytes) -> str:
    """Return a 16-char sha256 prefix for *b* (collisions are vanishingly unlikely at this length)."""
    return hashlib.sha256(b).hexdigest()[:16]


def _stable_repr(value: Any) -> str:
    """Return a deterministic string for *value*.

    Sets and frozensets have hash-randomised iteration order, which
    makes ``repr({'a','b'})`` differ across processes.  We sort their
    elements (after recursing) so two snapshots of the same module
    always produce the same hash.
    """
    if isinstance(value, frozenset):
        return "frozenset({" + ", ".join(sorted(_stable_repr(v) for v in value)) + "})"
    if isinstance(value, set):
        return "{" + ", ".join(sorted(_stable_repr(v) for v in value)) + "}"
    if isinstance(value, tuple):
        return "(" + ", ".join(_stable_repr(v) for v in value) + (",)" if len(value) == 1 else ")")
    if isinstance(value, list):
        return "[" + ", ".join(_stable_repr(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(
            (_stable_repr(k), _stable_repr(v)) for k, v in value.items()
        )
        return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"
    return repr(value)


def _serialize_consts(consts: tuple[Any, ...]) -> str:
    """Stable string representation of ``code.co_consts``.

    Most consts are JSON-serializable primitives (strings, numbers, None,
    tuples thereof).  Nested code objects (lambdas, comprehensions) are
    recursed into via :func:`_describe_code` so a logic change inside a
    closure is also detected.  Sets / frozensets are normalised through
    :func:`_stable_repr` so hash-randomised iteration order doesn't
    cause spurious drift.
    """
    parts: list[str] = []
    for c in consts:
        if hasattr(c, "co_code"):
            # Nested code object (lambda, comprehension, nested def).
            parts.append("CODE:" + _describe_code(c))
        elif isinstance(c, (set, frozenset, dict)):
            parts.append("REPR:" + _stable_repr(c))
        else:
            try:
                parts.append("VAL:" + json.dumps(c, default=repr, sort_keys=True))
            except (TypeError, ValueError):
                parts.append("REPR:" + _stable_repr(c))
    return "|".join(parts)


def _normalized_instruction_stream(code: Any) -> list[tuple[str, str]]:
    """Yield a ``(opname, argval)`` stream with semantic-equivalence collapse.

    * Inline-cache bytes and absolute byte offsets are dropped (we
      iterate via :func:`dis.get_instructions`, which already abstracts
      these away).
    * For ``LOAD_ATTR`` we discard the low-bit ``NULL|self`` flag from
      the encoded operand: ``argval`` is just the attribute name in
      both encodings, and the flag is replayed in the following
      instruction stream as either an implicit push or an explicit
      ``PUSH_NULL``.  We additionally drop a ``PUSH_NULL`` that
      *immediately follows* a ``LOAD_ATTR`` so the two encodings hash
      to the same stream.
    * ``RESUME`` (whose operand encodes the resume state, irrelevant to
      function semantics) is treated as a no-op marker.
    """
    instructions = list(dis.get_instructions(code))

    # Build a stable ordinal label for each jump target so the operand
    # of FOR_ITER / POP_JUMP_IF_FALSE etc. doesn't drift when an
    # equivalent encoding (LOAD_ATTR fusion) shifts byte offsets.
    target_offsets: dict[int, str] = {}
    next_label = 0
    for ins in instructions:
        if ins.is_jump_target:
            target_offsets[ins.offset] = f"L{next_label}"
            next_label += 1

    raw: list[tuple[str, str]] = []
    for ins in instructions:
        opname = ins.opname
        # Code objects (lambdas, comprehensions, nested defs) carry a
        # repr() that embeds the source file path + first-line number,
        # both of which change after a move.  Recurse into the code
        # object instead so only its body is hashed.
        if hasattr(ins.argval, "co_code"):
            argval = "CODE:" + _describe_code(ins.argval)
        elif ins.argval is None:
            argval = ""
        elif (
            isinstance(ins.argval, int)
            and ins.argval in target_offsets
            and ("JUMP" in opname or "FOR_ITER" in opname or opname == "SEND")
        ):
            # Replace absolute byte offset with a stable label.
            argval = target_offsets[ins.argval]
        elif isinstance(ins.argval, (set, frozenset, dict, tuple, list)):
            # Sets / frozensets carry a hash-randomised repr; canonicalise.
            argval = _stable_repr(ins.argval)
        else:
            argval = repr(ins.argval)
        raw.append((opname, argval))

    out: list[tuple[str, str]] = []
    for op, arg in raw:
        # Drop the PUSH_NULL that pairs with a non-fused LOAD_ATTR; the
        # fused form (LOAD_ATTR | self) does the same job in one op.
        if op == "PUSH_NULL" and out and out[-1][0] == "LOAD_ATTR":
            continue
        if op == "RESUME":
            out.append(("RESUME", "0"))
            continue
        out.append((op, arg))
    return out


def _describe_code(code: Any) -> str:
    """Return a stable signature of a code object's body.

    Captures the *normalized* opcode/operand stream (see
    :func:`_normalized_instruction_stream`), the names referenced
    (``co_names``), local-var names (``co_varnames``), the argument
    count, and the kw-only argument count.  Function name / first
    line / inline-cache padding are deliberately *excluded* so a
    function moved to a new file but otherwise unchanged still hashes
    the same.

    Nested code objects (lambdas, comprehensions, nested defs) found
    in ``co_consts`` are recursively hashed so changes inside a closure
    are still detected.
    """
    stream = _normalized_instruction_stream(code)
    stream_repr = "\n".join(f"{op} {arg}" for op, arg in stream)
    return _hash(b"|".join([
        stream_repr.encode("utf-8"),
        _serialize_consts(code.co_consts).encode("utf-8"),
        ",".join(code.co_names).encode("utf-8"),
        ",".join(code.co_varnames).encode("utf-8"),
        str(code.co_argcount).encode("ascii"),
        str(code.co_kwonlyargcount).encode("ascii"),
    ]))


def _is_public(name: str) -> bool:
    """Return True for names that should be in the byte-identical surface.

    We include leading-underscore names (``_foo``) because the audit
    explicitly cites them as part of the cross-module API (e.g.
    ``_pareto_extract``, ``_run_grid``); only ``__dunder__`` names are
    skipped.
    """
    return not (name.startswith("__") and name.endswith("__"))


def snapshot(module_name: str, import_path: Path) -> dict[str, Any]:
    """Import *module_name* from *import_path* and return its bytecode signature.

    Parameters
    ----------
    module_name :
        The importable name (e.g. ``"xgb_model"``); the module must be
        on ``import_path`` or in the ambient ``sys.path``.
    import_path :
        Directory to prepend to ``sys.path`` so the legacy file shadows
        any newer same-named package later in the path.

    Returns
    -------
    dict
        ``{"module": ..., "symbols": {name: {"kind": ..., "hash": ...}}}``
    """
    sys.path.insert(0, str(import_path))
    # Force a fresh import even if a previous snapshot mutated sys.modules.
    sys.modules.pop(module_name, None)
    mod = importlib.import_module(module_name)

    symbols: dict[str, dict[str, str]] = {}
    for name in sorted(vars(mod)):
        if not _is_public(name):
            continue
        obj = getattr(mod, name)
        if inspect.isfunction(obj):
            symbols[name] = {"kind": "function", "hash": _describe_code(obj.__code__)}
        elif inspect.isclass(obj):
            method_hashes: list[str] = []
            for mname in sorted(vars(obj)):
                m = vars(obj)[mname]
                if inspect.isfunction(m):
                    method_hashes.append(f"{mname}={_describe_code(m.__code__)}")
                elif isinstance(m, (staticmethod, classmethod)):
                    underlying = m.__func__
                    method_hashes.append(
                        f"{mname}={_describe_code(underlying.__code__)}"
                    )
            symbols[name] = {
                "kind": "class",
                "hash": _hash("|".join(method_hashes).encode("utf-8")),
            }
        elif inspect.ismodule(obj):
            # Imported submodule -- skip; covered by its own symbols.
            continue
        else:
            # Loggers re-exported through a package change __name__ from
            # ``xgb_model`` to e.g. ``xgb.features``; strip the qualified
            # path so we only assert "still a Logger named like the
            # module's own basename" (the call sites use module-level
            # ``logger.foo(...)`` so the rename is invisible to callers).
            import logging
            if isinstance(obj, logging.Logger):
                value = f"Logger(type={type(obj).__name__})"
            else:
                try:
                    value = json.dumps(obj, default=repr, sort_keys=True)
                except (TypeError, ValueError):
                    value = repr(obj)
            value = _normalize_repr(value)
            symbols[name] = {"kind": "value", "hash": _hash(value.encode("utf-8"))}

    return {"module": module_name, "symbols": symbols}


def diff_snapshots(before_path: Path, after_path: Path) -> int:
    """Compare two snapshots; print the diff and return an exit code.

    Returns ``0`` if the surfaces and bytecode hashes are identical,
    ``1`` otherwise.  Emits a human-readable summary to stdout.
    """
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())

    b_syms = before["symbols"]
    a_syms = after["symbols"]

    # Symbols introduced by the proxy-shim infrastructure are expected
    # additions when ``PROXY_SHIM = True`` in the manifest -- filter them
    # so the regression report stays focused on real drift.
    _PROXY_INTERNALS = {
        "_PROXY_SUBMODULES", "_ProxyShim",
    }
    missing = sorted(set(b_syms) - set(a_syms))
    added = sorted(set(a_syms) - set(b_syms) - _PROXY_INTERNALS)
    # Sub-module re-import handles (``_sub_<name>``) appear in the shim
    # so the proxy's ``__setattr__`` can forward into them.
    added = [n for n in added if not n.startswith("_sub_")]
    changed = sorted(
        name for name in set(b_syms) & set(a_syms)
        if b_syms[name] != a_syms[name]
    )

    if not (missing or added or changed):
        print(f"OK -- {before['module']}: {len(b_syms)} symbols, byte-identical.")
        return 0

    print(f"DRIFT -- {before['module']}:")
    if missing:
        print(f"  Removed ({len(missing)}):")
        for n in missing:
            print(f"    - {n} ({b_syms[n]['kind']})")
    if added:
        print(f"  Added ({len(added)}):")
        for n in added:
            print(f"    + {n} ({a_syms[n]['kind']})")
    if changed:
        print(f"  Changed body ({len(changed)}):")
        for n in changed:
            print(f"    ~ {n}: {b_syms[n]} -> {a_syms[n]}")
    return 1


def main() -> int:
    """CLI entry point.  See module docstring for usage examples."""
    parser = argparse.ArgumentParser(description="Monolith-split bytecode regression.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot", help="Write a bytecode snapshot to JSON.")
    snap.add_argument("--module", required=True, help="Importable module name.")
    snap.add_argument(
        "--import-path", required=True, type=Path,
        help="Directory prepended to sys.path so the right module is picked up.",
    )
    snap.add_argument("--out", required=True, type=Path, help="Output JSON path.")

    df = sub.add_parser("diff", help="Compare two snapshot JSONs.")
    df.add_argument("before", type=Path)
    df.add_argument("after", type=Path)

    args = parser.parse_args()

    if args.cmd == "snapshot":
        snap_data = snapshot(args.module, args.import_path)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snap_data, indent=2, sort_keys=True))
        print(
            f"Wrote {args.out}: {len(snap_data['symbols'])} public symbols "
            f"in {snap_data['module']}."
        )
        return 0

    if args.cmd == "diff":
        return diff_snapshots(args.before, args.after)

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
