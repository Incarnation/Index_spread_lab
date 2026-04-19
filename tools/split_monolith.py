"""Mechanical monolith-splitter.

Splits a single Python file into a package of submodules using an
explicit manifest of ``(target_file, start_line, end_line, label)``
tuples.  Each chunk is copied **verbatim** into its target submodule,
then the source file is replaced by a back-compat shim that re-exports
every symbol from the new package.

This script intentionally does **no AST manipulation**.  The line
ranges in the manifest must already be a clean partition of the
"definition" portion of the source file (everything between the
top-of-file imports and ``if __name__ == "__main__":``); any leftover
lines (blank lines, top-level comments, section banners) are dropped.

Usage::

    python tools/split_monolith.py --manifest tools/_split_manifests/xgb.py

The manifest module must define:

* ``SOURCE``: ``Path`` to the monolith file (relative to repo root).
* ``PACKAGE_DIR``: ``Path`` to the new package directory.
* ``HEADER_LINES``: how many leading lines of the monolith to copy to
  every submodule as the import / setup preamble (typically through
  the last top-level import).
* ``SUBMODULES``: list of ``(filename, [(start, end), ...], extra_imports)``
  describing what goes where.  Line ranges are 1-based inclusive.
* ``REEXPORTS``: list of public symbol names to re-export from
  ``__init__.py`` AND from the back-compat shim.
* ``SHIM_MODULE``: the original-file replacement; usually one line
  ``from <package> import *`` plus an ``__all__`` mirror.
* ``CLI_DISPATCH``: tuple ``(submodule_name, main_callable_name)`` so the
  shim can keep ``python scripts/<monolith>.py`` working.

Run the bytecode regression check from
``tools/monolith_split_regression.py`` before and after to verify the
move is byte-identical at the function level.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_manifest(path: Path) -> ModuleType:
    """Import a manifest file by absolute path."""
    spec = importlib.util.spec_from_file_location("split_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load manifest {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slice(lines: list[str], start: int, end: int) -> list[str]:
    """Return lines[start-1 : end] preserving trailing newlines."""
    if start < 1 or end > len(lines) or start > end:
        raise ValueError(f"Bad slice {start}-{end} for file of {len(lines)} lines")
    return lines[start - 1 : end]


def split(manifest_path: Path, dry_run: bool = False) -> int:
    """Execute the split described by *manifest_path*.

    Returns the number of submodules written (0 in dry-run mode if
    nothing would change).
    """
    repo_root = Path(__file__).resolve().parents[1]
    m = _load_manifest(manifest_path)

    source = repo_root / m.SOURCE
    package_dir = repo_root / m.PACKAGE_DIR
    src_lines = source.read_text().splitlines(keepends=True)

    # --- preamble (imports + module state to copy verbatim into every submodule)
    header = "".join(src_lines[: m.HEADER_LINES])

    # --- emit each submodule
    written: list[Path] = []
    for filename, ranges, extra_imports in m.SUBMODULES:
        target = package_dir / filename
        body_chunks: list[str] = []
        for start, end in ranges:
            body_chunks.extend(_slice(src_lines, start, end))
        body = "".join(body_chunks)
        contents = header
        if extra_imports:
            contents += "\n" + extra_imports.rstrip() + "\n"
        contents += "\n\n" + body
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
        written.append(target)
        print(f"  wrote {target.relative_to(repo_root)} "
              f"({sum(e - s + 1 for s, e in ranges)} lines)")

    # --- emit __init__.py with re-exports
    init = package_dir / "__init__.py"
    init_lines = [
        '"""Auto-generated package re-exports.\n\n',
        f'Produced by ``tools/split_monolith.py`` from ``{m.SOURCE}``.\n',
        'Edit the underlying submodules, not this file.\n',
        '"""\n',
        'from __future__ import annotations\n\n',
    ]
    # Group re-exports by submodule for readability; each symbol exists in
    # exactly one submodule per the manifest.
    for sub_filename, _ranges, _ in m.SUBMODULES:
        sub_mod = sub_filename.replace(".py", "")
        names_in_sub = m.SYMBOL_OWNERS.get(sub_mod, [])
        if names_in_sub:
            init_lines.append(
                f"from .{sub_mod} import (\n"
                + "".join(f"    {n},\n" for n in sorted(names_in_sub))
                + ")\n"
            )
    init_lines.append("\n__all__ = [\n")
    for name in sorted(m.REEXPORTS):
        init_lines.append(f"    {name!r},\n")
    init_lines.append("]\n")
    if not dry_run:
        init.write_text("".join(init_lines))
    print(f"  wrote {init.relative_to(repo_root)} ({len(m.REEXPORTS)} re-exports)")

    # --- emit back-compat shim that REPLACES the monolith file in place
    pkg = m.PACKAGE_DIR.replace("backend/scripts/", "").rstrip("/")
    submodule_names = [f.replace(".py", "") for f, *_ in m.SUBMODULES]
    use_proxy = bool(getattr(m, "PROXY_SHIM", False))
    shim_lines = [
        '"""Back-compat shim.\n\n',
        f'The implementation moved to ``{m.PACKAGE_DIR.replace("backend/scripts/", "scripts.").rstrip("/")}`` --\n',
        'this module simply re-exports the public surface so existing\n',
        '``from xxx import yyy`` callers keep working without changes.\n',
        '"""\n',
        'from __future__ import annotations\n\n',
        f'from {pkg} import (  # noqa: F401\n',
    ]
    for name in sorted(m.REEXPORTS):
        shim_lines.append(f"    {name},\n")
    shim_lines.append(")\n")
    if use_proxy:
        # Import each submodule as well so the proxy ``__setattr__``
        # below can forward writes (test monkeypatches that target the
        # shim-level constant) into every submodule that has its own
        # copy of the symbol.  Without this forwarding, monkeypatching
        # ``shim.SOME_CONST`` would only rebind the shim's name and the
        # submodule that owns ``_consumer_function`` would still see the
        # original value via its own module-level globals.
        shim_lines.append("\n")
        for sub in submodule_names:
            shim_lines.append(f"from {pkg} import {sub} as _sub_{sub}  # noqa: F401\n")
        shim_lines.append("\n")
        shim_lines.append("import sys as _sys\n")
        shim_lines.append("import types as _types\n")
        shim_lines.append("\n\n")
        shim_lines.append(
            "_PROXY_SUBMODULES = (\n"
            + "".join(f"    _sub_{sub},\n" for sub in submodule_names)
            + ")\n\n\n"
        )
        shim_lines.append(
            "class _ProxyShim(_types.ModuleType):\n"
            "    \"\"\"Module subclass that forwards attribute writes to submodules.\n\n"
            "    Tests historically monkeypatch module-level constants via\n"
            "    ``monkeypatch.setattr(<shim>, 'SOME_CONST', value)``.  After\n"
            "    the split, those constants live in submodules (each got its\n"
            "    own copy from the verbatim header).  This proxy ensures that\n"
            "    a write on the shim is propagated to *every* submodule that\n"
            "    has the same attribute name, preserving the original runtime\n"
            "    behaviour of the monolith.\n"
            "    \"\"\"\n\n"
            "    def __setattr__(self, name, value):  # noqa: D401\n"
            "        super().__setattr__(name, value)\n"
            "        for _sub in _PROXY_SUBMODULES:\n"
            "            if hasattr(_sub, name):\n"
            "                setattr(_sub, name, value)\n\n\n"
            "_sys.modules[__name__].__class__ = _ProxyShim\n"
        )
    shim_lines.append(
        f"\n__all__ = [\n"
        + "".join(f"    {n!r},\n" for n in sorted(m.REEXPORTS))
        + "]\n"
    )
    if hasattr(m, "CLI_DISPATCH") and m.CLI_DISPATCH:
        sub, fn = m.CLI_DISPATCH
        shim_lines.append(
            f'\n\nif __name__ == "__main__":\n'
            f'    import sys\n'
            f'    try:\n'
            f'        {fn}()\n'
            f'    except KeyboardInterrupt:\n'
            f'        sys.exit(130)\n'
            f'    except Exception as exc:  # noqa: BLE001\n'
            f'        import logging\n'
            f'        logging.getLogger(__name__).error("Fatal: %s", exc, exc_info=True)\n'
            f'        sys.exit(1)\n'
        )
    if not dry_run:
        source.write_text("".join(shim_lines))
    print(f"  rewrote {source.relative_to(repo_root)} as shim "
          f"({len(m.REEXPORTS)} re-exports)")
    return len(written)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Mechanical monolith splitter.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    n = split(args.manifest, dry_run=args.dry_run)
    print(f"Done -- {n} submodules emitted ({'dry-run' if args.dry_run else 'real'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
