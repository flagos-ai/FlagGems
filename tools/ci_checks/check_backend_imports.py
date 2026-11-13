#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that vendor backends still import valid names from shared code.

Vendor backends under ``src/flag_gems/runtime/backend/_<vendor>/`` import
heavily from generic modules (``flag_gems.utils``, ``flag_gems.runtime``,
``flag_gems.ops`` ...). Backend CI jobs only run on the vendor's own hardware
when that vendor's directory changes, so renaming or removing a shared symbol
can break other backends without any CI signal.

This check needs no hardware. It statically resolves every
``from flag_gems[.x.y] import name`` found in backend directories and fails
when the target module does not exist or does not define ``name``.

Resolution is deliberately lenient to avoid false positives: a module that
uses ``from x import *``, calls ``globals()`` or defines a module-level
``__getattr__`` accepts any name, and ``try/except ImportError`` guarded imports are skipped.

Exit codes:
  0 - all imports resolve
  1 - unresolved imports found
  2 - script internal error
"""

import argparse
import ast
import sys
from functools import lru_cache
from pathlib import Path

SRC_DIR = Path("src")
PACKAGE = "flag_gems"
BACKEND_DIR = SRC_DIR / PACKAGE / "runtime" / "backend"
BASELINE_FILE = Path("tools/ci_checks/backend_imports_baseline.txt")
ANY = None  # sentinel: module accepts any name


def _module_path(module: str):
    """Map a dotted module name to its source file, or None if absent."""
    base = SRC_DIR.joinpath(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.is_dir():  # namespace-style package without __init__.py
        return base
    return None


def _target_names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _collect(body, names):
    """Collect names bound at module level, descending into if/try/with."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if node.name == "__getattr__":
                return True
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names.update(_target_names(t))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return True
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names.update(_target_names(node.target))
            if _collect(node.body + node.orelse, names):
                return True
        elif isinstance(node, ast.If):
            if _collect(node.body + node.orelse, names):
                return True
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            if _collect(node.body, names):
                return True
        elif isinstance(node, ast.Try):
            blocks = node.body + node.orelse + node.finalbody
            for h in node.handlers:
                blocks += h.body
            if _collect(blocks, names):
                return True
    return False


@lru_cache(maxsize=None)
def _defined_names(module: str):
    """Return the set of names a module defines, ANY, or raise FileNotFoundError."""
    path = _module_path(module)
    if path is None:
        raise FileNotFoundError(module)
    if path.is_dir():
        return ANY
    source = path.read_text(encoding="utf-8")
    if "globals()" in source:  # names may be generated dynamically
        return ANY
    tree = ast.parse(source, filename=str(path))
    names = set()
    if _collect(tree.body, names):
        return ANY
    return names


_IMPORT_ERRORS = ("ImportError", "ModuleNotFoundError", "Exception")


def _catches_import_error(node: ast.Try) -> bool:
    for h in node.handlers:
        if h.type is None:
            return True
        types = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        if any(isinstance(t, ast.Name) and t.id in _IMPORT_ERRORS for t in types):
            return True
    return False


def _iter_imports(body, guarded=False):
    """Yield imports not protected by ``try/except ImportError``.

    Imports are statements, so only statement lists are walked. Descending into
    expressions (the bulk of any AST) would be wasted work.
    """
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not guarded:
                yield node
            continue
        is_try = isinstance(node, ast.Try)
        for field in ("body", "orelse", "finalbody", "handlers", "cases"):
            children = getattr(node, field, None)
            if not isinstance(children, list):  # e.g. Lambda.body is an expression
                continue
            inner = guarded or (
                field == "body" and is_try and _catches_import_error(node)
            )
            yield from _iter_imports(children, inner)


def _check_file(path: Path):
    """Return a list of (lineno, message) problems in *path*."""
    source = path.read_text(encoding="utf-8")
    if PACKAGE not in source:  # cheap pre-filter before the costly parse
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [(exc.lineno or 0, f"syntax error: {exc.msg}")]

    problems = []
    for node in _iter_imports(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (
                    alias.name == PACKAGE or alias.name.startswith(PACKAGE + ".")
                ) and _module_path(alias.name) is None:
                    problems.append((node.lineno, f"module '{alias.name}' not found"))
        else:
            if node.level or not node.module:
                continue  # relative imports stay inside the backend
            if node.module != PACKAGE and not node.module.startswith(PACKAGE + "."):
                continue
            try:
                defined = _defined_names(node.module)
            except FileNotFoundError:
                problems.append((node.lineno, f"module '{node.module}' not found"))
                continue
            except SyntaxError as exc:
                problems.append(
                    (node.lineno, f"'{node.module}' has a syntax error: {exc.msg}")
                )
                continue
            if defined is ANY:
                continue
            for alias in node.names:
                if alias.name == "*" or alias.name in defined:
                    continue
                if _module_path(f"{node.module}.{alias.name}") is not None:
                    continue  # importing a submodule
                problems.append(
                    (node.lineno, f"'{alias.name}' is not defined in '{node.module}'")
                )
    return problems


def _load_baseline(path: Path):
    """Known, pre-existing problems as ``<file>::<message>`` lines."""
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Limit to a backend directory name (e.g. _kunlunxin). Repeatable.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_FILE,
        help="File of known problems to tolerate (one '<file>::<message>' per line).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline with the current problems and exit 0.",
    )
    args = parser.parse_args(argv)

    if not BACKEND_DIR.is_dir():
        print(
            f"::error::Cannot find {BACKEND_DIR}; run from the repo root",
            file=sys.stderr,
        )
        return 2

    roots = [
        d
        for d in sorted(BACKEND_DIR.iterdir())
        if d.is_dir() and d.name.startswith("_") and d.name != "__pycache__"
    ]
    if args.backend:
        roots = [d for d in roots if d.name in args.backend]

    found = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            for lineno, msg in _check_file(path):
                found.setdefault(f"{path.as_posix()}::{msg}", lineno)

    if args.update_baseline:
        header = (
            "# Known unresolved flag_gems imports in vendor backends.\n"
            "# Format: <file>::<message>. Fix the entry and delete the line;\n"
            "# do not add new ones.\n"
        )
        args.baseline.write_text(
            header + "\n".join(sorted(found)) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(found)} entries to {args.baseline}")
        return 0

    baseline = _load_baseline(args.baseline)
    new = sorted(k for k in found if k not in baseline)
    for key in new:
        file, msg = key.split("::", 1)
        print(f"::error file={file},line={found[key]}::{msg}")
    if not args.backend:  # stale detection is only valid on a full scan
        for key in sorted(baseline - set(found)):
            print(f"::warning::baseline entry no longer needed, remove it: {key}")
    if new:
        print(
            f"\n{len(new)} backend import(s) no longer resolve against shared code. "
            "Update the backend, or keep a compatible alias in the shared module.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: backend imports resolve ({len(roots)} backend dirs checked)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(
            f"::error::check_backend_imports internal error: {exc!r}", file=sys.stderr
        )
        sys.exit(2)
