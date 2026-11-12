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

"""Check that test files do not import dispatchable operators directly from
flag_gems.ops submodules.

Rule: If a name is in ``flag_gems.ops.__all__``, it is registered in the ATen
dispatcher via ``SpecOpRegistrar`` and surfaces on ``flag_gems.<name>``.
Importing it with ``from flag_gems.ops[.<submod>] import <name>`` binds the
*generic* implementation at module load time, before the backend override is
applied, so vendor overrides are silently bypassed.

Tests must reference such names via ``flag_gems.<name>`` at call time.

Exit codes:
  0 - all checks pass
  1 - violations found
  2 - script internal error
"""

import argparse
import ast
import re
import sys
from pathlib import Path

TESTS_DIR = Path("tests")
BENCHMARK_DIR = Path("benchmark")
OPS_INIT = Path("src/flag_gems/ops/__init__.py")


def _load_ops_all(ops_init: Path) -> set:
    """Return the set of names listed in flag_gems.ops.__all__."""
    if not ops_init.exists():
        print(f"::error::Cannot find {ops_init}", file=sys.stderr)
        sys.exit(2)
    with open(ops_init) as f:
        tree = ast.parse(f.read(), filename=str(ops_init))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return {
                            elt.s
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.s, str)
                        }
    return set()


def _scan_file(path: Path, dispatchable: set) -> list:
    """Return list of (line_no, name, source_line) violations in *path*."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"::warning::Cannot read {path}: {exc}", file=sys.stderr)
        return []

    # Quick pre-filter: skip files with no suspicious import at all
    if "flag_gems.ops" not in source:
        return []

    violations = []
    for lineno, raw_line in enumerate(source.splitlines(), 1):
        line = raw_line.strip()
        # Match: from flag_gems.ops[.anything] import name[, name ...]
        # Honour ``# noqa: flag_gems_dispatch`` to allow deliberate wiring tests.
        m = re.match(r"from\s+flag_gems\.ops[\w.]*\s+import\s+(.+)", line)
        if not m:
            continue
        if re.search(r"#\s*noqa:.*flag_gems_dispatch", raw_line, re.IGNORECASE):
            continue
        imports_str = m.group(1)
        # Strip trailing comments
        imports_str = imports_str.split("#")[0]
        # Flatten multi-line parens: treat as single line (we match per line)
        imports_str = imports_str.replace("(", "").replace(")", "")
        for token in imports_str.split(","):
            token = token.strip()
            if not token:
                continue
            # Handle "name as alias" — the original name is what matters
            name = token.split(" as ")[0].strip() if " as " in token else token
            if name in dispatchable:
                violations.append((lineno, name, raw_line.rstrip()))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check tests/benchmarks for direct imports of dispatchable ops."
    )
    parser.add_argument(
        "--tests-dir",
        default=str(TESTS_DIR),
        help="Root directory of test files (default: tests/)",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Optional benchmark directory to also scan (not scanned by default).",
    )
    parser.add_argument(
        "--ops-init",
        default=str(OPS_INIT),
        help="Path to flag_gems/ops/__init__.py",
    )
    args = parser.parse_args()

    dispatchable = _load_ops_all(Path(args.ops_init))
    if not dispatchable:
        print(
            "::warning::flag_gems.ops.__all__ is empty; skipping check.",
            file=sys.stderr,
        )
        return 0

    scan_dirs = [Path(args.tests_dir)]
    if args.benchmark_dir is not None:
        scan_dirs.append(Path(args.benchmark_dir))
    all_violations: list[tuple[Path, int, str, str]] = []

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for path in sorted(scan_dir.rglob("*.py")):
            for lineno, name, src_line in _scan_file(path, dispatchable):
                all_violations.append((path, lineno, name, src_line))

    if not all_violations:
        print("check_direct_ops_imports: OK")
        return 0

    print(
        "FAIL: The following test/benchmark files import dispatchable ops directly\n"
        "      from flag_gems.ops submodules. This bypasses backend vendor overrides.\n"
        "      Use ``flag_gems.<name>`` at call time instead.\n",
        file=sys.stderr,
    )
    for path, lineno, name, src_line in all_violations:
        print(
            f"  {path}:{lineno}: '{name}' is dispatchable — import via flag_gems.{name}",
            file=sys.stderr,
        )
        print(f"    {src_line}", file=sys.stderr)
    print(
        f"\n{len(all_violations)} violation(s) in "
        f"{len(set(p for p, *_ in all_violations))} file(s).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
