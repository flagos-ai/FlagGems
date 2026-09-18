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

"""Post-install smoke checks for the built flag-gems distro package.

Run this inside a clean container *after* installing the .deb/.rpm. It
validates the two things that ``apt-get install`` plus ``dpkg -l`` cannot:
that every Python subpackage actually shipped, and that the dependency
versions the distro resolved are new enough for the code to run.

Both checks are deliberately CPU-only and stdlib-only: importing
``flag_gems`` needs a real accelerator (``flag_gems.runtime`` probes for a
device before the rest of the package is imported), so a genuine import
smoke can only run on a GPU runner. These checks catch packaging defects
on any runner, in seconds.

Usage:
    python3 packaging/verify-installed-package.py [--source-dir src/flag_gems]
"""

import argparse
import importlib.metadata
import importlib.util
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_installed_root() -> Path:
    """Locate the installed ``flag_gems`` package without importing it."""
    spec = importlib.util.find_spec("flag_gems")
    if spec is None or not spec.origin:
        raise SystemExit(
            "FAIL: flag_gems is not installed (find_spec returned nothing)"
        )
    return Path(spec.origin).parent


def python_dirs(root: Path) -> set:
    """Directories under ``root`` that hold at least one module, relative to it."""
    return {
        py.parent.relative_to(root)
        for py in root.rglob("*.py")
        if "__pycache__" not in py.parts
    }


def check_subpackages_shipped(source_dir: Path, installed_root: Path) -> List[str]:
    """Every source directory containing modules must exist in the install.

    Catches directories silently dropped by ``find_packages()`` because they
    lack an ``__init__.py`` -- the tree still imports them, so the installed
    package raises ModuleNotFoundError at import time.
    """
    missing = sorted(python_dirs(source_dir) - python_dirs(installed_root))
    return [f"subpackage not shipped: flag_gems/{d}" for d in missing]


def parse_pinned_dependencies(pyproject: Path) -> Dict[str, Tuple[str, str]]:
    """Extract ``name -> (operator, version)`` from pyproject ``dependencies``.

    Intentionally a small regex rather than a TOML parse: Ubuntu 22.04 ships
    Python 3.10, which has no ``tomllib``, and this script must stay
    dependency-free so it can run in a bare install container.
    """
    text = pyproject.read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
    if not block:
        return {}
    pinned = {}
    for raw in re.findall(r'"([^"]+)"', block.group(1)):
        spec = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=)\s*([0-9][^,\s]*)", raw)
        if spec:
            pinned[spec.group(1).lower().replace("-", "_")] = (
                spec.group(2),
                spec.group(3),
            )
    return pinned


def version_tuple(v: str) -> Tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


def installed_version(name: str) -> Optional[str]:
    for candidate in (name, name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def check_dependency_versions(pyproject: Path) -> Tuple[List[str], List[str]]:
    """Distro-resolved dependencies must be new enough for the pinned API.

    A bare ``Depends: python3-foo`` is satisfied by whatever the distro
    happens to ship, which can be years older than what the code calls.

    Reported as warnings unless ``--strict-deps`` is passed: distro packaging
    deliberately relaxes upstream ``==`` pins, so most gaps here are benign
    and only some break at runtime. Treat this output as a review list.
    """
    errors: List[str] = []
    notes: List[str] = []
    for name, (op, want) in parse_pinned_dependencies(pyproject).items():
        have = installed_version(name)
        if have is None:
            notes.append(f"{name}: no installed metadata found, skipped")
            continue
        if version_tuple(have) < version_tuple(want):
            errors.append(
                f"{name}: installed {have} is older than the pinned {op}{want}"
            )
        else:
            notes.append(f"{name}: {have} satisfies {op}{want}")
    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("src/flag_gems"),
        help="flag_gems source tree to compare the install against",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="pyproject.toml holding the upstream dependency pins",
    )
    parser.add_argument(
        "--strict-deps",
        action="store_true",
        help="fail (not just warn) when a resolved dependency is older than its pin",
    )
    args = parser.parse_args()

    installed_root = find_installed_root()
    print(f"installed flag_gems: {installed_root}")

    errors: List[str] = []

    if args.source_dir.is_dir():
        errors += check_subpackages_shipped(args.source_dir, installed_root)
    else:
        print(f"WARN: {args.source_dir} not found, skipping completeness check")

    warnings: List[str] = []

    if args.pyproject.is_file():
        dep_errors, notes = check_dependency_versions(args.pyproject)
        (errors if args.strict_deps else warnings).extend(dep_errors)
        for note in notes:
            print(f"  dep {note}")
    else:
        print(f"WARN: {args.pyproject} not found, skipping dependency check")

    if warnings:
        print("\nWARN: dependency versions behind their upstream pins:")
        for w in warnings:
            print(f"  - {w}")
        print("  (informational; re-run with --strict-deps to treat as failures)")

    if errors:
        print("\nFAIL: package smoke checks found problems:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK: package smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
