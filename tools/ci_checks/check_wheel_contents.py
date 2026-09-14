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

"""Check that a built wheel contains every fused Python source module.

Directories without an ``__init__.py`` work as namespace packages from an
editable checkout, but ``setuptools.find_packages`` omits them from a regular
wheel.  This used to omit ``flag_gems.fused.DSA`` and make ``import flag_gems``
fail after a non-editable install (issue #6145).

Exit codes:
  0 - every fused Python source module is present in the wheel
  1 - one or more fused Python source modules are missing
  2 - invalid input or script error
"""

import argparse
import sys
import zipfile
from pathlib import Path


def expected_fused_members(source_root: Path) -> set[str]:
    """Return wheel member names for Python files in ``flag_gems.fused``."""
    fused_root = source_root / "flag_gems" / "fused"
    if not fused_root.is_dir():
        raise FileNotFoundError(f"fused package directory not found: {fused_root}")

    return {
        source_file.relative_to(source_root).as_posix()
        for source_file in fused_root.rglob("*.py")
    }


def missing_fused_members(wheel: Path, source_root: Path) -> list[str]:
    """Return fused source modules that are absent from ``wheel``."""
    expected = expected_fused_members(source_root)
    with zipfile.ZipFile(wheel) as archive:
        packaged = set(archive.namelist())
    return sorted(expected - packaged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="wheel file to inspect")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("src"),
        help="project source root (default: src)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.wheel.is_file():
        print(f"::error::Wheel file not found: {args.wheel}")
        return 2

    try:
        missing = missing_fused_members(args.wheel, args.source_root)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"::error::Unable to check wheel contents: {error}")
        return 2

    if missing:
        for member in missing:
            print(
                f"::error file={args.source_root / member}::"
                f"Python source module is missing from {args.wheel.name}"
            )
        return 1

    print(
        f"All {len(expected_fused_members(args.source_root))} fused Python source "
        f"modules are present in {args.wheel.name}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
