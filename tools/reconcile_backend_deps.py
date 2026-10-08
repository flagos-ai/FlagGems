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

"""Reconcile a backend's pinned dependencies against what's already installed.

Used by CI jobs that run inside a prebuilt runtime image (which ships flagtree,
torch, etc. preinstalled). Rather than blindly trusting the image or blindly
reinstalling everything, this compares each pin in backends.yaml against the
version actually importable in the current environment:

  - version matches  -> skip (leave the image's package in place)
  - version differs / missing -> reinstall at the backends.yaml version

backends.yaml is the single source of truth (same file setup.sh reads), so this
keeps the runtime image and the repo's declared versions in agreement without
reinstalling packages that already match.

The compiler package (flagtree, or triton when flagtree is absent) is included
in the reconciliation; flagtree and triton are mutually exclusive (flagtree
installs into the triton namespace), so only the one declared for the backend is
considered. `triton` is deliberately NOT reconciled when flagtree is present.

Usage:
    python3 tools/reconcile_backend_deps.py <backend> [--dry-run]

Prints the packages that need (re)installing, one requirement per line, to
stdout. With --dry-run it only reports; otherwise it also runs the install.
Diagnostics go to stderr so stdout stays machine-consumable.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BACKENDS_YAML = ROOT / "src" / "flag_gems" / "backends.yaml"

# Requirement spec like "torch==2.9.0+cu129" or "flagtree==0.7.0+xpu3.6".
# We only reconcile exact-pinned (==) requirements; anything else is passed
# through to the installer untouched (we cannot meaningfully compare a range
# against an installed version here).
PIN_RE = re.compile(r"^([A-Za-z0-9._-]+)==(.+)$")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_backend(backend: str) -> dict:
    with open(BACKENDS_YAML) as f:
        data = yaml.safe_load(f)
    backends = data.get("backends", {})
    if backend not in backends:
        log(f"::error::unknown backend '{backend}' in {BACKENDS_YAML}")
        log("available: " + ", ".join(sorted(backends)))
        sys.exit(2)
    return backends[backend]


def collect_requirements(cfg: dict) -> list[str]:
    """Compiler package + deps. flagtree/triton are mutually exclusive."""
    reqs = list(cfg.get("deps", []))
    flagtree = cfg.get("flagtree")
    triton = cfg.get("triton")
    # Match setup.sh / sync_pyproject_extras: prefer flagtree, else triton.
    if flagtree:
        reqs.append(flagtree)
    elif triton:
        reqs.append(triton)
    return reqs


def normalize(version: str) -> str:
    """Normalize a version for comparison (strip whitespace)."""
    return version.strip()


def base_version(version: str) -> str:
    """Return the public version, dropping any PEP 440 local segment ('+...').

    e.g. '1.0.0.1+20260605' -> '1.0.0.1', '0.7.0+xpu3.6' -> '0.7.0'.
    """
    return normalize(version).split("+", 1)[0]


def versions_match(want: str, have: str) -> bool:
    """Whether an installed version satisfies a pinned '==' version.

    Exact string equality counts. In addition, a pin without a local segment
    is treated as satisfied by an installed build that adds one — i.e. the pin
    'xmlir==1.0.0.1' is satisfied by an installed '1.0.0.1+20260605', since the
    image's build is the same public release plus a local build tag. This
    avoids a spurious reinstall of a version (…+local) that isn't even
    published to the index. A pin that DOES carry a local segment must match
    exactly, so we never mistake one local build for another.
    """
    want, have = normalize(want), normalize(have)
    if want == have:
        return True
    if "+" not in want and base_version(have) == want:
        return True
    return False


def installed_version(dist_name: str) -> str | None:
    """Return the installed version of a distribution, or None if absent.

    Tries the requirement name as-is and with '-'/'_' swapped, since PyPI
    normalizes separators (e.g. 'flash_attn' vs 'flash-attn').
    """
    candidates = {dist_name, dist_name.replace("_", "-"), dist_name.replace("-", "_")}
    for name in candidates:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def reconcile(reqs: list[str]) -> tuple[list[str], list[str]]:
    """Split requirements into (to_install, up_to_date)."""
    to_install: list[str] = []
    up_to_date: list[str] = []
    for req in reqs:
        m = PIN_RE.match(req.strip())
        if not m:
            # Non-pinned spec: cannot compare, hand to installer to resolve.
            log(f"  {req}: not an exact pin, will (re)install")
            to_install.append(req)
            continue
        name, want = m.group(1), normalize(m.group(2))
        have = installed_version(name)
        if have is None:
            log(f"  {name}: not installed -> install {want}")
            to_install.append(req)
        elif versions_match(want, have):
            log(f"  {name}: {have} matches -> skip")
            up_to_date.append(req)
        else:
            log(f"  {name}: installed {have} != pinned {want} -> reinstall")
            to_install.append(req)
    return to_install, up_to_date


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile backend deps against the installed environment"
    )
    parser.add_argument("backend", help="backend key in backends.yaml, e.g. kunlunxin")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only report packages needing (re)install; do not install",
    )
    parser.add_argument(
        "--index",
        default="",
        help="extra index URL passed to pip when installing mismatches",
    )
    args = parser.parse_args()

    cfg = load_backend(args.backend)
    reqs = collect_requirements(cfg)
    log(f"Reconciling {len(reqs)} pinned dep(s) for backend '{args.backend}':")

    to_install, up_to_date = reconcile(reqs)

    log(
        f"\nSummary: {len(up_to_date)} up-to-date, "
        f"{len(to_install)} to (re)install."
    )
    # Machine-readable: the specs that need installing, one per line.
    for req in to_install:
        print(req)

    if args.dry_run or not to_install:
        return

    # Reinstall only the mismatches. --no-deps so we don't perturb the rest of
    # the image's carefully assembled environment (e.g. reinstalling torch's
    # transitive deps); we are reconciling explicit pins, not resolving a tree.
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--reinstall" if _pip_supports_reinstall() else "--force-reinstall",
        *to_install,
    ]
    if args.index:
        cmd += ["--extra-index-url", args.index]
    log("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _pip_supports_reinstall() -> bool:
    """pip uses --force-reinstall; uv's pip shim uses --reinstall.

    Default to pip's flag; this helper exists so the intent is documented and
    easy to adjust if the CI environment standardizes on uv.
    """
    return False


if __name__ == "__main__":
    main()
