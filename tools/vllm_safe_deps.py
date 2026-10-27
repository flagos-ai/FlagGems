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

"""Extract vllm's pip-installable dependencies, excluding packages that
conflict with the FlagOS runtime (torch, triton, nvidia-*, flashinfer, …).

Usage
-----
    # Print safe dependency specs, one per line:
    python tools/vllm_safe_deps.py vllm==0.21.0

    # Use with pip/uv:
    python tools/vllm_safe_deps.py vllm==0.21.0 | xargs uv pip install

    # Custom blacklist (extend the built-in one):
    python tools/vllm_safe_deps.py vllm==0.21.0 --exclude numba --exclude xgrammar

The script fetches metadata from PyPI (no download/install), filters out
hardware-specific packages, and prints one safe dependency per line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

# ── Packages provided by the FlagOS runtime layer ──────────────
# These must NOT be installed/upgraded when adding vllm on top.
BUILTIN_BLACKLIST: set[str] = {
    # torch ecosystem — provided by vendor base image + FlagOS runtime
    "torch",
    "torchaudio",
    "torchvision",
    # triton / flagtree — compiler layer
    "triton",
    "triton-ascend",
    "triton-gcu",
    # NVIDIA-specific packages
    "flashinfer-python",
    "flashinfer-cubin",
    "nvidia-cudnn-frontend",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "humming-kernels",
    "tokenspeed-mla",
    "tokenspeed-triton",
    # may pull nvidia-cuda-* runtime packages
    "numba",
    # tvm / tilelang — tightly coupled to hardware
    "apache-tvm-ffi",
    "tilelang",
    # quack-kernels — CUDA-specific
    "quack-kernels",
}


def _normalize(name: str) -> str:
    """PEP 503 normalize: lowercase, replace [-_.] with -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_dep_name(spec: str) -> str:
    """Extract the package name from a requires_dist spec string.

    Examples:
        "torch==2.11.0"          → "torch"
        "fastapi[standard]>=0.1" → "fastapi"
        "six>=1.16; ..."         → "six"
    """
    # strip environment markers
    spec = spec.split(";")[0].strip()
    # strip extras and version
    match = re.match(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)", spec)
    return match.group(1) if match else spec


def _strip_version(spec: str) -> str:
    """Return the spec without environment markers."""
    return spec.split(";")[0].strip()


def fetch_requires_dist(package_spec: str) -> list[str]:
    """Fetch requires_dist from PyPI JSON API."""
    # "vllm==0.21.0" → name="vllm", version="0.21.0"
    if "==" in package_spec:
        name, version = package_spec.split("==", 1)
    else:
        name = package_spec
        version = None

    if version:
        url = f"https://pypi.org/pypi/{name}/{version}/json"
    else:
        url = f"https://pypi.org/pypi/{name}/json"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"Error fetching metadata for {package_spec}: {exc}", file=sys.stderr)
        sys.exit(1)

    return data.get("info", {}).get("requires_dist") or []


def filter_safe_deps(
    requires_dist: list[str],
    blacklist: set[str],
) -> list[str]:
    """Filter out blacklisted and extras-only dependencies."""
    safe = []
    blacklist_normalized = {_normalize(p) for p in blacklist}

    for spec in requires_dist:
        # Skip extras-only deps (e.g. 'zentorch ; extra == "zen"')
        if "extra ==" in spec or "extra ==" in spec:
            continue

        dep_name = _parse_dep_name(spec)
        if _normalize(dep_name) in blacklist_normalized:
            continue

        safe.append(_strip_version(spec))

    return safe


def main():
    parser = argparse.ArgumentParser(
        description="Extract safe vllm deps for FlagOS environments",
    )
    parser.add_argument(
        "package",
        help='Package spec, e.g. "vllm==0.21.0"',
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional package names to exclude (can repeat)",
    )
    parser.add_argument(
        "--show-blocked",
        action="store_true",
        help="Also print blocked packages (prefixed with #)",
    )
    args = parser.parse_args()

    blacklist = BUILTIN_BLACKLIST | {_normalize(p) for p in args.exclude}
    requires_dist = fetch_requires_dist(args.package)

    if args.show_blocked:
        blacklist_normalized = {_normalize(p) for p in blacklist}
        for spec in requires_dist:
            if "extra ==" in spec or "extra ==" in spec:
                continue
            dep_name = _parse_dep_name(spec)
            stripped = _strip_version(spec)
            if _normalize(dep_name) in blacklist_normalized:
                print(f"# BLOCKED: {stripped}")
            else:
                print(stripped)
    else:
        safe = filter_safe_deps(requires_dist, blacklist)
        for dep in safe:
            print(dep)


if __name__ == "__main__":
    main()
