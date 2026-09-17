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

"""flaggems-setup — Install vendor-specific dependencies for FlagGems.

Usage:
    flaggems-setup <backend>           # e.g. nvidia-cuda128, ascend-cann900
    flaggems-setup --list              # show available backends
    flaggems-setup <backend> --dry-run # show what would be installed

Compiler selection (FlagTree / Triton):
    By default a compiler is installed automatically — FlagTree when the
    backend provides one, otherwise Triton. Override with the COMPILER
    environment variable or the --compiler flag:
        flaggems-setup <backend> --compiler triton
        COMPILER=triton flaggems-setup <backend>

This is intentionally a *top-level* module rather than part of the
``flag_gems`` package: importing ``flag_gems`` pulls in torch and triton at
load time, but this tool runs *before* those are installed, so its entry
point must not import the package.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def locate_backends_yaml():
    """Find backends.yaml without importing the flag_gems package.

    ``flag_gems/__init__.py`` imports torch and triton at module load; this CLI
    runs *before* those exist, so it must not import the package.
    ``importlib.util.find_spec`` locates the package directory without executing
    its ``__init__``.
    """
    # Adjacent to this module (e.g. a script copied next to the config).
    local = Path(__file__).with_name("backends.yaml")
    if local.exists():
        return local
    try:
        spec = importlib.util.find_spec("flag_gems")
    except (ImportError, ValueError):
        spec = None
    for loc in (spec.submodule_search_locations or []) if spec else []:
        cand = Path(loc) / "backends.yaml"
        if cand.exists():
            return cand
    return None


def load_config():
    """Load backends.yaml from the installed package or the source tree."""
    path = locate_backends_yaml()
    if path is None:
        print("Error: backends.yaml not found", file=sys.stderr)
        sys.exit(1)
    return yaml.safe_load(path.read_text())


def as_list(value):
    """Normalize a YAML field that may be absent, a string, or a list."""
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def derive_vendor(backend_key):
    """Derive vendor name from backend key.

    'ascend-cann900' → 'ascend'
    'nvidia'         → 'nvidia'
    """
    return backend_key.rsplit("-", 1)[0] if "-" in backend_key else backend_key


def get_index_url(vendor, cfg):
    """Generate the PyPI index URL for a vendor."""
    return cfg["pypi_base"].format(vendor=vendor)


def detect_pip():
    """Detect available pip command: prefer 'uv pip', fall back to 'pip'."""
    if shutil.which("uv"):
        return ["uv", "pip"]
    if shutil.which("pip"):
        return ["pip"]
    print("Error: neither 'uv' nor 'pip' found in PATH", file=sys.stderr)
    sys.exit(1)


def run(cmd, dry_run=False, check=True):
    """Run a command, or print it if dry_run."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [dry-run] {cmd_str}")
        return
    print(f"  $ {cmd_str}")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(
            f"Error: command failed with exit code {result.returncode}", file=sys.stderr
        )
        sys.exit(result.returncode)


def install_pkgs(pip, pkgs, index, mirror, dry_run=False):
    """Install packages, then their transitive dependencies.

    Step 1 pulls the exact (often vendor-local) builds from the vendor index
    with --no-deps; step 2 resolves any transitive dependencies from the
    mirror (the already-installed builds are left untouched).
    """
    if not pkgs:
        return
    run([*pip, "install", "--no-deps", "--index-url", index, *pkgs], dry_run=dry_run)
    run([*pip, "install", "--index-url", mirror, *pkgs], dry_run=dry_run)


def uninstall(pip, pkg, dry_run=False):
    """Uninstall a package if present. Not being installed is not an error.

    'uv pip uninstall' rejects '-y' (it never prompts), whereas plain 'pip'
    requires it, so the flag is tool-specific.
    """
    if pip[0] == "uv":
        cmd = [*pip, "uninstall", pkg]
    else:
        cmd = [*pip, "uninstall", "-y", pkg]
    run(cmd, dry_run=dry_run, check=False)


def resolve_compiler(backend, backend_key, override):
    """Decide which compiler to install: 'flagtree' or 'triton'.

    Mirrors setup.sh: an explicit choice (--compiler / COMPILER) wins;
    otherwise auto-select FlagTree when available, else fall back to Triton.
    """
    flagtree = as_list(backend.get("flagtree"))
    choice = (override or os.environ.get("COMPILER", "")).strip().lower()
    if not choice:
        if flagtree:
            choice = "flagtree"
        else:
            print(
                f"WARNING: FlagTree is not available for {backend_key}, "
                "falling back to Triton."
            )
            choice = "triton"
    if choice not in ("flagtree", "triton"):
        print(
            f"Error: unknown compiler '{choice}' (expected 'flagtree' or 'triton')",
            file=sys.stderr,
        )
        sys.exit(1)
    return choice


def install_compiler(pip, backend, backend_key, choice, index, mirror, dry_run=False):
    """Install the selected compiler, removing the other to avoid conflicts."""
    flagtree = as_list(backend.get("flagtree"))
    triton = as_list(backend.get("triton"))
    triton_post = as_list(backend.get("triton_post_install"))

    if choice == "flagtree":
        if not flagtree:
            print(
                f"Error: compiler=flagtree but FlagTree is not available for "
                f"'{backend_key}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("[Step 3] Installing FlagTree compiler ...")
        uninstall(pip, "triton", dry_run=dry_run)
        install_pkgs(pip, flagtree, index, mirror, dry_run=dry_run)
    else:  # triton
        if not triton:
            print(
                f"Error: compiler=triton but no Triton packages configured for "
                f"'{backend_key}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("[Step 3] Installing Triton compiler ...")
        uninstall(pip, "flagtree", dry_run=dry_run)
        install_pkgs(pip, triton, index, mirror, dry_run=dry_run)
        for pkg in triton_post:
            print(f"[Step 3] Triton post-install: {pkg} ...")
            install_pkgs(pip, [pkg], index, mirror, dry_run=dry_run)
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="flaggems-setup",
        description="Install vendor-specific dependencies for FlagGems.",
    )
    parser.add_argument(
        "backend",
        nargs="?",
        help="Backend to install (e.g. nvidia-cuda128, ascend-cann900)",
    )
    parser.add_argument("--list", action="store_true", help="List available backends")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )
    parser.add_argument(
        "--compiler",
        choices=["flagtree", "triton"],
        default=None,
        help="Compiler to install (default: COMPILER env var, or auto — "
        "FlagTree if available, otherwise Triton)",
    )
    parser.add_argument(
        "--pip", default=None, help="pip command to use (default: auto-detect)"
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.list:
        print("Available backends:\n")
        for key, info in cfg["backends"].items():
            vendor = derive_vendor(key)
            n_deps = len(as_list(info.get("deps")))
            compiler = "flagtree" if info.get("flagtree") else "triton"
            print(
                f"  {key:<20s}  python={info['python']}  vendor={vendor}  "
                f"({n_deps} deps, {compiler})"
            )
        return

    if not args.backend:
        parser.print_help()
        sys.exit(1)

    backend_key = args.backend
    if backend_key not in cfg["backends"]:
        print(f"Error: unknown backend '{backend_key}'", file=sys.stderr)
        print("Run 'flaggems-setup --list' to see available backends.", file=sys.stderr)
        sys.exit(1)

    backend = cfg["backends"][backend_key]
    vendor = derive_vendor(backend_key)
    index = backend.get("index") or get_index_url(vendor, cfg)
    mirror = cfg["mirror"]
    deps = as_list(backend.get("deps"))
    post_install = backend.get("post_install", [])
    pip = args.pip.split() if args.pip else detect_pip()
    compiler = resolve_compiler(backend, backend_key, args.compiler)

    print(f"Backend:  {backend_key}")
    print(f"Vendor:   {vendor}")
    print(f"Python:   {backend['python']}")
    print(f"Index:    {index}")
    print(f"Mirror:   {mirror}")
    print(f"Deps:     {len(deps)} packages")
    print(f"Compiler: {compiler}")
    print()

    # Step 1: Install vendor packages (torch + vendor runtime)
    print("[Step 1] Installing vendor packages ...")
    install_pkgs(pip, deps, index, mirror, dry_run=args.dry_run)
    print()

    # Step 2: Post-install overrides
    if post_install:
        installs = [p for p in post_install if not isinstance(p, dict)]
        uninstalls = [
            p["uninstall"]
            for p in post_install
            if isinstance(p, dict) and "uninstall" in p
        ]
        if installs:
            print("[Step 2] Post-install overrides ...")
            for pkg in installs:
                run([*pip, "install", "--index-url", index, pkg], dry_run=args.dry_run)
            print()
        if uninstalls:
            print("[Step 2] Post-install uninstalls ...")
            for pkg in uninstalls:
                uninstall(pip, pkg, dry_run=args.dry_run)
            print()

    # Step 3: Install the compiler (FlagTree or Triton)
    install_compiler(
        pip, backend, backend_key, compiler, index, mirror, dry_run=args.dry_run
    )

    print(
        f"FlagGems vendor dependencies installed for {backend_key} "
        f"(compiler: {compiler})"
    )


if __name__ == "__main__":
    main()
