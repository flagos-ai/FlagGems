#!/usr/bin/env python3
"""flaggems-setup — Install vendor-specific dependencies for FlagGems.

Usage:
    flaggems-setup <backend>          # e.g. nvidia, ascend-cann900
    flaggems-setup --list             # show available backends
    flaggems-setup <backend> --dry-run # show what would be installed
"""
import argparse
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import yaml


def load_config():
    """Load backends.yaml from the installed package."""
    try:
        data = files("flag_gems").joinpath("backends.yaml").read_text()
    except Exception:
        # Fallback: try loading from source tree
        p = Path(__file__).parent / "backends.yaml"
        if not p.exists():
            print("Error: backends.yaml not found", file=sys.stderr)
            sys.exit(1)
        data = p.read_text()
    return yaml.safe_load(data)


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


def run(cmd, dry_run=False):
    """Run a command, or print it if dry_run."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [dry-run] {cmd_str}")
        return
    print(f"  $ {cmd_str}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"Error: command failed with exit code {result.returncode}", file=sys.stderr
        )
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        prog="flaggems-setup",
        description="Install vendor-specific dependencies for FlagGems.",
    )
    parser.add_argument(
        "backend",
        nargs="?",
        help="Backend to install (e.g. nvidia, ascend-cann900)",
    )
    parser.add_argument("--list", action="store_true", help="List available backends")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
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
            n_deps = len(info.get("deps", []))
            post = " + post_install" if info.get("post_install") else ""
            print(
                f"  {key:<20s}  python={info['python']}  vendor={vendor}  ({n_deps} deps{post})"
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
    deps = backend.get("deps", [])
    post_install = backend.get("post_install", [])
    pip = args.pip.split() if args.pip else detect_pip()

    print(f"Backend:  {backend_key}")
    print(f"Vendor:   {vendor}")
    print(f"Python:   {backend['python']}")
    print(f"Index:    {index}")
    print(f"Mirror:   {mirror}")
    print(f"Deps:     {len(deps)} packages")
    print()

    # Step 1: Install vendor deps (no transitive deps)
    print("[Step 1] Installing vendor packages (--no-deps) ...")
    run(
        [*pip, "install", "--no-deps", "--index-url", index, *deps],
        dry_run=args.dry_run,
    )
    print()

    # Step 2: Install transitive deps from mirror
    print("[Step 2] Installing transitive dependencies ...")
    run([*pip, "install", "--index-url", mirror, *deps], dry_run=args.dry_run)
    print()

    # Step 3: Post-install overrides
    if post_install:
        installs = [p for p in post_install if not isinstance(p, dict)]
        uninstalls = [
            p["uninstall"]
            for p in post_install
            if isinstance(p, dict) and "uninstall" in p
        ]
        if installs:
            print("[Step 3] Post-install overrides ...")
            for pkg in installs:
                run([*pip, "install", "--index-url", index, pkg], dry_run=args.dry_run)
            print()
        if uninstalls:
            print("[Step 3] Post-install uninstalls ...")
            for pkg in uninstalls:
                run([*pip, "uninstall", "-y", pkg], dry_run=args.dry_run)
            print()

    print(f"FlagGems vendor dependencies installed for {backend_key}")


if __name__ == "__main__":
    main()
