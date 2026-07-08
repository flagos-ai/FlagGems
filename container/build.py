#!/usr/bin/env python3
"""Build FlagGems container images.

Reads container/configs.yaml and src/flag_gems/backends.yaml to resolve
build arguments, then invokes `docker build` with the appropriate
--build-arg values.

Usage:
    python container/build.py <backend-key> [options]

Examples:
    python container/build.py nvidia-cuda128 --dry-run
    python container/build.py ascend-cann900 --tag latest --push
    python container/build.py metax --base-image my-registry/metax:custom
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def find_project_root() -> Path:
    """Walk up from this script to find the project root (has pyproject.toml)."""
    d = Path(__file__).resolve().parent.parent
    if (d / "pyproject.toml").exists():
        return d
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd
    sys.exit("Error: cannot locate project root (pyproject.toml not found)")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_backend(backend_arg: str, configs: dict):
    """Resolve user input to (backends_yaml_key, variant).

    configs.yaml maps vendor → [variants]. For multi-variant vendors
    (nvidia, ascend), the backends.yaml key is "{vendor}-{variant}"
    (e.g. "nvidia-cuda128"). For single-variant vendors (cambricon),
    the backends.yaml key is just the vendor name (e.g. "cambricon").

    Returns (backends_yaml_key, variant_suffix) where variant_suffix
    is used for the base image name.
    """
    cfg_backends = configs.get("backends", {})

    # Try matching "{vendor}-{variant}" against the input
    for vendor, variants in cfg_backends.items():
        for variant in variants:
            full = f"{vendor}-{variant}"
            if backend_arg == full:
                # Multi-variant: backends.yaml key is the full name
                return full, variant
            if backend_arg == vendor and len(variants) == 1:
                # Single-variant shorthand: just the vendor name
                return vendor, variant

    # Try matching vendor name with multiple variants
    if backend_arg in cfg_backends:
        variants = cfg_backends[backend_arg]
        if len(variants) > 1:
            names = [f"{backend_arg}-{v}" for v in variants]
            sys.exit(
                f"Error: '{backend_arg}' has multiple variants: " f"{', '.join(names)}"
            )

    sys.exit(f"Error: '{backend_arg}' not found in configs.yaml backends")


def resolve_base_image(variant: str, configs: dict) -> str:
    """Derive base image name: {prefix}-{variant}:{tag}."""
    prefix = configs.get("base_image_prefix", "flagos-base")
    tag = configs.get("base_image_tag", "latest")
    return f"{prefix}-{variant}:{tag}"


def resolve_build_args(
    backend_key: str,
    variant: str,
    configs: dict,
    backends: dict,
    *,
    base_image_override: str | None = None,
    extra_pypi_override: str | None = None,
    include_tests_override: str | None = None,
) -> dict[str, str]:
    """Resolve all docker build-arg values for a given backend."""

    backend_info = backends.get("backends", {}).get(backend_key)
    if backend_info is None:
        sys.exit(f"Error: '{backend_key}' not found in backends.yaml")

    vendor = backend_key.split("-")[0]
    pypi_base = backends.get("pypi_base", "")
    mirror = backends.get("mirror", "https://mirrors.aliyun.com/pypi/simple")

    args = {
        "BASE_IMAGE": base_image_override or resolve_base_image(variant, configs),
        "PYTHON_VERSION": backend_info.get("python", "3.12"),
        "FLAGOS_PYPI": pypi_base.format(vendor=vendor) if pypi_base else "",
        "EXTRA_PYPI": extra_pypi_override or mirror,
        "EXTRAS_GROUP": backend_key,
        "INCLUDE_TESTS": include_tests_override or "true",
    }

    # Optional hooks from backends.yaml
    for key in ("pre_install", "post_install"):
        val = backend_info.get(key, "")
        if isinstance(val, str):
            val = val.strip()
        if val:
            args[key.upper()] = val

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Build FlagGems container images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "backend",
        help="Backend key (e.g. nvidia-cuda128, ascend-cann900, metax)",
    )
    parser.add_argument("--base-image", help="Override base image")
    parser.add_argument("--extra-pypi", help="Override extra PyPI mirror URL")
    parser.add_argument(
        "--include-tests",
        choices=["true", "false"],
        help="Override whether to include tests",
    )
    parser.add_argument("--tag", "-t", help="Image tag")
    parser.add_argument("--push", action="store_true", help="Push after building")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print command without executing"
    )

    args = parser.parse_args()

    project_root = find_project_root()
    configs = load_yaml(project_root / "container" / "configs.yaml")
    backends = load_yaml(project_root / "src" / "flag_gems" / "backends.yaml")
    containerfile = project_root / "container" / "Containerfile"

    if not containerfile.exists():
        sys.exit(f"Error: {containerfile} not found")

    backend_key, variant = resolve_backend(args.backend, configs)

    build_args = resolve_build_args(
        backend_key,
        variant,
        configs,
        backends,
        base_image_override=args.base_image,
        extra_pypi_override=args.extra_pypi,
        include_tests_override=args.include_tests,
    )

    tag = args.tag or f"flaggems-{backend_key}:latest"

    cmd = ["docker", "build"]
    for key, value in build_args.items():
        cmd.extend(["--build-arg", f"{key}={value}"])
    cmd.extend(["-f", str(containerfile), "-t", tag, str(project_root)])

    if args.dry_run:
        print("Would run:")
        print("  " + " \\\n    ".join(cmd))
        print()
        print("Build args:")
        for k, v in build_args.items():
            print(f"  {k}={v}")
        return

    print(f"Building {tag}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"docker build failed with exit code {result.returncode}")

    if args.push:
        print(f"Pushing {tag}...")
        result = subprocess.run(["docker", "push", tag])
        if result.returncode != 0:
            sys.exit(f"docker push failed with exit code {result.returncode}")

    print(f"Done: {tag}")


if __name__ == "__main__":
    main()
