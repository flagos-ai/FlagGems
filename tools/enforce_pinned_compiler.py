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

"""Make backends.yaml authoritative for the compiler package at runtime.

Runs after reconcile_backend_deps.py on backends that execute inside a prebuilt
runtime image. Those images bake the compiler (flagtree, or triton when
flagtree is absent) OUTSIDE the venv — e.g. /opt/flagtree — and ahead of the
venv on sys.path. reconcile reinstalls the backends.yaml pin INTO the venv but
pip cannot uninstall the baked copy ("Not uninstalling flagtree at /opt/..."),
so `import flagtree` still resolves to the stale baked version and the test run
silently uses the wrong compiler build.

This script closes that gap:

  1. Read the compiler pin from backends.yaml (same source as reconcile).
  2. Import the compiler and find the version actually resolved + its on-disk
     location.
  3. If that version doesn't match the pin AND the resolving copy lives outside
     the venv, remove the shadowing copy (the package dir and any dist-info /
     .pth / egg-link that points at it), then re-resolve in a fresh interpreter.
  4. Hard-verify: the importable version must equal the pin's base version, or
     exit non-zero so the job fails instead of testing a stale build.

Only the declared compiler is touched; flagtree and triton share a namespace
and are mutually exclusive, so we never remove the one we're keeping. Nothing
here runs unless a mismatch is detected, so an image already carrying the
pinned build is left untouched.

Usage:
    python3 tools/enforce_pinned_compiler.py <backend>

Env:
    VIRTUAL_ENV  venv prefix; a copy resolving under here is considered the
                 pip-installed one and is never deleted. Falls back to
                 sys.prefix when unset.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the exact config-loading + version-compare logic reconcile uses, so the
# pin read here is byte-for-byte what reconcile acted on.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconcile_backend_deps as rc  # noqa: E402

# Requirement spec like "flagtree==0.7.1a1+ascend3.5"; we only enforce exact
# pins (same contract as reconcile — a range can't be compared to an import).
PIN_RE = rc.PIN_RE


def log(msg: str) -> None:
    print(f"[enforce-compiler] {msg}", file=sys.stderr)


def compiler_pin(cfg: dict) -> tuple[str, str] | None:
    """Return (dist_name, pinned_version) for the backend's compiler, or None.

    flagtree wins over triton (matches setup.sh / reconcile). Only exact '=='
    pins are enforceable.
    """
    spec = cfg.get("flagtree") or cfg.get("triton")
    if not spec:
        return None
    m = PIN_RE.match(spec.strip())
    if not m:
        log(f"compiler spec '{spec}' is not an exact pin; skipping enforcement")
        return None
    return m.group(1), rc.normalize(m.group(2))


# The import name can differ from the distribution name. flagtree installs into
# the `triton` namespace, so for both compilers the module to import is triton;
# its __version__ / distribution metadata still carries the flagtree version.
def _probe_module(mod_name: str) -> tuple[str | None, str | None]:
    """Return (version, file-path) as resolved by a FRESH interpreter.

    A fresh process is used so a just-deleted path can't linger in an already
    imported module cache.
    """
    code = (
        "import importlib,importlib.metadata as m\n"
        f"mod=importlib.import_module({mod_name!r})\n"
        "v=getattr(mod,'__version__',None)\n"
        "if v is None:\n"
        "    try: v=m.version('flagtree')\n"
        "    except Exception:\n"
        "        try: v=m.version('triton')\n"
        "        except Exception: v=None\n"
        "print(v or '')\n"
        "print(getattr(mod,'__file__','') or '')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    if out.returncode != 0:
        log(f"could not import {mod_name}: {out.stderr.strip()}")
        return None, None
    lines = out.stdout.splitlines()
    version = lines[0].strip() if lines else ""
    path = lines[1].strip() if len(lines) > 1 else ""
    return (version or None), (path or None)


def _venv_prefix() -> Path:
    return Path(os.environ.get("VIRTUAL_ENV") or sys.prefix).resolve()


def _under(path: Path, prefix: Path) -> bool:
    try:
        path.resolve().relative_to(prefix)
        return True
    except ValueError:
        return False


def _remove_shadow(mod_file: Path, dist_name: str) -> bool:
    """Delete the package dir resolving the import + its metadata/path hooks.

    Returns True if anything was removed. Scoped to the shadowing location only;
    the venv copy is never the one passed here (caller checks _under first).

    Removing just the resolved package directory is sufficient to flip
    resolution: a fresh interpreter re-imports with the shadow dir gone, and
    even if a .pth still prepends the shadow's parent to sys.path, that entry no
    longer contains the package, so the import falls through to the venv copy.
    Python silently ignores a non-existent/empty sys.path entry, so we do NOT
    touch .pth files (they may inject unrelated paths too) — we only delete the
    package dir and its sibling dist-info/egg-info metadata.
    """
    removed = False
    # mod_file is .../<site_dir>/triton/__init__.py -> package dir is its parent.
    pkg_dir = mod_file.parent
    site_dir = pkg_dir.parent  # the directory on sys.path holding the package
    if pkg_dir.is_dir():
        log(f"removing shadowing package dir: {pkg_dir}")
        shutil.rmtree(pkg_dir, ignore_errors=True)
        removed = True
    # Sibling metadata for either namespace name, so importlib.metadata no
    # longer reports the stale version from this location.
    for name in {dist_name, "flagtree", "triton"}:
        for pat in (f"{name}-*.dist-info", f"{name}-*.egg-info"):
            for p in site_dir.glob(pat):
                log(f"removing stale metadata: {p}")
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(
                    missing_ok=True
                )
                removed = True
    return removed


def main() -> None:
    if len(sys.argv) != 2:
        log("usage: enforce_pinned_compiler.py <backend>")
        sys.exit(2)
    backend = sys.argv[1]

    cfg = rc.load_backend(backend)
    pin = compiler_pin(cfg)
    if pin is None:
        log("no enforceable compiler pin for this backend; nothing to do")
        return
    dist_name, want = pin
    want_base = rc.base_version(want)
    # flagtree lives in the triton namespace; triton backends import triton too.
    mod_name = "triton"
    log(f"enforcing compiler pin {dist_name}=={want} (import '{mod_name}')")

    have, path = _probe_module(mod_name)
    if have is None or not path:
        log(f"compiler not importable yet (version={have!r}, path={path!r})")
        # Not fatal here: `pip install .[test]` runs next and may finish the
        # install. The post-install verify below is the real gate.
    else:
        log(f"resolved {mod_name} {have} from {path}")
        if rc.versions_match(want, have):
            log("resolved version already matches the pin; no shadow to clear")
        else:
            venv = _venv_prefix()
            mod_file = Path(path)
            if _under(mod_file, venv):
                log(
                    f"resolved copy is inside the venv ({venv}) but version "
                    f"{have} != pin {want}; reconcile should have fixed this"
                )
            else:
                log(
                    f"stale compiler {have} shadowing the venv from {mod_file} "
                    f"(outside {venv}); removing it so the pinned build wins"
                )
                if not _remove_shadow(mod_file, dist_name):
                    log("found nothing to remove; shadow may be unreachable")

    # Hard gate: re-resolve in a fresh interpreter and require the pin.
    have2, path2 = _probe_module(mod_name)
    if have2 is None:
        log("::error::compiler still not importable after enforcement")
        sys.exit(1)
    if not rc.versions_match(want, have2):
        log(
            f"::error::compiler version {have2} (from {path2}) does not match "
            f"pinned {want}; backends.yaml is the source of truth"
        )
        sys.exit(1)
    log(f"OK: compiler resolves to {have2} (base {want_base}) from {path2}")


if __name__ == "__main__":
    main()
