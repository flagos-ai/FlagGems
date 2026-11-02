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
so the test run still resolves the stale baked version and silently uses the
wrong compiler build (confirmed in ops-test run 40473319476: reconcile installed
flagtree 0.7.1a1 into /flagos but `import` still reported 0.7.0 from /opt).

This script closes that gap, working entirely off installed *distribution
metadata* rather than importing anything:

  1. Read the compiler pin (flagtree, else triton) from backends.yaml — same
     source reconcile acted on.
  2. Enumerate every install of that distribution via importlib.metadata.
     More than one means a shadow: typically the venv copy (the pin, from
     reconcile) plus the image's baked copy outside the venv.
  3. Delete each copy whose version != the pin AND whose files live outside the
     venv (the baked shadow), using the distribution's own RECORD so we remove
     exactly what it installed — never touching the venv copy or other packages.
  4. Hard-verify: after cleanup, exactly the pinned version must remain
     resolvable, or exit non-zero so the job fails instead of testing a stale
     build.

We never import the compiler (importing triton on Ascend prints warnings to
stdout and, more importantly, triton and flagtree are *separate* distributions
here with different versions — importing would conflate them). Metadata lookup
is side-effect free and reports each distribution's real on-disk version.

Usage:
    python3 tools/enforce_pinned_compiler.py <backend>

Env:
    VIRTUAL_ENV  venv prefix; a copy whose files live under here is the
                 pip-installed pin and is never deleted. Falls back to
                 sys.prefix when unset.
"""

from __future__ import annotations

import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

# Reuse the exact config-loading + version-compare logic reconcile uses, so the
# pin read here is byte-for-byte what reconcile acted on.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconcile_backend_deps as rc  # noqa: E402

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


def _norm(name: str) -> str:
    return name.replace("_", "-").lower()


def find_installs(dist_name: str) -> list[metadata.Distribution]:
    """All installed distributions matching dist_name (separator-insensitive).

    importlib.metadata returns one Distribution per install location, so a
    package present both in the venv and baked outside it shows up twice.
    """
    target = _norm(dist_name)
    found = []
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name and _norm(name) == target:
            found.append(dist)
    return found


def _dist_root(dist: metadata.Distribution) -> Path:
    """Directory on sys.path that holds this distribution (its *-info parent)."""
    # dist._path points at the .dist-info/.egg-info dir; its parent is the
    # site dir. _path is private but stable across supported CPythons; fall
    # back to locating a file if it's ever absent.
    p = getattr(dist, "_path", None)
    if p is not None:
        return Path(p).parent
    located = dist.locate_file("")
    return Path(str(located))


def _dist_version(dist: metadata.Distribution) -> str:
    return rc.normalize(dist.metadata["Version"] or "")


def remove_dist(dist: metadata.Distribution) -> None:
    """Delete everything a distribution installed, via its RECORD, then prune.

    RECORD lists every installed file (package modules + the *-info metadata),
    so removing those removes exactly this install and nothing else.
    """
    root = _dist_root(dist)
    paths: list[Path] = []
    files = dist.files or []
    for rel in files:
        try:
            paths.append(Path(dist.locate_file(rel)).resolve())
        except Exception:
            continue
    # Remove files first, then any now-empty dirs (deepest first).
    dirs: set[Path] = set()
    for f in paths:
        try:
            if f.is_file() or f.is_symlink():
                f.unlink(missing_ok=True)
            dirs.add(f.parent)
        except OSError as e:
            log(f"  could not remove {f}: {e}")
    if not files:
        # No RECORD (rare): fall back to removing the *-info dir only, so we at
        # least stop metadata from reporting the stale version.
        info = getattr(dist, "_path", None)
        if info:
            shutil.rmtree(info, ignore_errors=True)
            log(f"  removed metadata dir (no RECORD): {info}")
    for d in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            if d.is_dir() and not any(d.iterdir()) and d != root:
                d.rmdir()
        except OSError:
            pass
    log(f"  removed {len(paths)} file(s) under {root}")


def _venv_prefix() -> Path:
    return Path(os.environ.get("VIRTUAL_ENV") or sys.prefix).resolve()


def _under(path: Path, prefix: Path) -> bool:
    try:
        path.resolve().relative_to(prefix)
        return True
    except ValueError:
        return False


def resolved_version(dist_name: str) -> tuple[str | None, Path | None]:
    """Version + root of the install that WINS on the current sys.path.

    When a name is installed in several locations, the one earliest on sys.path
    is what `import` gets. We mirror that: pick the matching distribution whose
    root appears first in sys.path order.
    """
    installs = find_installs(dist_name)
    if not installs:
        return None, None
    order = [Path(p).resolve() for p in sys.path if p]

    def rank(dist: metadata.Distribution) -> int:
        root = _dist_root(dist).resolve()
        for i, entry in enumerate(order):
            if entry == root:
                return i
        return len(order)  # not on sys.path -> lowest priority

    winner = min(installs, key=rank)
    return _dist_version(winner), _dist_root(winner)


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
    log(f"enforcing compiler pin {dist_name}=={want}")

    venv = _venv_prefix()
    installs = find_installs(dist_name)
    if not installs:
        log(
            f"::error::{dist_name} not installed at all after reconcile; "
            f"expected {want}"
        )
        sys.exit(1)

    log(f"found {len(installs)} install(s) of {dist_name}:")
    for dist in installs:
        root = _dist_root(dist)
        log(f"  - {_dist_version(dist)} at {root} (in venv: {_under(root, venv)})")

    # Remove every copy that is both stale (version != pin) and outside the venv
    # — i.e. the image's baked shadow. A matching copy, or anything inside the
    # venv, is left alone.
    for dist in installs:
        ver = _dist_version(dist)
        root = _dist_root(dist)
        if rc.versions_match(want, ver):
            continue
        if _under(root, venv):
            log(
                f"stale {dist_name} {ver} is inside the venv ({venv}); leaving "
                f"it for reconcile/pip to manage, not deleting"
            )
            continue
        log(f"removing stale shadow {dist_name} {ver} at {root}")
        remove_dist(dist)

    # Hard gate: what resolves now must be the pin.
    have, root = resolved_version(dist_name)
    if have is None:
        log(f"::error::{dist_name} no longer resolvable after cleanup")
        sys.exit(1)
    if not rc.versions_match(want, have):
        log(
            f"::error::{dist_name} resolves to {have} (from {root}), not pinned "
            f"{want}; backends.yaml is the source of truth"
        )
        sys.exit(1)
    log(f"OK: {dist_name} resolves to {have} from {root}")


if __name__ == "__main__":
    main()
