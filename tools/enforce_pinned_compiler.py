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
import sysconfig
from importlib import metadata
from importlib.util import find_spec
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
    return collect_installs(dist_name)[dist_name]


def collect_installs(*names: str) -> dict[str, list[metadata.Distribution]]:
    """Single pass over metadata.distributions(), collecting every requested name.

    metadata.distributions() walks every *.dist-info / *.egg-info dir on sys.path
    and parses each METADATA — the only genuinely expensive step here. Calling it
    once for flagtree, again for triton, and a third time in resolved_version
    tripled that cost for no benefit, so callers that need more than one name
    (main()) batch them here and reuse the result.
    """
    targets = {name: _norm(name) for name in names}
    result: dict[str, list[metadata.Distribution]] = {name: [] for name in names}
    for dist in metadata.distributions():
        raw = dist.metadata["Name"]
        if not raw:
            continue
        normed = _norm(raw)
        for name, target in targets.items():
            if normed == target:
                result[name].append(dist)
    return result


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
            # locate_file already yields an absolute path; resolving adds a stat
            # per RECORD entry (hundreds for a compiler package) for nothing,
            # since unlink() takes any path that points at the file.
            paths.append(Path(dist.locate_file(rel)))
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


def _install_path_suppressor(venv: Path, path_to_suppress: str) -> None:
    """Write a .pth file into the venv that strips a sys.path entry at startup.

    Used when a bare triton/ shadow lives on a read-only mount and cannot be
    deleted. The .pth is processed by site.py on every Python startup that
    uses this venv, so pytest subprocesses inherit the clean path without
    needing an explicit PYTHONPATH edit in the CI workflow.
    """
    site_pkgs = Path(sysconfig.get_path("purelib"))
    if not _under(site_pkgs, venv):
        log(
            f"  WARNING: venv site-packages {site_pkgs} is not under {venv}; "
            f"suppressor not installed"
        )
        return
    pth = site_pkgs / "flaggems-suppress-triton-shadow.pth"
    # A .pth line starting with "import " is executed as Python code at startup.
    # Use a private alias (_s) so we never collide with user code bound to `sys`.
    code = (
        f"import sys as _s; "
        f"_s.path = [_p for _p in _s.path if _p != {path_to_suppress!r}]; "
        f"del _s\n"
    )
    pth.write_text(code)
    log(f"  installed sys.path suppressor: {pth}")


def resolved_version(
    dist_name: str,
    installs: list[metadata.Distribution],
) -> tuple[str | None, Path | None]:
    """Version + root of the install that WINS on the current sys.path.

    When a name is installed in several locations, the one earliest on sys.path
    is what `import` gets. We mirror that: pick the matching distribution whose
    root appears first in sys.path order.

    Takes the already-collected installs list so main() does not need a fourth
    pass through metadata.distributions() just for the hard-verify step.
    The sys.path index is built once as a dict (O(1) lookup) rather than
    searched linearly per distribution.
    """
    if not installs:
        return None, None
    order_index = {Path(p).resolve(): i for i, p in enumerate(sys.path) if p}

    def rank(dist: metadata.Distribution) -> int:
        return order_index.get(_dist_root(dist).resolve(), len(order_index))

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
    # Hygon FlagTree owns triton; Ascend can provide it separately.
    hygon_flagtree = dist_name == "flagtree" and rc.vendor_of(backend) == "hygon"
    log(f"enforcing compiler pin {dist_name}=={want}")

    venv = _venv_prefix()

    # Single pass over metadata.distributions() collecting both the compiler
    # package and triton (needed for the flagtree shadow-eviction step below).
    # This avoids three separate full scans — one for the compiler pin, one for
    # triton shadows, and one inside resolved_version — cutting metadata I/O to
    # a single walk of the site-packages *.dist-info directories.
    collected = collect_installs(dist_name, "triton")
    installs = collected[dist_name]

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

    # When the pin is flagtree, also evict any triton package baked outside the
    # venv, except for the pinned Hygon FlagTree's own namespace. A separate
    # triton ahead of FlagTree on sys.path can shadow it even when the metadata
    # version is correct. setup.sh guards this with an explicit
    # 'uv pip uninstall triton'; the reconcile path has no such step.
    #
    # Two forms of shadow must be handled:
    #
    #   1. pip-installed triton with dist-info: found in collected["triton"],
    #      removed the same way as a stale flagtree shadow.
    #
    #   2. Bare directory on PYTHONPATH with no dist-info (e.g. /opt/triton
    #      injected via PYTHONPATH=[/opt/triton]): importlib.metadata never sees
    #      it, so collected["triton"] is empty. We scan every sys.path entry
    #      that (a) is outside the venv and (b) contains a triton/ subdirectory
    #      that looks like a real package (has __init__.py), then remove that
    #      subdirectory directly with shutil.rmtree.
    #
    # We only act when flagtree is the declared compiler and only on copies
    # outside the venv — triton inside the venv and backends that use triton
    # directly are never touched.
    if dist_name == "flagtree":
        # A baked, correctly pinned FlagTree can own triton/ even when it
        # lives outside the venv and has no separate triton dist-info.
        pinned_roots = {
            _dist_root(dist).resolve()
            for dist in installs
            if hygon_flagtree and rc.versions_match(want, _dist_version(dist))
        }
        # Form 1: pip-installed triton with dist-info outside the venv.
        triton_dist_shadows = [
            d
            for d in collected["triton"]
            if not _under(_dist_root(d), venv)
            and _dist_root(d).resolve() not in pinned_roots
        ]
        if triton_dist_shadows:
            log(
                f"flagtree is the pinned compiler; removing {len(triton_dist_shadows)} "
                f"pip-installed triton shadow(s) outside the venv:"
            )
            for d in triton_dist_shadows:
                log(f"  removing triton {_dist_version(d)} at {_dist_root(d)}")
                remove_dist(d)
        else:
            log("no pip-installed triton shadows outside the venv found")

        # Form 2: bare triton/ directory on sys.path with no dist-info.
        # Covers images that inject /opt/triton via PYTHONPATH.
        bare_removed = []
        for entry in sys.path:
            if not entry:
                continue
            p = Path(entry).resolve()
            if _under(p, venv) or p in pinned_roots:
                continue
            triton_pkg = p / "triton"
            if (triton_pkg / "__init__.py").exists():
                log(
                    f"found bare triton/ package at {triton_pkg} (no dist-info); "
                    f"removing it"
                )
                try:
                    shutil.rmtree(triton_pkg)
                    bare_removed.append(str(triton_pkg))
                    log(f"  removed {triton_pkg}")
                except OSError as e:
                    log(f"  could not remove {triton_pkg}: {e}")
                    log("  falling back to sys.path suppressor")
                    _install_path_suppressor(venv, entry)
                    # Patch current process too so the hard-verify below sees a
                    # clean sys.path, not the read-only shadow we could not delete.
                    sys.path = [
                        sp for sp in sys.path if Path(sp).resolve() != p.resolve()
                    ]
                    bare_removed.append(str(triton_pkg))
        if not bare_removed:
            log("no bare triton/ directories found outside the venv")

    # Hard gate: what resolves now must be the pin.
    # Re-collect: the list above was taken before the cleanup and still holds
    # the copies this run deleted, whose metadata now reads back empty -- they
    # would win the sys.path ranking and fail the gate on a removed install.
    installs = collect_installs(dist_name)[dist_name]
    have, root = resolved_version(dist_name, installs)
    if have is None:
        log(f"::error::{dist_name} no longer resolvable after cleanup")
        sys.exit(1)
    if not rc.versions_match(want, have):
        log(
            f"::error::{dist_name} resolves to {have} (from {root}), not pinned "
            f"{want}; backends.yaml is the source of truth"
        )
        sys.exit(1)
    if hygon_flagtree:
        spec = find_spec("triton")
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        expected = (root / "triton" / "__init__.py").resolve()
        if origin != expected:
            log(f"::error::triton resolves to {origin}, expected {expected}")
            sys.exit(1)
    log(f"OK: {dist_name} resolves to {have} from {root}")


if __name__ == "__main__":
    main()
