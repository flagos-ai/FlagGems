# FlagGems Packaging

Debian (.deb) and RPM (.rpm) packaging for FlagGems.

## Status

- **.deb (Phase 1, pure Python)**: builds locally on ubuntu:22.04 via the
  multi-stage `Dockerfile.deb` (`wheel-builder` → `deb-assembler`); the
  resulting `python3-flag-gems_*.deb` installs cleanly on ubuntu:24.04
  with `apt -f` resolving the distro Depends.
- **.rpm (Phase 1)**: builds locally on fedora:43; `dnf install` succeeds
  on a fresh fedora:43 container.
- **CI**: GitHub Actions `build-deb.yml` is currently `continue-on-error`
  (see workflow comment).

## Upstream source structure

Since the scikit-build retirement upstream, the repository builds two
kinds of distributions from clearly separated trees:

- **Python-only wheel** — `pyproject.toml` + `setup.py` at the repo root,
  plain `setuptools.build_meta` backend, sources under `src/flag_gems/`.
  The wheel is `py3-none-any` (no native code); `setup.py` additionally
  maps `tests/` → `flaggems_tests` and `benchmark/` → `flaggems_benchmark`
  so the operator test and benchmark suites ship inside the wheel.
- **C++ wrapped operators** — a separate scikit-build-core project under
  `cpp/`, built per vendor as `flag-gems-cpp-<vendor>` native wheels
  (cuda / musa / npu / gcu / ix). Each drops its `.so` files into the
  `flag_gems/` namespace and depends on the matching `flag-gems` version.
  Users opt in via the `flag-gems[cpp-<vendor>]` extras.

## Two phases

### Phase 1 (this scaffold) — `python3-flag-gems`, Python only

Single noarch binary package containing the Python module (Triton kernel
sources, Python wrappers under `/usr/lib/python3/dist-packages/flag_gems/`),
plus the bundled `flaggems_tests` / `flaggems_benchmark` suites and the
`flaggems-setup` console script. This mirrors upstream's default
`pip install flag_gems` behavior — no native extension involved.

### Phase 2 (deferred) — C++ operator packages from `cpp/`

Package the per-vendor native extension (upstream's `flag-gems-cpp-*`
wheels) as distro packages, e.g.:

- `python3-flag-gems-cpp-<vendor>` — the pybind11 extension `.so`s
  installed into the `flag_gems/` namespace, or alternatively
- `libflaggems` / `libflaggems-dev` — the C++ runtime + headers for
  downstream C++ consumers, via a direct `cmake --install` of `cpp/`.

Phase 2 needs a CUDA-/vendor-SDK-aware build environment (libtorch,
libtriton-jit) and is intentionally out of scope here.

## Layout

```
packaging/
├── debian/
│   ├── control          # python3-flag-gems (Phase 2 stanzas commented out)
│   ├── rules            # pip --target unpack of the wheel
│   ├── changelog, copyright, source/format
│   └── build-helpers/
│       ├── Dockerfile.deb       # 2-stage: wheel build → deb assemble
│       ├── build-flaggems.sh    # entrypoint
│       └── local-deps/          # Phase 2: drop libtriton-jit*.deb here
└── rpm/
    ├── specs/flag-gems.spec
    ├── dockerfiles/Dockerfile.rpm
    └── build-flag-gems-rpm.sh
```

## Build prerequisites

- Docker with networking access to PyPI (the wheel build only needs
  `setuptools` + `setuptools-scm`; no torch, cmake or vendor SDK).

Build outputs land in `debian-packages/` and `rpm-packages/` at the repo
root (both gitignored).

## Why Python-only in Phase 1

The upstream Python wheel is pure Python by design — the C++ operator
runtime lives in the separate `cpp/` tree and is distributed as optional
per-vendor `flag-gems-cpp-*` wheels. Packaging those requires per-vendor
toolchains (CUDA, MUSA, CANN, …) and a libtorch/libtriton-jit build
environment, so Phase 1 ships the universally useful Python package to
unblock users; Phase 2 adds the native packages for downstream
Debian/Fedora acceptance.
