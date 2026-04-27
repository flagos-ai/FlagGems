# FlagGems Packaging

Debian (.deb) and RPM (.rpm) packaging for FlagGems.

## Status — scaffold only, NOT yet validated

The directory tree is in place and the control/spec/Dockerfile reflect
the intended design. **No build has run yet.** Expect iteration before
the first .deb lands; the FlagTree packaging required ~8 attempts to
catch all edge cases, and FlagGems is a similar shape.

## Two phases

### Phase 1 (this scaffold) — bundled `python3-flag-gems`

Single binary package containing:
- Python module (Triton kernel sources, Python wrappers under
  `/usr/lib/python3/dist-packages/flag_gems/`)
- `liboperators.so` bundled inside the Python package directory

This is the path of least resistance: scikit-build-core builds the
wheel including the C++ extension, and `dpkg-buildpackage` wraps the
wheel into a single .deb. The `libflaggems` and `libflaggems-dev`
stanzas in `debian/control` are commented out for Phase 1.

### Phase 2 (deferred) — split into 3 binaries

- `libflaggems` — `/usr/lib/<libdir>/liboperators.so` (runtime)
- `libflaggems-dev` — `/usr/include/flag_gems/` headers + cmake config
- `python3-flag-gems` — Python module only, links against system `libflaggems`

Phase 2 needs:
- A separate `cmake --install` step before the wheel build, into
  the package staging dir
- `debian/python3-flag-gems.install`, `debian/libflaggems.install`,
  `debian/libflaggems-dev.install` files to slice the staging dir
- A patch to FlagGems' own loader so the Python module finds
  `liboperators.so` at `/usr/lib/<libdir>/` rather than alongside the
  Python files

## Layout

```
packaging/
├── debian/
│   ├── control          # 3 packages declared (2 commented out for Phase 1)
│   ├── rules            # pip --target unpack of the wheel
│   ├── changelog, copyright, source/format
│   └── helpers/
│       ├── Dockerfile.deb       # 2-stage: wheel build → deb assemble
│       ├── build-flaggems.sh    # entrypoint
│       └── local-deps/          # drop libtriton-jit*.deb here pre-build
└── rpm/
    ├── specs/flag-gems.spec     # to be written; mirrors Dockerfile.rpm
    └── helpers/
        ├── Dockerfile.rpm
        └── build-flaggems-rpm.sh
```

## Build prerequisites

- libtriton-jit + libtriton-jit-dev .debs (from the FlagOS APT repo, or
  the upstream libtriton_jit's CI artifacts) — drop into
  `packaging/debian/helpers/local-deps/` before running the build
- Docker with networking access for PyPI + PyTorch CDN

## Why bundled in Phase 1

FlagGems' `scikit-build-core` build packs `liboperators.so` inside
`flag_gems/_C/` (the Python package dir). To produce a separate
`libflaggems` we'd need to either:

1. Run cmake twice (once with the Python tree disabled, once with it
   enabled), or
2. Rewrite the Python module's loader to look for `liboperators.so`
   in `/usr/lib` rather than adjacent to the .py files

Both are real but non-trivial. Phase 1 ships everything in one .deb
to unblock users; Phase 2 polishes for downstream Debian/Fedora
acceptance.
