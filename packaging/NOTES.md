# FlagGems packaging notes (Phase 1)

## DEB / RPM asymmetry (intentional, but tracked)

Phase 1 currently ships two slightly different artifacts:

| target | C++ extension (`liboperators.so`) | wheel arch tag | distro arch |
|--------|------------------------------------|----------------|-------------|
| Debian | **ON** — bundled, links libtorch/libcuda | `linux_x86_64` | `amd64` |
| RPM    | **OFF** — Python-only path             | `linux_x86_64` (scikit-build-core always tags) | `x86_64` |

Why the split exists today:
- RPM `%pyproject_wheel` defaults to `FLAGGEMS_BUILD_C_EXTENSIONS=OFF`
- Debian `rules` invokes the full build with the C++ ext, mirroring upstream `setup.sh`

Plan to converge (Phase 2):
- Either enable C++ ext on the RPM side (`rpmbuild --define '_build_ext 1'` + extra `BuildRequires:`)
- Or disable it on the Debian side (`DEB_BUILD_OPTIONS=noext`)

Until that is decided, keep the RPM `BuildArch: x86_64` (not `noarch`) so the
wheel's actual arch tag and the distro arch line up — `noarch` + an x86_64
wheel placed under `%{python3_sitearch}` is internally inconsistent and trips
rpmlint.

## libtriton-jit dependency for CI

`Dockerfile.deb` requires `libtriton-jit*.deb` to be staged at
`packaging/debian/helpers/local-deps/` before the Debian build runs.

The CI workflow does **not** yet stage that artifact, so the `build-deb`
workflow is expected to fail until either:
- a `libtriton_jit` upstream release publishes the .deb (then `download-artifact`
  it here), or
- the FlagOS APT repo goes live and `Dockerfile.deb` can `apt-get install
  libtriton-jit-dev`.

The workflow is marked `continue-on-error: true` so a failing build does not
block downstream gating; remove that flag once the dependency is wired up.

## Merge order with FlagTree

`debian/control` declares `Recommends: python3-flagtree-nvidia`. Until
FlagTree's matching PR (`flagos-ai/FlagTree#607`) merges and the resulting
`python3-flagtree-nvidia` is available, `apt install python3-flag-gems` on a
fresh system will print a "Recommends not satisfiable" warning (not an error).

Recommended merge order: **FlagTree first → FlagGems second**.

## TODO

- `debian/copyright` does not yet enumerate third-party components linked into
  `liboperators.so` (nlohmann-json, pybind11, etc.). Audit the link line in
  `setup.sh` / CMake and add Files-Excluded / per-file copyright stanzas.
