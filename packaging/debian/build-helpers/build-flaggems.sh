#!/usr/bin/env bash
# Build python3-flag-gems_*.deb (Phase 1: bundled package, pure-Python wheel).
#
# Phase 1 sets FLAGGEMS_BUILD_C_EXTENSIONS=OFF (default in CMakeLists.txt:32),
# so the wheel does not link against libtriton-jit and no local-deps staging
# is needed. Phase 2 (C++ extension on) will reintroduce that dependency.
#
# Output: ./debian-packages/python3-flag-gems_*.deb

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mkdir -p dist
docker build \
    --network=host \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

# Mirror the rpm/build-flag-gems-rpm.sh layout: surface artifacts under
# debian-packages/ so CI upload-artifact and local users find them in the
# same place across DEB and RPM workflows.
mkdir -p debian-packages
cp dist/output/*.deb debian-packages/

echo ""
echo ">>> Output:"
ls -lh debian-packages/
