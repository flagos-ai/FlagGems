#!/usr/bin/env bash
# Build python3-flag-gems_*.deb (Phase 1: pure-Python wheel).
#
# The upstream Python wheel is pure Python (setuptools backend); the C++
# operators live in the separate cpp/ tree and are out of Phase 1 scope,
# so no libtriton-jit / local-deps staging is needed here. Phase 2
# (packaging the cpp/ per-vendor extensions) will reintroduce that.
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
