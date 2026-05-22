#!/usr/bin/env bash
# Build python3-flag-gems_*.deb (Phase 1: bundled package).
#
# Pre-requisite: a libtriton-jit*.deb must be in
# packaging/debian/build-helpers/local-deps/. Get it from the FlagOS APT repo
# (once live) or from the libtriton_jit upstream's CI artifacts.
#
# Output: ./dist/output/python3-flag-gems_*.deb

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DEPS_DIR="packaging/debian/build-helpers/local-deps"
if [ ! -d "$DEPS_DIR" ] || [ -z "$(ls "$DEPS_DIR"/libtriton-jit*.deb 2>/dev/null)" ]; then
    echo "ERROR: $DEPS_DIR/libtriton-jit*.deb is required for the build."
    echo "Copy or symlink the libtriton-jit{,-dev}_*.deb files there first."
    echo "Hint: set LIBTRITON_JIT_DEB_DIR to a directory holding the .deb files,"
    echo "      e.g. export LIBTRITON_JIT_DEB_DIR=\$HOME/git/libtriton_jit/output/deb"
    exit 1
fi

mkdir -p dist
docker build \
    --network=host \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --target deb-output \
    --output "type=local,dest=${REPO_ROOT}/dist" \
    .

echo ""
echo ">>> Output:"
ls -lh dist/output/ 2>/dev/null || ls -lh dist/
