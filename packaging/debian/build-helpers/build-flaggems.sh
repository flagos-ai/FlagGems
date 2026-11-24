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

# The wheel takes its version from setuptools_scm, which reads git. The
# build context has no .git (see .dockerignore), so derive the version here
# and hand it to the Dockerfile; the throwaway container keeps the host free
# of a setuptools_scm install. FLAGGEMS_VERSION in the environment overrides,
# for callers that stamp the version themselves.
if [ -z "${FLAGGEMS_VERSION:-}" ]; then
    FLAGGEMS_VERSION="$(docker run --rm \
        --network=host \
        -v "${REPO_ROOT}:/src:ro" -w /src \
        python:3.12-slim bash -euo pipefail -c '
            apt-get update -qq >/dev/null
            apt-get install -y -qq --no-install-recommends git >/dev/null
            pip install -q "setuptools-scm>=8"
            git config --global --add safe.directory /src
            python -m setuptools_scm' 2>/dev/null | tail -1)"
fi
if [ -z "${FLAGGEMS_VERSION}" ]; then
    echo "error: could not derive the wheel version with setuptools_scm" >&2
    exit 1
fi
echo ">>> wheel version: ${FLAGGEMS_VERSION}"

mkdir -p dist
docker build \
    --network=host \
    -f packaging/debian/build-helpers/Dockerfile.deb \
    --target deb-output \
    --build-arg "FLAGGEMS_VERSION=${FLAGGEMS_VERSION}" \
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
