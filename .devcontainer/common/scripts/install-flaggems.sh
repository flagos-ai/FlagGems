#!/usr/bin/env bash
set -euo pipefail

echo "==> Syncing git submodules..."
git submodule sync
git submodule update --init --recursive

echo "==> Installing FlagGems with the following configuration:"
echo "    CMAKE_ARGS=${CMAKE_ARGS:-<none>}"
echo "    FLAGGEMS_BACKEND=${FLAGGEMS_BACKEND:-CUDA}"
echo "    PIP_INDEX_URL=${PIP_INDEX_URL:-<default>}"

# Install FlagGems with the provided CMAKE_ARGS.
# Use sudo so we can replace any root-owned pre-installation from the base image.
CMAKE_ARGS="${CMAKE_ARGS:-}" \
sudo -E pip3 install -v --no-build-isolation -e .

echo "==> FlagGems installation completed successfully!"
