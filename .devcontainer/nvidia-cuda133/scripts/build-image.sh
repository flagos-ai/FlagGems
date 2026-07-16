#!/usr/bin/env bash
# Build the FlagGems NVIDIA CUDA 13.3 runtime image (and its base image) using build-infra.
#
# Called by devcontainer.json initializeCommand — runs on the host before the
# container is created. Each stage is skipped if its image already exists locally.
#
# Environment variables:
#   FLAGGEMS_BASE_IMAGE    override base image tag
#   FLAGGEMS_IMAGE         override runtime image tag

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_INFRA="$REPO_ROOT/.devcontainer/build-infra"

BASE_BACKEND="nvidia-cuda13.3"
RUNTIME_BACKEND="nvidia-cuda13.3"

# Derive base image tag from the LABEL in the base Containerfile
_base_version=$(grep '^LABEL org.opencontainers.image.version' "$BUILD_INFRA/base/$BASE_BACKEND" | grep -oP '"\K[^"]+')
_base_revision=$(grep '^LABEL org.opencontainers.image.revision' "$BUILD_INFRA/base/$BASE_BACKEND" | grep -oP '"\K[^"]+')
BASE_IMAGE="${FLAGGEMS_BASE_IMAGE:-flagos-base-${BASE_BACKEND}:${_base_version}-${_base_revision}}"
RUNTIME_IMAGE="${FLAGGEMS_IMAGE:-flagos-runtime-${RUNTIME_BACKEND}:latest}"

# ── Stage 1: base image ───────────────────────────────────────────────────────
if docker image inspect "$BASE_IMAGE" > /dev/null 2>&1; then
    echo "Base image $BASE_IMAGE already exists, skipping."
else
    echo "Building base image $BASE_IMAGE..."
    python3 "$BUILD_INFRA/base/build.py" "$BASE_BACKEND" --tag "$BASE_IMAGE"
fi

# ── Stage 2: runtime image ────────────────────────────────────────────────────
if docker image inspect "$RUNTIME_IMAGE" > /dev/null 2>&1; then
    echo "Runtime image $RUNTIME_IMAGE already exists, skipping."
else
    echo "Building runtime image $RUNTIME_IMAGE..."
    python3 "$BUILD_INFRA/runtime/build.py" "$RUNTIME_BACKEND" \
        --flaggems-dir "$REPO_ROOT" \
        --base-image "$BASE_IMAGE" \
        --tag "$RUNTIME_IMAGE"
fi
