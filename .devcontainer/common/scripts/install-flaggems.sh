#!/usr/bin/env bash
set -euo pipefail

echo "==> Syncing git submodules..."
git submodule sync
git submodule update --init --recursive

echo "==> Installing FlagGems in editable mode with the following configuration:"
echo "    CMAKE_ARGS=${CMAKE_ARGS:-<none>}"
echo "    FLAGGEMS_BACKEND=${FLAGGEMS_BACKEND:-CUDA}"
echo "    NO_BUILD_ISOLATION=${FLAGGEMS_NO_BUILD_ISOLATION:-0}"

# Uninstall the non-editable copy baked into the container image so the
# workspace source tree takes precedence at import time.
pip uninstall -y flaggems 2>/dev/null || true

# Build the pip install command.
# --no-build-isolation is only supported on platforms that ship all required
# build-time headers inside the container image (nvidia, iluvatar, mthreads,
# ascend). Other platforms must use the default isolated build.
INSTALL_ARGS="-e ."
if [ "${FLAGGEMS_NO_BUILD_ISOLATION:-0}" = "1" ]; then
    INSTALL_ARGS="--no-build-isolation ${INSTALL_ARGS}"
fi

# Prefer uv (present in /venv-based images); fall back to sudo pip for images
# where pip lives in the system Python (e.g. nvcr pytorch base images).
if command -v uv &>/dev/null; then
    CMAKE_ARGS="${CMAKE_ARGS:-}" \
    uv pip install ${INSTALL_ARGS}
else
    CMAKE_ARGS="${CMAKE_ARGS:-}" \
    sudo -E pip3 install ${INSTALL_ARGS}
fi

echo "==> FlagGems editable installation completed successfully!"
