#!/usr/bin/env bash
# Install FlagGems in editable mode inside the devcontainer.
#
# This script is called by postCreateCommand via the vendor-specific
# install-dev-tools.sh wrapper, which sources flaggems.env first to set
# FLAGGEMS_DEVCONTAINER_BACKEND (e.g. "nvidia-cuda128", "hygon").
#
# What this script does:
#   1. Syncs git submodules (libtriton_jit etc.)
#   2. Reads backends.yaml to determine whether C++ extensions are supported.
#   3. Installs FlagGems in editable mode using the Python already present in
#      the container image (no new venv — the image already ships a full
#      Python/PyTorch/Triton environment under /flagos or similar).
#   4. Appends backend environment variables to ~/.bashrc so they are active
#      in every subsequent shell session.
#
# C++ extensions are built automatically when the backend declares a
# cmake_backend value in backends.yaml (e.g. CUDA, NPU).  Set ENABLE_CPP=0
# to force-disable them even for supported backends.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BACKENDS_YAML="src/flag_gems/backends.yaml"
BACKEND="${FLAGGEMS_DEVCONTAINER_BACKEND:?FLAGGEMS_DEVCONTAINER_BACKEND must be set}"
export BACKEND

# ── 1. Init submodules ───────────────────────────────────────────────────────
# The workspace is a bind mount; .git/modules may be owned by root if a
# previous container run executed as root (e.g. a failed first attempt before
# updateRemoteUserUID aligned the UID).  Fix ownership before git tries to
# write core.worktree into the submodule config files.
echo "==> Initialising git submodules..."
if [ -d .git/modules ]; then
    GIT_MODULES_OWNER=$(stat -c '%u' .git/modules)
    CURRENT_UID=$(id -u)
    if [ "${GIT_MODULES_OWNER}" != "${CURRENT_UID}" ]; then
        echo "    Fixing .git/modules ownership (was uid=${GIT_MODULES_OWNER}, now uid=${CURRENT_UID})..."
        sudo chown -R "${CURRENT_UID}:$(id -g)" .git/modules
    fi
fi
git submodule update --init --recursive --quiet

# ── 2. Determine C++ support from backends.yaml ───────────────────────────────
echo "==> Reading backend config for '${BACKEND}'..."

CMAKE_BACKEND=$(python3 - <<'EOF'
import yaml, sys, os
backend = os.environ["BACKEND"]
cfg = yaml.safe_load(open("src/flag_gems/backends.yaml"))
b = cfg["backends"].get(backend)
if b is None:
    print(f"Error: unknown backend '{backend}'", file=sys.stderr)
    sys.exit(1)
print(b.get("cmake_backend", ""))
EOF
)

ENABLE_CPP="${ENABLE_CPP:-}"

if [ -n "${CMAKE_BACKEND}" ]; then
    # Backend supports C++ — default ON, override with ENABLE_CPP=0
    if [ "${ENABLE_CPP}" = "0" ]; then
        ENABLE_CPP=0
        echo "==> C++ extensions: DISABLED (ENABLE_CPP=0)"
    else
        ENABLE_CPP=1
        echo "==> C++ extensions: ENABLED (backend=${CMAKE_BACKEND})"
    fi
else
    # Backend does not support C++ extensions
    if [ "${ENABLE_CPP}" = "1" ]; then
        echo "Warning: ENABLE_CPP=1 but backend '${BACKEND}' has no cmake_backend — ignoring."
    fi
    ENABLE_CPP=0
    echo "==> C++ extensions: not supported for '${BACKEND}'"
fi

# ── 3. Build CMAKE_ARGS ───────────────────────────────────────────────────────
if [ "${ENABLE_CPP}" = "1" ]; then
    export CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=${CMAKE_BACKEND} -DCMAKE_BUILD_TYPE=Release"
else
    export CMAKE_ARGS=""
fi

echo "==> CMAKE_ARGS: ${CMAKE_ARGS:-<none>}"

# ── 4. Ensure build tools are present ────────────────────────────────────────
# Read build-time dependencies from pyproject.toml rather than hard-coding
# them here:
#   • build-system.requires  — the standard PEP 517 field (scikit-build-core,
#                              pybind11, setuptools, setuptools-scm)
#   • tool.flaggems.no-isolation-deps.deps — packages excluded from requires
#                              because they must exist in the active env before
#                              --no-build-isolation is used (cmake, ninja)
echo "==> Ensuring build tools (from pyproject.toml)..."
BUILD_DEPS=$(python3 - <<'PYEOF'
import sys

# tomllib is in the stdlib since 3.11; fall back to tomli for older versions.
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("tomllib/tomli not available — install tomli first", file=sys.stderr)
        sys.exit(1)

with open("pyproject.toml", "rb") as f:
    cfg = tomllib.load(f)

deps = list(cfg["build-system"]["requires"])
deps += cfg.get("tool", {}).get("flaggems", {}).get("no-isolation-deps", {}).get("deps", [])
print("\n".join(deps))
PYEOF
)

if command -v uv &>/dev/null; then
    echo "${BUILD_DEPS}" | xargs uv pip install -q
else
    echo "${BUILD_DEPS}" | xargs pip install -q
fi

# ── 5. Uninstall any baked-in non-editable copy ───────────────────────────────
pip uninstall -y flaggems 2>/dev/null || true

# ── 6. Editable install ───────────────────────────────────────────────────────
# --no-build-isolation reuses the build tools already present in the image.
echo "==> Installing FlagGems in editable mode (backend: ${BACKEND})..."
if command -v uv &>/dev/null; then
    CMAKE_ARGS="${CMAKE_ARGS}" \
    uv pip install --no-build-isolation -e .
else
    CMAKE_ARGS="${CMAKE_ARGS}" \
    pip install --no-build-isolation -e .
fi

# ── 7. Write backend environment variables to ~/.bashrc ───────────────────────
echo "==> Writing backend environment to ~/.bashrc..."
python3 - <<EOF
import yaml, os

backend = os.environ["BACKEND"]
cfg = yaml.safe_load(open("src/flag_gems/backends.yaml"))
b = cfg["backends"][backend]

lines = [
    "",
    f"# --- FlagGems devcontainer environment ({backend}) ---",
]
for k, v in b.get("env", {}).items():
    lines.append(f'export {k}="{v}"')
for script in b.get("env_source", []):
    lines.append(f'[ -f {script} ] && source {script} || true')
lines.append("# --- end FlagGems environment ---")

bashrc = os.path.expanduser("~/.bashrc")
with open(bashrc, "a") as f:
    f.write("\n".join(lines) + "\n")
print(f"  Written to {bashrc}")
EOF

echo "==> FlagGems editable installation completed."
