#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok()   { printf " ${GREEN}[OK]${NC}\n"; }
fail() { printf " ${RED}[FAILED]${NC}\n"; exit 1; }

BACKENDS_YAML="src/flag_gems/backends.yaml"

# ── Validate argument ─────────────────────────────────────────
[ "$#" -eq 1 ] || { echo "Usage: $0 <BACKEND>"; exit 1; }

BACKEND="${1}"

# ── Read config from backends.yaml ────────────────────────────
if [ ! -f "$BACKENDS_YAML" ]; then
  echo "Error: $BACKENDS_YAML not found. Run from the FlagGems repo root."
  exit 1
fi

# Phase 1: Extract only python version and vendor using grep/awk
# (no pyyaml dependency — runs before venv creation)
PYTHON_VERSION=$(awk "/^  ${BACKEND}:/{found=1} found && /python:/{print \$2; exit}" "$BACKENDS_YAML" | tr -d '"')
if [ -z "${PYTHON_VERSION}" ]; then
  echo "Error: unknown backend '${BACKEND}'"
  echo "Available backends:"
  awk '/^  [a-z].*:$/{gsub(/:$/,""); print "  "$1}' "$BACKENDS_YAML"
  exit 1
fi

VENDOR=$(echo "${BACKEND}" | sed 's/-[^-]*$//')
[ "${VENDOR}" = "${BACKEND}" ] && VENDOR="${BACKEND}"
PYPI_BASE=$(grep '^pypi_base:' "$BACKENDS_YAML" | sed 's/^pypi_base: *"//;s/"$//')
FLAGOS_PYPI=$(echo "${PYPI_BASE}" | sed "s/{vendor}/${VENDOR}/")
MIRROR=$(grep '^mirror:' "$BACKENDS_YAML" | sed 's/^mirror: *"//;s/"$//')

printf "Backend: ${BACKEND} (vendor: ${VENDOR})"
ok

# ── Detect or install uv ─────────────────────────────────────
UV_VERSION="0.11.22"
UV_MIRROR="https://resource.flagos.net/repository/flagos-filestore/utils"

printf "Checking uv ..."
if command -v uv &>/dev/null; then
  printf " $(uv --version)"
  ok
else
  printf " not found, installing ...\n"
  export HOME=$(eval echo ~"$(whoami)")
  ARCH=$(uname -m)
  mkdir -p "$HOME/.local/bin"
  curl -sSf "${UV_MIRROR}/uv-${ARCH}-${UV_VERSION}-linux-gnu.tar.gz" \
    | tar xz -C "$HOME/.local/bin" 2>/dev/null \
    || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv &>/dev/null || { printf "uv installation"; fail; }
  printf "Installed $(uv --version)"
  ok
fi

# ── Install Python via uv ────────────────────────────────────
printf "Installing Python ${PYTHON_VERSION} ..."
uv python install "${PYTHON_VERSION}" --python-preference only-managed -q || fail
ok

# ── Create virtual environment ────────────────────────────────
printf "Creating virtual environment ..."
uv venv .venv --python "${PYTHON_VERSION}" --python-preference only-managed -q || fail
ok
source .venv/bin/activate

printf "Python: $(python --version)"
ok

# ── Source vendor environment ─────────────────────────────────
export USE_TRITON="${USE_TRITON:-}"
source tools/env.sh "${BACKEND}"

# ── Install build tools ──────────────────────────────────────
printf "Installing build tools ..."
uv pip install -q \
  "setuptools>=64.0" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0" \
  "PyYAML>=6.0" \
  --index "${MIRROR}" \
  || fail
ok

# ── Phase 2: Full YAML parse (pyyaml now available in venv) ──
eval $(python3 -c "
import yaml, sys

cfg = yaml.safe_load(open('${BACKENDS_YAML}'))
b = cfg['backends']['${BACKEND}']

cmake_backend = b.get('cmake_backend', '')
print(f'CMAKE_BACKEND={cmake_backend}')

ft = b.get('flagtree', '')
if isinstance(ft, list):
    ft = ' '.join(ft)
print(f'FLAGTREE_PKGS=\"{ft}\"')

tr = b.get('triton', '')
if isinstance(tr, list):
    tr = ' '.join(tr)
print(f'TRITON_PKGS=\"{tr}\"')

post_install = []
post_uninstall = []
for item in b.get('post_install', []):
    if isinstance(item, dict) and 'uninstall' in item:
        post_uninstall.append(item['uninstall'])
    else:
        post_install.append(item)
print(f'POST_INSTALL=\"{\" \".join(post_install)}\"')
print(f'POST_UNINSTALL=\"{\" \".join(post_uninstall)}\"')
")

# ── C++ extensions ───────────────────────────────────────────
# Set ENABLE_CPP=1 to build C++ wrapped operators.
# Default: OFF (C++ extensions require vendor SDK and toolchain).
if [ "${ENABLE_CPP:-0}" = "1" ]; then
  if [ -z "${CMAKE_BACKEND}" ]; then
    echo "Error: ENABLE_CPP=1 but backend '${BACKEND}' does not support C++ extensions"
    exit 1
  fi
  export CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=${CMAKE_BACKEND}"
  printf "C++ extensions: ON (${CMAKE_BACKEND})"
  ok
else
  printf "C++ extensions: OFF"
  ok
fi

# ── Install FlagGems ──────────────────────────────────────────
# Use --no-build-isolation so the build process reuses the build tools
# already installed in the current venv.
printf "Installing FlagGems [${BACKEND}] ..."
uv pip install --no-build-isolation ".[${BACKEND}]" \
  --default-index "${FLAGOS_PYPI}" \
  --index "${MIRROR}" \
  || fail
ok

# ── Compiler selection ───────────────────────────────────────
# COMPILER controls which Triton-compatible compiler to use:
#   COMPILER=flagtree → use FlagTree (default when available)
#   COMPILER=triton   → use vendor Triton
#   unset             → auto: FlagTree if available, otherwise Triton
COMPILER="${COMPILER:-}"

if [ -z "${COMPILER}" ]; then
  if [ -n "${FLAGTREE_PKGS}" ]; then
    COMPILER=flagtree
  else
    COMPILER=triton
  fi
fi

if [ "${COMPILER}" = "flagtree" ]; then
  if [ -n "${FLAGTREE_PKGS}" ]; then
    # FlagTree installs into site-packages/triton, so any existing
    # triton-prefixed packages must be removed first.
    TRITON_INSTALLED=$(uv pip list 2>/dev/null | awk '{print $1}' | grep -i '^triton' || true)
    if [ -n "${TRITON_INSTALLED}" ]; then
      printf "Replacing Triton with FlagTree ..."
      echo "${TRITON_INSTALLED}" | xargs uv pip uninstall -q 2>/dev/null || true
      ok
    fi
    printf "Installing FlagTree ..."
    uv pip install -q ${FLAGTREE_PKGS} --default-index "${FLAGOS_PYPI}" || fail
    ok
  else
    printf "FlagTree not available for ${BACKEND}, using Triton\n"
    COMPILER=triton
  fi
fi

if [ "${COMPILER}" = "triton" ] && [ -n "${TRITON_PKGS}" ]; then
  printf "Installing Triton ..."
  uv pip install -q ${TRITON_PKGS} --default-index "${FLAGOS_PYPI}" || fail
  ok
fi

if [ "${COMPILER}" != "flagtree" ] && [ "${COMPILER}" != "triton" ]; then
  echo "Error: unknown COMPILER value '${COMPILER}' (expected 'flagtree' or 'triton')"
  exit 1
fi

# ── Vendor-specific post-install ──────────────────────────────
if [ -n "${POST_INSTALL}" ]; then
  for pkg in ${POST_INSTALL}; do
    printf "Post-install: ${pkg} ..."
    uv pip install -q "${pkg}" --default-index "${FLAGOS_PYPI}" || fail
    ok
  done
fi

if [ -n "${POST_UNINSTALL}" ]; then
  for pkg in ${POST_UNINSTALL}; do
    printf "Post-uninstall: ${pkg} ..."
    uv pip uninstall -q "${pkg}" 2>/dev/null || true
    ok
  done
fi

# ── Install test dependencies ─────────────────────────────────
printf "Installing test dependencies ..."
uv pip install -q ".[test]" --index "${MIRROR}" || fail
ok

printf "\n${GREEN}FlagGems setup complete for ${BACKEND}${NC}\n"
