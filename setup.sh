#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
export PATH="${HOME}/.local/bin:$PATH"
if command -v uv &>/dev/null; then
  printf " $(uv --version)"
  ok
else
  printf " not found, installing ...\n"
  ARCH=$(uname -m)
  mkdir -p "$HOME/.local/bin"
  curl -sSf "${UV_MIRROR}/uv-${ARCH}-${UV_VERSION}-linux-gnu.tar.gz" \
    | tar xz -C "$HOME/.local/bin" 2>/dev/null \
    || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  command -v uv &>/dev/null || { printf "uv installation"; fail; }
  printf "Installed $(uv --version)"
  ok
fi
# Persist PATH for subsequent GitHub Actions steps
[ -n "${GITHUB_PATH:-}" ] && echo "$HOME/.local/bin" >> "$GITHUB_PATH"

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
  "setuptools-scm>=8" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0" \
  "PyYAML==6.0.3" \
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

triton_post = []
for item in b.get('triton_post_install', []):
    if isinstance(item, str):
        triton_post.append(item)
print(f'TRITON_POST_INSTALL=\"{\" \".join(triton_post)}\"')
")

# ── C++ extensions ───────────────────────────────────────────
# Set ENABLE_CPP=1 to build C++ wrapped operators.
# Default: OFF (C++ extensions require vendor SDK and toolchain).
# The main package is pure Python; when ENABLE_CPP=1 the native extension is
# built and installed separately from the cpp/ subdirectory (see below).
if [ "${ENABLE_CPP:-0}" = "1" ]; then
  if [ -z "${CMAKE_BACKEND}" ]; then
    echo "Error: ENABLE_CPP=1 but backend '${BACKEND}' does not support C++ extensions"
    exit 1
  fi
  printf "C++ extensions: ON (${CMAKE_BACKEND})"
  ok
else
  printf "C++ extensions: OFF"
  ok
fi

# ── Install FlagGems ──────────────────────────────────────────
# Use --no-build-isolation so the build process reuses the build tools
# already installed in the current venv.
# Fetch tags and deepen history for setuptools-scm version detection.
# Shallow clones (CI default) lack tag reachability.
git fetch --tags --unshallow --quiet 2>/dev/null \
  || git fetch --tags --depth=500 --quiet 2>/dev/null \
  || git fetch --tags --quiet 2>/dev/null \
  || true
printf "Installing FlagGems [${BACKEND}] ..."
uv pip install --no-build-isolation ".[${BACKEND}]" \
  --default-index "${FLAGOS_PYPI}" \
  --index "${MIRROR}" \
  || fail
ok

# ── Install C++ wrapped operators (optional) ─────────────────
# The native extension is a separate distribution (flag-gems-cpp-<vendor>) that
# installs its .so files into the flag_gems/ namespace. Build it from cpp/ with
# the vendor name injected into cpp/pyproject.toml.
if [ "${ENABLE_CPP:-0}" = "1" ]; then
  vendor_lc="$(echo "${CMAKE_BACKEND}" | tr '[:upper:]' '[:lower:]')"
  tools/set_cpp_vendor.sh "${vendor_lc}" || fail
  printf "Installing FlagGems C++ extensions [${CMAKE_BACKEND}] ..."
  CMAKE_ARGS="-DFLAGGEMS_BUILD_C_EXTENSIONS=ON -DFLAGGEMS_BACKEND=${CMAKE_BACKEND}" \
    uv pip install --no-build-isolation ./cpp \
    --default-index "${FLAGOS_PYPI}" \
    --index "${MIRROR}" \
    || fail
  ok
fi

# ── Compiler selection ───────────────────────────────────────
# COMPILER controls which Triton-compatible compiler to use:
#   COMPILER=flagtree → use FlagTree (error if unavailable)
#   COMPILER=triton   → use vendor Triton
#   unset             → auto: FlagTree if available, otherwise Triton
COMPILER="${COMPILER:-}"

if [ -z "${COMPILER}" ]; then
  if [ -n "${FLAGTREE_PKGS}" ]; then
    COMPILER=flagtree
  else
    printf "WARNING: FlagTree is not available for ${BACKEND}, falling back to Triton.\n"
    COMPILER=triton
  fi
fi

if [ "${COMPILER}" = "flagtree" ]; then
  if [ -n "${FLAGTREE_PKGS}" ]; then
    printf "Installing FlagTree ..."
    uv pip install -q ${FLAGTREE_PKGS} --default-index "${FLAGOS_PYPI}" || fail
    ok
  else
    echo "Error: COMPILER=flagtree but FlagTree is not available for '${BACKEND}'."
    exit 1
  fi
fi

if [ "${COMPILER}" = "triton" ] && [ -n "${TRITON_PKGS}" ]; then
  printf "Installing Triton ..."
  uv pip install -q ${TRITON_PKGS} --default-index "${FLAGOS_PYPI}" || fail
  ok
elif [ "${COMPILER}" = "triton" ] && [ -z "${TRITON_PKGS}" ]; then
  echo "Error: COMPILER=triton but no triton packages configured for '${BACKEND}'"
  exit 1
fi

if [ "${COMPILER}" != "flagtree" ] && [ "${COMPILER}" != "triton" ]; then
  echo "Error: unknown COMPILER value '${COMPILER}' (expected 'flagtree' or 'triton')"
  exit 1
fi

# ── Triton-specific post-install ──────────────────────────────
if [ -n "${TRITON_POST_INSTALL}" ] && [ "${COMPILER}" = "triton" ]; then
  for pkg in ${TRITON_POST_INSTALL}; do
    printf "Triton post-install: ${pkg} ..."
    uv pip install -q "${pkg}" --default-index "${FLAGOS_PYPI}" --index "${MIRROR}" || fail
    ok
  done
fi

# ── Install test dependencies ─────────────────────────────────
printf "Installing test dependencies ..."
uv pip install -q ".[test]" --index "${MIRROR}" || fail
ok

# ── Write env into .venv/bin/activate ────────────────────────
# So that `source .venv/bin/activate` sets up the full environment.
printf "Writing environment to .venv/bin/activate ..."
python3 -c "
import yaml

cfg = yaml.safe_load(open('${BACKENDS_YAML}'))
b = cfg['backends']['${BACKEND}']

lines = []
lines.append('')
lines.append('# --- FlagGems environment (${BACKEND}) ---')

for k, v in b.get('env', {}).items():
    lines.append(f'export {k}={v}')

for script in b.get('env_source', []):
    lines.append(f'[ -f {script} ] && source {script} || true')

lines.append('# --- end FlagGems environment ---')

with open('.venv/bin/activate', 'a') as f:
    f.write('\n'.join(lines) + '\n')
" || fail
ok

printf "\n${GREEN}FlagGems setup complete for ${BACKEND}${NC}\n"
printf "Run: ${GREEN}source .venv/bin/activate${NC}\n"
