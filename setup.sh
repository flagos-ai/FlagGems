#!/bin/bash
set -euo pipefail

SUPPORTED_VENDORS=(
  "ascend-cann850"
  "ascend-cann900"
  "cambricon"
  "enflame"
  "hygon"
  "iluvatar"
  "kunlunxin"
  "metax"
  "mthreads"
  "nvidia"
  "spacemit"
  "sunrise"
  "thead"
  "tsingmicro"
)

declare -A PYTHON_SUPPORTED=(
  ["ascend-cann850"]="3.11"
  ["ascend-cann900"]="3.11"
  ["cambricon"]="3.10"
  ["enflame"]="3.12"
  ["hygon"]="3.10"
  ["iluvatar"]="3.10"
  ["kunlunxin"]="3.10"
  ["metax"]="3.12"
  ["mthreads"]="3.10"
  ["nvidia"]="3.12"
  ["spacemit"]="3.12"
  ["sunrise"]="3.10"
  ["thead"]="3.12"
  ["tsingmicro"]="3.10"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok()   { printf " ${GREEN}[OK]${NC}\n"; }
fail() { printf " ${RED}[FAILED]${NC}\n"; exit 1; }

valid_vendor() {
  local needle=$1
  for item in "${SUPPORTED_VENDORS[@]}"; do
    [ "$item" == "$needle" ] && return 0
  done
  return 1
}

# ── Validate argument ─────────────────────────────────────────
[ "$#" -eq 1 ] || { echo "Usage: $0 <VENDOR>"; exit 1; }

VENDOR=${1}
valid_vendor "$VENDOR" || {
  echo "Invalid vendor '${VENDOR}'"
  echo "Supported: ${SUPPORTED_VENDORS[*]}"
  exit 1
}
printf "Vendor: ${VENDOR}"
ok

PYTHON_VERSION=${PYTHON_SUPPORTED[$VENDOR]}

# ── Detect or install uv ─────────────────────────────────────
UV_VERSION="0.11.22"
UV_MIRROR="https://resource.flagos.net/repository/flagos-filestore/utils"

printf "Checking uv ..."
if command -v uv &>/dev/null; then
  printf " $(uv --version)"
  ok
else
  printf " not found, installing ...\n"
  ARCH=$(uname -m)
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
source tools/env.sh "${VENDOR}"

# ── FLAGOS PyPI index ─────────────────────────────────────────
PYPI_VENDOR=${VENDOR}
if [[ "$VENDOR" == ascend-* ]]; then
  PYPI_VENDOR="ascend"
fi
export FLAGOS_PYPI="https://resource.flagos.net/repository/flagos-pypi-${PYPI_VENDOR}/simple"
ALIYUN_PYPI="https://mirrors.aliyun.com/pypi/simple"

# ── Install build tools ──────────────────────────────────────
printf "Installing build tools ..."
uv pip install -q \
  "setuptools>=64.0" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0" \
  --index "${ALIYUN_PYPI}" \
  || fail
ok

# ── Install FlagGems ──────────────────────────────────────────
printf "Installing FlagGems [${VENDOR}] ..."
uv pip install ".[${VENDOR}]" \
  --default-index "${FLAGOS_PYPI}" \
  --index https://mirrors.aliyun.com/pypi/simple \
  || fail
ok

# ── Vendor-specific post-install ──────────────────────────────
if [ "$VENDOR" = "ascend-cann900" ]; then
  printf "Overriding triton with triton-ascend 3.2.1 ..."
  uv pip install -q "triton-ascend==3.2.1" --index "${FLAGOS_PYPI}" || fail
  ok
fi

if [ "$VENDOR" = "kunlunxin" ]; then
  printf "Replacing triton with Kunlunxin override ..."
  uv pip install -q "triton==3.0.0+a48aedef" --index "${FLAGOS_PYPI}" || fail
  uv pip uninstall -q pytest-repeat 2>/dev/null || true
  ok
fi

# ── Install test dependencies ─────────────────────────────────
printf "Installing test dependencies ..."
uv pip install -q ".[test]" --index "${ALIYUN_PYPI}" || fail
ok

printf "\n${GREEN}FlagGems setup complete for ${VENDOR}${NC}\n"
