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

eval $(python3 -c "
import yaml, sys, json

cfg = yaml.safe_load(open('${BACKENDS_YAML}'))
key = '${BACKEND}'

if key not in cfg['backends']:
    avail = ', '.join(cfg['backends'].keys())
    print(f'echo \"Unknown backend: {key}\"; echo \"Available: {avail}\"; exit 1')
    sys.exit()

b = cfg['backends'][key]
vendor = key.rsplit('-', 1)[0] if '-' in key else key
index_url = b.get('index') or cfg['pypi_base'].format(vendor=vendor)

print(f'PYTHON_VERSION={b[\"python\"]}')
print(f'VENDOR={vendor}')
print(f'FLAGOS_PYPI={index_url}')
print(f'MIRROR={cfg[\"mirror\"]}')

# Encode deps and post_install as space-separated strings
deps = ' '.join(b.get('deps', []))
post = ' '.join(b.get('post_install', []))
print(f'BACKEND_DEPS=\"{deps}\"')
print(f'POST_INSTALL=\"{post}\"')
")

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
  --index "${MIRROR}" \
  || fail
ok

# ── Install FlagGems ──────────────────────────────────────────
printf "Installing FlagGems [${BACKEND}] ..."
uv pip install ".[${BACKEND}]" \
  --default-index "${FLAGOS_PYPI}" \
  --index "${MIRROR}" \
  || fail
ok

# ── Vendor-specific post-install ──────────────────────────────
if [ -n "${POST_INSTALL}" ]; then
  for pkg in ${POST_INSTALL}; do
    printf "Post-install: ${pkg} ..."
    uv pip install -q "${pkg}" --default-index "${FLAGOS_PYPI}" || fail
    ok
  done
fi

# Kunlunxin-specific cleanup
if [ "$BACKEND" = "kunlunxin" ]; then
  uv pip uninstall -q pytest-repeat 2>/dev/null || true
fi

# ── Install test dependencies ─────────────────────────────────
printf "Installing test dependencies ..."
uv pip install -q ".[test]" --index "${MIRROR}" || fail
ok

printf "\n${GREEN}FlagGems setup complete for ${BACKEND}${NC}\n"
