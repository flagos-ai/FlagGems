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

# Usage:
#   source tools/wgrad_preflight.sh
#   source tools/wgrad_preflight.sh --skip-clear-cache
#
# Why "source":
#   We need exports (CUDA_VISIBLE_DEVICES / PYTHONPATH) to persist in caller shell.

set -euo pipefail

SKIP_CLEAR_CACHE=0
if [[ "${1:-}" == "--skip-clear-cache" ]]; then
  SKIP_CLEAR_CACHE=1
fi

REPO_ROOT="/workspace/FlagGems"
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[wgrad-preflight] error: expected repo at $REPO_ROOT"
  return 1 2>/dev/null || exit 1
fi

cd "$REPO_ROOT"

# Pin to GPU4 to avoid accidentally occupying other cards.
export CUDA_VISIBLE_DEVICES=4
# Only add FlagGems src path; do not inject full dist-packages.
export PYTHONPATH="$REPO_ROOT/src"
# Optional strict mode: refuse silent Torch fallback (recommended for bench).
# export FLAGGEMS_WGRAD_REQUIRE_GEMMEX=1
# Optional: consecutive GemmEx runtime failures before permanent fallback.
# export FLAGGEMS_WGRAD_GEMMEX_FAIL_LIMIT=3
# Optional: reject non-contiguous main_grad (avoids silent densify+copy cost).
# export FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD=1

CSRC_CPP="$REPO_ROOT/src/flag_gems/csrc/wgrad_gemm_accum.cpp"
CSRC_HDR="$REPO_ROOT/src/flag_gems/csrc/wgrad_gemm_accum_kernel.h"

if [[ $SKIP_CLEAR_CACHE -eq 0 ]]; then
  # Clear only wgrad-related Torch extension cache to avoid stale JIT binaries.
  if [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]]; then
    rm -rf "${TORCH_EXTENSIONS_DIR}/flag_gems_wgrad_gemm_accum" 2>/dev/null || true
  fi
  rm -rf "$HOME/.cache/torch_extensions/flag_gems_wgrad_gemm_accum" 2>/dev/null || true
  rm -rf /tmp/torch_extensions/flag_gems_wgrad_gemm_accum 2>/dev/null || true
fi

echo "[wgrad-preflight] cwd=$PWD"
echo "[wgrad-preflight] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[wgrad-preflight] PYTHONPATH=$PYTHONPATH"
echo "[wgrad-preflight] FLAGGEMS_WGRAD_REQUIRE_GEMMEX=${FLAGGEMS_WGRAD_REQUIRE_GEMMEX:-0}"
echo "[wgrad-preflight] FLAGGEMS_WGRAD_GEMMEX_FAIL_LIMIT=${FLAGGEMS_WGRAD_GEMMEX_FAIL_LIMIT:-3}"
echo "[wgrad-preflight] FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD=${FLAGGEMS_WGRAD_REQUIRE_CONTIGUOUS_MAIN_GRAD:-0}"
echo "[wgrad-preflight] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-<default ~/.cache/torch_extensions>}"
echo "[wgrad-preflight] CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-<unset>}}"
if command -v nvcc >/dev/null 2>&1; then
  echo "[wgrad-preflight] nvcc=$(command -v nvcc)"
else
  echo "[wgrad-preflight] nvcc=<not on PATH> (JIT GemmEx may fall back to Torch)"
fi
if [[ -f "$CSRC_CPP" && -f "$CSRC_HDR" ]]; then
  echo "[wgrad-preflight] csrc OK: $CSRC_CPP"
else
  echo "[wgrad-preflight] WARN: missing package-data sources:"
  echo "[wgrad-preflight]   cpp=$CSRC_CPP present=$([[ -f $CSRC_CPP ]] && echo yes || echo NO)"
  echo "[wgrad-preflight]   hdr=$CSRC_HDR present=$([[ -f $CSRC_HDR ]] && echo yes || echo NO)"
fi
if [[ $SKIP_CLEAR_CACHE -eq 0 ]]; then
  echo "[wgrad-preflight] cleared wgrad JIT cache"
else
  echo "[wgrad-preflight] skipped cache clear"
fi

# Optional one-liner diag (ignore failure if import path broken).
python - <<'PY' 2>/dev/null || true
from flag_gems.ops.wgrad_gemm_accum import wgrad_gemmex_diag
d = wgrad_gemmex_diag()
print(
    "[wgrad-preflight] diag:",
    f"csrc_cpp_present={d['csrc_cpp_present']}",
    f"csrc_header_present={d['csrc_header_present']}",
    f"nvcc={d['nvcc']}",
    f"backend={d['backend']}",
    f"load_error={d['load_error']}",
)
PY
