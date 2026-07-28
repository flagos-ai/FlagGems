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
if [[ $SKIP_CLEAR_CACHE -eq 0 ]]; then
  echo "[wgrad-preflight] cleared wgrad JIT cache"
else
  echo "[wgrad-preflight] skipped cache clear"
fi
