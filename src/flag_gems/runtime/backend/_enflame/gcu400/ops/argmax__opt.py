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

"""argmax candidate v2: inner path nw=4 plus fp32/bf16 tile optimization for non_inner.

Both independent paths come from constexpr sweeps (bit-identical to the CPU reference):

1. inner (K==1): the production heuristic gives num_warps=1/2 for M>128, which is underconfigured.
   `G48 / BM8 / TILE_N=npo2(N) / nw4` on (4096,4096) gives
   fp16 1.88x, fp32 2.10x, bf16 2.13x.
2. non_inner (K>1) with fp32/bfloat16 dtype, K>=512 and N>=512:
   production uses TILE_N=128/nw2(or 4); measured `TILE_K=512 / TILE_N=64 /
   nw4 / ONE_TILE_PER_CTA=False` gives 1.47~1.48x on (64,512,512) and
   1.23x on (1024,1024,512). fp16's production config is already optimal (1.000x),
   and larger TILE_N degrades severely (16~18ms), so fp16 always delegates to production.

All other cases (dim=None flat path, keepdim, dtype argument, empty tensors,
non-contiguous input, fp16 non_inner, non_inner below the size threshold)
delegate to production `argmax`, guaranteeing full correctness coverage and unchanged edge semantics.

TRACKED argmax.py is not modified, nor are the global HEURISTICS_CONFIGS:
the original JITFunction is taken from `LibEntry.fn.fn` and passed constexpr explicitly.
Runtime patching of `Heuristics.values` has no effect (LibEntry cache is frozen), measured 1.00x.
"""

import logging
import math

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.argmax import argmax as _prod_argmax
from flag_gems.runtime.backend._enflame.gcu400.ops.argmax import (
    argmax_kernel_inner as _ki_wrapped,
)
from flag_gems.runtime.backend._enflame.gcu400.ops.argmax import (
    argmax_kernel_non_inner as _kn_wrapped,
)

logger = logging.getLogger(__name__)


_KI = _ki_wrapped.fn.fn
_KN = _kn_wrapped.fn.fn

__all__ = ["argmax"]


def _inner_constexpr(M, N):
    TILE_N = triton.next_power_of_2(N) if N <= (32 * 1024) else 4096
    cap = max(1, 32768 // TILE_N)
    BLOCK_M = min(8, cap)
    if M <= 128:
        BLOCK_M = min(BLOCK_M, max(1, triton.next_power_of_2(M // 16)))
    return TILE_N, BLOCK_M


def _out_index(inp, shape_list, dim, keepdim):
    out = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    return out if keepdim else torch.squeeze(out, dim)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME_GCU400 ARGMAX")
    if (
        dim is None
        or dtype is not None
        or keepdim
        or inp.numel() == 0
        or not inp.is_contiguous()
    ):
        return _prod_argmax(inp, dim, keepdim, dtype=dtype)

    shape = inp.shape
    d = dim % inp.ndim
    N = shape[d]
    M = math.prod(shape[:d])
    K = inp.numel() // M // N

    shape_list = list(shape)
    shape_list[d] = 1

    with torch_device_fn.device(inp.device):
        if K == 1:
            out_index = _out_index(inp, shape_list, dim, keepdim)
            TILE_N, BLOCK_M = _inner_constexpr(M, N)
            grid = (min(48, triton.cdiv(M, BLOCK_M)), 1, 1)
            _KI[grid](
                inp,
                out_index,
                M,
                N,
                TILE_N=TILE_N,
                BLOCK_M=BLOCK_M,
                GRID_SIZE=48,
                NUM_STAGES=4,
                num_warps=4,
            )
            return out_index

        if (
            d == 1
            and inp.dtype in (torch.float32, torch.bfloat16)
            and K >= 512
            and N >= 512
        ):
            out_index = _out_index(inp, shape_list, dim, keepdim)
            TILE_K = min(triton.next_power_of_2(K), 512)
            TILE_N = 64
            grid = (min(48, M * triton.cdiv(K, TILE_K)), 1, 1)
            _KN[grid](
                inp,
                out_index,
                M,
                N,
                K,
                TILE_K=TILE_K,
                TILE_N=TILE_N,
                ONE_TILE_PER_CTA=False,
                GRID_SIZE=48,
                NUM_STAGES=4,
                num_warps=4,
            )
            return out_index

        return _prod_argmax(inp, dim, keepdim, dtype=dtype)
