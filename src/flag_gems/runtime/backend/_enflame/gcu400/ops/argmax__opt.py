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

"""argmax 候选 v2：inner 路径 nw=4 + non_inner 的 fp32/bf16 tile 优化。

两条独立路径均来自 constexpr 扫描（与 CPU 参考逐位一致）：

1. inner (K==1)：生产 heuristic 对 M>128 给 num_warps=1/2，实测欠配。
   `G48 / BM8 / TILE_N=npo2(N) / nw4` 在 (4096,4096) 上
   fp16 1.88x、fp32 2.10x、bf16 2.13x。
2. non_inner (K>1) 且 dtype 为 fp32/bfloat16 且 K>=512 且 N>=512：
   生产用 TILE_N=128/nw2(或4)，实测 `TILE_K=512 / TILE_N=64 /
   nw4 / ONE_TILE_PER_CTA=False` 在 (64,512,512) 上 1.47~1.48x、
   (1024,1024,512) 上 1.23x。fp16 的生产配置已是最优（1.000x），
   更大 TILE_N 全部严重劣化（16~18ms），故 fp16 一律委托生产。

其余情况（dim=None 的 flat 路径、keepdim、dtype 参数、空张量、
非连续输入、fp16 的 non_inner、不满足尺寸门槛的 non_inner）
全部委托生产 `argmax`，保证 correctness 全覆盖与边界语义不变。

不修改 TRACKED 的 argmax.py，也不改全局 HEURISTICS_CONFIGS：
直接取 `LibEntry.fn.fn` 原始 JITFunction 显式传 constexpr。
运行时 patch `Heuristics.values` 无效（LibEntry 缓存已固化），实测 1.00x。
"""

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
