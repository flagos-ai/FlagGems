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

"""softmax_out 候选 v1：K>1（内层归约维）路径按算子覆盖 constexpr。

生产 gcu400 `softmax_out`（11-03 TRACKED 深度优化实现）对 `K>1`
委托公共 `flag_gems.ops.softmax.softmax_out`，其
`softmax_kernel_non_inner` 用全局 heuristic：

    TILE_K 贪心取大（waves>1 就翻倍，上限 min(K,8192)）
    TILE_N = cdiv(8192, TILE_K)
    num_warps = 4

在 benchmark 的 3D case 上 TILE_K 被推到 512/1024，导致
TILE_N=16/8，每个 CTA 沿 N 串行循环 32/128 次：

    (64,512,512)      fp16 0.247ms  (torch 0.075ms)
    (1024,1024,1024)  fp16 12.16ms (torch 5.35ms)

81 组 + 14 组 constexpr 扫描（与 torch.softmax 容差比对）显示
`TILE_K=256 / TILE_N=64 / num_warps=2 / ONE_TILE_PER_CTA=False`
是全局最优且普适：

    (64,512,512)      fp16/bf16 1.46x、fp32 1.27x
    (1024,1024,1024)  fp16 1.37x（生产隐含配置实测 1.00x）

注意 `num_warps=2` 是收益关键：`nw4` 同一 tile 配置只有 0.96x；
`TILE_K<256` 或 `TILE_N>128` 普遍劣化，`TILE_N=256` 触发
OutOfResources(dsm memory)。

实现：复用公共 kernel 的原始 JITFunction（`LibEntry.fn.fn`）
显式传 constexpr，不修改 TRACKED 的 softmax.py、不动全局
HEURISTICS_CONFIGS。

**只接管最窄的安全情形**：浮点、连续、非 half_to_float、
out 连续且 dtype 一致、K>1 且 K>=256、N>=64；其余
（空张量、N==1、K==1、K<256、N<64、整型、half_to_float、
非连续输入/out、负 dim 之外的越界 dim、out 与 self 别名）
全部委托生产 `softmax_out`，保证 correctness 全覆盖。
"""

import triton

from flag_gems.ops.softmax import softmax_kernel_non_inner as _kernel_wrapped
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.softmax import (
    softmax_out as _prod_softmax_out,
)

_K = _kernel_wrapped.fn.fn

__all__ = ["softmax_out"]


def softmax_out(self, dim, half_to_float=False, *, out):
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    if self.numel() == 0:
        return _prod_softmax_out(self, dim, half_to_float, out=out)

    d = dim % self.ndim
    M = 1
    N = self.shape[d]
    for i in range(d):
        M *= self.shape[i]
    Kv = self.numel() // M // N

    if not (
        self.is_floating_point()
        and self.is_contiguous()
        and out.is_contiguous()
        and not half_to_float
        and out.dtype == self.dtype
        and out.data_ptr() != self.data_ptr()
        and Kv > 1
        and Kv >= 256
        and N >= 64
    ):
        return _prod_softmax_out(self, dim, half_to_float, out=out)

    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)

    TILE_K, TILE_N, nw = 256, 64, 2
    grid = (M, triton.cdiv(Kv, TILE_K), 1)
    with torch_device_fn.device(self.device):
        _K[grid](
            out,
            self,
            M,
            N,
            Kv,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            ONE_TILE_PER_CTA=(TILE_N >= N),
            num_warps=nw,
        )
    return out
