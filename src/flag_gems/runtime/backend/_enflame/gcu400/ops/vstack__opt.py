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

"""vstack 候选 v1：大 shape 用 BLOCK=65536 的线性拼接 kernel。

生产 `vstack` 只是 `cat(aligned, dim=0)` 的包装；第 21 项落地的
`cat__opt.try_fast_cat` 用 `BLOCK = 16384 if max_n<65536 else 32768`。
30 组扫描（BLOCK × num_warps × 2D/1D grid，与 torch.cat 逐位一致）显示：
单输入元素量 >= 2^24 时 `BLOCK=65536 / nw4` 稳定优于 32768：

    fp16/bf16 (4096,4096) / (64,512,512)  0.1128 -> 0.0891  1.27x
    fp32      同 shape                     0.1728 -> 0.1577  1.10x

小 shape (64,64) 各配置只有 0.0037~0.0044ms 的微秒级差异（1.05~1.19x，
抖动量级），故**只接管大 shape**，小 shape 与所有边界
（输入数 >4、非连续、dtype/device 不一致、float8、t0.ndim<2、
非 cat 维 shape 不同）全部委托生产 `vstack`，保证 correctness 全覆盖
且不引入 p10 抖动回归。

复用 `cat__opt` 的 `_can_fast_path` 边界判据与 `_linear_cat_kernel`，
不改 cat.py / cat__opt.py（避免影响已闭环的第 21 项 concat 与
第 30 项 cat_out），只按算子选择更大的 BLOCK。
"""

from typing import List, Tuple, Union

import torch
import triton

from flag_gems.runtime.backend._enflame.gcu400.ops.cat import cat as _cat
from flag_gems.runtime.backend._enflame.gcu400.ops.cat__opt import (
    _can_fast_path,
    _linear_cat_kernel,
)

__all__ = ["vstack"]

_LARGE = 1 << 20


def vstack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
) -> torch.Tensor:
    n = len(tensors)
    if n == 0:
        raise RuntimeError("vstack expected a non-empty TensorList")

    t0 = tensors[0]
    if t0.ndim < 2:
        aligned = list(torch.atleast_2d(tensors))
    else:
        aligned = list(tensors) if not isinstance(tensors, list) else tensors

    A = list(aligned)
    # 只接管「大 shape + 严格满足 cat fast path 边界」的情形
    if A[0].numel() < _LARGE or t0.ndim < 2 or not _can_fast_path(A, 0):
        # 与生产 vstack 逐字同构的直接调用，零额外转发开销
        return _cat(aligned, dim=0)

    ns = [A[j].numel() if j < n else 0 for j in range(4)]
    ptrs = [A[j] if j < n else A[0] for j in range(4)]
    offs = [0] * 4
    acc = 0
    for j in range(4):
        offs[j] = acc
        acc += ns[j]
    out_shape = (sum(t.shape[0] for t in A),) + tuple(A[0].shape[1:])
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)
    BLOCK = 65536
    grid = (triton.cdiv(max(ns), BLOCK), n)
    _linear_cat_kernel[grid](out, *ptrs, *ns, *offs, NT=n, BLOCK=BLOCK, num_warps=4)
    return out
