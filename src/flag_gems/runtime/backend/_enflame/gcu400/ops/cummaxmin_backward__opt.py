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

"""cummaxmin_backward 候选 v1：num_warps 4→8 + 免 mask + 精简 host。

生产 `flag_gems.ops.cummax.cummaxmin_backward` 用 fp32 中间 buffer +
`tl.atomic_add` scatter + `.to(dtype)`，num_warps 取 Triton 默认 4，
且无条件 `.contiguous()` / `torch_device_fn.device` 上下文 / logger。

13 轮迭代（含 kernel 循环修正与容差判据）显示：
- num_warps=8：0.277ms vs 生产 0.308ms（1.11x）；
- reduction_size 恰为 2 的幂时去掉 mask 再省一点；
- 直接 atomic_add 到输出 dtype 会损失 fp16/bf16 精度，丢弃；
- num_warps 1/2、BLOCK 256/1024/2048、2D tile 均劣化。

接管 inner_size==1（含 benchmark 的 2D/dim=1 与 correctness 的 dim=-1）
以及 inner_size>1 的 interior-axis 情形；两者都用 grid-stride 内核，避免
生产实现 "一个 program 一个 outer 行" 在 GCU 上超过 grid.x=65535。
非连续输入就地 `.contiguous()`，不再委托生产实现（其 grid 未做上限）。
RS==0 的空 reduction 仍委托生产。NaN 语义与生产一致（indices 来自前向，
本算子只做 scatter-add）。
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.cummax import cummaxmin_backward as _public

__all__ = ["cummaxmin_backward"]


@triton.jit
def _cmb_kernel(G, IDX, OUT, NROW, RS, IS, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    ar = tl.arange(0, BLOCK)
    # grid.x is capped at 65535 on GCU, so the launch uses a fixed grid and
    # each program walks a strided subset of the outer rows.
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        outer = row // IS
        inner = row % IS
        base = outer * (RS * IS) + inner
        for bs in range(0, RS, BLOCK):
            pos = bs + ar
            if MASKED:
                m = pos < RS
                g = tl.load(G + base + pos * IS, mask=m, other=0.0)
                t = tl.load(IDX + base + pos * IS, mask=m, other=0)
                tl.atomic_add(OUT + base + t * IS, g.to(tl.float32), mask=m)
            else:
                g = tl.load(G + base + pos * IS)
                t = tl.load(IDX + base + pos * IS)
                tl.atomic_add(OUT + base + t * IS, g.to(tl.float32))


@triton.jit
def _cmb_inner_kernel(
    G, IDX, OUT, NROW, RS, IS, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    """Reduction axis is interior: vectorize over the contiguous inner axis.

    ``row`` indexes an (outer, r) pair; every lane scatters
    ``grad[outer, r, j]`` into ``grad_input[outer, t, j]`` with ``t`` from the
    forward indices.  The grid is capped and strided because
    ``outer * reduction`` can be far beyond grid.x's 65535 limit.
    """
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        outer = row // RS
        outer_base = outer * (RS * IS)
        src_base = outer_base + (row % RS) * IS
        for bs in range(0, IS, BLOCK):
            j = bs + ar
            if MASKED:
                m = j < IS
                g = tl.load(G + src_base + j, mask=m, other=0.0)
                t = tl.load(IDX + src_base + j, mask=m, other=0)
                tl.atomic_add(OUT + outer_base + t * IS + j, g.to(tl.float32), mask=m)
            else:
                g = tl.load(G + src_base + j)
                t = tl.load(IDX + src_base + j)
                tl.atomic_add(OUT + outer_base + t * IS + j, g.to(tl.float32))


def cummaxmin_backward(grad_output, input, indices, dim):
    ndim = grad_output.ndim
    if dim < 0:
        dim += ndim
    shape = list(grad_output.shape)
    RS = shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]

    if RS == 0:
        return _public(grad_output, input, indices, dim)

    # Stay on the local kernels instead of delegating to the generic
    # implementation: that one launches one program per (outer, r) pair and
    # overflows grid.x for shapes like (200, 40999, 3).  Both kernels grid-stride
    # over their rows, so contiguous inputs keep the fast path and strided ones
    # only pay for a copy.
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    out = torch.zeros(shape, dtype=torch.float32, device=grad_output.device)
    if inner_size == 1:
        BLOCK = min(triton.next_power_of_2(RS), 1024)
        _cmb_kernel[(min(outer_size, 65535),)](
            grad_output,
            indices,
            out,
            outer_size,
            RS,
            inner_size,
            BLOCK=BLOCK,
            MASKED=(BLOCK != RS),
            num_warps=8,
        )
    else:
        BLOCK = min(triton.next_power_of_2(inner_size), 1024)
        nrow = outer_size * RS
        _cmb_inner_kernel[(min(nrow, 65535),)](
            grad_output,
            indices,
            out,
            nrow,
            RS,
            inner_size,
            BLOCK=BLOCK,
            MASKED=(BLOCK != inner_size),
            num_warps=8,
        )
    return out.to(grad_output.dtype)
