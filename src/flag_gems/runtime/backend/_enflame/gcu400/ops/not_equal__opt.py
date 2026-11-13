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

"""not_equal 候选 v1：手写 flat 单调 grid kernel（替换 grid=12 的 pointwise_dynamic 路径）。

根因（2026-11-13 探针）：生产 not_equal 走通用 flag_gems.utils.pointwise_dynamic，
其 ENFLAME 档 CodeGenConfig 为 max_grid_size=(12,1,1) + max_tile_size=4096 ——
全设备 48 SP 仅起 12 个 CTA、每 CTA 21846 个 tile 的 GSL 长循环，1G fp16 实测
13.19ms（有效带宽 379GB/s，torch 原生 1.92ms = 2.6TB/s）。

本文件改为手写 1D flat kernel（pid*BLOCK+arange、int32 偏移、原生 dtype 比较）：
- 语义等价：fp16/bf16->fp32 为单射，!= 结果与 ALWAYS_BOOL 提升后逐位一致；
  int8/16/32 原生比较反而修复了 fp32 cast 在 |x|>2^24 时的精度缺陷。
- 分档（探针实测，GCU400/libra，do_bench w1000/r100 median）：
  * n <= 8192        : BLOCK=next_pow2(n)（16..8192）、nw4、单 CTA（launch 开销主导）
  * n <= 2^18        : BLOCK=8192、nw4（2^16 实测 4.03us vs pd 4.70us）
  * n <= 2^20        : BLOCK=clamp(131072//es, 8192, 65536)、nw4（2^20 7.29us vs pd 16.07us）
  * 其余             : 同 BLOCK、nw2（2^30 fp16 1.95ms vs pd 13.18ms = 6.7x；fp32
                        BLOCK=32768 3.86ms vs pd 13.02ms = 3.4x，DSM 甜点 2 tile ~=256KB）
  * n % BLOCK == 0   : EVEN 无 mask 快径（2^30 再省 ~1.7%）
- 非法形态（非连续/广播/异 dtype/es>4/complex/fp8/n>=2^31/grid>65535）回退
  生产 not_equal_func（pointwise_dynamic），保证零行为回归。
- DSM 预算：峰值 = 2*BLOCK*es（输入）+ BLOCK*1B（bool 输出）<= 320KB < 448KB/program。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device

from .not_equal import not_equal_func

logger = logging.getLogger(__name__)
device = device.name

_FAST_DTYPES = {
    dt
    for dt in (
        getattr(torch, name, None)
        for name in (
            "bool",
            "uint8",
            "int8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "float16",
            "bfloat16",
            "float32",
        )
    )
    if dt is not None
}

_INT32_LIMIT = 0x7FFFFFFF
_GRID_LIMIT = 65535


@triton.jit(do_not_specialize=["N"])
def _not_equal_flat_kernel(X, Y, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        tl.store(Out + offs, x != y)
    else:
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        tl.store(Out + offs, x != y, mask=mask)


def not_equal(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS_ENFLAME NOT_EQUAL")
    if (
        A.dtype is B.dtype
        and A.dtype in _FAST_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        n = A.numel()
        if n == 0:
            return torch.empty_like(A, dtype=torch.bool)
        if n <= 8192:
            block = max(16, 1 << (n - 1).bit_length())
            num_warps = 4
        else:
            es = A.element_size()
            block = max(8192, min(65536, 131072 // es))
            num_warps = 4 if n <= (1 << 20) else 2
        grid = (triton.cdiv(n, block),)
        if n < _INT32_LIMIT and grid[0] <= _GRID_LIMIT:
            out = torch.empty_like(A, dtype=torch.bool)
            _not_equal_flat_kernel[grid](
                A,
                B,
                out,
                n,
                BLOCK=block,
                EVEN=(n % block == 0),
                num_warps=num_warps,
            )
            return out
    return not_equal_func(A, B)
