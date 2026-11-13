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

"""square_ 候选 v2：inplace 1D 手写 kernel。

生产 square_ 是 `_kernel_for(A.dtype)(A, out0=A)`：fp16/bf16 升 fp32 相乘
再回写窄 dtype，fp32 直接相乘。pointwise_dynamic 的 ENFLAME
max_grid_size=(12,1,1) 让大 shape 并行度不足（实测约 6ms vs torch 0.76ms）。
三档 BLOCK 沿用 square/square_out 已验证的策略。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _square_inplace_kernel(X, N, BLOCK: tl.constexpr, HP: tl.constexpr):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = out_offset < N
    x = tl.load(X + out_offset, mask=in_range, other=0)
    if HP:
        x_wide = x.to(tl.float32)
        result = x_wide * x_wide
    else:
        result = x * x
    tl.store(X + out_offset, result, mask=in_range)


def square_(A):
    n = A.numel()
    if n < 65536 or not A.is_contiguous():
        # 小 shape 是 launch 开销下限，复用生产同一底层调用保证严格 1.000x
        from flag_gems.ops.square import _kernel_for

        _kernel_for(A.dtype)(A, out0=A)
        return A
    wide_product = A.dtype in (torch.float16, torch.bfloat16)
    if n < (1 << 24):
        BLOCK, num_warps = 16384, 2
    else:
        BLOCK, num_warps = 32768, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _square_inplace_kernel[grid](
        A, n, BLOCK=BLOCK, HP=wide_product, num_warps=num_warps
    )
    return A
