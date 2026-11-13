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

"""rsub_scalar 候选 v1：手写 1D kernel，按 shape 分档。

生产（公共 + gcu400）走 pointwise_dynamic rsub_func_tensor_scalar:
    return y - x * alpha   (x=张量 A, y=标量 B, alpha=标量)
ENFLAME max_grid_size=(12,1,1) 让大 shape 并行度不足。
沿用 square/square_out/clamp_max_ 已验证的分档策略。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rsub_scalar_kernel(
    X, Out, N, B: tl.constexpr, ALPHA: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0).to(tl.float32)
    r = B - x * ALPHA
    tl.store(Out + off, r, mask=m)


def rsub_scalar(A, B, alpha=1):
    n = A.numel()
    # 只有非连续输入才回退；小 shape 用手写小 BLOCK kernel，
    # 避免「回退公共实现」在生产路径多一层 Python 转发开销。
    if not A.is_contiguous():
        from flag_gems.ops.rsub import rsub_scalar as _public

        return _public(A, B, alpha)
    out = torch.empty_like(A)
    if n < 65536:
        BLOCK, nw = 1024, 2
    elif n < (1 << 24):
        BLOCK, nw = 16384, 2
    else:
        BLOCK, nw = 32768, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _rsub_scalar_kernel[grid](
        A, out, n, float(B), float(alpha), BLOCK=BLOCK, num_warps=nw
    )
    return out
