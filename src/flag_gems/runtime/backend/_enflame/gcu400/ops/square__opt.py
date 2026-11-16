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

"""square_ candidate v2: in-place 1D hand-written kernel.

Production square_ is `_kernel_for(A.dtype)(A, out0=A)`: fp16/bf16 promote to fp32, multiply
then write back to the narrow dtype; fp32 multiplies directly. pointwise_dynamic's ENFLAME
max_grid_size=(12,1,1) leaves large shapes under-parallelized (measured ~6ms vs torch 0.76ms).
Three BLOCK tiers reuse the strategy validated by square/square_out.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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
    logger.debug("GEMS_ENFLAME_GCU400 SQUARE_")
    n = A.numel()
    if n < 65536 or not A.is_contiguous():
        # Small shapes are at the launch overhead floor, so reuse production's exact call for strict 1.000x
        from flag_gems.ops.square import _kernel_for

        _kernel_for(A.dtype)(A, out0=A)
        return A
    wide_product = A.dtype in (torch.float16, torch.bfloat16)
    if n < (1 << 24):
        BLOCK, num_warps = 16384, 2
    else:
        BLOCK, num_warps = 65536, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _square_inplace_kernel[grid](
        A, n, BLOCK=BLOCK, HP=wide_product, num_warps=num_warps
    )
    return A
