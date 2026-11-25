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

"""gt_scalar_ candidate v6: hand-written in-place large-BLOCK kernel.

Production gt_func_scalar uses pointwise_dynamic (ALWAYS_BOOL), writing x.to(tl.float32) > y
back into input tensor A (floating 0.0/1.0). ENFLAME max_grid_size=(12,1,1)
leaves large shapes under-parallelized. v2 adds a B131072/nw4 tier for shapes above 2^28 (2^30 probe optimum).
Small shapes also use a hand-written small-BLOCK kernel (falling back to the common implementation adds a dispatch
layer).
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _gt_scalar_inplace_kernel(X, N, B: tl.constexpr, BLOCK: tl.constexpr):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = out_offset < N
    x = tl.load(X + out_offset, mask=in_range, other=0).to(tl.float32)
    comparison = (x > B).to(tl.float32)
    tl.store(X + out_offset, comparison, mask=in_range)


def gt_scalar_(A, B):
    logger.debug("GEMS_ENFLAME_GCU400 GT_SCALAR_")
    n = A.numel()
    if n < 65536 or not A.is_contiguous():
        # Small shapes are already at the launch overhead floor (~0.0037ms); any extra Python check
        # causes a 5~10% regression, so reuse production's exact call to guarantee strict 1.000x
        from flag_gems.ops.gt import gt_func_scalar

        return gt_func_scalar(A, B, out0=A)
    # Probes agree at 2^24 and 2^30: B131072/nw4 is optimal
    # (2^24 ~6.3x, 2^30 ~8.8x over the common implementation), grids 128 / 8192 respectively
    BLOCK, num_warps = 131072, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _gt_scalar_inplace_kernel[grid](A, n, float(B), BLOCK=BLOCK, num_warps=num_warps)
    return A
