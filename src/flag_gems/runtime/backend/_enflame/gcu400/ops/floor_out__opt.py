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

"""floor_out candidate v1: hand-written 1D kernel, large shapes use B131072/nw4.

Production floor_func uses pointwise_dynamic: tl.floor(x.to(tl.float32)).to(x.dtype),
and ENFLAME max_grid_size=(12,1,1) leaves large shapes under-parallelized.
On gt_scalar_, B131072/nw4 was verified clearly better than B32768/nw4 at the 2^24/2^30 tiers;
microsecond small shapes reuse production's same underlying call to avoid regressions from extra Python checks.
"""

import logging

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _floor_out_kernel(X, Out, N, BLOCK: tl.constexpr):
    out_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = out_offset < N
    x = tl.load(X + out_offset, mask=in_range, other=0)
    result = tl.floor(x.to(tl.float32)).to(x.dtype)
    tl.store(Out + out_offset, result, mask=in_range)


def floor_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME_GCU400 FLOOR_OUT")
    if out is None:
        from flag_gems.ops.floor import floor_out as _public

        return _public(A, out=None)
    n = A.numel()
    if n < 65536 or not A.is_contiguous() or not out.is_contiguous():
        from flag_gems.ops.floor import floor_func

        floor_func(A, out0=out)
        return out
    BLOCK, num_warps = 131072, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 262144
        grid = (triton.cdiv(n, BLOCK),)
    _floor_out_kernel[grid](A, out, n, BLOCK=BLOCK, num_warps=num_warps)
    return out
