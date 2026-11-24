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

"""rsub_scalar candidate v1: hand-written 1D kernel tiered by shape.

Production (common + gcu400) uses pointwise_dynamic rsub_func_tensor_scalar:
    return y - x * alpha   (x=tensor A, y=scalar B, alpha=scalar)
ENFLAME max_grid_size=(12,1,1) leaves large shapes under-parallelized.
Reuse the tiered strategy validated by square/square_out/clamp_max_.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


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
    logger.debug("GEMS_ENFLAME_GCU400 RSUB_SCALAR")
    n = A.numel()
    # Only non-contiguous inputs fall back; small shapes use a hand-written small-BLOCK kernel,
    # avoiding the extra Python forwarding overhead of "fall back to the common implementation" on the production path.
    if not A.is_contiguous():
        from flag_gems.ops.rsub import rsub_scalar as _public

        return _public(A, B, alpha)
    out = torch.empty_like(A)
    if n < 65536:
        BLOCK, nw = 1024, 2
    elif n < (1 << 24):
        BLOCK, nw = 16384, 2
    else:
        BLOCK, nw = 65536, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _rsub_scalar_kernel[grid](
        A, out, n, float(B), float(alpha), BLOCK=BLOCK, num_warps=nw
    )
    return out
