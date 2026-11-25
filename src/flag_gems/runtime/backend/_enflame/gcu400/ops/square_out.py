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

"""square_out candidate v2: on top of v1, small shapes fall back to the common implementation.

v1's 3 benchmark rounds: overall 4.15x / fp16 4.32x / fp16_large 4.34x,
but small shapes (64x64) showed slight 0.920~0.987 regressions, triggering the per-case p10>=0.95
gate and FAIL. v2 sends n<65536 straight to common pointwise_dynamic,
removing that regression; everything else keeps v1's large-BLOCK hand-written kernel.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _square_kernel(X, Y, N, BLOCK: tl.constexpr, HP: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0)
    if HP:
        xf = x.to(tl.float32)
        y = xf * xf
    else:
        y = x * x
    tl.store(Y + off, y, mask=m)


def _public(A, out):
    from flag_gems.ops.square import square_out as _impl

    return _impl(A, out=out)


def square_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME_GCU400 SQUARE_OUT")
    if out is None:
        out = torch.empty_like(A)
    n = A.numel()
    # Small shapes and non-contiguous inputs fall back to the common implementation, avoiding any tiering regression /
    # semantic degradation
    if n < 65536 or not A.is_contiguous() or not out.is_contiguous():
        return _public(A, out)
    hp = A.dtype in (torch.float16, torch.bfloat16)
    if n < (1 << 24):
        BLOCK, nw = 16384, 2
    else:
        BLOCK, nw = 65536, 4
    grid = (triton.cdiv(n, BLOCK),)
    if grid[0] > 65535:
        BLOCK = 65536
        grid = (triton.cdiv(n, BLOCK),)
    _square_kernel[grid](A, out, n, BLOCK=BLOCK, HP=hp, num_warps=nw)
    return out
