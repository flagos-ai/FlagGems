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

"""C05 multi-tier candidate v3: bitwise_or_scalar_ grid-stride-loop.

Tiered by dtype + element count; configs come from the 12-tier sweep in t4b_bor_tune.py:
- n <= 8192                 -> fall back to original pointwise_dynamic (hand-written kernel measured worse)
- bool and n <= 2^20        -> fall back (this tier degrades GSL by 0.906x)
- int16 n > 2^16           -> BLOCK=16384 / SIPS=48 / nw4
- int16 n <= 2^16          -> BLOCK=8192  / SIPS=48 / nw4
- int32 n <= 2^16          -> BLOCK=8192  / SIPS=48 / nw8
- others (incl. large int32) -> BLOCK=8192 / SIPS=48 / nw4
- non-contiguous input       -> fall back to the original implementation
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _bitwise_or_gsl(X, N, S: tl.constexpr, BLOCK: tl.constexpr, SIPS: tl.constexpr):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK)
    for t in range(pid, n_tiles, SIPS):
        off = t * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        x = tl.load(X + off, mask=m, other=0)
        tl.store(X + off, x | S, mask=m)


def _orig(A, B):
    from flag_gems.runtime.backend._enflame.gcu400.ops.bitwise_or import (
        bitwise_or_scalar_ as fn,
    )

    return fn(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_SCALAR_ GSL_V3")
    n = A.numel()
    if not A.is_contiguous() or n <= 8192:
        return _orig(A, B)
    dt = A.dtype
    if dt == torch.bool:
        if n <= (1 << 22):
            return _orig(A, B)
        block, sips, nw = 16384, 48, 4
    elif dt == torch.int16:
        block, sips, nw = (16384, 48, 4) if n > (1 << 16) else (8192, 48, 4)
    elif n <= (1 << 16):
        block, sips, nw = 8192, 48, 8
    else:
        block, sips, nw = 8192, 48, 4
    _bitwise_or_gsl[(sips,)](A, n, int(B), BLOCK=block, SIPS=sips, num_warps=nw)
    return A
