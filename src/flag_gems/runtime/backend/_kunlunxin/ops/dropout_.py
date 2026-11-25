# Copyright 2026, The FlagOS Contributors.
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

import logging

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

UNROLL = 4


@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset"])
def dropout_inplace_kernel(
    X,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Processes one full ``BLOCK * UNROLL``-element tile with no boundary mask.
    # The host launches exactly ``N // (BLOCK * UNROLL)`` programs, so every
    # load/store address below is guaranteed in-bounds and the compiler keeps
    # the contiguous DMA path.
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    mask0 = r0 > p
    mask1 = r1 > p
    mask2 = r2 > p
    mask3 = r3 > p
    p = 1.0 / (1.0 - p)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    x0 = tl.load(X + off_0)
    x1 = tl.load(X + off_1)
    x2 = tl.load(X + off_2)
    x3 = tl.load(X + off_3)

    y0 = x0 * p * mask0
    y1 = x1 * p * mask1
    y2 = x2 * p * mask2
    y3 = x3 * p * mask3

    tl.store(X + off_0, y0)
    tl.store(X + off_1, y1)
    tl.store(X + off_2, y2)
    tl.store(X + off_3, y3)


@triton.jit(do_not_specialize=["p", "philox_seed", "philox_offset", "start"])
def dropout_inplace_tail_kernel(
    X,
    N,
    start,
    p,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Handles the remaining ``[start, N)`` elements with a single masked tile.
    # ``philox_offset`` is advanced past the full-tile kernel's counters so the
    # random stream stays continuous.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)

    mask = r0 > p
    p = 1.0 / (1.0 - p)

    off = start + i4
    x = tl.load(X + off, mask=off < N, other=0.0)
    y = x * p * mask
    tl.store(X + off, y, mask=off < N)


def dropout_(input, p=0.5, train=True):
    """In-place dropout (aten::dropout_) on kunlunxin / XPU.

    Same semantics as the generic op, but the launch uses precomputed tuple
    grids instead of a ``lambda meta: ...`` callable grid.  On XPU a callable
    grid makes triton re-specialize the launch on every call (~190ms/launch),
    so BLOCK/num_warps are computed here and passed explicitly.

    The bulk is handled by a full-tile ``UNROLL=4`` kernel (four contiguous
    loads/stores per program, no boundary mask) and the remainder by a single
    masked-tile kernel.  This keeps the fast unrolled path for the bulk while
    avoiding the XPU tail-store bug where fully out-of-range sibling loads
    cause the partially-masked tail store to be dropped.
    """
    logger.debug("GEMS DROPOUT_")
    if not train or p == 0.0:
        return input
    if p == 1.0:
        return input.zero_()
    assert p > 0.0 and p < 1.0, "p must be in [0, 1]"
    assert input.is_contiguous(), "dropout_ requires a contiguous tensor"
    device = input.device
    N = input.numel()

    # Mirror the generic dropout heuristic (heuristics_config_utils.dropout_heur_*).
    block = 512 if N <= 512 else 1024
    if N <= 512:
        num_warps = 4
    elif N <= 1024:
        num_warps = 8
    else:
        num_warps = 16

    tile = block * UNROLL
    n_full = N // tile
    rem = N - n_full * tile
    n_tail = triton.cdiv(rem, block) if rem > 0 else 0

    # Advance the Philox state by the exact number of counter draws consumed by
    # the full-tile and tail kernels so successive calls never reuse a stream.
    increment = n_full * block + n_tail * block
    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        if n_full > 0:
            dropout_inplace_kernel[(n_full,)](
                input, p, philox_seed, philox_offset, BLOCK=block, num_warps=num_warps
            )
        if n_tail > 0:
            dropout_inplace_tail_kernel[(n_tail,)](
                input,
                N,
                n_full * tile,
                p,
                philox_seed,
                philox_offset + n_full * block,
                BLOCK=block,
                num_warps=num_warps,
            )
    return input


__all__ = ["dropout_", "dropout_inplace_kernel", "dropout_inplace_tail_kernel"]
