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

"""Kunlunxin override for ``aten::rrelu_``.

The evaluation path is delegated to the generic implementation (it is correct
on XPU3).  The training path in the generic implementation uses one kernel with
four unrolled load/store pairs (``UNROLL = 4``).  On XPU3 that multi-store form
miscompiles for element counts that are not a multiple of ``BLOCK * UNROLL``:
with a partially masked (or all-false) trailing store the other three stores
are silently dropped, so ``rrelu_(x, training=True)`` becomes a no-op for any
``n <= 4*BLOCK`` (verified: ``n = 3072 -> slope == 1`` everywhere).  It only
works when every one of the four stores is fully unmasked.

This override keeps the exact RReLU training semantics but emits a single
masked load/store per element, the same shape as the (working) eval kernel.
The slope still comes from the real ``tl.philox`` stream via
``philox_backend_seed_offset`` / ``uint_to_uniform_float`` -- no CPU or ATen
random source is involved.
"""

import logging

import triton
import triton.language as tl

from flag_gems.ops.rrelu_ import rrelu_ as _generic_rrelu_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

# One masked load/store per element avoids the XPU3 multi-store miscompile. A
# 4K tile keeps the per-program element count equal to the generic kernel's
# 4 x 1K unrolled tile, so the program count matches.
_TRAIN_BLOCK_SIZE = 4096
_TRAIN_NUM_WARPS = 4
# Aligned fast path: same geometry as the generic unrolled kernel.
_UNROLL_BLOCK = 1024
_UNROLL = 4


@triton.jit(do_not_specialize=["lower", "upper", "philox_seed", "philox_offset"])
def _rrelu_train_kernel_single(
    x_ptr,
    n_elements,
    lower,
    upper,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, _r1, _r2, _r3 = tl.philox(philox_seed, c0, c1, _O, _O)

    u = uint_to_uniform_float(r0)
    lower_f = lower.to(tl.float32)
    upper_f = upper.to(tl.float32)
    slope = lower_f + u * (upper_f - lower_f)
    if EVEN:
        x = tl.load(x_ptr + i4)
        tl.store(x_ptr + i4, tl.where(x >= 0, x, x * slope).to(x.dtype))
    else:
        mask = i4 < n_elements
        # Clamped unmasked load: a masked tail load returns garbage here.
        x = tl.load(x_ptr + tl.minimum(i4, n_elements - 1))
        tl.store(
            x_ptr + i4, tl.where(x >= 0, x, x * slope).to(x.dtype), mask=mask
        )


@triton.jit(do_not_specialize=["lower", "upper", "philox_seed", "philox_offset"])
def _rrelu_train_kernel_unrolled4(
    x_ptr,
    n_elements,
    lower,
    upper,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """Aligned fast path: every one of the four stores is fully unmasked.

    The XPU3 miscompile only drops stores when a trailing mask is partial or
    all-false; with all four masks true (n divisible by BLOCK*UNROLL) this is
    the same kernel shape the generic implementation uses, and it is correct.
    """
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    lower_f = lower.to(tl.float32)
    upper_f = upper.to(tl.float32)
    u0 = uint_to_uniform_float(r0)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u3 = uint_to_uniform_float(r3)
    base = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_0 = base
    off_1 = base + BLOCK
    off_2 = base + 2 * BLOCK
    off_3 = base + 3 * BLOCK
    x0 = tl.load(x_ptr + off_0)
    x1 = tl.load(x_ptr + off_1)
    x2 = tl.load(x_ptr + off_2)
    x3 = tl.load(x_ptr + off_3)
    lo = lower_f
    hi = upper_f
    tl.store(x_ptr + off_0, tl.where(x0 >= 0, x0, x0 * (lo + u0 * (hi - lo))).to(x0.dtype))
    tl.store(x_ptr + off_1, tl.where(x1 >= 0, x1, x1 * (lo + u1 * (hi - lo))).to(x1.dtype))
    tl.store(x_ptr + off_2, tl.where(x2 >= 0, x2, x2 * (lo + u2 * (hi - lo))).to(x2.dtype))
    tl.store(x_ptr + off_3, tl.where(x3 >= 0, x3, x3 * (lo + u3 * (hi - lo))).to(x3.dtype))


def _rrelu_train_(A, lower, upper, generator=None):
    n_elements = A.numel()
    if n_elements == 0:
        return A
    align = _UNROLL_BLOCK * _UNROLL
    with torch_device_fn.device(A.device):
        if n_elements % align == 0:
            # Every store is full: reuse the wide unrolled tile (fast, correct).
            grid = (n_elements // align,)
            philox_seed, philox_offset = philox_backend_seed_offset(
                n_elements // _UNROLL, generator=generator
            )
            _rrelu_train_kernel_unrolled4[grid](
                A,
                n_elements,
                float(lower),
                float(upper),
                philox_seed,
                philox_offset,
                BLOCK=_UNROLL_BLOCK,
                UNROLL=_UNROLL,
                num_warps=_TRAIN_NUM_WARPS,
            )
        else:
            # Partial trailing block: one masked load/store per element.
            grid = (triton.cdiv(n_elements, _TRAIN_BLOCK_SIZE),)
            philox_seed, philox_offset = philox_backend_seed_offset(
                n_elements, generator=generator
            )
            _rrelu_train_kernel_single[grid](
                A,
                n_elements,
                float(lower),
                float(upper),
                philox_seed,
                philox_offset,
                BLOCK=_TRAIN_BLOCK_SIZE,
                EVEN=(n_elements % _TRAIN_BLOCK_SIZE) == 0,
                num_warps=_TRAIN_NUM_WARPS,
            )
    return A


def rrelu_(
    A,
    lower=0.125,
    upper=0.33333333333333331,
    training=False,
    generator=None,
):
    logger.debug("GEMS_KUNLUNXIN RRELU_")
    if training:
        return _rrelu_train_(A, lower, upper, generator)
    return _generic_rrelu_(A, lower, upper, training, generator)
