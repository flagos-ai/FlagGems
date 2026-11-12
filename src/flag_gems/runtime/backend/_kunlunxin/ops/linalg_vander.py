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
"""Kunlunxin (XPU) vander.

The generic kernel launches one flat program per ``BLOCK`` consecutive output
elements and recovers ``(row, col)`` with integer ``%``/``//``, reloading
``x[row]`` for every lane.  On this backend that layout costs ~7x on the
largest benchmark shape (n=1024 -> 1M outputs): the vector integer div/mod and
the repeated gather dominate, not the power itself.

This override uses a 2-D launch (row x column-block) so the exponent is
``program_id(1) * BLOCK + arange`` with no div/mod, and evaluates ``x**j`` by
repeated squaring over the exponent bits -- ``NBITS = bit_length(N - 1)``
multiply/select pairs per lane -- instead of ``libdevice.pow``.  Both are
necessary: switching only the layout still leaves the libdevice pow call as
~70% of the kernel time.  Repeated squaring is also slightly more accurate
than libdevice ``pow`` here (max relative error 2.2e-6 vs 5.9e-5 on random
fp32 inputs), because every step is an exact multiplication of a power of x.

Complex and fp64 inputs keep the generic implementation (the backend has no
fp64 support and the generic complex kernel already computes in fp64).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_vander import linalg_vander as _generic_vander

logger = logging.getLogger(__name__)

# Columns per program for the row-major kernel: 1024 lanes keeps the unrolled
# exponent chain at 10 steps and gives one program per row when N == 1024.
_BLOCK = 1024
# Elements per program for the flat kernel (see ``_vander_flat_kernel``).
_FLAT_BLOCK = 4096
# Below this column count the flat kernel wins, above it the row-major one does.
# The crossover is measured on XPU: with few columns the row-major kernel pays a
# large fixed cost per program (one program per input element), while the flat
# kernel keeps the store coalesced and amortises that cost over several rows.
_FLAT_MAX_N = 128


@triton.jit
def _vander_pow_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, NBITS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)

    acc = tl.full((BLOCK,), 1.0, tl.float32)
    power = tl.load(x_ptr + row).to(tl.float32)
    for bit in tl.static_range(NBITS):
        acc = tl.where(((col >> bit) & 1) != 0, acc * power, acc)
        power = power * power

    mask = col < N
    tl.store(out_ptr + row * N + col, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _vander_flat_kernel(
    x_ptr,
    out_ptr,
    TOTAL,
    N: tl.constexpr,
    POW2: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    NBITS: tl.constexpr,
):
    # Output is contiguous, so a flat block spans whole rows: ``row``/``col``
    # are recovered by shift/and when N is a power of two (the common case) and
    # by integer div/mod otherwise.
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = idx < TOTAL
    if POW2:
        row = idx >> SHIFT
        col = idx & (N - 1)
    else:
        row = idx // N
        col = idx % N

    acc = tl.full((BLOCK,), 1.0, tl.float32)
    power = tl.load(x_ptr + row, mask=inb, other=1.0).to(tl.float32)
    for bit in tl.static_range(NBITS):
        acc = tl.where(((col >> bit) & 1) != 0, acc * power, acc)
        power = power * power

    tl.store(out_ptr + idx, acc.to(out_ptr.dtype.element_ty), mask=inb)


def linalg_vander(x, N=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_VANDER")
    if N is None:
        N = x.shape[-1]

    if x.is_complex() or x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return _generic_vander(x, N)

    n = x.shape[-1]
    out_shape = x.shape[:-1] + (n, N)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    if n == 0 or N == 0:
        return out

    nbits = max(1, (N - 1).bit_length())
    x_flat = x.reshape(-1)
    # One program row per input element: leading batch dims collapse into the
    # row axis, so a (batch..., n) input yields prod(batch) * n rows of N.
    rows = x_flat.numel()

    if N <= _FLAT_MAX_N:
        total = rows * N
        pow2 = (N & (N - 1)) == 0
        shift = N.bit_length() - 1 if pow2 else 0
        _vander_flat_kernel[(triton.cdiv(total, _FLAT_BLOCK),)](
            x_flat,
            out.reshape(-1),
            total,
            N=N,
            POW2=pow2,
            SHIFT=shift,
            BLOCK=_FLAT_BLOCK,
            NBITS=nbits,
            num_warps=8,
        )
        return out

    out_flat = out.reshape(rows, N)
    _vander_pow_kernel[(rows, triton.cdiv(N, _BLOCK))](
        x_flat, out_flat, N, BLOCK=_BLOCK, NBITS=nbits, num_warps=8
    )
    return out
