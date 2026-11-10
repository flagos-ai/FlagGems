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

"""Kunlunxin backend for linalg_vander.

The generic ``vander_complex_kernel`` computes (a + b j)**p via the polar
form r**p * (cos(p*theta) + j sin(p*theta)) with ``atan2`` in fp64.  The XPU
libdevice only implements ``atan2``/``pow``/``cos``/``sin`` for fp32 operands
(``[unsupported-on-xpu] libdevice function ... has no XPU implementation for
these operand types``), so that kernel fails to legalize.  This backend
replaces the polar-form computation with binary exponentiation: a sequence of
fp64 multiplies/adds only, which lowers everywhere, and is exact for integer
exponents (the output equals (a + b j)**p by construction).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.linalg_vander import vander_kernel

logger = logging.getLogger(__name__)


@triton.jit
def vander_complex_kernel(
    x_ri_ptr,
    out_ri_ptr,
    N,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    STORE_FP64: tl.constexpr,
    NBITS: tl.constexpr,
):
    """Complex vander via binary exponentiation: out = (a + b j)**p, p = col.

    The generic kernel uses the polar form r**p * (cos(p*theta) + j*sin(p*theta))
    which requires libdevice ``atan2``/``pow``/``cos``/``sin`` in fp64; the XPU
    libdevice only provides those for fp32.  Binary exponentiation needs only
    fp64 multiply/add, so it legalizes on XPU and is exact for the integer
    exponent p (a*b for col == 0 yields 1 + 0 j, matching torch).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    col = offsets % N  # exponent p
    row = offsets // N  # index into the flat complex input

    base = row * 2
    real = tl.load(x_ri_ptr + base, mask=mask, other=0.0).to(tl.float64)
    imag = tl.load(x_ri_ptr + base + 1, mask=mask, other=0.0).to(tl.float64)

    # result = (1 + 0j); base = (a + bj); at step k multiply result by base
    # when bit k of p is set, then square base.  NBITS covers any p < 2**NBITS.
    rr = tl.full((BLOCK_SIZE,), 1.0, tl.float64)
    ii = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    br = real
    bi = imag
    p = col
    for k in tl.static_range(NBITS):
        bit = (p >> k) & 1
        nrr = rr * br - ii * bi
        nii = rr * bi + ii * br
        rr = tl.where(bit == 1, nrr, rr)
        ii = tl.where(bit == 1, nii, ii)
        nbr = br * br - bi * bi
        nbi = 2.0 * br * bi
        br = nbr
        bi = nbi

    store_dtype = tl.float64 if STORE_FP64 else tl.float32
    out_base = offsets * 2
    tl.store(out_ri_ptr + out_base, rr.to(store_dtype), mask=mask)
    tl.store(out_ri_ptr + out_base + 1, ii.to(store_dtype), mask=mask)


def linalg_vander(x, N=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_VANDER")

    assert (
        x.dtype.is_floating_point or x.dtype.is_complex
    ), f"Unsupported dtype {x.dtype}"

    if N is None:
        N = x.shape[-1]

    batch_dims = x.shape[:-1]
    n = x.shape[-1]

    x_flat = x.reshape(-1)
    final_shape = batch_dims + (n, N)

    total_elements = x_flat.numel() * N
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    if x.is_complex():
        x_ri = torch.view_as_real(x_flat.contiguous()).reshape(-1)

        strides = []
        stride = 1
        for dim in reversed(final_shape):
            strides.append(stride)
            stride *= dim
        strides.reverse()

        out = torch.empty_strided(
            final_shape, tuple(strides), dtype=x.dtype, device=x.device
        )
        out_flat = out.reshape(-1)
        out_ri = torch.view_as_real(out_flat).reshape(-1)

        store_fp64 = x.dtype == torch.complex128
        vander_complex_kernel[grid](
            x_ri,
            out_ri,
            N,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            STORE_FP64=store_fp64,
            NBITS=max(1, (N - 1).bit_length()),
        )
    else:
        out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
        vander_kernel[grid](x_flat, out, N, total_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
