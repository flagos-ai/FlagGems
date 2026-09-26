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

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)


@triton.jit
def vander_kernel(
    x_ptr,
    out_ptr,
    N,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    col = offsets % N
    row = offsets // N

    x_val = tl.load(x_ptr + row, mask=mask)
    compute_dtype = tl.float64 if out_ptr.dtype.element_ty == tl.float64 else tl.float32
    result = tl_extra_shim.pow(x_val.to(compute_dtype), col.to(compute_dtype))

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def vander_complex_kernel(
    x_ri_ptr,
    out_ri_ptr,
    N,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    STORE_FP64: tl.constexpr,
    POW_BITS: tl.constexpr,
):
    """Complex vander: each output element is (a + b j)**p with integer p in [0, N).

    Instead of the polar form (r**p * (cos(p*theta) + j sin(p*theta))), which on
    XPU requires the crashing tl_extra_shim.{cos,sin,atan2} externs, this computes
    the integer power directly by exponentiation-by-squaring using complex
    multiplication. This is exact for integer exponents and touches no externs.

    Inputs/outputs are viewed as interleaved real/imag float arrays via
    torch.view_as_real so the kernel only handles real dtypes (Triton cannot
    specialise on complex tensors).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    col = offsets % N
    row = offsets // N

    base = row * 2
    real = tl.load(x_ri_ptr + base, mask=mask, other=0.0)
    imag = tl.load(x_ri_ptr + base + 1, mask=mask, other=0.0)

    base_re = real.to(tl.float64)
    base_im = imag.to(tl.float64)

    res_re = tl.zeros_like(base_re) + 1.0
    res_im = tl.zeros_like(base_im)
    c = col
    for _ in tl.static_range(POW_BITS):
        odd = (c & 1) == 1
        nr = res_re * base_re - res_im * base_im
        ni = res_re * base_im + res_im * base_re
        res_re = tl.where(odd, nr, res_re)
        res_im = tl.where(odd, ni, res_im)
        br = base_re * base_re - base_im * base_im
        bi = 2.0 * base_re * base_im
        base_re = br
        base_im = bi
        c = c >> 1

    store_dtype = tl.float64 if STORE_FP64 else tl.float32
    out_real_store = res_re.to(store_dtype)
    out_imag_store = res_im.to(store_dtype)

    out_base = offsets * 2
    tl.store(out_ri_ptr + out_base, out_real_store, mask=mask)
    tl.store(out_ri_ptr + out_base + 1, out_imag_store, mask=mask)


def linalg_vander(x, N=None):
    logger.debug("GEMS_KUNLUNXIN LINALG_VANDER")

    # fmt: off
    assert x.dtype.is_floating_point or x.dtype.is_complex, f"Unsupported dtype {x.dtype}"
    # fmt: on

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
        pow_bits = max(1, int(N - 1).bit_length())
        vander_complex_kernel[grid](
            x_ri,
            out_ri,
            N,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            STORE_FP64=store_fp64,
            POW_BITS=pow_bits,
        )
    else:
        out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
        vander_kernel[grid](x_flat, out, N, total_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
