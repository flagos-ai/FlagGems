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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# Flat 1-D tiles that span whole output rows. The output [M, N] is row-major, so
# a program walking a contiguous span of it stores with unit stride (contiguous
# DMA on XPU3); a 2-D tile whose row stride is the *runtime* N instead lowers to
# one short strided DMA per row (~20x slower on P800). N is a constexpr so the
# row/column split (off // N, off % N) strength-reduces to shifts and the bit
# loop unrolls completely.
_BLOCK_FLOAT = 4096
_BLOCK_DD = 2048
_BLOCK_INT = 1024
_BLOCK_COMPLEX = 2048


@libentry()
@triton.jit
def _vander_flat_kernel(
    x_ptr,
    out_ptr,
    numel,
    N: tl.constexpr,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # out[i, j] = x[i] ** p, p = j (increasing) or N-1-j. The exponent is a
    # column-dependent integer, so the power is built by binary exponentiation:
    # each step multiplies the running result by the current square only where
    # the corresponding exponent bit is set. This avoids the transcendental
    # expansion of a general pow and keeps the rounding chain logarithmic in p.
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    valid = off < numel
    row = off // N
    col = off % N

    x_val = tl.load(x_ptr + row * x_stride, mask=valid, other=1.0).to(COMPUTE_DTYPE)
    exponent = (col if increasing else (N - 1 - col)).to(tl.int32)

    result = tl.full([BLOCK], 1.0, dtype=COMPUTE_DTYPE)
    base = x_val
    for i in tl.static_range(NBITS):
        bit = (exponent >> i) & 1
        result = tl.where(bit != 0, result * base, result)
        base = base * base

    tl.store(out_ptr + off, result, mask=valid)


@libentry()
@triton.jit
def _vander_flat_dd_kernel(
    x_ptr,
    out_ptr,
    numel,
    N: tl.constexpr,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # float32 path with a double-float (hi, lo) accumulator. Binary
    # exponentiation in plain float32 accumulates one rounding per step, which is
    # not enough for large exponents; carrying the residual of every product
    # (exact via fma) keeps ~48 bits and makes the stored float32 correctly
    # rounded without needing a wider device type.
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    valid = off < numel
    row = off // N
    col = off % N

    x_val = tl.load(x_ptr + row * x_stride, mask=valid, other=1.0).to(tl.float32)
    exponent = (col if increasing else (N - 1 - col)).to(tl.int32)

    res_hi = tl.full([BLOCK], 1.0, dtype=tl.float32)
    res_lo = tl.zeros([BLOCK], dtype=tl.float32)
    base_hi = x_val
    base_lo = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(NBITS):
        bit = (exponent >> i) & 1
        take = bit != 0
        prod = res_hi * base_hi
        err = tl.math.fma(res_hi, base_hi, -prod)
        err = err + (res_hi * base_lo + res_lo * base_hi)
        hi = prod + err
        lo = err - (hi - prod)
        res_hi = tl.where(take, hi, res_hi)
        res_lo = tl.where(take, lo, res_lo)

        sq = base_hi * base_hi
        sq_err = tl.math.fma(base_hi, base_hi, -sq)
        sq_err = sq_err + 2.0 * base_hi * base_lo
        base_hi = sq + sq_err
        base_lo = sq_err - (base_hi - sq)

    tl.store(out_ptr + off, res_hi, mask=valid)


@libentry()
@triton.jit
def _vander_flat_int_kernel(
    x_ptr,
    out_ptr,
    numel,
    N: tl.constexpr,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Integer path: exact int64 powers including two's-complement wrap-around,
    # matching ATen's int64 cumprod semantics.
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    valid = off < numel
    row = off // N
    col = off % N

    x_val = tl.load(x_ptr + row * x_stride, mask=valid, other=1).to(tl.int64)
    exponent = (col if increasing else (N - 1 - col)).to(tl.int32)

    result = tl.full([BLOCK], 1, dtype=tl.int64)
    base = x_val
    for i in tl.static_range(NBITS):
        bit = (exponent >> i) & 1
        result = tl.where(bit != 0, result * base, result)
        base = base * base

    tl.store(out_ptr + off, result, mask=valid)


@libentry()
@triton.jit
def _vander_flat_complex_kernel(
    xr_ptr,
    xi_ptr,
    outr_ptr,
    outi_ptr,
    numel,
    N: tl.constexpr,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Complex path: (re, im) carried separately, powers built with the same
    # binary exponentiation and the complex squaring (a+bi)^2 = (a^2-b^2, 2ab).
    pid = ext.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    valid = off < numel
    row = off // N
    col = off % N

    xr = tl.load(xr_ptr + row * x_stride, mask=valid, other=1.0).to(COMPUTE_DTYPE)
    xi = tl.load(xi_ptr + row * x_stride, mask=valid, other=0.0).to(COMPUTE_DTYPE)
    exponent = (col if increasing else (N - 1 - col)).to(tl.int32)

    resr = tl.full([BLOCK], 1.0, dtype=COMPUTE_DTYPE)
    resi = tl.zeros([BLOCK], dtype=COMPUTE_DTYPE)
    baser = xr
    basei = xi
    for i in tl.static_range(NBITS):
        bit = (exponent >> i) & 1
        sel = bit != 0
        newr = resr * baser - resi * basei
        newi = resr * basei + resi * baser
        resr = tl.where(sel, newr, resr)
        resi = tl.where(sel, newi, resi)
        baser, basei = baser * baser - basei * basei, 2.0 * baser * basei

    tl.store(outr_ptr + off, resr, mask=valid)
    tl.store(outi_ptr + off, resi, mask=valid)


def _nbits(n):
    return max(1, int(n - 1).bit_length())


def _grid(numel, block):
    return (triton.cdiv(numel, block),)


def vander(x, N=None, increasing=False):
    logger.debug("GEMS_KUNLUNXIN VANDER")

    if x.ndim != 1:
        raise RuntimeError("x must be a one-dimensional tensor.")

    if N is not None and N < 0:
        raise RuntimeError("N must be non-negative.")

    M = x.shape[0]
    if N is None:
        N = M

    if x.is_complex():
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        if M == 0 or N == 0:
            return out
        xc = x.contiguous()
        xr = xc.real.contiguous()
        xi = xc.imag.contiguous()
        outr = torch.empty((M, N), dtype=xr.dtype, device=x.device)
        outi = torch.empty((M, N), dtype=xr.dtype, device=x.device)
        compute_dtype = tl.float32 if xr.dtype == torch.float32 else tl.float64
        numel = M * N
        _vander_flat_complex_kernel[_grid(numel, _BLOCK_COMPLEX)](
            xr,
            xi,
            outr,
            outi,
            numel,
            N=N,
            x_stride=xr.stride(0),
            increasing=increasing,
            NBITS=_nbits(N),
            BLOCK=_BLOCK_COMPLEX,
            COMPUTE_DTYPE=compute_dtype,
        )
        out.real.copy_(outr)
        out.imag.copy_(outi)
        return out

    out_dtype = torch.promote_types(x.dtype, torch.long)
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    if M == 0 or N == 0:
        return out

    numel = M * N
    nbits = _nbits(N)

    if x.dtype == torch.float32:
        _vander_flat_dd_kernel[_grid(numel, _BLOCK_DD)](
            x,
            out,
            numel,
            N=N,
            x_stride=x.stride(0),
            increasing=increasing,
            NBITS=nbits,
            BLOCK=_BLOCK_DD,
        )
    elif x.is_floating_point():
        compute_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32
        block = _BLOCK_INT if x.dtype == torch.float64 else _BLOCK_FLOAT
        _vander_flat_kernel[_grid(numel, block)](
            x,
            out,
            numel,
            N=N,
            x_stride=x.stride(0),
            increasing=increasing,
            NBITS=nbits,
            BLOCK=block,
            COMPUTE_DTYPE=compute_dtype,
        )
    else:
        _vander_flat_int_kernel[_grid(numel, _BLOCK_INT)](
            x,
            out,
            numel,
            N=N,
            x_stride=x.stride(0),
            increasing=increasing,
            NBITS=nbits,
            BLOCK=_BLOCK_INT,
        )
    return out
