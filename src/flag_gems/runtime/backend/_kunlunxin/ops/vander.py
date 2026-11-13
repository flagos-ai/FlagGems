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

_BLOCK_M = 16
_BLOCK_N = 64


@libentry()
@triton.jit
def _vander_pow_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # out[i, j] = x[i] ** p, p = j (increasing) or N-1-j. The exponent is a
    # column-dependent integer, so the power is built by binary exponentiation:
    # each step multiplies the running result by the current square only where
    # the corresponding exponent bit is set. This avoids the transcendental
    # expansion of a general pow and keeps the rounding chain logarithmic in p.
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rows < M
    mask_n = cols < N

    x_val = tl.load(x_ptr + rows * x_stride, mask=mask_m, other=1.0).to(COMPUTE_DTYPE)
    powers = cols if increasing else (N - 1 - cols)
    powers = tl.maximum(powers, 0).to(tl.int32)[None, :]
    exponent = powers + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    result = tl.full([BLOCK_M, BLOCK_N], 1.0, dtype=COMPUTE_DTYPE)
    base = tl.zeros([BLOCK_M, BLOCK_N], dtype=COMPUTE_DTYPE) + x_val[:, None]
    for i in range(0, NBITS):
        bit = (exponent >> i) & 1
        result = tl.where(bit != 0, result * base, result)
        base = base * base

    out_off = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_off, result, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@triton.jit
def _vander_dd_pow_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # float32 path with a double-float (hi, lo) accumulator. Binary
    # exponentiation in plain float32 accumulates one rounding per step, which is
    # not enough for large exponents; carrying the residual of every product
    # (exact via fma) keeps ~48 bits and makes the stored float32 correctly
    # rounded without needing a wider device type.
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rows < M
    mask_n = cols < N

    x_val = tl.load(x_ptr + rows * x_stride, mask=mask_m, other=1.0).to(tl.float32)
    powers = cols if increasing else (N - 1 - cols)
    powers = tl.maximum(powers, 0).to(tl.int32)[None, :]
    exponent = powers + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    res_hi = tl.full([BLOCK_M, BLOCK_N], 1.0, dtype=tl.float32)
    res_lo = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    base_hi = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + x_val[:, None]
    base_lo = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for i in range(0, NBITS):
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

    out_off = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_off, res_hi, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@triton.jit
def _vander_int_pow_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Integer path: exact int64 powers including two's-complement wrap-around,
    # matching ATen's int64 cumprod semantics.
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rows < M
    mask_n = cols < N

    x_val = tl.load(x_ptr + rows * x_stride, mask=mask_m, other=1)
    powers = cols if increasing else (N - 1 - cols)
    powers = tl.maximum(powers, 0).to(tl.int32)[None, :]
    exponent = powers + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    result = tl.full([BLOCK_M, BLOCK_N], 1, dtype=tl.int64)
    base = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int64) + x_val[:, None].to(tl.int64)
    for i in range(0, NBITS):
        bit = (exponent >> i) & 1
        result = tl.where(bit != 0, result * base, result)
        base = base * base

    out_off = rows[:, None] * N + cols[None, :]
    tl.store(out_ptr + out_off, result, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@triton.jit
def _vander_complex_kernel(
    xr_ptr,
    xi_ptr,
    outr_ptr,
    outi_ptr,
    M,
    N,
    x_stride,
    increasing: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # Complex path: (re, im) carried separately, powers built with the same
    # binary exponentiation and the complex squaring (a+bi)^2 = (a^2-b^2, 2ab).
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rows < M
    mask_n = cols < N

    xr = tl.load(xr_ptr + rows * x_stride, mask=mask_m, other=1.0).to(COMPUTE_DTYPE)
    xi = tl.load(xi_ptr + rows * x_stride, mask=mask_m, other=0.0).to(COMPUTE_DTYPE)
    powers = cols if increasing else (N - 1 - cols)
    powers = tl.maximum(powers, 0).to(tl.int32)[None, :]
    exponent = powers + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    resr = tl.full([BLOCK_M, BLOCK_N], 1.0, dtype=COMPUTE_DTYPE)
    resi = tl.zeros([BLOCK_M, BLOCK_N], dtype=COMPUTE_DTYPE)
    baser = tl.zeros([BLOCK_M, BLOCK_N], dtype=COMPUTE_DTYPE) + xr[:, None]
    basei = tl.zeros([BLOCK_M, BLOCK_N], dtype=COMPUTE_DTYPE) + xi[:, None]
    for i in range(0, NBITS):
        bit = (exponent >> i) & 1
        sel = bit != 0
        newr = resr * baser - resi * basei
        newi = resr * basei + resi * baser
        resr = tl.where(sel, newr, resr)
        resi = tl.where(sel, newi, resi)
        baser, basei = baser * baser - basei * basei, 2.0 * baser * basei

    out_off = rows[:, None] * N + cols[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(outr_ptr + out_off, resr, mask=mask)
    tl.store(outi_ptr + out_off, resi, mask=mask)


def _nbits(n):
    return max(1, int(n - 1).bit_length())


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
        _vander_complex_kernel[(triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))](
            xr,
            xi,
            outr,
            outi,
            M,
            N,
            xr.stride(0),
            increasing=increasing,
            NBITS=_nbits(N),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            COMPUTE_DTYPE=compute_dtype,
        )
        out.real.copy_(outr)
        out.imag.copy_(outi)
        return out

    out_dtype = torch.promote_types(x.dtype, torch.long)
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    if M == 0 or N == 0:
        return out

    if x.dtype == torch.float32:
        _vander_dd_pow_kernel[(triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))](
            x,
            out,
            M,
            N,
            x.stride(0),
            increasing=increasing,
            NBITS=_nbits(N),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
        )
    elif x.is_floating_point():
        compute_dtype = tl.float64 if x.dtype == torch.float64 else tl.float32
        _vander_pow_kernel[(triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))](
            x,
            out,
            M,
            N,
            x.stride(0),
            increasing=increasing,
            NBITS=_nbits(N),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            COMPUTE_DTYPE=compute_dtype,
        )
    else:
        _vander_int_pow_kernel[(triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))](
            x,
            out,
            M,
            N,
            x.stride(0),
            increasing=increasing,
            NBITS=_nbits(N),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
        )
    return out
