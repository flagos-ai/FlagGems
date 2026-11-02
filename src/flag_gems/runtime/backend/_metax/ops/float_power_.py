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

logger = logging.getLogger(__name__)


_BLOCK = 1024
_BLOCK_SMALL = 512
_SMALL_N = 65536
_MID_N = 1 << 21  # fp32 mid-size class [65536, 2^21) prefers BLOCK=512
_NUM_WARPS_SMALL = 8


@triton.jit
def _fast_pow32(x, e):
    """|x|^e via exp2(e*log2(|x|)) with IEEE pow edge handling (fp32).

    Zero-base cases fall out of exp2/log2 naturally: log2(0) = -inf gives
    exp2(e * -inf) = 0 for e > 0 and +inf for e < 0; e == 0 (including 0^0)
    is forced to 1.0.  Negative bases get NaN for non-integer exponents and a
    sign flip for odd integer exponents.  Odd/integer tests use the integer
    bit pattern of e (valid for |e| < 2^23, always true for float16/bf16
    exponent data) instead of floor operations.
    """
    r = tl.math.exp2(e * tl.math.log2(tl.abs(x)))
    r = tl.where(e == 0.0, 1.0, r)
    neg = x < 0.0
    ei = e.to(tl.int32)
    is_int = e == ei.to(tl.float32)
    odd = (ei & 1) != 0
    r = tl.where(neg & odd, -r, r)
    r = tl.where(neg & ~is_int, float("nan"), r)
    return r


@triton.jit
def _float_power_tt_kernel(
    a_ptr,
    e_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    IS_FP64: tl.constexpr,
    FAST: tl.constexpr,
):
    """In-place A = pow(A, exponent) elementwise (exponent is a same-shape tensor)."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(a_ptr + offs, mask=mask)
    e = tl.load(e_ptr + offs, mask=mask)
    if IS_FP64:
        r = tl.extra.libdevice.pow(x, e)
    else:
        xf = x.to(tl.float32)
        ef = e.to(tl.float32)
        if FAST:
            r = _fast_pow32(xf, ef)
        else:
            r = tl.extra.libdevice.pow(xf, ef)
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _float_power_ts_kernel(
    a_ptr, e_val, n_elements, BLOCK: tl.constexpr, IS_FP64: tl.constexpr
):
    """In-place A = pow(A, exponent) with exponent a scalar kernel argument."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(a_ptr + offs, mask=mask)
    if IS_FP64:
        r = tl.extra.libdevice.pow(x, e_val.to(tl.float64))
    else:
        r = tl.extra.libdevice.pow(x.to(tl.float32), e_val)
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _float_power_t1_kernel(
    a_ptr,
    e_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    IS_FP64: tl.constexpr,
    FAST: tl.constexpr,
):
    """In-place A = pow(A, exponent) with exponent a 1-element tensor (broadcast)."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(a_ptr + offs, mask=mask)
    e = tl.load(e_ptr)
    if IS_FP64:
        r = tl.extra.libdevice.pow(x, e)
    else:
        xf = x.to(tl.float32)
        ef = e.to(tl.float32)
        if FAST:
            r = _fast_pow32(xf, ef)
        else:
            r = tl.extra.libdevice.pow(xf, ef)
    tl.store(a_ptr + offs, r, mask=mask)


def float_power_tensor_scalar_(A, exponent):
    n = A.numel()
    if n == 0:
        return A
    is_fp64 = A.dtype == torch.float64
    fast = A.dtype in (torch.float16, torch.bfloat16)
    if n < _SMALL_N:
        block = _BLOCK_SMALL
        num_warps = _NUM_WARPS_SMALL
    elif fast:
        # fp16/bf16 fast pow: 2 warps measured 2-3% faster than 4 on all
        # n >= 64K shapes (fewer warps per block, more resident blocks)
        block = _BLOCK
        num_warps = 2
    elif n < _MID_N:
        # fp32 mid-size (e.g. 1M elements): BLOCK=512 measured ~5% faster than 1024
        block = _BLOCK_SMALL
        num_warps = 4
    else:
        block = _BLOCK
        num_warps = 4
    grid = (triton.cdiv(n, block),)
    if isinstance(exponent, torch.Tensor):
        if exponent.numel() == 1:
            _float_power_t1_kernel[grid](
                A,
                exponent,
                n,
                BLOCK=block,
                IS_FP64=is_fp64,
                FAST=fast,
                num_warps=num_warps,
            )
        else:
            _float_power_tt_kernel[grid](
                A,
                exponent,
                n,
                BLOCK=block,
                IS_FP64=is_fp64,
                FAST=fast,
                num_warps=num_warps,
            )
    else:
        _float_power_ts_kernel[grid](
            A,
            float(exponent),
            n,
            BLOCK=block,
            IS_FP64=is_fp64,
            num_warps=num_warps,
        )
    return A


# The single kernel entry handles both scalar and tensor exponents
# (it branches on isinstance(exponent, torch.Tensor)), so the aten
# .Tensor overload shares the same implementation.
float_power_tensor_tensor_ = float_power_tensor_scalar_
