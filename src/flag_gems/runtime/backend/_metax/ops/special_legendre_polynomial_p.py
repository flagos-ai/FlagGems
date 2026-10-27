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


_STATIC_MAX = 32


@triton.jit
def _legendre_p_static(
    x_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    N: tl.constexpr,
    DTYPE: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(DTYPE)
    else:
        x = tl.load(x_ptr + offs).to(DTYPE)

    # Legendre recurrence, fully unrolled at compile time (N constexpr):
    #   P_0(x) = 1, P_1(x) = x
    #   P_k(x) = ((2k-1) * x * P_{k-1}(x) - (k-1) * P_{k-2}(x)) / k
    if N == 0:
        res = tl.full([BLOCK], 1.0, DTYPE)
    elif N == 1:
        res = x
    else:
        p_prev2 = tl.full([BLOCK], 1.0, DTYPE)  # P_0(x)
        p_prev1 = x  # P_1(x)
        for k in tl.static_range(2, N + 1):
            kf = float(k)
            p_new = ((2.0 * kf - 1.0) * x * p_prev1 - (kf - 1.0) * p_prev2) / kf
            p_prev2 = p_prev1
            p_prev1 = p_new
        res = p_prev1

    if MASKED:
        tl.store(out_ptr + offs, res, mask=mask)
    else:
        tl.store(out_ptr + offs, res)


@triton.jit
def _legendre_p3_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
    MASKED: tl.constexpr,
):
    # Closed form P_3(x) = (5*x^3 - 3*x)/2 = x*(5*x^2 - 3)/2
    # evaluated with the shortest f32 dependency chain (2 muls + 2 FMAs).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(DTYPE)
    else:
        x = tl.load(x_ptr + offs).to(DTYPE)
    x2 = x * x
    res = x * (5.0 * x2 - 3.0) * 0.5
    if MASKED:
        tl.store(out_ptr + offs, res, mask=mask)
    else:
        tl.store(out_ptr + offs, res)


@triton.jit
def _legendre_p3_nomask(
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Leanest n=3 form: closed-form P_3(x) = x*(5*x^2 - 3)/2, no numel arg,
    # no bounds mask (only used when numel is an exact multiple of BLOCK).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(DTYPE)
    x2 = x * x
    res = x * (5.0 * x2 - 3.0) * 0.5
    tl.store(out_ptr + offs, res)


@triton.jit
def _legendre_p_dyn(
    x_ptr,
    out_ptr,
    n,
    numel,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(DTYPE)

    p_prev2 = tl.full([BLOCK], 1.0, DTYPE)  # P_0(x)
    p_prev1 = x  # P_1(x)
    for k in tl.range(2, n + 1):
        kf = k.to(DTYPE)
        p_new = ((2.0 * kf - 1.0) * x * p_prev1 - (kf - 1.0) * p_prev2) / kf
        p_prev2 = p_prev1
        p_prev1 = p_new

    res = tl.where(n == 0, p_prev2, p_prev1)
    # torch returns zeros for negative n
    res = tl.where(n < 0, 0.0, res)
    tl.store(out_ptr + offs, res, mask=mask)


def _pick(numel):
    # Measured on MetaX C550 (n=3, do_bench):
    #   >= 67M elements : BLOCK=512, num_warps=4 (near copy BW ceiling)
    #   >= 64K elements : BLOCK=1024, num_warps=4
    #   <  64K elements : BLOCK=256, num_warps=8 (launch-bound)
    if numel >= (1 << 26):
        return 512, 4
    if numel >= (1 << 16):
        return 1024, 4
    return 256, 8


def special_legendre_polynomial_p(x, n):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError("x must be floating point")
    if isinstance(n, torch.Tensor):
        n = n.item()
    n = int(n)
    out = torch.empty_like(x)
    numel = x.numel()
    if numel == 0:
        return out
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype == torch.float64:
        DTYPE = tl.float64
    elif x.dtype == torch.float16:
        DTYPE = tl.float32  # compute in fp32 for fp16 inputs
    else:
        DTYPE = tl.float32
    if n == 3:
        # Closed-form P_3(x) = x*(5*x^2 - 3)/2: shortest dependency chain.
        BLOCK, NW = _pick(numel)
        if numel % BLOCK == 0:
            grid = (numel // BLOCK,)
            _legendre_p3_nomask[grid](x, out, BLOCK=BLOCK, DTYPE=DTYPE, num_warps=NW)
        else:
            grid = (triton.cdiv(numel, BLOCK),)
            _legendre_p3_kernel[grid](
                x,
                out,
                numel,
                BLOCK=BLOCK,
                DTYPE=DTYPE,
                MASKED=True,
                num_warps=NW,
            )
    elif 0 <= n <= _STATIC_MAX:
        BLOCK, NW = _pick(numel)
        masked = numel % BLOCK != 0
        grid = (triton.cdiv(numel, BLOCK),)
        _legendre_p_static[grid](
            x,
            out,
            numel,
            BLOCK=BLOCK,
            N=n,
            DTYPE=DTYPE,
            MASKED=masked,
            num_warps=NW,
        )
    else:
        BLOCK = 1024
        grid = (triton.cdiv(numel, BLOCK),)
        _legendre_p_dyn[grid](x, out, n, numel, BLOCK=BLOCK, DTYPE=DTYPE)
    return out
