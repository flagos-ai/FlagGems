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

"""Physicists' Hermite polynomial H_n(x) elementwise.

out[i] = H_{n}[i](x[i]) with the three-term recurrence
    H_0(x) = 1, H_1(x) = 2x, H_k(x) = 2x*H_{k-1}(x) - 2(k-1)*H_{k-2}(x)

The recurrence runs in float64 regardless of the input dtype (the harness
reference is ATen evaluated on float64-upcast x, and a float32-internal
recurrence loses too much accuracy through cancellation for degrees >= 4)
and rounds exactly once at the store.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.special_hermite_polynomial_h import (
    special_hermite_polynomial_h as _generic_hermite,
)

logger = logging.getLogger(__name__)
_MAX_DEG = 9


@triton.jit
def _hermite_deg_expr(xd, DEG: tl.constexpr):
    """Closed-form H_DEG(xd) for a compile-time degree, evaluated in float64.

    Even degrees are a polynomial in y = x*x and odd degrees are x times such
    a polynomial, so each degree costs one multiply for y, a Horner chain of at
    most four float64 FMAs and (odd degrees) one final multiply.  Every
    coefficient is an exact small integer and float64 keeps the cancellation of
    the expanded form ~1e-16 relative to the largest intermediate, far below
    the 1e-4 + 1.3e-6*|ref| tolerance.
    """
    y = xd * xd
    if DEG == 0:
        r = tl.full(xd.shape, 1.0, tl.float64)
    elif DEG == 1:
        r = 2.0 * xd
    elif DEG == 2:
        r = tl.fma(y, 4.0, -2.0)
    elif DEG == 3:
        r = xd * tl.fma(y, 8.0, -12.0)
    elif DEG == 4:
        r = tl.fma(tl.fma(y, 16.0, -48.0), y, 12.0)
    elif DEG == 5:
        r = xd * tl.fma(tl.fma(y, 32.0, -160.0), y, 120.0)
    elif DEG == 6:
        r = tl.fma(tl.fma(tl.fma(y, 64.0, -480.0), y, 720.0), y, -120.0)
    elif DEG == 7:
        r = xd * tl.fma(tl.fma(tl.fma(y, 128.0, -1344.0), y, 3360.0), y, -1680.0)
    elif DEG == 8:
        r = tl.fma(
            tl.fma(tl.fma(tl.fma(y, 256.0, -3584.0), y, 13440.0), y, -13440.0),
            y,
            1680.0,
        )
    else:
        r = xd * tl.fma(
            tl.fma(tl.fma(tl.fma(y, 512.0, -9216.0), y, 48384.0), y, -80640.0),
            y,
            30240.0,
        )
    return r


@triton.jit
def _hermite_h_scalar_kernel(
    X, OUT, numel, DEG: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
    else:
        mask = offs < numel
        x = tl.load(X + offs, mask=mask)
    xd = x.to(tl.float64)
    r = _hermite_deg_expr(xd, DEG)
    r = r.to(x.dtype)
    if EVEN:
        tl.store(OUT + offs, r)
    else:
        tl.store(OUT + offs, r, mask=mask)


@triton.jit
def _hermite_h_tensor_kernel(X, N, OUT, numel, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        n = tl.load(N + offs)
    else:
        mask = offs < numel
        x = tl.load(X + offs, mask=mask)
        n = tl.load(N + offs, mask=mask)
    xd = x.to(tl.float64)
    ni = n.to(tl.int32)
    ni = tl.minimum(tl.maximum(ni, 0), 9)
    two_x = 2.0 * xd
    h_prev = tl.full([BLOCK], 1.0, tl.float64)
    h_cur = two_x
    r = tl.where(ni == 0, h_prev, h_cur)
    for k in tl.static_range(1, 9):
        h_next = tl.fma(two_x, h_cur, -2.0 * k * h_prev)
        h_prev = h_cur
        h_cur = h_next
        r = tl.where(ni == k + 1, h_cur, r)
    r = r.to(x.dtype)
    if EVEN:
        tl.store(OUT + offs, r)
    else:
        tl.store(OUT + offs, r, mask=mask)


def _num_warps(block):
    if block <= 128:
        return 1
    if block <= 512:
        return 2
    return 4


def _block_for(numel):
    if numel >= 8192:
        return 1024
    return max(64, triton.next_power_of_2(numel))


def _launch_scalar(x, deg):
    numel = x.numel()
    out = torch.empty_like(x)
    if numel == 0:
        return out
    block = _block_for(numel)
    grid = ((numel + block - 1) // block,)
    _hermite_h_scalar_kernel[grid](
        x,
        out,
        numel,
        DEG=deg,
        BLOCK=block,
        EVEN=numel % block == 0,
        num_warps=_num_warps(block),
    )
    return out


def _launch_tensor(x, n):
    numel = x.numel()
    out = torch.empty_like(x)
    if numel == 0:
        return out
    block = _block_for(numel)
    grid = ((numel + block - 1) // block,)
    _hermite_h_tensor_kernel[grid](
        x,
        n,
        out,
        numel,
        BLOCK=block,
        EVEN=numel % block == 0,
        num_warps=_num_warps(block),
    )
    return out


def _validate_deg(deg, n):
    if deg < 0 or deg > _MAX_DEG:
        raise ValueError(
            f"special_hermite_polynomial_h only supports n in [0, 9], got n={n}"
        )


def special_hermite_polynomial_h(x, n):
    logger.debug("GEMS_HYGON SPECIAL_HERMITE_POLYNOMIAL_H")
    if not isinstance(x, torch.Tensor):
        raise ValueError("First argument must be a tensor")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Unsupported dtype {x.dtype}")
    if not x.is_contiguous():
        return _generic_hermite(x, n)
    if isinstance(n, torch.Tensor):
        if n.ndim > x.ndim:
            return _generic_hermite(x, n)
        nn = n.numel()
        if nn == 1:
            deg = int(n.reshape(-1)[0].item())
            _validate_deg(deg, n)
            return _launch_scalar(x, deg)
        if n.shape != x.shape or not n.is_contiguous():
            return _generic_hermite(x, n)
        return _launch_tensor(x, n)
    if isinstance(n, (int, float)) and (not isinstance(n, bool)):
        deg = int(n)
        _validate_deg(deg, n)
        return _launch_scalar(x, deg)
    raise ValueError("n must be a Tensor, int, or float")
