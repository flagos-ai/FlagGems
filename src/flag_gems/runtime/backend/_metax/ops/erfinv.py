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


@triton.jit
def _erfinv_poly(x):
    ax = tl.abs(x)
    rt_arg = 0.5 * (1.0 - ax)
    u2 = tl.sqrt(-tl.log2(rt_arg))
    sp = u2 * 0.47574549209011296 - 1.4857142857142856
    p = -0.0019877972081303596
    p = p * sp + 0.0024571786634624004
    p = p * sp + 0.002318469574674964
    p = p * sp + -0.0015593586722388864
    p = p * sp + -0.005279253702610731
    p = p * sp + 0.006560060195624828
    p = p * sp + -0.00670156953856349
    p = p * sp + 0.013129789382219315
    p = p * sp + -0.02516971156001091
    p = p * sp + 0.046528931707143784
    p = p * sp + -0.08964996784925461
    p = p * sp + 0.18467749655246735
    p = p * sp + -0.4460960328578949
    w = u2 * 0.8325546111576978 + p
    z = tl.where(x < 0.0, -w, w)
    z = tl.where(ax == 1.0, tl.where(x < 0.0, -float("inf"), float("inf")), z)
    return z


@triton.jit
def _erfinv_poly_lp(x):
    ax = tl.abs(x)
    rt_arg = 0.5 * (1.0 - ax)
    u2 = tl.sqrt(-tl.log2(rt_arg))
    sp = u2 * 0.47574549209011296 - 1.4857142857142856
    p = 0.014825913251096496
    p = p * sp + -0.01808701172502101
    p = p * sp + 0.0017262853332131532
    p = p * sp + -0.01693443778446957
    p = p * sp + 0.05110630684047027
    p = p * sp + -0.09149050445762476
    p = p * sp + 0.18420427797781141
    p = p * sp + -0.44603328015421
    w = u2 * 0.8325546111576978 + p
    z = tl.where(x < 0.0, -w, w)
    z = tl.where(ax == 1.0, tl.where(x < 0.0, -float("inf"), float("inf")), z)
    return z


@triton.jit
def _erfinv_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, DIVISIBLE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if DIVISIBLE:
        x = tl.load(x_ptr + offsets)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
    y = _erfinv_poly(x)
    if DIVISIBLE:
        tl.store(out_ptr + offsets, y)
    else:
        tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _erfinv_kernel_lp(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, DIVISIBLE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if DIVISIBLE:
        x = tl.load(x_ptr + offsets)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    y = _erfinv_poly_lp(xf)
    y = y.to(x.dtype)
    if DIVISIBLE:
        tl.store(out_ptr + offsets, y)
    else:
        tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _erfinv_ld_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.extra.cuda.libdevice.erfinv(x)
    tl.store(out_ptr + offsets, y, mask=mask)


_SMALL_N = 1 << 20
_TINY_N = 1 << 16


def erfinv(x):
    out = torch.empty_like(x)
    n = x.numel()
    if x.dtype == torch.float64:
        grid = (triton.cdiv(n, 1024),)
        _erfinv_ld_kernel[grid](x, out, n, BLOCK_SIZE=1024, num_warps=2)
    elif x.dtype in (torch.float16, torch.bfloat16):
        if n > _SMALL_N:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _erfinv_kernel_lp[grid](
                x, out, n, BLOCK_SIZE=BLOCK, num_warps=2, DIVISIBLE=(n % BLOCK == 0)
            )
        elif n > _TINY_N:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _erfinv_kernel_lp[grid](
                x, out, n, BLOCK_SIZE=BLOCK, num_warps=4, DIVISIBLE=(n % BLOCK == 0)
            )
        else:
            BLOCK = 256
            grid = (triton.cdiv(n, BLOCK),)
            _erfinv_kernel_lp[grid](
                x, out, n, BLOCK_SIZE=BLOCK, num_warps=2, DIVISIBLE=(n % BLOCK == 0)
            )
    else:
        if n > _SMALL_N:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _erfinv_kernel[grid](
                x, out, n, BLOCK_SIZE=BLOCK, num_warps=2, DIVISIBLE=(n % BLOCK == 0)
            )
        elif n > _TINY_N:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _erfinv_kernel[grid](
                x, out, n, BLOCK_SIZE=BLOCK, num_warps=4, DIVISIBLE=(n % BLOCK == 0)
            )
        else:
            BLOCK = 256
            grid = (triton.cdiv(n, BLOCK),)
            _erfinv_kernel[grid](
                x, out, n, BLOCK_SIZE=BLOCK, num_warps=2, DIVISIBLE=(n % BLOCK == 0)
            )
    return out
