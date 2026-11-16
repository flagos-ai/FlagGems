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

"""Kunlunxin XPU3 override for `aten::infinitely_differentiable_gelu_backward`.

Semantics: `dx = grad * dydx` where
    dydx(x) = 0.5*(1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)

The generic implementation is a `@pointwise_dynamic` kernel whose body calls
`tl.math.erf` and `tl.exp`.  On XPU3 `tl.math.erf` lowers to the libdevice soft
extern (`_ZN3xpu5erffEf`): one `llvm.call` per lane, ~3.2 ns/element, i.e. the
target kernel is ~50x slower than torch (profiling: 16.7M fp16 = 54 ms,
268M = 900 ms).  This override replaces erf with the same odd-polynomial
`x*P(x^2)` (deg-12 LSQ, max abs err 3.7e-5 on |x|<=3, hard cut to +/-1 beyond)
that `_kunlunxin/ops/erf.py` already uses, and keeps the core `tl.exp` (which
lowers fast and correctly in a plain elementwise kernel).  `tl.exp2` is *not*
usable here (wrong values on this backend).

Layout: contiguous same-dtype fp16/bf16/fp32 inputs take a flat 1-D tile with a
numel-adaptive BLOCK.  The tail mask is a constexpr branch so shapes that divide
the tile exactly never touch the (non-coalesced, ~40% slower) masked-memory
path.  Anything else (non-contiguous / mixed dtype / other dtypes) falls back to
the still-on-device generic Gems kernel.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Same LSQ odd-polynomial as `_kunlunxin/ops/erf.py`: erf(s) = s * P(s^2) on
# |s| <= 3, saturated to +/-1 outside.  No transcendental in erf itself.
_CUT = tl.constexpr(3.0)
_SCALE1 = tl.constexpr(0.7071067811)  # 1 / sqrt(2)
_SCALE2 = tl.constexpr(0.3989422803)  # 1 / sqrt(2*pi)

# numel-adaptive flat tile (do_bench sweep over the benchmark matrix; above
# ~16K the tile width is flat within ~1%, small shapes are launch-bound and
# prefer many tiny programs).
_MIN_TILE = 2048
_MID_TILE = 4096
_BIG_TILE = 16384
_HUGE_TILE = 32768


def _pick_block(n):
    if n <= 65536:
        return _MIN_TILE
    if n <= 4_194_304:
        return _MID_TILE
    if n <= 33_554_432:
        return _BIG_TILE
    return _HUGE_TILE


@triton.jit
def _gelu_bwd_tile_kernel(g_ptr, x_ptr, o_ptr, n, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = off < n
        g = tl.load(g_ptr + off, mask=m, other=0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=m, other=0).to(tl.float32)
    else:
        g = tl.load(g_ptr + off).to(tl.float32)
        x = tl.load(x_ptr + off).to(tl.float32)

    s = _SCALE1 * x
    t = s * s
    p = 2.0958534e-12
    p = p * t + -1.4200718e-10
    p = p * t + 4.475963e-09
    p = p * t + -8.813736e-08
    p = p * t + 1.2336866e-06
    p = p * t + -1.3277817e-05
    p = p * t + 0.00011584865
    p = p * t + -0.00084552215
    p = p * t + 0.005211773
    p = p * t + -0.0268563
    p = p * t + 0.112833545
    p = p * t + -0.37612554
    p = p * t + 1.1283791
    erf_s = s * p
    erf_s = tl.where(s > _CUT, 1.0, erf_s)
    erf_s = tl.where(s < -_CUT, -1.0, erf_s)

    e = tl.exp(-t)
    dydx = _SCALE2 * x * e + 0.5 * erf_s + 0.5
    dx = dydx * g

    if MASKED:
        tl.store(o_ptr + off, dx.to(o_ptr.dtype.element_ty), mask=m)
    else:
        tl.store(o_ptr + off, dx.to(o_ptr.dtype.element_ty))


def _launch(grad, x, out, n):
    block = _pick_block(n)
    masked = (n % block) != 0
    grid = (triton.cdiv(n, block),)
    num_warps = 8 if block >= 8192 else 4
    _gelu_bwd_tile_kernel[grid](
        grad,
        x,
        out,
        n,
        BLOCK=block,
        MASKED=masked,
        num_warps=num_warps,
    )


def _supported(t):
    return t.dtype in (torch.float16, torch.bfloat16, torch.float32)


def infinitely_differentiable_gelu_backward(grad, self_input):
    logger.debug("GEMS_KUNLUNXIN INFINITELY_DIFFERENTIABLE_GELU_BACKWARD")
    if (
        grad.is_contiguous()
        and self_input.is_contiguous()
        and grad.dtype == self_input.dtype
        and _supported(grad)
    ):
        n = grad.numel()
        out = torch.empty_like(grad)
        if n == 0:
            return out
        _launch(grad, self_input, out, n)
        return out
    # Non-contiguous / mixed-dtype / unsupported dtype: real device kernel from
    # the generic Gems pointwise implementation (no CPU/ATen fallback).
    from flag_gems.ops.infinitely_differentiable_gelu_backward import (
        infinitely_differentiable_gelu_backward as _generic,
    )

    return _generic(grad, self_input)
