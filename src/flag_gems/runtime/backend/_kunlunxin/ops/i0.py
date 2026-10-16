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

# NOTE(2026-10-14): Kunlunxin (XPU) implementation of i0 / i0_out.
#
# Vendor-delegation evaluation (rejected): torch.library.get_kernel(aten.i0.out,
# "CUDA") at import time returns the torch_xmlir device kernel
# (ts_eager_fallback.cpp:375).  It is bit-exact with torch.i0 but 30-40x SLOWER
# than the triton polynomial on this backend (68.9ms vs 2.24ms for (4096,4096)
# f16) -- it IS the slow native path the benchmark measures as baseline.  So
# unlike amax / log_softmax_out, vendor delegation is a regression here.
#
# Optimizations vs the generic (flag_gems.ops) implementation (all verified
# against torch.i0, maxrel f32=4.8e-7 << rtol 1.3e-6; f16/bf16 bit-exact):
#   1. 3 fdiv/elem -> 1 fdiv + 1 sqrt: inv_ax = 1/ax, yb = 3.75*inv_ax,
#      res_big = exp(ax)*p_big*sqrt(inv_ax)  (sqrt(1/x) == 1/sqrt(x), 1 ulp).
#      fdiv lowers to a slow software sequence on xpu3; the LLIR confirmed the
#      generic kernel emits 3 scalar fdiv per lane.
#   2. t = x*(1/3.75) instead of x/3.75 (1 more fdiv removed).
#   3. BLOCK_SIZE 1024 -> 8192: 1.8-2.5x on every measured shape (the masked
#      tile cost is amortized over 8x fewer programs).
#   4. Mask-free kernel when n % 8192 == 0 (all benchmark shapes): ~8% faster.
#   5. Constant-tuple grid: `grid = lambda meta:` rebuilds the launch grid per
#      call which participates in the xpu3 compile-cache key (per-call
#      recompilation, see special_logit/hypot precedent).
#
# Semantics: torch.i0(+-inf) == NaN on this platform (vendor kernel computes
# inf*0); the polynomial reproduces that (no special-casing, do NOT "fix").
import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)

_BLOCK = 8192


@triton.jit
def _i0_poly(x_f32):
    # i0(x) for |x| <= 3.75 (Abramowitz-Stegun 9.8.1) and
    # exp(ax)*p_big/sqrt(ax) for |x| > 3.75 (A&S 9.8.2).
    ax = tl.abs(x_f32)
    inv_ax = 1.0 / ax
    t = x_f32 * 0.26666666666666666  # x / 3.75
    y = t * t
    p_small = 1.0 + y * (
        3.5156229
        + y
        * (
            3.0899424
            + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
        )
    )
    yb = 3.75 * inv_ax  # 3.75 / ax
    p_big = 0.39894228 + yb * (
        0.01328592
        + yb
        * (
            0.00225319
            + yb
            * (
                -0.00157565
                + yb
                * (
                    0.00916281
                    + yb
                    * (
                        -0.02057706
                        + yb * (0.02635537 + yb * (-0.01647633 + yb * 0.00392377))
                    )
                )
            )
        )
    )
    res_big = tl.exp(ax) * p_big * tl.sqrt(inv_ax)  # exp(ax)*p_big/sqrt(ax)
    return tl.where(ax <= 3.75, p_small, res_big)


@triton.jit
def i0_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = _i0_poly(x.to(tl.float32))
    tl.store(out_ptr + offsets, res, mask=mask)


@triton.jit
def i0_kernel_nomask(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    res = _i0_poly(x.to(tl.float32))
    tl.store(out_ptr + offsets, res)


def _launch_i0(out: torch.Tensor, x: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Input and output must be {flag_gems.device} tensors")
    assert (
        out.numel() == x.numel()
    ), "Input and output must have the same number of elements"
    assert out.device == x.device, "Input and output must be on the same device"

    # Red line: no ATen implementation calls allowed.  The kernel promotes the
    # loaded value to fp32 for the polynomial and stores through the output
    # pointer's dtype (triton casts on store), so neither an input dtype
    # conversion (`x.to(...)`) nor a `contiguous()` / `copy_()` round-trip is
    # needed -- a non-contiguous tensor is rejected up front (same contract as
    # i0_).
    if not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("i0: input and output tensors must be contiguous")

    n_elements = out.numel()
    if n_elements % _BLOCK == 0 and n_elements > 0:
        grid = (n_elements // _BLOCK,)
        i0_kernel_nomask[grid](x, out, n_elements, BLOCK_SIZE=_BLOCK)
    else:
        grid = (triton.cdiv(n_elements, _BLOCK),)
        i0_kernel[grid](x, out, n_elements, BLOCK_SIZE=_BLOCK)
    return out


def i0(x: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN I0")
    if x.device.type != flag_gems.device:
        raise ValueError(f"i0: input tensor must be on {flag_gems.device} device")
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x, dtype=out_dtype)
    _launch_i0(out, x)
    return out


def i0_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN I0_OUT")
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(
            f"i0_out: input and output tensors must be on {flag_gems.device} device"
        )
    if not out.is_floating_point():
        raise TypeError("i0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError(
            "i0_out: input and output must have the same number of elements"
        )
    _launch_i0(out, x)
    return out
