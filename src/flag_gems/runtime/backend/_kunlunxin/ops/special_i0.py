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
#
# Kunlunxin (XPU) override of special_i0 / special_i0_out.
#
# Why an override exists at all (performance):
#   the generic `flag_gems/ops/special_i0.py` launches its kernel with a
#   `grid = lambda meta: ...` built inside `_launch_special_i0`, which on the
#   Triton xpubin backend is baked into the compilation cache key, so every
#   call recompiles the whole kernel (~130-200 ms/case measured, independent
#   of tensor size; see harness/perf_ir/prof_special_i0_pre.log for the
#   pre-profiling evidence: 93.9% of device time in `_special_i0_kernel`,
#   while the kernel itself only costs ~2.2 ms/call on the device).
#
# Why *this* evaluation scheme (performance):
#   same recipe as the proven sibling special_bessel_j0 override (which also
#   evaluates a Bessel function with core Triton primitives only): the
#   vendor `pointwise_dynamic` 1D-tile path with a fixed
#   CodeGenConfig(512, (65536,)*3, ..., kunlunAutoGrid=True, unroll_num=8)
#   and a plain-tuple grid (no lambda) - no per-call recompilation, and the
#   tail-masked 1D tile is generated once per shape and reused.
#   The body is the Abramowitz & Stegun 9.8.1 / 9.8.2 pair (max rel err
#   ~2e-7), evaluated in fp32 and stored back cast to the input dtype,
#   matching the reference semantics used by tests/test_special_i0.py
#   (fp16/bf16: fp32 computation then cast to input dtype).
#
# Numeric edge cases (matching torch semantics):
#   i0(0) = 1, i0(+-inf) = +inf, i0(nan) = nan.  The large-argument branch
#   is clamped to axl = max(ax, 3.75) so 3.75/axl can never divide by zero
#   (the small-region lanes pick p_small through the final select anyway),
#   and an explicit `ax == inf` fixup maps +-inf to +inf (i0 is an even
#   function); NaN falls through as NaN because NaN == inf is false.
# No CPU/ATen/native/composite fallback.
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def special_i0_func(x):
    # i0(x) = I0(|x|) is an even function; |x| <= 3.75 uses the A&S 9.8.1
    # polynomial (|err| < 1.6e-7), |x| > 3.75 uses the A&S 9.8.2
    # exp(|x|) * poly(3.75/|x|) / sqrt(|x|) (|err| < 1.9e-7).
    xf = x.to(tl.float32)
    ax = tl.abs(xf)
    is_small = ax <= 3.75

    t = ax / 3.75
    y = t * t
    p_small = 1.0 + y * (
        3.5156229
        + y
        * (
            3.0899424
            + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))
        )
    )

    # Clamp the small-region lanes to 3.75 so 3.75/axl can never divide by
    # zero; those lanes are discarded by the final select.
    axl = tl.where(is_small, 3.75, ax)
    yb = 3.75 / axl
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
    res_big = tl.exp(ax) * p_big / tl.sqrt(axl)

    result = tl.where(is_small, p_small, res_big)
    # i0(+-inf) = +inf (matches torch); NaN stays NaN (NaN == inf is false).
    result = tl.where(ax == float("inf"), float("inf"), result)
    return result.to(x.dtype)


def _special_i0_impl(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"special_i0 is not implemented for {x.dtype}")
    if out.dtype != x.dtype:
        raise RuntimeError("special_i0.out expects out to have the input dtype")
    if x.device != out.device:
        raise RuntimeError("special_i0.out expects out to be on the input device")
    special_i0_func(x, out0=out)
    return out


def special_i0(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0")
    return _special_i0_impl(x, torch.empty_like(x))


def special_i0_out(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_I0_OUT")
    return _special_i0_impl(x, out)
