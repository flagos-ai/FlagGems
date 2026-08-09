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
# Kunlunxin (XPU) override of special_bessel_y0 (aten::special_bessel_y0).
#
# Why an override exists at all:
#   the generic `flag_gems/ops/special_bessel_y0.py` calls `tl_extra_shim.y0(x)`,
#   which resolves to the XPU vendor libdevice `y0` and fails to *link*:
#     ld.lld: error: undefined symbol: Unsupported
#     >>> referenced by special_bessel_y0.py:31 ... special_bessel_y0_func_kernel_rank_1
#   so every call raises `RuntimeError: Command failed (1): [... xpu3-elfconv-triton ...]`.
#   Same failure class as libdevice j0 / lgamma / log1p / polygamma on this backend.
#   Consequence at HEAD: 6/6 accuracy cases fail and the benchmark aborts on its
#   first cell, i.e. the operator is entirely unavailable on XPU.  Y0 must
#   therefore be evaluated with core Triton primitives only (log/sin/cos/sqrt/
#   where all link on XPU).
#
# Why *this* evaluation scheme (performance):
#   TritonXPU runs a pointwise body on a fixed 12-program grid, so the kernel is
#   ALU-bound, not bandwidth-bound: a trivial `x*2` reaches ~1033 GB/s at
#   [4096,4096] fp32 (0.130 ms) while this Y0 body needs 2.07 ms (65 GB/s), ~16x
#   the memory floor.  Measured unit costs on this backend (16.78 M lanes):
#   fma ~0.057 ms, tl.where ~0.11 ms, sin/cos ~0.12 ms, div/sqrt ~0.033 ms,
#   tl.log ~0.006 ms (essentially free).  The scheme below is the
#   Numerical-Recipes rational pair, which costs
#     ~28 fma / 3 div / 3 transcendental / 1 sqrt / 1 tl.where
#   (predicted 2.20 ms from the unit-cost table, measured 2.07 ms):
#     x <  8 : Y0 = R(x^2)/S(x^2) + (2/pi) * log(x) * (P(x^2)/Q(x^2))
#              with P/Q the degree-5/5 J0 rational (same one the sibling
#              special_bessel_j0 override uses -- it is inlined here on purpose
#              rather than imported, so no gems-level function binding is
#              involved and the vendor override cannot be bypassed)
#     x >= 8 : Y0 = sqrt(2/(pi x)) * ( sin(x-pi/4) * P1(w)
#                                    + (8/x) * cos(x-pi/4) * Q1(w) ),  w = (8/x)^2
#   Both branches are always evaluated and merged with a single `tl.where`
#   (an elementwise python `if` is not expressible inside a Triton tile, and a
#   bool->float conversion would trip the XPU `arith.uitofp` check at
#   BLOCK >= 256).
#   Measured non-levers (see harness/solution/performance/special_bessel_y0_xpu1_20260830.md):
#   max_tile_size 256/512/1024/2048 all land at 2.062-2.068 ms; num_ctas is
#   hard-wired to 12 by the vendor pointwise_dynamic 1d-tile path.  Dropping the
#   rationals to degree 4/4 (refit in u=(x/8)^2) would save ~2 fma but costs an
#   order of magnitude of accuracy, because the J0 factor's error is amplified by
#   |(2/pi) log x| as x -> 0+ (assembled Y0 max abs error 3.1e-06 -> 1.27e-04).
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

# Numeric constants are written inline below: a Triton @jit body cannot read
# module-level python globals (only tl.constexpr ones), so keep them literal.
#   0.785398163397448279  = pi/4
#   0.636619772367581343  = 2/pi
#   0.0795774715459476678 = (2/pi) * (1/8), folding z = 8/x into sqrt(2/(pi x))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def special_bessel_y0_func(x):
    xf = x.to(tl.float32)
    is_small = xf < 8.0

    # ----- x < 8 : R/S + (2/pi) log(x) J0(x), both rationals degree 5/5 in x^2 -
    y = xf * xf
    r = -2957821389.0 + y * (
        7062834065.0
        + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733)))
    )
    s = 40076544269.0 + y * (
        745249964.8
        + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0)))
    )
    pj = 57568490574.0 + y * (
        -13362590354.0
        + y
        * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456))))
    )
    qj = 57568490411.0 + y * (
        1029532985.0
        + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0)))
    )
    ans_small = r / s + 0.636619772367581343 * tl.log(xf) * (pj / qj)

    # ----- x >= 8 : asymptotic expansion in w = (8/x)^2 -----
    # No clamp on the division: the lanes with x <= 0 produce +-inf/NaN here but
    # are discarded by the final select, and a guarding `tl.where` measured 9%
    # slower (2.275 vs 2.066 ms at [4096,4096], reproduced 3x).
    z = 8.0 / xf
    w = z * z
    xx = xf - 0.785398163397448279
    p1 = 1.0 + w * (
        -0.1098628627e-2
        + w * (0.2734510407e-4 + w * (-0.2073370639e-5 + w * 0.2093887211e-6))
    )
    q1 = -0.1562499995e-1 + w * (
        0.1430488765e-3
        + w * (-0.6911147651e-5 + w * (0.7621095161e-6 + w * (-0.934945152e-7)))
    )
    ans_large = tl.sqrt(0.0795774715459476678 * z) * (
        tl.sin(xx) * p1 + z * tl.cos(xx) * q1
    )

    # ----- combine -----
    # Domain edges need no explicit fixup and match torch's CPU reference exactly:
    #   x == 0        -> small branch, log(0) = -inf, J0(0) = 1  -> -inf
    #   x <  0        -> small branch, log(x) = NaN              -> NaN
    #   x == -inf     -> small branch, log(-inf) = NaN           -> NaN
    #   x == +inf     -> large branch, sin(inf) = NaN, and the sqrt factor is
    #                    exactly 0, 0 * NaN = NaN                -> NaN
    #   x == NaN      -> NaN < 8 is False, large branch          -> NaN
    # (verified against torch.special.bessel_y0 on CPU float64, which returns
    #  -inf / NaN / NaN / NaN / NaN for those inputs.)
    return tl.where(is_small, ans_small, ans_large).to(x.dtype)


def special_bessel_y0(A):
    logger.debug("GEMS SPECIAL_BESSEL_Y0")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"special_bessel_y0: unsupported dtype {A.dtype}, only float32 and float64 are supported"
        )
    return special_bessel_y0_func(A)
