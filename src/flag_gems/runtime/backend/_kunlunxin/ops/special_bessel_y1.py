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
# Kunlunxin (XPU) override of special_bessel_y1 (aten::special_bessel_y1).
#
# Why an override exists at all:
#   the generic `flag_gems/ops/special_bessel_y1.py:35` calls `tl_extra_shim.y1(x)`,
#   which resolves to the XPU vendor libdevice `y1` and fails to *link*:
#     ld.lld: error: undefined symbol: Unsupported
#     >>> referenced by special_bessel_y1.py:35 ... special_bessel_y1_func_kernel_rank_1
#   so every call raises `RuntimeError: Command failed (1): [... xpu3-elfconv-triton ...]`.
#   `hasattr(tl_extra_shim, "y1")` is True, so only an actual compile exposes it --
#   the same trap as libdevice y0 / j0 / lgamma / integer-exponent pow on this
#   backend.  Consequence at HEAD: 10/10 accuracy cases fail and the benchmark
#   aborts on its first cell, i.e. the operator is entirely unavailable on XPU.
#   Y1 must therefore be evaluated with core Triton primitives only
#   (log / sin / cos / sqrt / where all link on XPU).
#
# Why *this* evaluation scheme (performance):
#   TritonXPU runs a pointwise body on a fixed 12-program grid, so the kernel is
#   ALU-bound, not bandwidth-bound: a trivial `x*2` reaches ~1033 GB/s at
#   [4096,4096] fp32 (0.130 ms) while this Y1 body needs ~1.92 ms in the same
#   probe, ~15x the memory floor.  Op count is therefore the only lever.  The
#   scheme is the Numerical-Recipes `bessy1` rational pair, Horner-step count
#   ~42 fma / 4 div / 1 sqrt / 3 transcendental (log, sin, cos) / 1 tl.where:
#     x <  8 : Y1 = x*(P/Q) + (2/pi)*( log(x)*x*(Pj/Qj) - 1/x )
#              rearranged to  x*( P/Q + (2/pi)*log(x)*(Pj/Qj) ) - (2/pi)/|x|
#              so the leading x is applied once instead of twice; that
#              rearrangement is worth 4.8% (1.874 vs 1.963 ms at [4096,4096]).
#              P is degree 5 and Q degree 6 in y = x^2.  Pj/Qj is the degree-5/5
#              J1 rational -- the same one the sibling special_bessel_j1 override
#              uses.  It is inlined here on purpose rather than imported, so no
#              gems-level function reference exists that SpecOpRegistrar could
#              leave pointing at a stale object.
#     x >= 8 : Y1 = sqrt(2/(pi x)) * ( sin(x-3pi/4) * P1(w)
#                                    + (8/x) * cos(x-3pi/4) * Q1(w) ),  w = (8/x)^2
#              with NR's *order-1* P1/Q1 coefficients (different from the order-0
#              ones the Y0 sibling uses) and the exact 3pi/4 instead of NR's
#              truncated 2.356194491.
#   Both branches are always evaluated and merged with a single `tl.where` (an
#   elementwise python `if` is not expressible inside a Triton tile, and a
#   bool->float conversion would trip the XPU `arith.uitofp` check at
#   BLOCK >= 256).  No clamp on either division: the lanes that fall in the wrong
#   branch produce +-inf/NaN and are discarded by the select, and a guarding
#   `tl.where` measured 9% slower on the Y0 sibling.
#
# Why `tl.log(xf + 1e-37)` and `tl.abs(xf)` (correctness, not cosmetics):
#   torch.randn on this device emits exact +0.0 at a measured rate of ~1.0e-07
#   (harness/probe/special_bessel_y1_nanhunt_probe.py, 20 hits in 200M lanes), and
#   the two largest SPECIAL_SHAPES in tests/ are 167.8M and 5.9M lanes -- so Y1(0)
#   is reached in practice and must be -inf like torch, not NaN.  Written plainly,
#   `xf * inner` is 0 * (-inf) = NaN at x == 0 because log(0) = -inf.  Nudging the
#   log argument by 1e-37 keeps `inner` finite, so `xf*inner` is 0 and the
#   `-(2/pi)/|x|` term supplies -inf; `tl.abs` extends that to x == -0.0.  The
#   nudge is a no-op for every |x| > ~1.7e-30 (it is below half an fp32 ulp there)
#   and is multiplied by x below that, so it costs no accuracy: 1e6 randn lanes
#   are bit-identical to the unnudged body.  Cost 2.4% (1.918 vs 1.874 ms) versus
#   17.2% (2.197 ms) for the obvious `tl.where(xf == 0.0, -inf, res)`.
#   The literal magnitude matters: 1e-38 and below make the XPU backend lower
#   tl.log to an unlinkable `log2` (`ld.lld: error: undefined symbol: log2`),
#   1e-37 and above link -- see harness/probe/special_bessel_y1_logtrap_probe.py.
#
#   Measured non-levers (see harness/solution/performance/special_bessel_y1_xpu1_20260830.md):
#   max_tile_size 256/512/1024/2048 land within 0.06% of each other; num_ctas is
#   hard-wired to 12 by the vendor pointwise_dynamic 1d-tile path; unroll_num /
#   buffer_size_limit / isCloseVectorization / kunlunAutoGrid / num_warps were
#   measured to be non-levers for this exact code shape on the Y0 sibling.
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
#   2.356194490192344929  = 3*pi/4
#   0.636619772367581343  = 2/pi
#   0.0795774715459476678 = (2/pi) * (1/8), folding z = 8/x into sqrt(2/(pi x))


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def special_bessel_y1_func(x):
    xf = x.to(tl.float32)
    is_small = xf < 8.0

    # ----- x < 8 : x*(P/Q) + (2/pi)*( log(x)*J1(x) - 1/x ) -------------------
    y = xf * xf
    p = -4900604943000.0 + y * (
        1275274390000.0
        + y
        * (-51534381390.0 + y * (734926455.1 + y * (-4237922.726 + y * 8511.937935)))
    )
    q = 24995805700000.0 + y * (
        424441966400.0
        + y
        * (
            3733650367.0
            + y * (22459040.02 + y * (102042.605 + y * (354.9632885 + y * 1.0)))
        )
    )
    # the J1 rational, inlined; its leading x is hoisted into `xf * inner` below
    pj = 72362614232.0 + y * (
        -7895059235.0
        + y
        * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))
    )
    qj = 144725228442.0 + y * (
        2300535178.0
        + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0)))
    )
    inner = p / q + 0.636619772367581343 * (tl.log(xf + 1e-37) * (pj / qj))
    ans_small = xf * inner - 0.636619772367581343 / tl.abs(xf)

    # ----- x >= 8 : asymptotic expansion in w = (8/x)^2 -----
    z = 8.0 / xf
    w = z * z
    xx = xf - 2.356194490192344929
    p1 = 1.0 + w * (
        0.183105e-2
        + w * (-0.3516396496e-4 + w * (0.2457520174e-5 + w * (-0.240337019e-6)))
    )
    q1 = 0.04687499995 + w * (
        -0.2002690873e-3
        + w * (0.8449199096e-5 + w * (-0.88228987e-6 + w * 0.105787412e-6))
    )
    ans_large = tl.sqrt(0.0795774715459476678 * z) * (
        tl.sin(xx) * p1 + z * tl.cos(xx) * q1
    )

    # ----- combine -----
    # Domain map, verified lane by lane against torch.special.bessel_y1 on CPU
    # float64 (harness/probe/special_bessel_y1_zerofix_probe.py):
    #   x == +0 / -0 -> small branch, inner finite thanks to the 1e-37 nudge, so
    #                   0*inner - (2/pi)/|0| = -inf                  (matches)
    #   x <  0       -> small branch, log(x) = NaN -> NaN            (matches)
    #   x == -inf    -> small branch, log(-inf) = NaN -> NaN         (matches)
    #   x == +inf    -> large branch, sin(inf) = NaN and the sqrt factor is
    #                   exactly 0, 0*NaN = NaN                       (matches)
    #   x == NaN     -> NaN < 8 is False, large branch -> NaN        (matches)
    # The one residual divergence is a platform property, not a formula defect:
    # the device flushes subnormals *and* the smallest normal to zero, so every
    # 0 < |x| <= 1.17549e-38 lane is literally the x == 0 lane inside the kernel
    # and returns -inf, where torch returns -inf only below ~1.87e-39 and a
    # finite ~-6e37 above it (and NaN for negative subnormals).
    return tl.where(is_small, ans_small, ans_large).to(x.dtype)


def special_bessel_y1(A):
    logger.debug("GEMS SPECIAL_BESSEL_Y1")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"special_bessel_y1: unsupported dtype {A.dtype}, only float32 and float64 are supported"
        )
    return special_bessel_y1_func(A)
