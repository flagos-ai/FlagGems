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
# Kunlunxin (XPU) override of special_bessel_j0 (aten::special_bessel_j0).
#
# Why an override exists at all:
#   the generic `flag_gems/ops/special_bessel_j0.py` calls `tl_extra_shim.j0(x)`,
#   which resolves to the XPU vendor libdevice `j0` and fails at link time with
#   `ld.lld: error: undefined symbol: Unsupported` (same class of failure as
#   lgamma / log1p / polygamma on this backend).  So J0 must be evaluated with
#   core Triton primitives only (sin/cos/sqrt/abs/where all link on XPU).
#
# Why *this* particular evaluation scheme (performance):
#   TritonXPU runs this pointwise body on a fixed 12-program grid, so the kernel
#   is ALU-bound, not bandwidth-bound: a trivial `x*2` pointwise reaches
#   ~1040 GB/s at [4096,4096] fp32 (0.129 ms) while the J0 body needs 4.78 ms
#   (28 GB/s), i.e. ~37x the memory floor.  Measured per-element costs on this
#   backend (16.7M elements, harness/probe/special_bessel_j0_perf_probe.py):
#   fma ~0.057 ms, tl.where ~0.11 ms, sin/cos ~0.12 ms, div/sqrt ~0.033 ms.
#   The previous body spent that budget on a 20-term Taylor series (19 fma)
#   *plus* an always-evaluated fdlibm pzero/qzero asymptotic block (~22 fma,
#   7 divisions) *plus* 7 `tl.where`s.
#   This version uses the Hart / Numerical-Recipes rational scheme, which needs
#   only 18 fma, 3 divisions, 2 transcendentals, 1 sqrt and 2 `tl.where`s:
#     |x| <  8 : J0 = P(x^2) / Q(x^2)                 (degree 5/5)
#     |x| >= 8 : J0 = sqrt(2/(pi*|x|)) *
#                     (cos(|x|-pi/4) * P1(w) - (8/|x|) * sin(|x|-pi/4) * Q1(w))
#                with w = (8/|x|)^2
#   Both branches are always evaluated and combined with `tl.where` (elementwise
#   python `if` is not usable inside a Triton tile, and a bool->float conversion
#   would trip the XPU `arith.uitofp` check at BLOCK >= 256).
#   Measured on XPU 5: [4096,4096] fp32 4.781 ms -> 1.762 ms (2.71x),
#   [1024,65536] 18.94 ms -> 6.93 ms (2.73x); accuracy also improves, max abs
#   error over a 4M-point grid on [-8,8] drops from 9.5e-6 to 1.0e-6 (suite
#   tolerance is atol=1e-4), see
#   harness/probe/special_bessel_j0_accuracy_probe.py.
#   Grid/tile geometry is *not* a lever here: num_ctas is hard-wired to 12 by the
#   vendor pointwise_dynamic 1d-tile path and num_warps changes timing by <1%.
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
#   0.0795774715459476678 = (2/pi) * (1/8), the 1/8 folding z = 8/|x| into 1/|x|


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def special_bessel_j0_func(x):
    xf = x.to(tl.float32)
    ax = tl.abs(xf)
    is_small = ax < 8.0

    # ----- |x| < 8 : degree-5/5 rational in y = x^2 -----
    # NaN propagates naturally through this branch (NaN < 8 is False selects the
    # large branch, whose ax stays NaN), so no explicit NaN fixup is needed.
    y = ax * ax
    p = 57568490574.0 + y * (
        -13362590354.0
        + y
        * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456))))
    )
    q = 57568490411.0 + y * (
        1029532985.0
        + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0)))
    )
    ans_small = p / q

    # ----- |x| >= 8 : asymptotic expansion in w = (8/|x|)^2 -----
    # Clamp the small lanes to 8 so that 8/ax can never divide by zero; those
    # lanes are discarded by the final select.
    axl = tl.where(is_small, 8.0, ax)
    z = 8.0 / axl
    w = z * z
    xx = axl - 0.785398163397448279
    p1 = 1.0 + w * (
        -0.1098628627e-2
        + w * (0.2734510407e-4 + w * (-0.2073370639e-5 + w * 0.2093887211e-6))
    )
    q1 = -0.1562499995e-1 + w * (
        0.1430488765e-3
        + w * (-0.6911147651e-5 + w * (0.7621095161e-6 + w * (-0.934935152e-7)))
    )
    ans_large = tl.sqrt(0.0795774715459476678 * z) * (
        tl.cos(xx) * p1 - z * tl.sin(xx) * q1
    )

    # ----- combine; J0(+-inf) = 0 (matches the previous XPU override) -----
    result = tl.where(is_small, ans_small, ans_large)
    result = tl.where(ax == float("inf"), 0.0, result)
    return result.to(x.dtype)


def special_bessel_j0(A):
    logger.debug("GEMS SPECIAL_BESSEL_J0")
    if A.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"special_bessel_j0: unsupported dtype {A.dtype}, only float32 and float64 are supported"
        )
    return special_bessel_j0_func(A)
