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
# Kunlunxin (XPU) override of special_ndtri (aten::special_ndtri).
#
# Why an override exists at all:
#   the generic `flag_gems/ops/special_ndtri.py:31` binds
#   `tl_extra_shim.normcdfinv` and calls it from the pointwise body.  On XPU that
#   attribute exists (`hasattr(...) is True`) but the symbol does not, so the
#   kernel fails to *link*:
#     ld.lld: error: undefined symbol: Unsupported
#     >>> referenced by ... :(probe_func_kernel_rank_1)
#   surfaced as `RuntimeError: Command failed (1): [... xpu3-elfconv-triton ...]`.
#   Measured at HEAD (harness/probe/ndtri_xpu1_20260830/): 8/10 accuracy cases
#   fail on that single compile error and the benchmark aborts on its very first
#   cell, i.e. the operator is completely unavailable on this backend.  Same trap
#   as libdevice y0 / y1 / j0 / lgamma / integer-exponent pow.
#
# Replacement scheme:
#   Wichura's AS241 PPND7 (the single-precision variant) piecewise rational
#   approximation of the standard normal quantile, ~7 significant digits, which
#   is the whole of fp32.  Three regions, all evaluated with core Triton
#   primitives only (sub / mul / add / div / minimum / abs / log / sqrt / where
#   all link on XPU):
#     |q| <= 0.425   (q = p - 0.5, i.e. p in [0.075, 0.925]):
#         r = 0.180625 - q*q ;  ndtri = q * A(r)/B(r)   (deg 3 / deg 3)
#     otherwise, with r = sqrt(-log(min(p, 1-p))):
#         r <= 5   ->  ndtri = sign(q) * C(r-1.6)/D(r-1.6)   (deg 3 / deg 2)
#         r >  5   ->  ndtri = sign(q) * E(r-5)  /F(r-5)     (deg 3 / deg 2)
#   Both tail rationals are always evaluated and merged with one `tl.where`; an
#   elementwise python `if` is not expressible inside a Triton tile and a
#   bool->float conversion would trip the XPU `arith.uitofp` check at
#   BLOCK >= 256.  All coefficients are inline literals on purpose: a
#   @triton.jit body cannot read module-level python globals
#   (`NameError: Cannot access global variable ... from within @jit'ed function`),
#   and inlining everything into a single jit body also means the module globals
#   hold no other gems operator that SpecOpRegistrar could leave stale.
#
# Domain map, and why it needs only two selects (verified lane by lane against
# CPU float64 torch.special.ndtri, harness/probe/ndtri_xpu1_20260830/):
#   p == 0.5  -> central branch, q is exactly 0.0, so q * (A/B) is exactly +0.0
#   p == 0    -> tail, q < 0 so pt = p = 0, tl.log(0) = -inf, r = +inf.  `r > 5`
#                selects the r>5 rational, whose argument is clamped by
#                `1.0 / (1.0 / r + 1e-16)`: at 1e16 the *degree 3* numerator
#                overflows fp32 to +inf while the *degree 2* denominator stays
#                finite (1.2258e30), so the ratio is +inf and sign(q) = -1 gives
#                -inf, matching torch.  A plain unclamped +inf would give
#                inf/inf = nan.
#   p == 1    -> q >= 0 so pt = 1 - p = 0, same path, sign(q) = +1 -> +inf
#   p <  0 or p > 1 -> pt < 0, tl.log(negative) = nan, r = nan.  `nan > 5` is
#                False on XPU (measured), so the nan lane takes the r<=5
#                rational whose argument r-1.6 is *unclamped* nan -> nan.  This
#                is why the clamp is applied only to the r>5 argument.
#   p == nan  -> `tl.abs(nan) <= 0.425` is False, pt = 1 - nan = nan, same path.
#   The `sign(q)` factor is `q / tl.abs(q)` rather than a third select: a divide
#   costs ~0.089 ms versus ~0.24 ms for one more select at [4096,4096] fp32, and
#   the 0/0 = nan it produces at p == 0.5 lands only in the tail value, which the
#   central select discards.
#
# `1.0 - p` carries no cancellation error: for 0.5 <= p <= 1 the subtraction is
# exact (Sterbenz), and for p < 0.5 the select picks p itself.
#
# Cost model, *measured on this backend* rather than taken from the harness note
# (harness/probe/ndtri_xpu1_20260830/probe_[fghi]*.log, [4096,4096] fp32, vendor
# pointwise_dynamic, do_bench median; trivial `x*2` body = 0.1325 ms):
#     fma 0.076 | div 0.089 | log 0.074 | sqrt 0.074 | abs 0.027
#     first select in the body 0.98 (!) | each further select on a new
#     condition ~0.24 | on an existing condition ~0.17 | tl.minimum ~1.10 each
# The published table (fma 0.057, where 0.11, div/sqrt 0.033) underestimates a
# select on this code shape by ~9x, which is why the shape of the optimisation is
# "spend fma/div, never spend a select":
#   AS241 with tl.minimum for both pt and the clamp .......... 4.973 ms
#   + select-free pt (2u/(1+sqrt(1-4u))) and reciprocal clamp . 3.496 ms
#   + pt back to a tl.where, reciprocal clamp kept (shipped) .. 2.616 ms
#   (tl.where clamp instead of reciprocal 2.908; single tail division 2.758)
# 20 fma-class + 4 div + 1 log + 1 sqrt + 1 abs + 2 selects predicts ~3.4 ms
# against 2.616 measured, i.e. the model is now conservative rather than 2.5x
# optimistic.
#
# Measured non-levers on this exact code shape: max_tile_size 256/512/1024/2048
# land within 0.02% of each other (4.9748/4.9750/4.9755/4.9747 on the first
# body); unroll_num, buffer_size_limit, isCloseVectorization, kunlunAutoGrid and
# num_warps were measured to be inert on the sibling y0/y1 overrides.
# No CPU/ATen/native/composite fallback.
import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

# The logger name is deliberately the *generic* module name: tests/
# test_special_ndtri.py does `caplog.at_level("DEBUG", logger=
# "flag_gems.ops.special_ndtri")`, and a record emitted on
# `_kunlunxin.ops.special_ndtri` would inherit root's WARNING level and be
# dropped, breaking the `assert "GEMS SPECIAL_NDTRI" in caplog.text` in all four
# test functions.  `record.pathname` still points at this file, which is what the
# dispatch evidence uses.
logger = logging.getLogger("flag_gems.ops.special_ndtri")

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

# ndtri is implemented for single and double precision only in PyTorch
# (ndtri_cpu / ndtri_cuda both raise NotImplementedError for Half and BFloat16);
# the same gate is kept so this override matches the reference.  Note that on
# this device torch.float64 silently degrades to float32 (a tensor created with
# dtype=torch.float64 reports .dtype == torch.float32 and element_size() == 4),
# so the float64 entry only matters for callers that pass the dtype through.
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def special_ndtri_func(x):
    p = x.to(tl.float32)
    q = p - 0.5
    aq = tl.abs(q)

    # ----- central region: |q| <= 0.425  ->  q * A(r)/B(r), r = 0.180625 - q^2
    rc = 0.180625 - q * q
    num_c = 3.3871327179 + rc * (50.434271938 + rc * (159.29113202 + rc * 59.109374720))
    den_c = 1.0 + rc * (17.895169469 + rc * (78.757757664 + rc * 67.187563600))
    x_central = q * (num_c / den_c)

    # ----- tails: r = sqrt(-log(min(p, 1-p)))
    pt = tl.where(q < 0.0, p, 1.0 - p)
    r = tl.sqrt(-tl.log(pt))

    # r <= 5 branch; argument intentionally unclamped so nan survives
    r1 = r - 1.6
    num_1 = 1.4234372777 + r1 * (
        2.7568153900 + r1 * (1.3067284816 + r1 * 0.17023821103)
    )
    den_1 = 1.0 + r1 * (0.73700164250 + r1 * 0.12021132975)

    # r > 5 branch; r is clamped to 1e16 so that r == +inf yields +inf instead of
    # inf/inf = nan.  The clamp is arithmetic on purpose: 1/(1/inf + 1e-16) is
    # exactly 1e16, while for every finite r in [0.83, 9.4] the 1e-16 sits nine
    # orders of magnitude below an fp32 ulp of 1/r, so it is a no-op there.  A
    # `tl.minimum` would cost ~1.0 ms at [4096,4096] (measured) and would also
    # destroy nan; a `tl.where` costs ~0.29 ms; this costs ~0.11 ms.
    r2 = 1.0 / (1.0 / r + 1e-16) - 5.0
    num_2 = 6.6579051150 + r2 * (
        3.0812263860 + r2 * (0.42868294337 + r2 * 0.017337203997)
    )
    den_2 = 1.0 + r2 * (0.24197894225 + r2 * 0.012258202635)

    x_tail = (q / aq) * tl.where(r > 5.0, num_2 / den_2, num_1 / den_1)

    return tl.where(aq <= 0.425, x_central, x_tail).to(x.dtype)


def special_ndtri(self):
    logger.debug("GEMS SPECIAL_NDTRI")
    if self.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"\"ndtri\" not implemented for '{self.dtype}'; supported dtypes are "
            f"{_SUPPORTED_DTYPES}"
        )
    return special_ndtri_func(self)
