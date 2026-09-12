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
# Kunlunxin (XPU) override of special_log_ndtr / special_log_ndtr_
# (aten::special_log_ndtr), log Phi(x) for the standard normal CDF Phi.
#
# Why an override exists (two independent HEAD defects, both measured on XPU 4):
#
# 1. CORRECTNESS.  The generic `flag_gems/ops/special_log_ndtr.py:41` forms Phi
#    first and takes the log afterwards: `log(0.5 * tl_extra_shim.erfc(-x/sqrt2))`.
#    The XPU libdevice `erfc` (`_ZN3xpu4erfcEf`) links fine, but it is evidently
#    `1 - erf`, so its *absolute* resolution is one ulp of 1 and its relative
#    error grows like 1.2e-7/erfc(a) -- measured against a CPU float64 oracle
#    (harness/probe/special_log_ndtr_erfc_accuracy_probe.py):
#        a=1.99 -> 3.0e-5   a=2.48 -> 2.6e-4   a=3.00 -> 1.8e-3
#        a>=3.92 -> erfc returns *exactly 0.0*, so log gives -inf.
#    Taking the log of that reproduces the HEAD accuracy failures exactly
#    (4 of 6 cases, `Greatest absolute difference: 0.1029`, mismatch 0.1% of
#    lanes, i.e. the |x| > ~3.1 tail of randn), and makes every x < -5.55 lane
#    return -inf where torch returns a large finite value.
# 2. PERFORMANCE.  There was no vendor file at all, so the generic module kept
#    the *generic* `flag_gems.utils.pointwise_dynamic` codegen (verified:
#    `flag_gems.special_log_ndtr.__globals__["pointwise_dynamic"].__module__`
#    was `flag_gems.utils.pointwise_dynamic`).  That path runs at ~2.3 GB/s
#    here: 57.9 ms for [4096,4096] fp32.
#
# Scheme (never forms Phi on the left tail, so it cannot underflow to -inf):
#     za  = |x| / sqrt(2)
#     tt  = 1 / (1 + za/2)
#     P   = Numerical-Recipes `erfcc` Chebyshev polynomial in tt (10 coeffs),
#           for which  erfc(za) = tt * exp(P - za^2)  with |frac err| < 1.2e-7
#     e   = P - za^2                       ( = log(erfc(za)) - log(tt) )
#     x <= 0 :  log Phi(x) = e + log(0.5*tt)           -- pure log space, no exp
#     x >  0 :  log Phi(x) = log1p(q), q = -0.5*tt*exp(e)  -- 1-Phi is the small
#                                                            quantity here
# Validated in float32 arithmetic against a CPU float64 oracle over
# x in [-40, 10] (harness/probe/special_log_ndtr_scheme_check.py): max relative
# error 5.7e-7 on x <= 1, and log Phi(-40) = -804.6084 instead of -inf.
#
# log1p is NOT taken from the platform, for a measured reason
# (harness/probe/special_log_ndtr_log1p_probe.py, buckets of |y|):
#   tl_extra_shim.log1p links, but it behaves like plain log(1+y): relative
#   error 2.9e-3 at |y|=1e-5, 2.0e-1 at 1.5e-7, and it returns *exactly 0* for
#   |y| < ~3e-8.  Root cause is one level down: tl.log(s) near s=1 carries an
#   absolute error of ~3.5e-8 (half an ulp of 1), and below 1-s ~ 1e-4 it
#   returns literally s-1 (measured error == d^2/2 exactly, d = 1-s).
#   So log1p(q) is built here as  tl.log(s) + (q - (s-1))/s  with s = 1+q:
#   s-1 is exact by Sterbenz, so (q - (s-1)) is precisely the bit that rounding
#   dropped, and adding it back divided by s recovers what tl.log lost.  Measured
#   |y|=1e-9..1e-4: relative error falls to <= |y|/2 (i.e. absolute error
#   <= y^2/2 <= 5e-9) versus 1e0..3e-4 for the shim, and it is never worse.
#   Residual: for |y| >~ 1e-4 the answer inherits tl.log's own 3.5e-8 absolute
#   floor, so relative accuracy peaks around 3.5e-4 near x ~ 3.7 where
#   |log Phi| ~ 1e-4.  Left as is on purpose: the absolute error there is
#   3.5e-8, one ulp of 1, which is the platform's log floor, and buying relative
#   accuracy would cost a third branch (extra tl.where ~0.11 ms + 3 fma).
#
# Symbol availability was established by real *compile* probes, one process per
# case (harness/probe/special_log_ndtr_symbol_probe.py) -- `hasattr` proves
# nothing on this backend.  On XPU `tl_extra_shim` is
# `triton.language.extra.xpu.libdevice` (no `__nv_*` anywhere in it); each entry
# maps a dtype to an external symbol and the unavailable ones are literally
# spelled "Unsupported", which is what the linker then reports:
#     erfc fp32 -> _ZN3xpu4erfcEf     OK        log1p fp32 -> _ZN3xpu6log1pfEf  OK
#     erf  fp32 -> _ZN3xpu3erfEf      OK        exp/log    -> Unsupported       FAIL
#     y0/y1/j0/j1/lgamma              FAIL (matches the sibling bessel overrides)
# Hence exp/log below are core `tl.*` (both measured accurate to ~1 ulp away
# from 1), and no libdevice symbol is referenced by this file at all.
#
# Cost model (increment form, [4096,4096] fp32): 9 fma (Horner) + 1 div + 1 tl.log
# + 1 tl.exp + 1 div + 1 tl.where + ~7 mul/add on top of the ~0.13 ms pointwise
# floor -> ~1.2-1.5 ms predicted.
#
# Numeric constants are inline literals on purpose: a Triton @jit body cannot
# read module-level python globals, and inlining also means SpecOpRegistrar has
# no gems-level function reference left to leave pointing at a stale object.
#   0.7071067811865476 = 1/sqrt(2)
# No CPU/ATen/native/composite fallback; the whole evaluation is in the kernel.
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
def special_log_ndtr_func(x):
    xf = x.to(tl.float32)
    za = tl.abs(xf) * 0.7071067811865476
    tt = 1.0 / (1.0 + 0.5 * za)

    # Numerical Recipes `erfcc` polynomial: erfc(za) = tt * exp(P - za*za)
    p = -1.26551223 + tt * (
        1.00002368
        + tt
        * (
            0.37409196
            + tt
            * (
                0.09678418
                + tt
                * (
                    -0.18628806
                    + tt
                    * (
                        0.27886807
                        + tt
                        * (
                            -1.13520398
                            + tt * (1.48851587 + tt * (-0.82215223 + tt * 0.17087277))
                        )
                    )
                )
            )
        )
    )
    e = p - za * za

    # x <= 0: log(0.5*erfc(|x|/sqrt2)) evaluated entirely in log space.
    #   x == +-0.0 -> tt = 1, P(1) ~ 1e-9  -> log(0.5)   = -0.6931472  (matches)
    #   x == -inf  -> e = -inf, log(0.5*0) = -inf -> -inf             (matches)
    #   x == NaN   -> NaN > 0 is False, so this branch, NaN           (matches)
    left = e + tl.log(0.5 * tt)

    # x > 0: q = -(1 - Phi(x)) is the small quantity, so build it and take
    # log1p(q) with the rounding bit of (1+q) added back explicitly.
    #   x == +inf  -> tt = 0, e = -inf, exp(e) = 0, s = 1 -> 0.0            (matches)
    q = -0.5 * tt * tl.exp(e)
    s = 1.0 + q
    right = tl.log(s) + (q - (s - 1.0)) / s

    return tl.where(xf > 0.0, right, left).to(x.dtype)


def special_log_ndtr(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG_NDTR")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"special_log_ndtr: unsupported dtype {A.dtype}, "
            "only float16, bfloat16 and float32 are supported"
        )
    return special_log_ndtr_func(A)


def special_log_ndtr_(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_LOG_NDTR_")
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"special_log_ndtr_: unsupported dtype {A.dtype}, "
            "only float16, bfloat16 and float32 are supported"
        )
    special_log_ndtr_func(A, out0=A)
    return A
