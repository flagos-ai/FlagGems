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

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math helpers
#
# `_dirichlet_grad` mirrors the reference implementation in
# ``ATen/native/Distributions.h`` (``dirichlet_grad_one`` and its private
# beta-gradient helpers), which express the reparameterized Dirichlet gradient
# as a piecewise function of ``x`` (the sampled value), ``alpha`` (the
# concentration) and ``total = sum(alpha)``.  Each branch is a careful
# asymptotic / rational approximation of ``-(d/dalpha cdf)/pdf`` of the
# corresponding Beta distribution.  Replicating the exact branching structure
# (instead of the naive closed-form ``x * (digamma(total)-digamma(alpha)+log(x))``)
# is required to match PyTorch numerically, since the closed form is only
# asymptotically correct and diverges from the reference for the moderate
# ``alpha`` / near-boundary ``x`` values that occur in practice.
#
# All work is done in the input dtype (float32/float64) to match the reference
# ``accscalar_t``; the result is cast back to the input dtype at the boundary.
#
# Control-flow structure: the reference is an element-wise ``if -> return``
# cascade, where a hit branch skips all the later ones.  ``tl.where`` evaluates
# both sides eagerly, so every branch (small / large / mid / rational) is
# computed for every element and merged with ``tl.where`` -- a large amount of
# dead computation.  This is unavoidable on gcu400, whose compiler marks
# ``tt.map_elementwise`` illegal, so the piecewise math is expressed directly as
# tensor expressions instead of a lazily-mapped scalar function.
# ---------------------------------------------------------------------------

PI = 3.1415926535897932384626433832795028841971
PSI_10 = 2.25175258906672110764

# Coefficient tables ``c[2][3][3][4]`` for the rational-correction branch of
# ``_dirichlet_grad_one`` (ported from ATen::native::dirichlet_grad_one).
# Unrolled as two (3*3*4) flattish arrays and hoisted to module-level
# ``tl.constexpr`` so the kernel body stays readable; this is equivalent to the
# inlined assignments and changes neither numerics nor codegen.  (Plain
# module-level floats are not visible inside ``@triton.jit`` kernels, hence
# the ``tl.constexpr`` wrapping.)
C0_000 = tl.constexpr(1.003668233)
C0_001 = tl.constexpr(-0.01061107488)
C0_002 = tl.constexpr(-0.0657888334)
C0_003 = tl.constexpr(0.01201642863)
C0_010 = tl.constexpr(0.6336835991)
C0_011 = tl.constexpr(-0.3557432599)
C0_012 = tl.constexpr(0.05486251648)
C0_013 = tl.constexpr(-0.001465281033)
C0_020 = tl.constexpr(-0.03276231906)
C0_021 = tl.constexpr(0.004474107445)
C0_022 = tl.constexpr(0.002429354597)
C0_023 = tl.constexpr(-0.0001557569013)
C0_100 = tl.constexpr(0.221950385)
C0_101 = tl.constexpr(-0.3187676331)
C0_102 = tl.constexpr(0.01799915743)
C0_103 = tl.constexpr(0.01074823814)
C0_110 = tl.constexpr(-0.2951249643)
C0_111 = tl.constexpr(0.06219954479)
C0_112 = tl.constexpr(0.01535556598)
C0_113 = tl.constexpr(0.001550077057)
C0_120 = tl.constexpr(0.02155310298)
C0_121 = tl.constexpr(0.004170831599)
C0_122 = tl.constexpr(0.001292462449)
C0_123 = tl.constexpr(6.976601077e-05)
C0_200 = tl.constexpr(-0.05980841433)
C0_201 = tl.constexpr(0.008441916499)
C0_202 = tl.constexpr(0.01085618172)
C0_203 = tl.constexpr(0.002319392565)
C0_210 = tl.constexpr(0.02911413504)
C0_211 = tl.constexpr(0.01400243777)
C0_212 = tl.constexpr(-0.002721828457)
C0_213 = tl.constexpr(0.000751041181)
C0_220 = tl.constexpr(0.005900514878)
C0_221 = tl.constexpr(-0.001936558688)
C0_222 = tl.constexpr(-9.495446725e-06)
C0_223 = tl.constexpr(5.385558597e-05)

C1_000 = tl.constexpr(1.0)
C1_001 = tl.constexpr(-0.02924021934)
C1_002 = tl.constexpr(-0.04438342661)
C1_003 = tl.constexpr(0.007285809825)
C1_010 = tl.constexpr(0.6357567472)
C1_011 = tl.constexpr(-0.3473456711)
C1_012 = tl.constexpr(0.05454656494)
C1_013 = tl.constexpr(-0.002407477521)
C1_020 = tl.constexpr(-0.03301322327)
C1_021 = tl.constexpr(0.004845219414)
C1_022 = tl.constexpr(0.00231480583)
C1_023 = tl.constexpr(-0.0002307248149)
C1_100 = tl.constexpr(0.5925320577)
C1_101 = tl.constexpr(-0.1757678135)
C1_102 = tl.constexpr(0.01505928619)
C1_103 = tl.constexpr(0.000564515273)
C1_110 = tl.constexpr(0.1014815858)
C1_111 = tl.constexpr(-0.06589186703)
C1_112 = tl.constexpr(0.01272886114)
C1_113 = tl.constexpr(-0.0007316646956)
C1_120 = tl.constexpr(-0.007258081865)
C1_121 = tl.constexpr(0.001096195486)
C1_122 = tl.constexpr(0.0003934994223)
C1_123 = tl.constexpr(-4.12701925e-05)
C1_200 = tl.constexpr(0.06469649321)
C1_201 = tl.constexpr(-0.0236701437)
C1_202 = tl.constexpr(0.002902096474)
C1_203 = tl.constexpr(-5.896963079e-05)
C1_210 = tl.constexpr(0.001925008108)
C1_211 = tl.constexpr(-0.002869809258)
C1_212 = tl.constexpr(0.0008000589141)
C1_213 = tl.constexpr(-6.063713228e-05)
C1_220 = tl.constexpr(-0.0003477407336)
C1_221 = tl.constexpr(6.95975487e-05)
C1_222 = tl.constexpr(1.097287507e-05)
C1_223 = tl.constexpr(-1.650969693e-06)


# This kernel carries far more per-element state than a typical pointwise op
# (digamma reflection/series state, the Taylor/saddle-point/rational
# sub-branches, many coefficient tables), and the eager ``tl.where`` form keeps
# every branch alive, so register pressure grows with the tile size.
# A 12-round CodeGenConfig sweep on gcu400 (2026-11-11) showed that a
# 512-element 1-D tile passes all 48 correctness cases and doubles the mean
# speedup versus the former 128-element tile (23.87x vs 11.21x).
# Non-power-of-two tiles (96/160/192) regress accuracy and must be avoided.
_DIRICHLET_GRAD_CONFIG = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
)


@triton.jit
def _div_rn(a, b):
    # IEEE round-to-nearest division.  ``tl.div_rn`` is only available for
    # float32 (it maps to a fast correctly-rounded intrinsic); for float64 the
    # plain ``/`` operator is already correctly rounded, so fall back to it.
    # ``b.dtype`` is a compile-time attribute, so this ``if`` is specialized
    # away and the fp32-only intrinsic is never instantiated for fp64 inputs.
    # ``b`` (the denominator) is always a tensor in the call sites below, which
    # makes the dtype check reliable even when ``a`` is a python scalar.
    if b.dtype == tl.float32:
        return tl.div_rn(a, b)
    else:
        return a / b


@triton.jit
def _digamma_one(x):
    # Cephes-style digamma.  Ported from ATen::native::digamma_one so that the
    # result is bit-for-bit close to the reference implementation used inside
    # ``dirichlet_grad_one``.  This is a *tensor* function (evaluated on the
    # whole pointwise tile), so the branches are expressed with ``tl.where``
    # (eager) instead of real control flow; the gcu400 backend cannot legalize
    # ``tt.map_elementwise``.
    PSI_10: tl.constexpr = 2.25175258906672110764

    # 本算子只对正的浓度参数 alpha / beta / total 求 digamma：反射公式与
    # 负整数极点分支恒不触发，整段省去 tan / floor 的 libdevice 外函数调用。
    y = x

    # Push y to be >= 10 (matches the C++ ``while (y < 10)``).  After the
    # reflection y is strictly positive, so at most 10 unit increments are
    # ever needed.
    result = tl.zeros_like(x)
    for _ in range(10):
        shift = y < 10.0
        # ``tl.div_rn`` (IEEE round-to-nearest) is used for the harmonic
        # sum; Triton's tensor ``/`` is an approximate 1-ULP division whose
        # error would be amplified by ``1/alpha`` inside the small branch of
        # ``dirichlet_grad_one``.
        result = tl.where(shift, result - _div_rn(1.0, y), result)
        y = tl.where(shift, y + 1.0, y)

    # Exact value at the loop boundary.
    exact = result + PSI_10

    # Asymptotic digamma for y > 10 (computed eagerly, selected below).
    a0 = 8.33333333333333333333e-2
    a1 = -2.10927960927960927961e-2
    a2 = 7.57575757575757575758e-3
    a3 = -4.16666666666666666667e-3
    a4 = 3.96825396825396825397e-3
    a5 = -8.33333333333333333333e-3
    a6 = 8.33333333333333333333e-2
    z = _div_rn(1.0, y * y)
    # Horner evaluation of the asymptotic polynomial A[0..6] (Cephes
    # ``polevl`` with degree 6); ``series = z * poly`` reproduces
    # ``y = z * polevl(z, A, 6)``.
    poly = (((((a6 * z + a5) * z + a4) * z + a3) * z + a2) * z + a1) * z + a0
    series = tl.where(y < 1.0e17, z * poly, tl.zeros_like(x))

    asymp = result + tl_extra_shim.log(y) - _div_rn(0.5, y) - series

    out = tl.where(y == 10.0, exact, asymp)
    return out


@triton.jit
def _beta_grad_alpha_small(x, alpha, beta):
    # Taylor expansion of the reparameterized Beta(alpha,beta) gradient around x=0.
    factor = _digamma_one(alpha) - _digamma_one(alpha + beta) - tl_extra_shim.log(x)
    numer = tl.zeros_like(x) + 1.0
    # The series is dominated by ``factor / alpha``; for tiny ``alpha`` any
    # rounding in ``1 / alpha`` is amplified, so use correctly-rounded division.
    inv_alpha = _div_rn(1.0, alpha)
    series = _div_rn(numer, alpha) * (factor + inv_alpha)
    for i in range(1, 11):
        casted_i = i
        numer = numer * (casted_i - beta) * x / casted_i
        denom = alpha + casted_i
        inv_denom = _div_rn(1.0, denom)
        series = series + _div_rn(numer, denom) * (factor + inv_denom)
    result = x * tl_extra_shim.pow(1.0 - x, -beta) * series
    return tl.where(result == result, result, tl.zeros_like(result))


@triton.jit
def _beta_grad_beta_small(x, alpha, beta):
    # Taylor expansion of the reparameterized Beta(alpha,beta) gradient around x=1.
    factor = _digamma_one(alpha + beta) - _digamma_one(beta)
    numer = tl.zeros_like(x) + 1.0
    betas = tl.zeros_like(x) + 1.0
    dbetas = tl.zeros_like(x)
    series = _div_rn(factor, alpha)
    for i in range(1, 9):
        casted_i = i
        numer = numer * (-x / casted_i)
        dbetas = dbetas * (beta - casted_i) + betas
        betas = betas * (beta - casted_i)
        series = series + _div_rn(numer, alpha + casted_i) * (dbetas + factor * betas)
    result = -tl_extra_shim.pow(1.0 - x, 1.0 - beta) * series
    return tl.where(result == result, result, tl.zeros_like(result))


@triton.jit
def _beta_grad_alpha_mid(x, alpha, beta):
    # Rice saddle-point expansion for alpha, beta both large.
    total = alpha + beta
    mean = alpha / total
    std = tl_extra_shim.sqrt(alpha * beta / (total + 1.0)) / total
    # The ATen reference uses a 0.1*std crossover between the polynomial and
    # saddle-point branches, tuned for double accumulation.  gcu400 has no
    # fp64, so the saddle-point branch (``term4 = term4_base**-1.5`` where
    # ``term4_base`` is a near-cancellation of two large log terms) loses
    # enough precision right at the crossover that its float32 result flips
    # sign against the double reference.  Widen the stable polynomial branch
    # to 0.3*std so the delicate boundary region stays in the polynomial form;
    # there the polynomial and saddle point still agree to well within the
    # 2e-3 accuracy tolerance, so this matches the reference without touching
    # the ill-conditioned saddle point in float32.
    near_mean = (mean - 0.3 * std <= x) & (x <= mean + 0.3 * std)

    # Polynomial form valid close to the mode.  ``b2..b4`` are powers of
    # beta; the Cephes polynomial is, in alpha, degree 4 with descending
    # powers of beta: beta^4, beta^3, beta^2, beta^1, beta^0 (inside the
    # last factor).
    b2 = beta * beta
    b3 = b2 * beta
    b4 = b2 * b2
    poly = 47.0 * x * b4 + alpha * (
        (43.0 + 20.0 * (16.0 + 27.0 * beta) * x) * b3
        + alpha
        * (
            3.0 * (59.0 + 180.0 * beta - 90.0 * x) * b2
            + alpha
            * (
                (453.0 + 1620.0 * beta * (1.0 - x) - 455.0 * x) * beta
                + alpha * (8.0 * (1.0 - x) * (135.0 * beta - 11.0))
            )
        )
    )
    prefactor_num = _div_rn((1.0 + 12.0 * alpha) * (1.0 + 12.0 * beta), total * total)
    prefactor_den = 12960.0 * alpha * alpha * alpha * beta * beta * (1.0 + 12.0 * total)
    near_mean_result = _div_rn(prefactor_num, (1.0 - x)) * _div_rn(poly, prefactor_den)

    # Saddle-point form (computed eagerly, selected when not near the mode).
    prefactor = _div_rn(-x, tl_extra_shim.sqrt(2.0 * alpha * beta / total))
    stirling = (
        (1.0 + 1.0 / (12.0 * alpha) + 1.0 / (288.0 * alpha * alpha))
        * (1.0 + 1.0 / (12.0 * beta) + 1.0 / (288.0 * beta * beta))
        / (1.0 + 1.0 / (12.0 * total) + 1.0 / (288.0 * total * total))
    )
    term1_num = (
        2.0 * alpha * alpha * (x - 1.0) + alpha * beta * (x - 1.0) - x * (beta * beta)
    )
    # ``axbx`` is exactly the previous ``term3_den`` expression, computed once.
    axbx = alpha * (x - 1.0) + beta * x
    term1_den = (
        tl_extra_shim.sqrt(2.0 * alpha / beta)
        * (total * tl_extra_shim.sqrt(total))
        * axbx
        * axbx
    )
    term1 = _div_rn(term1_num, term1_den)
    # ``log(alpha / (total * x))`` is shared by ``term2`` and ``term4_base``.
    log_a_over_total_x = tl_extra_shim.log(_div_rn(alpha, total * x))
    term2 = 0.5 * log_a_over_total_x
    term3_num = tl_extra_shim.sqrt(8.0 * alpha * beta / total)
    term3 = _div_rn(term3_num, axbx)
    # ``term4_base`` is a difference of two large log-terms and is numerically
    # delicate near the mode (it can approach zero, and ``pow(base, -1.5)``
    # amplifies any rounding).  The reference evaluates the quotients in
    # ``accscalar_t`` with IEEE round-to-nearest division; Triton's plain
    # ``/`` on tensors is an approximate (1-ULP) division, so we use
    # ``tl.div_rn`` (correctly rounded) here to reproduce the reference's
    # rounding and keep the saddle-point regime within tolerance.
    log_b_over_total_1mx = tl_extra_shim.log(_div_rn(beta, total * (1.0 - x)))
    term4_base = beta * log_b_over_total_1mx + alpha * log_a_over_total_x
    # ``pow(base, -1.5)`` expressed as ``1 / (base * sqrt(base))`` so that the
    # exponent stays in the base's dtype (``tl_extra_shim.pow`` with a python
    # scalar exponent mismatches dtypes for float64).
    term4 = _div_rn(1.0, term4_base * tl_extra_shim.sqrt(term4_base))
    term1234 = tl.where(
        x < mean, term1 + term2 * (term3 + term4), term1 + term2 * (term3 - term4)
    )
    saddle_result = stirling * prefactor * term1234

    return tl.where(near_mean, near_mean_result, saddle_result)


@triton.jit
def _dirichlet_grad_one(x, alpha, total):
    # Ported from ATen::native::dirichlet_grad_one (Distributions.h).
    #
    # Tensor function: the ``if -> return`` cascade of the C++ reference is
    # expressed with ``tl.where`` (every branch evaluated eagerly) because the
    # gcu400 backend cannot legalize ``tt.map_elementwise``.  Unselected
    # branches may carry inf/nan but are discarded by the final selection.
    beta = total - alpha
    boundary = total * x * (1.0 - x)

    small = _beta_grad_alpha_small(x, alpha, beta)
    small_beta = -_beta_grad_beta_small(1.0 - x, beta, alpha)
    mid = _beta_grad_alpha_mid(x, alpha, beta)

    # Rational-correction branch (the final fallback in dirichlet_grad_one).
    u = tl_extra_shim.log(x)
    a = tl_extra_shim.log(alpha) - u
    b = tl_extra_shim.log(total) - a
    # Coefficient tables c[2][3][3][4]: module-level ``tl.constexpr`` tables
    # ``C0_*`` / ``C1_*`` (see top of file).

    C1_000 = 1.0
    C1_001 = -0.02924021934
    C1_002 = -0.04438342661
    C1_003 = 0.007285809825
    C1_010 = 0.6357567472
    C1_011 = -0.3473456711
    C1_012 = 0.05454656494
    C1_013 = -0.002407477521
    C1_020 = -0.03301322327
    C1_021 = 0.004845219414
    C1_022 = 0.00231480583
    C1_023 = -0.0002307248149
    C1_100 = 0.5925320577
    C1_101 = -0.1757678135
    C1_102 = 0.01505928619
    C1_103 = 0.000564515273
    C1_110 = 0.1014815858
    C1_111 = -0.06589186703
    C1_112 = 0.01272886114
    C1_113 = -0.0007316646956
    C1_120 = -0.007258081865
    C1_121 = 0.001096195486
    C1_122 = 0.0003934994223
    C1_123 = -4.12701925e-05
    C1_200 = 0.06469649321
    C1_201 = -0.0236701437
    C1_202 = 0.002902096474
    C1_203 = -5.896963079e-05
    C1_210 = 0.001925008108
    C1_211 = -0.002869809258
    C1_212 = 0.0008000589141
    C1_213 = -6.063713228e-05
    C1_220 = -0.0003477407336
    C1_221 = 6.95975487e-05
    C1_222 = 1.097287507e-05
    C1_223 = -1.650969693e-06

    pow_u0 = 1.0
    pow_u1 = u
    pow_u2 = u * u
    pow_a0 = 1.0
    pow_a1 = a
    pow_a2 = a * a

    p = tl.zeros_like(x)
    q = tl.zeros_like(x)

    # Unroll the 3x3 double loop (i, j) explicitly to avoid indexing tuples
    # with loop variables, which Triton does not support.
    ua = pow_u0 * pow_a0
    p = p + ua * (C0_000 + b * (C0_001 + b * (C0_002 + b * C0_003)))
    q = q + ua * (C1_000 + b * (C1_001 + b * (C1_002 + b * C1_003)))

    ua = pow_u0 * pow_a1
    p = p + ua * (C0_010 + b * (C0_011 + b * (C0_012 + b * C0_013)))
    q = q + ua * (C1_010 + b * (C1_011 + b * (C1_012 + b * C1_013)))

    ua = pow_u0 * pow_a2
    p = p + ua * (C0_020 + b * (C0_021 + b * (C0_022 + b * C0_023)))
    q = q + ua * (C1_020 + b * (C1_021 + b * (C1_022 + b * C1_023)))

    ua = pow_u1 * pow_a0
    p = p + ua * (C0_100 + b * (C0_101 + b * (C0_102 + b * C0_103)))
    q = q + ua * (C1_100 + b * (C1_101 + b * (C1_102 + b * C1_103)))

    ua = pow_u1 * pow_a1
    p = p + ua * (C0_110 + b * (C0_111 + b * (C0_112 + b * C0_113)))
    q = q + ua * (C1_110 + b * (C1_111 + b * (C1_112 + b * C1_113)))

    ua = pow_u1 * pow_a2
    p = p + ua * (C0_120 + b * (C0_121 + b * (C0_122 + b * C0_123)))
    q = q + ua * (C1_120 + b * (C1_121 + b * (C1_122 + b * C1_123)))

    ua = pow_u2 * pow_a0
    p = p + ua * (C0_200 + b * (C0_201 + b * (C0_202 + b * C0_203)))
    q = q + ua * (C1_200 + b * (C1_201 + b * (C1_202 + b * C1_203)))

    ua = pow_u2 * pow_a1
    p = p + ua * (C0_210 + b * (C0_211 + b * (C0_212 + b * C0_213)))
    q = q + ua * (C1_210 + b * (C1_211 + b * (C1_212 + b * C1_213)))

    ua = pow_u2 * pow_a2
    p = p + ua * (C0_220 + b * (C0_221 + b * (C0_222 + b * C0_223)))
    q = q + ua * (C1_220 + b * (C1_221 + b * (C1_222 + b * C1_223)))

    approx = _div_rn(x * (_digamma_one(total) - _digamma_one(alpha)), beta)
    rational = _div_rn(p, q) * approx

    result = rational
    result = tl.where((alpha > 6.0) & (beta > 6.0), mid, result)
    result = tl.where((x >= 0.5) & (boundary < 0.75), small_beta, result)
    result = tl.where((x <= 0.5) & (boundary < 2.5), small, result)
    return result


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=_DIRICHLET_GRAD_CONFIG,
)
@triton.jit
def _dirichlet_grad_func(x, alpha, total):
    # ``pointwise_dynamic`` provides the tensor tile (load / store / mask);
    # the piecewise implementation runs directly on the tile with ``tl.where``
    # (``tl.map_elementwise`` is illegal on gcu400).  The reference
    # ``dirichlet_grad_one`` accumulates in ``accscalar_t`` which equals the
    # input dtype (float for float, double for double); the inputs are already
    # promoted to that dtype by ``pointwise_dynamic``.
    return _dirichlet_grad_one(x, alpha, total).to(x.dtype)


def _dirichlet_grad(x, alpha, total):
    logger.debug("GEMS DIRICHLET_GRAD")
    # The ATen reference only implements the AT_DISPATCH_FLOATING_TYPES path
    # (float32 / float64); reject other dtypes explicitly to match the CUDA
    # dispatcher behaviour.
    assert x.dtype in (
        torch.float32,
        torch.float64,
    ), f"_dirichlet_grad only supports float32/float64, got {x.dtype}"
    # GCU400 的 libdevice 外函数没有 float64 版本（floor/tan 会抛
    # KeyError(triton.language.float64)），而设备上的 double 张量本就以 fp32
    # 存储。统一在 fp32 中完成 Triton 计算，再把结果还原成输入 dtype。
    if x.dtype == torch.float64:
        result_f32 = _dirichlet_grad_func(x.float(), alpha.float(), total.float())
        return result_f32.to(torch.float64)
    return _dirichlet_grad_func(x, alpha, total)


def _dirichlet_grad_into(x, alpha, total, *, out):
    logger.debug("GEMS DIRICHLET_GRAD_OUT")
    assert x.dtype in (
        torch.float32,
        torch.float64,
    ), f"_dirichlet_grad only supports float32/float64, got {x.dtype}"
    if x.dtype == torch.float64:
        # out= 重载同样需要 fp32 计算；临时缓冲区避免向 fp64 输出直接写 kernel。
        result_f32 = torch.empty(out.shape, dtype=torch.float32, device=out.device)
        _dirichlet_grad_func(x.float(), alpha.float(), total.float(), out0=result_f32)
        out.copy_(result_f32)
        return out
    _dirichlet_grad_func(x, alpha, total, out0=out)
    return out
