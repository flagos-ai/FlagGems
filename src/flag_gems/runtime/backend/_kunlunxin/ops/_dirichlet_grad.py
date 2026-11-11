# Copyright 2026, The FlagOS Contributors.
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
# Kunlunxin (XPU) override of ``flag_gems.ops._dirichlet_grad``.
#
# The generic implementation maps a scalar ``_dirichlet_grad_one`` over the
# tile with ``tl.map_elementwise`` (Triton >= 3.6) to get lazy ``if -> return``
# control flow.  The XPU backend has no lowering for the resulting
# ``tt.map_elementwise`` op, so ``PassManager::run`` fails with
# ``failed to legalize operation 'tt.map_elementwise'``.
#
# This override keeps the exact ATen ``dirichlet_grad_one`` math but evaluates
# it in plain vectorized Triton (``tl.where`` based selection instead of scalar
# control flow), which lowers on XPU.  Branch selection happens at the end, so
# every branch is evaluated for every element; that is the trade-off for not
# depending on ``tt.map_elementwise``.
#
# Two further XPU constraints shaped the code below:
#   * The XPU libdevice exposes ``log``/``exp`` (and the fp64 variants of
#     ``tan``/``pow``/``div_rn``) as ``Unsupported`` stubs that fail at link
#     time, and ``log1p`` trips ``triton_xpu.unpack reached lowering without a
#     bufPtr``.  Everything is therefore built from core ``tl`` math ops
#     (``tl.log``/``tl.exp``/``tl.sqrt``/``tl.floor``/``tl.sin``/``tl.cos``,
#     with ``exp(b*log(a))`` standing in for ``pow``).
#   * The device has no fp64 (torch silently stores everything as fp32), so the
#     saddle-point branch is evaluated in fp32 against a double-precision
#     reference.  ``_log1p`` plus the ``fma``-based ``axbx`` below keep the
#     cancellation in ``beta*log(beta/(total*(1-x))) + alpha*log(alpha/(total*x))``
#     from blowing past the test tolerance.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _div_rn(a, b):
    # IEEE round-to-nearest division (matches the reference ``accscalar_t``
    # rounding).  ``tl.div_rn`` is float32-only; for float64 plain ``/`` is
    # already correctly rounded.  ``b.dtype`` is a compile-time attribute.
    if b.dtype == tl.float32:
        return tl.div_rn(a, b)
    else:
        return a / b


@triton.jit
def _digamma_one(x):
    # Vectorized Cephes-style digamma, ported from ATen::native::digamma_one.
    PI: tl.constexpr = 3.1415926535897932384626433832795028841971
    PSI_10: tl.constexpr = 2.25175258906672110764

    is_int = x == tl.floor(x)
    neg = x < 0.0

    # Reflection x -> 1 - x for negative x.
    additional_summand = tl.where(
        neg, _div_rn(-PI, tl.sin(PI * x) / tl.cos(PI * x)), tl.zeros_like(x)
    )
    y = tl.where(neg, 1.0 - x, x)

    # Push y to be >= 10 (C++ ``while (y < 10)``; at most 10 increments since
    # y > 1 after the reflection).
    result = tl.zeros_like(x)
    for _ in range(10):
        small = y < 10.0
        result = tl.where(small, result - _div_rn(1.0, y), result)
        y = tl.where(small, y + 1.0, y)

    # Asymptotic series.
    a0 = 8.33333333333333333333e-2
    a1 = -2.10927960927960927961e-2
    a2 = 7.57575757575757575758e-3
    a3 = -4.16666666666666666667e-3
    a4 = 3.96825396825396825397e-3
    a5 = -8.33333333333333333333e-3
    a6 = 8.33333333333333333333e-2
    z = _div_rn(1.0, y * y)
    poly = (((((a6 * z + a5) * z + a4) * z + a3) * z + a2) * z + a1) * z + a0
    series = tl.where(y < 1.0e17, z * poly, tl.zeros_like(x))

    out = result + tl.log(y) - _div_rn(0.5, y) - series + additional_summand
    # Exact value at the loop boundary.
    out = tl.where(y == 10.0, result + PSI_10 + additional_summand, out)
    # x == 0 or a negative integer maps to +inf.
    out = tl.where((x <= 0.0) & is_int, float("inf"), out)
    return out


@triton.jit
def _beta_grad_alpha_small(x, alpha, beta):
    # Taylor expansion of the reparameterized Beta(alpha,beta) gradient at x=0.
    factor = _digamma_one(alpha) - _digamma_one(alpha + beta) - tl.log(x)
    numer = tl.zeros_like(x) + 1.0
    inv_alpha = _div_rn(1.0, alpha)
    series = _div_rn(numer, alpha) * (factor + inv_alpha)
    for i in range(1, 11):
        casted_i = i
        numer = numer * (casted_i - beta) * x / casted_i
        denom = alpha + casted_i
        inv_denom = _div_rn(1.0, denom)
        series = series + _div_rn(numer, denom) * (factor + inv_denom)
    result = x * tl.exp(-beta * tl.log(1.0 - x)) * series
    return tl.where(result != result, tl.zeros_like(result), result)


@triton.jit
def _beta_grad_beta_small(x, alpha, beta):
    # Taylor expansion of the reparameterized Beta(alpha,beta) gradient at x=1.
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
    result = -tl.exp((1.0 - beta) * tl.log(1.0 - x)) * series
    return tl.where(result != result, tl.zeros_like(result), result)


@triton.jit
def _log1p(v):
    # Accurate log(1 + v) without an extern call.
    #
    # ``tl.log(1.0 + v)`` loses relative accuracy for small ``|v|`` because
    # ``1.0 + v`` rounds away the low bits of ``v`` (log(r) ~ v, so the error
    # of the rounded sum is amplified by 1/v).  The saddle-point branch below
    # feeds it ratios that are close to 1 and multiplies the (small) log by a
    # large ``alpha``/``beta``, so this matters.  For small ``|v|`` use the
    # Taylor series; otherwise ``1 + v`` is far enough from 1 that the plain
    # log keeps full relative accuracy.
    small = tl.abs(v) < 0.25
    # Taylor series, truncated after v^10 (error < 1e-8 at |v| = 0.25).
    poly = tl.zeros_like(v)
    term = v
    for k in range(1, 11):
        poly = poly + term * (1.0 / k)
        term = -term * v
    return tl.where(small, poly, tl.log(1.0 + v))


@triton.jit
def _beta_grad_alpha_mid(x, alpha, beta):
    # Rice saddle-point expansion for alpha, beta both large.
    total = alpha + beta
    mean = alpha / total
    std = tl.sqrt(alpha * beta / (total + 1.0)) / total
    near_mean = (mean - 0.1 * std <= x) & (x <= mean + 0.1 * std)

    # Polynomial form valid close to the mode.
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
    near_result = _div_rn(prefactor_num, (1.0 - x)) * _div_rn(poly, prefactor_den)

    # Saddle-point form.
    prefactor = _div_rn(-x, tl.sqrt(2.0 * alpha * beta / total))
    stirling = (
        (1.0 + 1.0 / (12.0 * alpha) + 1.0 / (288.0 * alpha * alpha))
        * (1.0 + 1.0 / (12.0 * beta) + 1.0 / (288.0 * beta * beta))
        / (1.0 + 1.0 / (12.0 * total) + 1.0 / (288.0 * total * total))
    )
    term1_num = (
        2.0 * alpha * alpha * (x - 1.0) + alpha * beta * (x - 1.0) - x * (beta * beta)
    )
    # ``axbx == alpha*(x-1) + beta*x``, i.e. ``total*x - alpha``.  Evaluate it
    # through an fma so the rounding of ``total*x`` does not survive: in the
    # saddle-point regime ``total*x`` and ``alpha`` nearly cancel, and ``axbx``
    # divides ``term1``/``term3`` and squares into ``term1_den``.
    axbx = tl.fma(total, x, -alpha)
    term1_den = tl.sqrt(2.0 * alpha / beta) * (total * tl.sqrt(total)) * axbx * axbx
    term1 = _div_rn(term1_num, term1_den)
    # ``log(alpha / (total*x))`` and ``log(beta / (total*(1-x)))`` are both
    # ratios of nearly equal quantities here; rewrite them as log1p of the
    # (small, accurately computed) relative difference ``d = total*x - alpha``
    # so the large alpha/beta multipliers below do not amplify the rounding of
    # a catastrophic cancellation.
    d = axbx
    log_a_over_total_x = _log1p(_div_rn(-d, total * x))
    term2 = 0.5 * log_a_over_total_x
    term3_num = tl.sqrt(8.0 * alpha * beta / total)
    term3 = _div_rn(term3_num, axbx)
    log_b_over_total_1mx = _log1p(_div_rn(d, total * (1.0 - x)))
    term4_base = beta * log_b_over_total_1mx + alpha * log_a_over_total_x
    term4 = _div_rn(1.0, term4_base * tl.sqrt(term4_base))
    term1234 = tl.where(
        x < mean, term1 + term2 * (term3 + term4), term1 + term2 * (term3 - term4)
    )
    far_result = stirling * prefactor * term1234

    return tl.where(near_mean, near_result, far_result)


@triton.jit
def _dirichlet_grad_tile(x, alpha, total):
    # Ported from ATen::native::dirichlet_grad_one (Distributions.h); the
    # scalar ``if -> return`` cascade becomes end-of-function ``tl.where``
    # selection so no ``tt.map_elementwise`` is needed.
    beta = total - alpha
    boundary = total * x * (1.0 - x)

    small_branch = (x <= 0.5) & (boundary < 2.5)
    beta_small_branch = (x >= 0.5) & (boundary < 0.75)
    mid_branch = (alpha > 6.0) & (beta > 6.0)

    r_small = _beta_grad_alpha_small(x, alpha, beta)
    r_beta_small = -_beta_grad_beta_small(1.0 - x, beta, alpha)
    r_mid = _beta_grad_alpha_mid(x, alpha, beta)

    # Rational-correction branch (final fallback in dirichlet_grad_one).
    u = tl.log(x)
    a = tl.log(alpha) - u
    b = tl.log(total) - a

    C0_000 = 1.003668233
    C0_001 = -0.01061107488
    C0_002 = -0.0657888334
    C0_003 = 0.01201642863
    C0_010 = 0.6336835991
    C0_011 = -0.3557432599
    C0_012 = 0.05486251648
    C0_013 = -0.001465281033
    C0_020 = -0.03276231906
    C0_021 = 0.004474107445
    C0_022 = 0.002429354597
    C0_023 = -0.0001557569013
    C0_100 = 0.221950385
    C0_101 = -0.3187676331
    C0_102 = 0.01799915743
    C0_103 = 0.01074823814
    C0_110 = -0.2951249643
    C0_111 = 0.06219954479
    C0_112 = 0.01535556598
    C0_113 = 0.001550077057
    C0_120 = 0.02155310298
    C0_121 = 0.004170831599
    C0_122 = 0.001292462449
    C0_123 = 6.976601077e-05
    C0_200 = -0.05980841433
    C0_201 = 0.008441916499
    C0_202 = 0.01085618172
    C0_203 = 0.002319392565
    C0_210 = 0.02911413504
    C0_211 = 0.01400243777
    C0_212 = -0.002721828457
    C0_213 = 0.000751041181
    C0_220 = 0.005900514878
    C0_221 = -0.001936558688
    C0_222 = -9.495446725e-06
    C0_223 = 5.385558597e-05

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
    r_rational = _div_rn(p, q) * approx

    out = tl.where(small_branch, r_small, r_rational)
    out = tl.where(beta_small_branch, r_beta_small, out)
    out = tl.where(mid_branch, r_mid, out)
    return out


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
)
@triton.jit
def _dirichlet_grad_func(x, alpha, total):
    return _dirichlet_grad_tile(x, alpha, total).to(x.dtype)


def _dirichlet_grad(x, alpha, total):
    logger.debug("GEMS_KUNLUNXIN DIRICHLET_GRAD")
    # The ATen reference only implements the AT_DISPATCH_FLOATING_TYPES path.
    assert x.dtype in (
        torch.float32,
        torch.float64,
    ), f"_dirichlet_grad only supports float32/float64, got {x.dtype}"
    return _dirichlet_grad_func(x, alpha, total)


def _dirichlet_grad_into(x, alpha, total, *, out):
    logger.debug("GEMS_KUNLUNXIN DIRICHLET_GRAD_OUT")
    assert x.dtype in (
        torch.float32,
        torch.float64,
    ), f"_dirichlet_grad only supports float32/float64, got {x.dtype}"
    _dirichlet_grad_func(x, alpha, total, out0=out)
    return out
