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

"""Triton implementation of ``_standard_gamma_grad`` (reparameterized Gamma gradient).

G(alpha, x) = -d/dalpha log(P(alpha, x)), following ATen's
``standard_gamma_grad_one``:

  * x < 0.8                -> Taylor series around x = 0
  * alpha > 8 (x >= 0.8)   -> Rice saddle point expansion (near / away mode)
  * otherwise              -> bivariate rational approximation

Computation is carried out in fp32 for fp16/bf16/fp32 inputs and in fp64 for
fp64 inputs (mirroring ATen's accscalar_t).

Every lane evaluates all three regions (Triton selects with ``tl.where``), so the
body minimises per-element divides / exp / log / sqrt:
  * the small-x series uses the algebraically identical factored form
    ``x*e^x*((psi(alpha) - ln x)*S1 + S2)``, which cancels both ``pow`` calls
    and the ``pow(x, alpha-1)*exp(-x)`` pdf of the literal ATen expression;
  * the six reciprocals ``1/(alpha+i)`` needed by the Taylor series *and* by the
    digamma shift recurrence are produced by one division plus a prefix/suffix
    product chain instead of nine divisions;
  * the digamma recurrence shifts to >= 4 using monotone conditions (alpha > 0
    implies alpha+4 >= 4, where the asymptotic expansion is accurate to ~1e-8)
    and reuses those same reciprocals;
  * ``pow(f, -1.5)`` becomes ``1/(f*sqrt(f))`` and ``sqrt(8*alpha)``,
    ``sqrt(2/alpha)``, ``1/(alpha-x)`` reuse shared subexpressions.

The log-ratio ``u = log(x/alpha)`` is always computed from the quotient (never
as ``log(x) - log(alpha)``): in the Rice branch it is multiplied by alpha, so
the ~5e-7 absolute error of a log subtraction would be amplified past the fp32
tolerance of the operator.  ``log(x)`` itself is derived as ``log(alpha) + u``.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4


class _ShapeMismatchError(RuntimeError, AssertionError):
    """Raised when the two operands do not have identical shapes."""


@triton.jit
def _digamma_one(alpha, inv_a, i1, i2, i3, i4):
    # digamma(x) for x > 0: shift recurrence up to >= 4, then asymptotic series.
    # The four shift conditions are monotone, so the shifted argument and its
    # reciprocal are recovered by selection from the precomputed values.
    c0 = alpha < 4.0
    c1 = alpha + 1.0 < 4.0
    c2 = alpha + 2.0 < 4.0
    c3 = alpha + 3.0 < 4.0
    res = alpha * 0.0
    res = res - tl.where(c0, inv_a, 0.0)
    res = res - tl.where(c1, i1, 0.0)
    res = res - tl.where(c2, i2, 0.0)
    res = res - tl.where(c3, i3, 0.0)
    x = tl.where(
        c0,
        tl.where(
            c1,
            tl.where(c2, tl.where(c3, alpha + 4.0, alpha + 3.0), alpha + 2.0),
            alpha + 1.0,
        ),
        alpha,
    )
    inv_x = tl.where(
        c0, tl.where(c1, tl.where(c2, tl.where(c3, i4, i3), i2), i1), inv_a
    )

    z = inv_x * inv_x
    poly = 8.33333333333333333333e-2
    poly = poly * z + (-2.10927960927960927961e-2)
    poly = poly * z + 7.57575757575757575758e-3
    poly = poly * z + (-4.16666666666666666667e-3)
    poly = poly * z + 3.96825396825396825397e-3
    poly = poly * z + (-8.33333333333333333333e-3)
    poly = poly * z + 8.33333333333333333333e-2
    y = z * poly
    return res + tl.log(x) - 0.5 * inv_x - y


@triton.jit
def _standard_gamma_grad_one(alpha, x):
    log_a = tl.log(alpha)
    u = tl.log(x / alpha)
    log_x = log_a + u
    inv_a = 1.0 / alpha

    # 1/(alpha+i) for i = 1..5 from one division (prefix/suffix product chain).
    d1 = alpha + 1.0
    d2 = alpha + 2.0
    d3 = alpha + 3.0
    d4 = alpha + 4.0
    d5 = alpha + 5.0
    t5 = d5
    t4 = d4 * t5
    t3 = d3 * t4
    t2 = d2 * t3
    invp = 1.0 / (d1 * t2)
    i1 = t2 * invp
    i2 = (d1 * t3) * invp
    p2d = d1 * d2
    i3 = (p2d * t4) * invp
    p3d = p2d * d3
    i4 = (p3d * t5) * invp
    i5 = (p3d * d4) * invp

    # ---- branch 1: Taylor series for small x (x < 0.8) ----
    # S1 = sum (-x)^n / (n! (alpha+n)),  S2 = sum (-x)^n / (n! (alpha+n)^2)
    s1 = inv_a
    s2 = inv_a * inv_a
    nx = -x
    s1 += nx * i1
    s2 += nx * i1 * i1
    nx = nx * (-0.5 * x)
    s1 += nx * i2
    s2 += nx * i2 * i2
    nx = nx * (-x / 3.0)
    s1 += nx * i3
    s2 += nx * i3 * i3
    nx = nx * (-0.25 * x)
    s1 += nx * i4
    s2 += nx * i4 * i4
    nx = nx * (-0.2 * x)
    s1 += nx * i5
    s2 += nx * i5 * i5
    small = (
        x * tl.exp(x) * ((_digamma_one(alpha, inv_a, i1, i2, i3, i4) - log_x) * s1 + s2)
    )
    small = tl.where(small != small, 0.0, small)

    # ---- branch 2: Rice saddle point expansion for alpha > 8 ----
    alpha2 = alpha * alpha
    numer_2 = (
        1440.0 * alpha2
        + 6.0 * x * (53.0 - 120.0 * x)
        - 65.0 * x * x * inv_a
        + alpha * (107.0 + 3600.0 * x)
    )
    near_mode = (
        (1.0 + 24.0 * alpha * (1.0 + 12.0 * alpha))
        * numer_2
        / (1244160.0 * alpha2 * alpha2)
    )

    sqrt_a = tl.sqrt(alpha)
    inv_sqrt_a = 1.0 / sqrt_a
    diff = alpha - x
    inv_diff = 1.0 / diff
    term2 = 2.8284271247461903 * sqrt_a * inv_diff
    fx = (x - alpha) - alpha * u
    term3 = 1.0 / (fx * tl.sqrt(fx))
    term23 = tl.where(x < alpha, term2 - term3, term2 + term3)
    term1 = u * term23 - (
        1.4142135623730951 * inv_sqrt_a * (alpha + x) * inv_diff * inv_diff
    )
    stirling = 1.0 + (inv_a * 0.08333333333333333) * (
        1.0 + inv_a * 0.041666666666666664
    )
    away = -stirling * x * term1 * (0.35355339059327373 * inv_sqrt_a)

    is_near = (0.9 * alpha <= x) & (x <= 1.1 * alpha)
    large = tl.where(is_near, near_mode, away)

    # ---- branch 3: bivariate rational approximation ----
    v = log_a
    cv0 = 0.16009398 + u * (0.53487893 + u * 0.040121004)
    cv1 = -0.094634809 + u * (0.1298071 + u * (-0.0065914022))
    cv2 = 0.025146376 + u * (0.065735949 + u * (-0.0026286047))
    cv3 = -0.0030648343 + u * (-0.0015649758 + u * (-0.0013441777))
    cv4 = 1.0 + u * (0.16639465 + u * 0.017050642)
    cv5 = 0.32668115 + u * (0.020070113 + u * (-0.0021309326))
    cv6 = 0.10406089 + u * (-0.0035938915 + u * 0.00085092367)
    cv7 = 0.0014179084 + u * (-0.00058392623 + u * (-1.5247877e-07))
    p = cv0 + v * (cv1 + v * (cv2 + v * cv3))
    q = cv4 + v * (cv5 + v * (cv6 + v * cv7))
    rational = tl.exp(p / q)

    return tl.where(x < 0.8, small, tl.where(alpha > 8.0, large, rational))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=1),
        triton.Config({"BLOCK": 256}, num_warps=2),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=2),
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _sgg_flat_kernel(
    a_ptr,
    x_ptr,
    o_ptr,
    n,
    BLOCK: tl.constexpr,
    IS_FP64: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        a = tl.load(a_ptr + offs)
        x = tl.load(x_ptr + offs)
    else:
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask, other=1.0)
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if IS_FP64:
        alpha = a
        xv = x
    else:
        alpha = a.to(tl.float32)
        xv = x.to(tl.float32)
    r = _standard_gamma_grad_one(alpha, xv)
    if EVEN:
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty))
    else:
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sgg_strided_kernel(
    a_ptr,
    x_ptr,
    o_ptr,
    numel,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    BLOCK: tl.constexpr,
    ND: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel

    idx = offs
    aoff = tl.zeros_like(offs)
    boff = tl.zeros_like(offs)
    if ND >= 6:
        i5 = idx % s5
        idx = idx // s5
        aoff += i5 * a5
        boff += i5 * b5
    if ND >= 5:
        i4 = idx % s4
        idx = idx // s4
        aoff += i4 * a4
        boff += i4 * b4
    if ND >= 4:
        i3 = idx % s3
        idx = idx // s3
        aoff += i3 * a3
        boff += i3 * b3
    if ND >= 3:
        i2 = idx % s2
        idx = idx // s2
        aoff += i2 * a2
        boff += i2 * b2
    if ND >= 2:
        i1 = idx % s1
        idx = idx // s1
        aoff += i1 * a1
        boff += i1 * b1
    aoff += idx * a0
    boff += idx * b0

    a = tl.load(a_ptr + aoff, mask=mask, other=1.0)
    x = tl.load(x_ptr + boff, mask=mask, other=1.0)
    if IS_FP64:
        alpha = a
        xv = x
    else:
        alpha = a.to(tl.float32)
        xv = x.to(tl.float32)
    r = _standard_gamma_grad_one(alpha, xv)
    tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=mask)


def standard_gamma_grad(self, output):
    a = self
    x = output
    if not a.is_floating_point() or not x.is_floating_point():
        raise RuntimeError(
            "_standard_gamma_grad: expected floating point tensors, got "
            f"self_grad={a.dtype}, output={x.dtype}"
        )
    if a.dtype != x.dtype:
        raise RuntimeError(
            "_standard_gamma_grad: expected self_grad and output to have the "
            f"same dtype, got {a.dtype} and {x.dtype}"
        )
    if a.shape != x.shape:
        raise _ShapeMismatchError(
            "_standard_gamma_grad: expected self_grad and output to have the "
            f"same shape, got {tuple(a.shape)} and {tuple(x.shape)}"
        )

    n = a.numel()
    out = torch.empty_like(a, memory_format=torch.contiguous_format)
    if n == 0:
        return out

    is_fp64 = a.dtype == torch.float64
    if a.is_contiguous() and x.is_contiguous():
        # 4096 is a multiple of every BLOCK in the autotune list, so a size
        # divisible by 4096 lets the kernel drop the tail mask entirely.
        even = (n & 4095) == 0
        i64 = n > 2147483647
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
        _sgg_flat_kernel[grid](a, x, out, n, IS_FP64=is_fp64, EVEN=even, I64=i64)
        return out

    shape = list(a.shape)
    nd = len(shape)
    if nd > 6:
        raise _ShapeMismatchError(
            "_standard_gamma_grad: noncontiguous inputs with more than 6 "
            f"dimensions are not supported (got {nd})"
        )
    asd = list(a.stride())
    bsd = list(x.stride())
    while len(shape) < 6:
        shape.append(1)
        asd.append(0)
        bsd.append(0)

    grid = (triton.cdiv(n, _BLOCK),)
    _sgg_strided_kernel[grid](
        a,
        x,
        out,
        n,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        asd[0],
        asd[1],
        asd[2],
        asd[3],
        asd[4],
        asd[5],
        bsd[0],
        bsd[1],
        bsd[2],
        bsd[3],
        bsd[4],
        bsd[5],
        BLOCK=_BLOCK,
        ND=nd,
        IS_FP64=is_fp64,
        num_warps=_NUM_WARPS,
    )
    return out
