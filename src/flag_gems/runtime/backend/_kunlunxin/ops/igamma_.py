# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of igamma_ (aten::igamma_, in-place lower
# regularized incomplete gamma P(a, x), a = self, x = other).
#
# HEAD state (this file was a 288-byte bare forwarder onto
# `flag_gems/ops/igamma_.py`) compiles and is numerically correct on XPU --
# the generic kernel's only `tl_extra_shim` use is `pow(v, -1.0)`, a
# *floating-point* exponent, which links on this backend (integer exponents
# do not), and its log-gamma is an inline Lanczos rather than the
# `__nv_lgammaf` shim that broke igammac.  So this is a pure performance
# rewrite, not a functional repair.
#
# Where the generic implementation's time went (measured, [4096, 4096] fp32):
#
#   1. `self.copy_(result)`.  The generic wrapper computes into a fresh
#      `torch.empty_like` and then copies back.  Inside `use_gems()` -- which
#      is exactly how `benchmark/test_igamma_.py` times this op, since that
#      file sets no `gems_op` -- `Tensor.copy_` is itself a FlagGems kernel
#      running at ~1.25 GB/s, so the copy alone costs tens of ms.  Writing
#      P(a, x) straight back into `self` removes both the copy and the
#      allocation.
#   2. Two 60-iteration runtime loops.  The kernel is 100% ALU bound (the
#      memory floor for [4096, 4096] fp32 is ~0.13 ms), so every per-lane
#      instruction inside those loops is worth ~0.06 ms of wall time.
#   3. The continued fraction used forward modified Lentz: two divisions,
#      two `tl.abs`, two comparisons and two `tl.where` re-normalisation
#      guards per step.  Those guards only exist to keep a
#      *convergence-tested* forward recurrence from dividing by zero; with a
#      fixed term count there is nothing to test.  `tl.where` costs ~2x an
#      fma on this backend.
#   4. Four `tl_extra_shim.pow(v, -1.0)` calls, one of them (`d = pow(d,-1)`)
#      inside the Lentz loop, i.e. once per element per iteration.  A
#      libdevice `pow` is far more expensive than the plain `1.0 / v` it is
#      equivalent to for exponent -1.
#
# The rewrite (same four levers that took the sibling out-of-place `igammac`
# override from 228 ms to 39 ms on this cell, plus lever 4 above which is
# specific to this file):
#   a) power series for P when x < a + 1, 24 terms, denominator carried
#      incrementally so the body holds no int->float conversion;
#   b) the Legendre continued fraction for Q evaluated *backwards* from the
#      N_CF-th convergent (1 division + 4 additions per step, no guards),
#      P = 1 - Q;
#   c) the DLMF 8.12.4 asymptotic expansion for large a with x ~ a, with the
#      coefficient table inlined as literals (a `@triton.jit` body cannot
#      read a module-level constant at all) and `log1p(s) - s` as a 10-term
#      Horner chain instead of a runtime Taylor loop;
#   d) `1.0 / v` in place of every `pow(v, -1.0)`;
#   e) an inlined Abramowitz & Stegun 7.1.26 `erf` in place of
#      `tl.math.erf`, which measured 10.9 ms per pass over [4096, 4096] fp32
#      against a 2.0 ms memory floor -- the most expensive single operation
#      found on this backend so far, and 13.8 ms of the kernel before this
#      change.  See `_erf_as`.
#
# Numerically the rewrite is *more* accurate than the generic kernel, not
# less: the asymptotic branch takes over the `a ~ 20, x ~ a + 1` region that
# dominated the old error, and for the backward recurrence a larger N_CF is
# actually worse (the coefficient a_N = N(a-N) grows like N^2 and loses fp32
# precision), so the shorter loop is both faster and tighter.
#
# It also fixes the out-of-domain semantics: the generic kernel clamped
# whatever the series produced for a <= 0 into [0, 1], while ATen returns NaN
# (and returns 1, not NaN, for a == 0 with x > 0).  The edge ladder below now
# follows ATen's `calc_igamma` order exactly, so x == 0, a == 0, a <= 0,
# x < 0, +-inf and NaN all agree with ATen CPU on a 198-point grid.  The one
# remaining difference is for huge *finite* a (a = 3.4e38, x = 3): ATen fp32
# overflows its own lgamma and reports NaN, whereas this kernel reports the
# mathematically correct 0 -- the generic kernel also reported 0, so that is
# not a change in behaviour.
import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _lgamma_pos(z):
    # Lanczos approximation of log-gamma for z > 0 (g=7, n=9 coefficients).
    # XPU Triton has no usable `lgamma` intrinsic (`tl_extra_shim.lgamma`
    # links to `undefined symbol: __nv_lgammaf`), so it is evaluated inline
    # in fp32.  Only a > 0 reaches here; a <= 0 yields NaN by construction.
    g = 7.0
    x = 0.99999999999980993
    x = x + 676.5203681218851 / (z + 0.0)
    x = x + (-1259.1392167224028) / (z + 1.0)
    x = x + 771.32342877765313 / (z + 2.0)
    x = x + (-176.61502916214059) / (z + 3.0)
    x = x + 12.507343278686905 / (z + 4.0)
    x = x + (-0.13857109526572012) / (z + 5.0)
    x = x + 9.9843695780195716e-6 / (z + 6.0)
    x = x + 1.5056327351493116e-7 / (z + 7.0)
    t = (z - 1.0) + g + 0.5
    half_log_2pi = 0.9189385332046727
    return half_log_2pi + ((z - 1.0) + 0.5) * tl.log(t) - t + tl.log(x)


@triton.jit
def _eta_sq(sigma):
    # -2 * (log(1 + s) - s) = s^2 * sum_{k>=0} 2 (-1)^k / (k+2) * s^k.
    #
    # Horner chain on `s` clamped to [-0.4, 0.4]: the caller only consumes
    # the result where |sigma| < 0.4, and clamping keeps the unselected lanes
    # finite (an unclamped s^9 overflows).  10 terms leave < 2e-9 relative
    # remainder at |s| = 0.4, below the fp32 rounding of the s^2 prefactor.
    # No division, no int->float conversion, no loop.
    s = tl.minimum(tl.maximum(sigma, -0.4), 0.4)
    p = -0.18181818181818182
    p = p * s + 0.2
    p = p * s + (-0.2222222222222222)
    p = p * s + 0.25
    p = p * s + (-0.2857142857142857)
    p = p * s + 0.3333333333333333
    p = p * s + (-0.4)
    p = p * s + 0.5
    p = p * s + (-0.6666666666666666)
    p = p * s + 1.0
    return tl.maximum((s * s) * p, 0.0)


@triton.jit
def _erf_as(z, em):
    """erf(z) via Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7 absolute.

    `em` is exp(-z*z), supplied by the caller because the asymptotic
    expansion needs exactly the same quantity (`exp(-a eta^2 / 2)`).

    `tl.math.erf` is by far the most expensive operation available on this
    backend -- measured on a trivial [4096, 4096] fp32 kernel it adds
    10.9 ms over the 2.0 ms memory floor, while `tl.exp`, `tl.sqrt`,
    `tl.log` and `tl.where` all add ~0.  Inside this kernel it accounted for
    13.8 ms of 37.4 ms; this replacement costs ~1.6 ms and reproduces the
    device oracle error to four significant digits (2.164e-06 / 3.508e-06
    with either implementation), because 1.5e-7 is already below the fp32
    rounding of the result.
    """
    az = tl.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    p = 1.061405429
    p = p * t + (-1.453152027)
    p = p * t + 1.421413741
    p = p * t + (-0.284496736)
    p = p * t + 0.254829592
    r = 1.0 - (p * t) * em
    return tl.where(z >= 0.0, r, -r)


@triton.jit
def igamma_kernel_xpu(
    a_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    N_SER: tl.constexpr,
    N_CF: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # other=1.0 / other=0.0: masked lanes never feed a valid result (their
    # stores are masked out too).  A clamped-address unmasked load would put
    # the runtime scalar `n_elements - 1` into the per-lane address
    # expression, which is a documented 15-168x cliff on this backend.
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    a_f = a.to(tl.float32)
    x_f = x.to(tl.float32)

    is_nan_x = x_f != x_f
    is_nan_a = a_f != a_f
    # `v * 0.0 != 0.0` is true for both inf and NaN; subtracting the NaN case
    # leaves a pure isinf test without an intrinsic.
    is_inf_x = ((x_f * 0.0) != 0.0) & ~is_nan_x
    is_inf_a = ((a_f * 0.0) != 0.0) & ~is_nan_a

    log_gamma_a = _lgamma_pos(a_f)
    log_x_term = a_f * tl.log(x_f) - x_f - log_gamma_a

    # Path 1: power series, P(a,x) = e^{-x} x^a sum_n x^n / Gamma(a+n+1),
    # directly (this is the regime where P is the small quantity, so the
    # series gives it with good *relative* accuracy).  Converges for
    # x < a + 1.  `ai` carries a + i so the body holds no int->float cast.
    term = 1.0 / a_f
    series_sum = term
    ai = a_f
    for _s in range(1, N_SER):
        ai = ai + 1.0
        term = term * (x_f / ai)
        series_sum = series_sum + term
    p_series = tl.exp(log_x_term) * series_sum

    # Path 2: Legendre continued fraction for Q(a, x),
    #   Q = e^{a log x - x - log Gamma(a)} / w,
    #   w = b0 + a1/(b1 + a2/(b2 + ...)),  a_n = n (a - n),  b_n = x+2n+1-a,
    # evaluated *backwards* from the N_CF-th convergent (w <- b_N, then
    # w <- b_{n-1} + a_n / w for n = N..1), with both coefficient sequences
    # carried incrementally (b_{n-1} = b_n - 2, a_{n-1} = a_n + (b_{n-1}-x)).
    # 1 division + 4 additions per step instead of Lentz's ~20 ops, and no
    # re-normalisation guard is needed at a fixed term count.  Bonus: at
    # integer a = n the coefficient a_n = n(a-n) is exactly 0, which
    # truncates the fraction at the mathematically right place.
    # P = 1 - Q; here Q is the small quantity, so this is accurate.
    w = x_f + (2.0 * N_CF + 1.0) - a_f
    b_prev = x_f + (2.0 * N_CF - 1.0) - a_f
    an = (a_f - (1.0 * N_CF)) * (1.0 * N_CF)
    for _c in range(N_CF):
        w = b_prev + an / w
        an = an + (b_prev - x_f)
        b_prev = b_prev - 2.0
    p_cf = 1.0 - tl.exp(log_x_term - tl.log(w))

    # Path 3: DLMF 8.12.4 asymptotic expansion for large a with x ~ a,
    #   Q(a,x) ~ 0.5 erfc(eta sqrt(a/2))
    #            + e^{-a eta^2 / 2} sum_k c_k(eta)/a^k / sqrt(2 pi a),
    # sigma = (x-a)/a, eta = sgn(sigma) sqrt(-2(log(1+sigma)-sigma)).
    # For a > 10 and fp32 output only c_0..c_3 clear the rounding floor, and
    # each is truncated at the term that falls below it (8/5/3/1
    # coefficients -> 17 literals, inlined because a `@triton.jit` body
    # cannot reference a module-level table).
    sigma = (x_f - a_f) / a_f
    root = tl.sqrt(_eta_sq(sigma))
    eta = tl.where(sigma > 0.0, root, tl.where(sigma < 0.0, -root, 0.0))

    c0 = -2.185448510679992e-06
    c0 = c0 * eta + 3.919263178522438e-05
    c0 = c0 * eta + (-0.0001787551440329218)
    c0 = c0 * eta + 0.0003527336860670194
    c0 = c0 * eta + 0.0011574074074074073
    c0 = c0 * eta + (-0.014814814814814815)
    c0 = c0 * eta + 0.08333333333333333
    c0 = c0 * eta + (-0.3333333333333333)

    c1 = 0.00020576131687242798
    c1 = c1 * eta + (-0.0009902263374485596)
    c1 = c1 * eta + 0.0026455026455026454
    c1 = c1 * eta + (-0.003472222222222222)
    c1 = c1 * eta + (-0.001851851851851852)

    c2 = 0.0007716049382716049
    c2 = c2 * eta + (-0.002681327160493827)
    c2 = c2 * eta + 0.004133597883597884

    c3 = 0.0006494341563786008

    a_inv = 1.0 / a_f
    poly_sum = c3 * a_inv + c2
    poly_sum = poly_sum * a_inv + c1
    poly_sum = poly_sum * a_inv + c0

    # `z = eta sqrt(a/2)`; `exp(-z*z)` is exactly the expansion's
    # `exp(-a eta^2 / 2)`, so it is computed once and shared with `_erf_as`.
    z = eta * tl.sqrt(a_f * 0.5)
    em = tl.exp(-z * z)
    q_asym = em * poly_sum / tl.sqrt(2.0 * 3.141592653589793 * a_f)
    q_asym = q_asym + 0.5 * (1.0 - _erf_as(z, em))
    p_asym = 1.0 - q_asym

    # Per-element path selection.  A nested
    # `tl.where(use_asym, ., tl.where(use_series, ., .))` blows the XPU
    # uni_sram pass with three distinct live values; the mask-and-sum form is
    # NaN-safe (unselected lanes are forced to 0.0 before summing) and
    # compiles.  Exactly one term survives, so the [0, 1] projection is
    # applied once to the sum.
    use_asym = (a_f > 10.0) & (tl.abs(sigma) < 0.4)
    use_series = (x_f < (a_f + 1.0)) & ~use_asym
    use_cf = ~use_asym & ~use_series
    computed = (
        tl.where(use_asym, p_asym, 0.0)
        + tl.where(use_series, p_series, 0.0)
        + tl.where(use_cf, p_cf, 0.0)
    )
    computed = tl.minimum(tl.maximum(computed, 0.0), 1.0)

    # Boundary / infinity ladder, in ATen's exact evaluation order
    # (`calc_igamma` in aten/src/ATen/native/Math.h):
    #   1. x < 0 or a < 0   -> NaN
    #   2. a == 0           -> 1 if x > 0 else NaN
    #   3. x == 0           -> 0
    #   4. isinf(a)         -> NaN if isinf(x) else 0
    #   5. isinf(x)         -> 1
    #   6. otherwise the series / CF / asymptotic value.
    # The NaN default has to be written explicitly: with a NaN input every
    # branch predicate above is false, so the mask-and-sum above would
    # quietly yield 0.0 rather than NaN.
    result = tl.where(is_nan_a | is_nan_x, float("nan"), computed)
    result = tl.where(is_inf_x, 1.0, result)
    result = tl.where(is_inf_a, tl.where(is_inf_x, float("nan"), 0.0), result)
    result = tl.where(x_f == 0.0, 0.0, result)
    result = tl.where(a_f == 0.0, tl.where(x_f > 0.0, 1.0, float("nan")), result)
    result = tl.where((x_f < 0.0) | (a_f < 0.0), float("nan"), result)

    tl.store(out_ptr + offsets, result.to(out_ptr.type.element_ty), mask=mask)


# BLOCK / trip counts: chosen by an independent-process `do_bench(median)`
# sweep on the *in-place* form (an out-of-place tuning cannot be reused --
# with a_ptr == out_ptr deeper per-program work behaves differently) over the
# official benchmark matrix.  See
# harness/solution/performance/igamma__xpu2_20260830.md.
_BLOCK = 512
_N_SER = 24
_N_CF = 18


def _launch(out: torch.Tensor, a: torch.Tensor, x: torch.Tensor):
    n = out.numel()
    if n == 0:
        return out
    grid = (triton.cdiv(n, _BLOCK),)
    igamma_kernel_xpu[grid](
        a,
        x,
        out,
        n,
        BLOCK_SIZE=_BLOCK,
        N_SER=_N_SER,
        N_CF=_N_CF,
        buffer_size_limit=2048,
    )
    return out


def igamma_(self, other):
    """In-place regularized lower incomplete gamma P(a, x), a=self, x=other.

    The result is written back into `self`; `self` is returned.
    """
    logger.debug("GEMS_KUNLUNXIN IGAMMA_")

    if not isinstance(self, torch.Tensor):
        raise TypeError("igamma_ expects a torch.Tensor as the first argument")
    if not isinstance(other, torch.Tensor):
        raise TypeError("igamma_ expects a torch.Tensor as the second argument")
    if self.device.type != flag_gems.device:
        raise ValueError(f"igamma_: self must be on {flag_gems.device}")
    if other.device.type != flag_gems.device:
        raise ValueError(f"igamma_: other must be on {flag_gems.device}")

    if self.numel() == 0:
        return self

    # In-place: the broadcast result must have `self`'s shape.
    output_shape = torch.broadcast_shapes(self.shape, other.shape)
    if tuple(output_shape) != tuple(self.shape):
        raise RuntimeError(
            "igamma_: output with shape "
            f"{tuple(self.shape)} doesn't match the broadcast shape "
            f"{tuple(output_shape)}"
        )

    x_t = other if other.shape == self.shape else other.broadcast_to(self.shape)
    if x_t.dtype != self.dtype:
        x_t = x_t.to(self.dtype)
    if not x_t.is_contiguous():
        x_t = x_t.contiguous()

    if self.is_contiguous():
        # Fast path: read and write `self` directly.  No temporary, no
        # `copy_` back (that copy was the dominant cost of the generic
        # implementation inside `use_gems()`).
        _launch(self, self, x_t)
        return self

    # `self` is strided: a strided block store is mis-lowered on this
    # backend, so compute into a contiguous temporary and let ATen's
    # `copy_` scatter it back.  This path is not reached by the official
    # test or benchmark matrices (both feed contiguous tensors).
    a_c = self.contiguous()
    tmp = torch.empty_like(a_c)
    _launch(tmp, a_c, x_t)
    self.copy_(tmp)
    return self
