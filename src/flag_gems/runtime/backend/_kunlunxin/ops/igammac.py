# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of igammac (aten::special_gammaincc, out-of-place).
#
# Root causes of the generic `flag_gems/ops/igammac.py` on XPU:
#   1. `_build_d_coeffs` builds the DLMF 8.12.4 coefficient tensor with
#      `torch.tensor(list, dtype=..., device=flag_gems.device)`. Inside
#      `use_gems()`/`enable()` the Kunlunxin `to_copy` override refuses
#      CPU->XPU copies -> `NotImplementedError` before any kernel runs.
#   2. `tl_extra_shim.lgamma` resolves to `undefined symbol: __nv_lgammaf` at
#      xpu3 link time (same root cause as lgamma / special_gammainc /
#      special_gammaln / igammac_ overrides).
#   3. `tl_extra_shim.log1p` resolves to `undefined symbol: __nv_log1pf` at
#      link time; no `tl.log1p` exists in the XPU Triton dialect.
#   4. `@triton.autotune` with 10 configs re-compiles per shape on XPU and
#      inflates IR (harness lesson 2.1); replaced with a single BLOCK config.
#
# Fix: dedicated XPU kernel identical in math to the generic one (power
# series for x < a+1, continued fraction for x >= a+1, DLMF 8.12.4
# asymptotic expansion for large a with x ~ a, plus inf/nan/domain
# handling), with:
#   - inline Lanczos g=7 log-gamma for a > 0 (`_lgamma_pos`, same helper as
#     lgamma / special_gammainc / mvlgamma_ overrides),
#   - fixed BLOCK=512, no autotune; fp32 only (XPU has no fp64).
#
# ---------------------------------------------------------------------------
# 2026-08-30 (XPU2) performance rewrite. The kernel is pure ALU (the memory
# floor for [4096, 4096] fp32 is ~0.13 ms, the kernel measured 228 ms), so
# every per-lane instruction inside the two runtime loops costs ~0.06 ms of
# wall time. Four changes, all validated against a CPU fp64 oracle at a
# tighter max error than the previous code (4.7e-6 -> see
# harness/solution/performance/igammac_xpu2_20260830.md):
#   a) The continued fraction is evaluated *bottom-up* (backward recurrence
#      on the N-term convergent) instead of forward modified-Lentz. Lentz
#      needs two divisions, two `tl.abs`, two comparisons and two `tl.where`
#      re-normalisation guards per step (~20 ops); the backward recurrence
#      needs one division and four additions (~6 ops) and needs no guard at
#      all because a fixed term count makes the "did it converge yet" test
#      unnecessary. `tl.where` costs ~2x an fma on this backend, so the two
#      guards alone were ~40% of the loop.
#   b) Loop trip counts cut using the fp32 error floor instead of a
#      worst-case guess: 50 -> 24 series terms and 50 -> 18 CF terms. Both
#      plateau well before that (series 22, CF 14) on a 400k-point grid.
#   c) `_log1p_minus_s`'s 29-iteration runtime Taylor loop (with a division
#      and an int->float conversion per step) is replaced by a 10-term
#      Horner polynomial in the *clamped* relative offset, so the asymptotic
#      eta is a straight-line fma chain.
#   d) The DLMF coefficient table no longer lives in device memory: the
#      kernel used to issue 64 scalar `tl.load`s from `d_ptr` per element.
#      Only c_0..c_3 matter for fp32 once a > 10, so the needed 17
#      coefficients are inlined as literals (Triton cannot reference
#      module-level constants at all - `NameError: Cannot access global
#      variable` - so they are written out longhand) and the `d_ptr`
#      argument is gone.
# In addition the asymptotic activation window is widened from
# `a > 20 & |sigma| < 0.3` to `a > 10 & |sigma| < 0.4`. That is not a
# speed change (all three branches are evaluated for every lane and merged
# with a mask-and-sum) but it moves the hardest series/CF inputs onto the
# cheap, more accurate expansion, which is what allows (b).
import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _lgamma_pos(z):
    # Lanczos approximation of log-gamma for z > 0 (g=7, n=9 coefficients).
    # XPU Triton has no `lgamma` intrinsic (undefined symbol at link time),
    # so it is evaluated inline in fp32. gammaincc feeds a > 0 into
    # _lgamma_pos (the reflection branch is unnecessary since out-of-domain
    # a <= 0 produces NaN regardless).
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
    # Evaluated as a Horner chain on `s` clamped to [-0.4, 0.4]: the caller
    # only consumes the result where |sigma| < 0.4, and clamping keeps the
    # unselected lanes finite (an unclamped s^9 overflows for the large-|s|
    # lanes and the resulting NaN, while masked out, is free to avoid).
    # 10 terms leave < 2e-9 relative remainder at |s| = 0.4, i.e. below the
    # fp32 rounding of the s^2 prefactor.
    #
    # This replaces `tl_extra_shim.log1p` (undefined symbol: __nv_log1pf on
    # XPU) *and* the previous 29-iteration runtime Taylor loop: no division,
    # no int->float conversion, no loop.
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
def igammac_kernel_xpu(
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
    # other=1.0 / other=0.0: masked lanes never feed valid results (their
    # stores are masked out as well).
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    a_f = a.to(tl.float32)
    x_f = x.to(tl.float32)

    # Detect inf and NaN (same edge semantics as the generic kernel).
    is_nan_x = x_f != x_f
    is_nan_a = a_f != a_f
    is_inf_or_nan_x = (x_f * 0.0) != 0.0
    is_inf_or_nan_a = (a_f * 0.0) != 0.0
    is_inf_x = is_inf_or_nan_x & ~is_nan_x
    is_inf_a = is_inf_or_nan_a & ~is_nan_a
    is_finite = ~is_inf_or_nan_x & ~is_inf_or_nan_a
    in_domain = (a_f > 0.0) & (x_f >= 0.0) & is_finite

    log_gamma_a = _lgamma_pos(a_f)
    log_x_term = a_f * tl.log(x_f) - x_f - log_gamma_a

    # Path 1: power series for P(a, x) = e^{-x} x^a sum_n x^n / Gamma(a+n+1),
    # then Q = 1 - P. Converges for x < a + 1. The denominator a + i is
    # carried incrementally (`ai`) so the loop body holds no int->float
    # conversion; 24 terms are past the fp32 error floor for every
    # (a, x) that reaches this branch (worst case a ~ 10, x ~ a + 1).
    term = 1.0 / a_f
    series_sum = term
    ai = a_f
    for _s in range(1, N_SER):
        ai = ai + 1.0
        term = term * (x_f / ai)
        series_sum = series_sum + term
    q_series = 1.0 - tl.exp(log_x_term) * series_sum

    # Path 2: the Legendre continued fraction for Q(a, x) directly,
    #   Q = e^{a log x - x - log Gamma(a)} / w,
    #   w = b0 + a1/(b1 + a2/(b2 + ...)),  a_n = n (a - n),  b_n = x+2n+1-a.
    # Evaluated *backwards* from the N_CF-th convergent (w <- b_N, then
    # w <- b_{n-1} + a_n / w for n = N..1). Compared with forward modified
    # Lentz this drops one division, two `tl.abs`, two comparisons and the
    # two `tl.where` re-normalisation guards per step: the guards only exist
    # to keep a *convergence-tested* forward recurrence from dividing by
    # zero, and with a fixed term count there is nothing to test.
    # Both coefficient sequences are carried incrementally:
    #   b_{n-1} = b_n - 2      and      a_{n-1} = a_n + (b_{n-1} - x).
    w = x_f + (2.0 * N_CF + 1.0) - a_f
    b_prev = x_f + (2.0 * N_CF - 1.0) - a_f
    an = (a_f - (1.0 * N_CF)) * (1.0 * N_CF)
    for _c in range(N_CF):
        w = b_prev + an / w
        an = an + (b_prev - x_f)
        b_prev = b_prev - 2.0
    q_cf = tl.exp(log_x_term - tl.log(w))

    # Path 3: DLMF 8.12.4 asymptotic expansion for large a with x ~ a:
    #   Q(a,x) ~ 0.5 erfc(eta sqrt(a/2)) + e^{-a eta^2 / 2}
    #            * sum_k c_k(eta) / a^k / sqrt(2 pi a),
    # with sigma = (x-a)/a, eta = sgn(x-a) sqrt(-2(log(1+sigma)-sigma)).
    # c_k(eta) = sum_n d[k,n] eta^n from DLMF table 8.12.1; for a > 10 and
    # fp32 output only c_0..c_3 are above the rounding floor, and each is
    # truncated at the term that falls below it (8/5/3/1 coefficients).
    # The literals are inlined because a `@triton.jit` body cannot read a
    # module-level table, and inlining also removes 64 scalar global loads
    # per element that the previous revision issued from `d_ptr`.
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

    q_asym = 0.5 * (1.0 - tl.math.erf(eta * tl.sqrt(a_f * 0.5)))
    q_asym = q_asym + tl.exp(-0.5 * a_f * eta * eta) * poly_sum / tl.sqrt(
        2.0 * 3.141592653589793 * a_f
    )

    # Per-element path selection. Note: a plain two-level
    # `tl.where(use_asym, q_asym, tl.where(use_series, q_series, q_cf))`
    # blows the XPU uni_sram pass (three distinct live values); the
    # mask-and-sum form is NaN-safe (unselected lanes are forced to 0.0
    # before summing) and compiles fine. Because exactly one term survives,
    # the [0, 1] projection is applied once to the sum rather than to each
    # branch.
    use_asym = (a_f > 10.0) & (tl.abs(sigma) < 0.4)
    use_series = (x_f < (a_f + 1.0)) & ~use_asym
    use_cf = ~use_asym & ~use_series
    computed = (
        tl.where(use_asym, q_asym, 0.0)
        + tl.where(use_series, q_series, 0.0)
        + tl.where(use_cf, q_cf, 0.0)
    )
    computed = tl.minimum(tl.maximum(computed, 0.0), 1.0)

    # Boundary and infinity handling:
    # Q(a, 0) = 1, Q(inf, x) = 1, Q(a, inf) = 0, Q(inf, inf) = NaN,
    # out-of-domain (a <= 0 or x < 0) gives NaN.
    inf_result = tl.where(
        is_inf_x & is_inf_a,
        float("nan"),
        tl.where(is_inf_x, 0.0, tl.where(is_inf_a, 1.0, float("nan"))),
    )
    result = tl.where(
        is_finite,
        tl.where(in_domain, computed, float("nan")),
        inf_result,
    )

    tl.store(out_ptr + offsets, result.to(out_ptr.type.element_ty), mask=mask)


def _launch(out: torch.Tensor, a: torch.Tensor, x: torch.Tensor):
    a_c = a.contiguous()
    x_c = x.contiguous()
    was_noncontig = not out.is_contiguous()
    out_c = out.contiguous() if was_noncontig else out

    n = out_c.numel()
    if n > 0:
        BLOCK = 512
        grid = (triton.cdiv(n, BLOCK),)
        igammac_kernel_xpu[grid](
            a_c,
            x_c,
            out_c,
            n,
            BLOCK_SIZE=BLOCK,
            N_SER=24,
            N_CF=18,
            buffer_size_limit=2048,
        )

    if was_noncontig:
        out.copy_(out_c)
    return out


def igammac(a: torch.Tensor, x: torch.Tensor, *, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN IGAMMAC")
    if a.device.type != flag_gems.device:
        raise ValueError(f"igammac: first input tensor must be on {flag_gems.device}")
    if x.device.type != flag_gems.device:
        raise ValueError(f"igammac: second input tensor must be on {flag_gems.device}")

    if not a.is_floating_point():
        a = a.to(torch.get_default_dtype())
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    if a.dtype not in (torch.float32, torch.float64) or x.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(
            f"igammac Triton kernel supports fp32/fp64, but got "
            f"a.dtype={a.dtype}, x.dtype={x.dtype}"
        )

    if out is None:
        out_dtype = torch.promote_types(a.dtype, x.dtype)
        out = torch.empty_like(a, dtype=out_dtype, device=a.device)
    else:
        if out.device.type != flag_gems.device:
            raise ValueError(
                f"igammac_out: output tensor must be on {flag_gems.device}"
            )
        if not out.is_floating_point():
            raise TypeError("igammac_out: output tensor must be a floating point type")
        if a.numel() != x.numel() or a.numel() != out.numel():
            raise ValueError(
                "igammac_out: input and output must have the same number of elements"
            )

    if a.dtype != out.dtype:
        a = a.to(out.dtype)
    if x.dtype != out.dtype:
        x = x.to(out.dtype)

    _launch(out, a, x)
    return out


def igammac_out(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN IGAMMAC_OUT")
    return igammac(a, x, out=out)
