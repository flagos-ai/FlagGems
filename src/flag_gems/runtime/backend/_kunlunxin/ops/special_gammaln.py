# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of special_gammaln / special_gammaln_out.
#
# Root cause: generic `flag_gems/ops/special_gammaln.py` calls
# `tl_extra_shim.lgamma(x)` which resolves to `undefined symbol: Unsupported`
# at xpu3 link time (shared gamma-family root cause).
#
# Fix: inline full-domain log|Gamma(z)|. Lanczos g=7 works directly for ALL
# z > 0 (no 0.5 split needed); the reflection identity
#   log|Gamma(z)| = log(pi) - log|sin(pi z)| - lgamma(1-z)
# is used only for z <= 0. Using reflection for small positive z (e.g. 1e-7)
# is catastrophically inaccurate because sin(pi z) underflows precision, so we
# restrict reflection to the negative half-line. The special_gammaln test
# feeds `torch.randn(...)` (can be negative), so the reflection branch is
# required. Both branches are evaluated unconditionally and blended with
# `tl.where` to stay XPU-safe (no data-dependent control flow).
import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _lgamma_pos(z):
    # Stirling-with-shift: for z>0, shift z up by N=8 so Stirling is accurate
    # (avoids fp32 catastrophic cancellation of the Lanczos polynomial), then
    # subtract log(product z*(z+1)*...*(z+N-1)) via the recurrence
    #   lgamma(z) = lgamma(z+N) - sum_i log(z+i).
    w = z + 8.0
    inv = 1.0 / w
    inv2 = inv * inv
    # Stirling asymptotic series through 1/w^9:
    # (1/12)/w - (1/360)/w^3 + (1/1260)/w^5 - (1/1680)/w^7 + (1/1188)/w^9
    corr = inv * (
        (1.0 / 12.0)
        - inv2 * (
            (1.0 / 360.0)
            - inv2 * (
                (1.0 / 1260.0)
                - inv2 * ((1.0 / 1680.0) - inv2 * (1.0 / 1188.0))
            )
        )
    )
    lg_w = (w - 0.5) * tl.log(w) - w + 0.9189385332046727 + corr
    lp = (
        tl.log(z)
        + tl.log(z + 1.0)
        + tl.log(z + 2.0)
        + tl.log(z + 3.0)
        + tl.log(z + 4.0)
        + tl.log(z + 5.0)
        + tl.log(z + 6.0)
        + tl.log(z + 7.0)
    )
    return lg_w - lp


@triton.jit
def _lgamma_full(z):
    pi = 3.141592653589793
    log_pi = 1.1447298858494002
    # Direct Lanczos for z > 0; reflection for z <= 0. Pick a safe positive
    # dummy for the disabled branch so we never divide by zero / negative.
    z_pos = tl.where(z > 0.0, z, 1.0 - z)
    lg_pos = _lgamma_pos(z_pos)
    # Argument-reduce sin(pi*z) to keep the argument in [0, pi/2] for maximum
    # fp32 accuracy near gamma poles. Use the signed distance to the NEAREST
    # integer r = z - round(z) in [-0.5, 0.5]; |sin(pi*z)| = sin(pi*|r|).
    # This avoids the fp32 cancellation of `1 - (z - floor(z))` that destroys
    # precision when z is just below an integer (e.g. z = -8e-6).
    half = tl.abs(z - tl.floor(z + 0.5))
    u = pi * half
    # XPU tl.sin is inaccurate for very small arguments (returns 3.55e-5
    # instead of 3.58e-5 for u ~ 3.6e-5, ~0.7% relative error). Compute
    # log(sin(u)) via sinc series to avoid it:
    #   log(sin(u)) = log(u) + log(sin(u)/u)
    # where sin(u)/u = sum_{k>=0} (-1)^k u^{2k} / (2k+1)!  is well-conditioned
    # on [0, pi/2] and its log() is accurate via the standard log().
    u2 = u * u
    # 7-term series through u^12: enough for ~1e-8 relative on [0, pi/2].
    sinc = (
        1.0
        - u2 * (
            (1.0 / 6.0)
            - u2 * (
                (1.0 / 120.0)
                - u2 * (
                    (1.0 / 5040.0)
                    - u2 * (
                        (1.0 / 362880.0)
                        - u2 * ((1.0 / 39916800.0) - u2 * (1.0 / 6227020800.0))
                    )
                )
            )
        )
    )
    log_sin = tl.log(u) + tl.log(sinc)
    lg_neg = log_pi - log_sin - lg_pos
    return tl.where(z > 0.0, lg_pos, lg_neg)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def special_gammaln_func(x):
    return _lgamma_full(x.to(tl.float32)).to(x.dtype)


def special_gammaln(A):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_GAMMALN")
    return special_gammaln_func(A)


def special_gammaln_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_GAMMALN_OUT")
    if out is None:
        return special_gammaln_func(A)
    special_gammaln_func(A, out0=out)
    return out
