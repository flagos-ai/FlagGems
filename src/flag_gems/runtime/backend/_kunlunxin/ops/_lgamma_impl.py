# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) inline log-gamma helpers.
#
# XPU Triton has no `lgamma` intrinsic (`tl_extra_shim.lgamma` resolves to
# `undefined symbol: Unsupported` at xpu3 link time). This module supplies
# fp32 inline replacements built on `tl.log` + `tl.sin` (both supported):
#
# - `_lgamma_pos(z)`: Lanczos g=7 for z > 0 (fastest, use when domain proven).
# - `_lgamma_full(z)`: full-domain log|Gamma(z)| using the reflection
#   identity  Γ(z)·Γ(1−z) = π / sin(π z)  for z < 0.5. Not exact at negative
#   integer poles (returns +inf via log(0)); torch.lgamma matches that.
import triton
import triton.language as tl


@triton.jit
def _lgamma_pos(z):
    x = 0.99999999999980993
    x = x + 676.5203681218851 / z
    x = x + (-1259.1392167224028) / (z + 1.0)
    x = x + 771.32342877765313 / (z + 2.0)
    x = x + (-176.61502916214059) / (z + 3.0)
    x = x + 12.507343278686905 / (z + 4.0)
    x = x + (-0.13857109526572012) / (z + 5.0)
    x = x + 9.9843695780195716e-6 / (z + 6.0)
    x = x + 1.5056327351493116e-7 / (z + 7.0)
    t = (z - 1.0) + 7.0 + 0.5
    return 0.9189385332046727 + ((z - 1.0) + 0.5) * tl.log(t) - t + tl.log(x)


@triton.jit
def _lgamma_full(z):
    # For z >= 0.5 evaluate Lanczos directly; for z < 0.5 use reflection:
    #   log|Gamma(z)| = log(pi) - log|sin(pi*z)| - log|Gamma(1-z)|
    # Evaluate both branches unconditionally, then blend with tl.where to
    # keep this XPU-safe (no data-dependent control flow).
    pi = 3.141592653589793
    log_pi = 1.1447298858494002
    lg_pos = _lgamma_pos(tl.where(z >= 0.5, z, 1.0 - z))
    sin_val = tl.abs(tl.sin(pi * z))
    lg_neg = log_pi - tl.log(sin_val) - lg_pos
    return tl.where(z >= 0.5, lg_pos, lg_neg)
