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

# Kunlunxin (XPU) override of polygamma (out-of-place, aten::polygamma).
#
# Root causes of the generic `flag_gems/ops/polygamma.py` on XPU (all
# confirmed at compile/link time and numerics on xpu3):
#   1. n=1 (trigamma): `tl_extra_shim.fast_dividef` resolves to
#      `undefined symbol: Unsupported` at link time (generic polygamma.py:86).
#   2. n>=2 (Hurwitz zeta): `tl_extra_shim.lgamma` resolves to
#      `undefined symbol: Unsupported` at link time (generic polygamma.py:227;
#      same root cause as the lgamma / special_gammainc / special_gammaln /
#      mvlgamma / igammac overrides).
#   3. `@triton.autotune` (6 configs, n=0/n=1 raw kernels) re-compiles per
#      shape on XPU and inflates IR (harness lesson 2.1); the vendor digamma
#      override also documents the raw-pointer recurrence crashing the
#      TritonXPUVectorize pass on bf16.
#   4. Numerics (discovered while porting, all verified against a double
#      precision reference that the CPU fp32 reference matches bit-for-bit):
#      a. `tl.sin(pi*x)` on XPU carries ~4e-8 absolute error, so near the
#         integer poles (|x - round(x)| < ~1e-7) the reflected trigamma term
#         pi^2/sin^2(pi x) is wrong by tens of %. Fixed with a Taylor sin^2
#         in the reduced argument d = |x - round(x)| (12th order is accurate
#         to < 1.3e-9 over the whole [0, 0.5] domain, probed).
#      b. The zeta/trigamma sums accumulate ~1e-6 relative error over ~16
#         positive terms at n>=3 on fp32 (n=8 polygamma ~40320 * zeta near
#         x=1 sits right at the 1.3e-6 rtol edge). Fixed with Kahan-style
#         compensated accumulation; XPU libdevice powf is bit-exact vs the
#         fp64 reference (probed over 1M samples), so the remaining error is
#         dominated by the additions only.
#
# Fix: dedicated pointwise_dynamic kernels (single config, no autotune) that
# keep the generic math bit-for-bit where it is sound:
#   - n=0: digamma body (tl.log/tl.cos/tl.sin only - all link on XPU).
#   - n=1: trigamma body with IEEE division, Taylor sin^2 reflection and
#     compensated accumulation.
#   - n>=2: zeta body with `tl_extra_shim.pow` (verified to link and run on
#     XPU) and an inline Lanczos g=7 log-gamma `_lgamma_pos` for the n!
#     factor exp(lgamma(n+1)); s = n+1 >= 2 is always positive so no
#     reflection branch is needed (same helper as lgamma / mvlgamma_ /
#     igammac overrides).
import logging

import triton
import triton.language as tl

from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_pow = tl_extra_shim.pow
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


# ---------------------------------------------------------------------------
# Shared math (jit helpers). PI is defined inside each helper: Triton forbids
# reading module-level globals from @jit'ed code.
# ---------------------------------------------------------------------------


@triton.jit
def _digamma_body(x_f32):
    pi = 3.1415926535897932384626433832795028841971
    reflect_mask = x_f32 < 0.5
    xr = tl.where(reflect_mask, 1.0 - x_f32, x_f32)
    s = tl.zeros_like(x_f32)
    y = xr
    for _ in range(8):
        m = y < 8.0
        s = s - tl.where(m, 1.0 / y, 0.0)
        y = tl.where(m, y + 1.0, y)
    r = 1.0 / y
    t2 = r * r
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    series = (
        (-0.5 * r)
        + (-1.0 / 12.0) * t2
        + (1.0 / 120.0) * t4
        + (-1.0 / 252.0) * t6
        + (1.0 / 240.0) * t8
    )
    psi_y = tl.log(y) + s + series
    cot_term = tl.cos(pi * x_f32) / tl.sin(pi * x_f32)
    return tl.where(reflect_mask, psi_y - pi * cot_term, psi_y)


@triton.jit
def _sin2_pi_x(x_f32):
    # sin^2(pi x) reproducing the float32 argument the CPU reference's
    # sinf(c10::pi<float> * x) uses, evaluated to ~1e-7 relative accuracy in
    # fp32-only arithmetic (XPU has no fp64 and its tl.sin carries ~2.5e-7
    # absolute error, both probed).
    #
    # p = fl32(pi*x) is the reference-identical quantized argument; theta is
    # p - n*pi recovered exactly to ~1e-13 through an fma chain with a split
    # pi = 3.1415927410125732421875 + (-8.742278e-8) (the fma rounds only
    # once per step, at the magnitude of the shrunken result). Then
    # sin^2(pi x) = sin^2(theta) via a 10th-order Taylor in theta^2
    # (|theta| <= 0.9, remainder < 1e-14), or = cos^2(pi/2 - theta) with the
    # pi/2-split for the remaining arc (probed: worst rel err 8.2e-7 over
    # 350k lanes of [-5,5] with pole concentrations, 0 lanes above 1e-6).
    p = x_f32 * 3.1415926535897932384626433832795028841971
    nv = tl.floor(p * 0.31830988618379067154 + 0.5)
    h1 = tl.math.fma(-nv, 3.1415927410125732421875, p)
    theta = tl.math.fma(-nv, -8.742278012618954e-8, h1)
    # fold into [-pi/2, pi/2] exactly (the fma rounding above is at
    # |theta| magnitude -- up to pi -- and would be an absolute ~1e-7 error
    # on the phase; folding keeps |theta| <= pi/2 so rounding stays ~1e-7
    # absolute, i.e. ~1e-7 relative on sin).
    sigma = tl.where(
        theta > 1.5707963267948966,
        1.0,
        tl.where(theta < -1.5707963267948966, -1.0, 0.0),
    )
    a1 = tl.math.fma(-sigma, 3.1415927410125732421875, theta)
    theta = tl.math.fma(-sigma, -8.742278012618954e-8, a1)
    s = tl.abs(theta)
    s2 = s * s
    poly = (
        1.0
        - s2 / 6.0
        + s2 * s2 / 120.0
        - s2 * s2 * s2 / 5040.0
        + s2 * s2 * s2 * s2 / 362880.0
        - s2 * s2 * s2 * s2 * s2 / 39916800.0
    )
    taylor = s2 * poly * poly
    # sin(s) = cos(pi/2 - s) for s in (0.9, pi/2]; pi/2 = 1.5707963267948966,
    # split pi2 = 1.5707963705062866 + 4.3711483e-8 (sub rounded at |u| scale).
    u = 1.5707963705062866 - s
    u2 = u * u
    cospol = (
        1.0
        - u2 / 2.0
        + u2 * u2 / 24.0
        - u2 * u2 * u2 / 720.0
        + u2 * u2 * u2 * u2 / 40320.0
    )
    return tl.where(s < 0.9, taylor, cospol * cospol)


@triton.jit
def _trigamma_body(x_f32):
    pi = 3.1415926535897932384626433832795028841971
    reflect_mask = x_f32 < 0.5
    # The reflection term pi^2/sin^2(pi x) must reproduce the fp32 CPU
    # reference bit-closely: near an integer pole the reference's sinf of the
    # quantized float32 pi*x product differs from the true value by O(1), so
    # the kernel has to evaluate the *same* quantized argument (see
    # _sin2_pi_x). tl.sin alone would fail there (absolute error ~4e-8 vs
    # pole phases below 1e-6).
    csc_sq = (pi * pi) / _sin2_pi_x(x_f32)
    result = tl.where(reflect_mask, -csc_sq, 0.0)
    y = tl.where(reflect_mask, 1.0 - x_f32, x_f32)
    # Kahan-style compensated accumulation (the 6 recurrence terms plus the
    # asymptotic tail all round at the running sum's magnitude).
    comp = tl.zeros_like(x_f32)
    for _ in range(6):
        term = 1.0 / (y * y)
        z = term - comp
        t = result + z
        comp = (t - result) - z
        result = t
        y += 1.0
    iyy = 1.0 / (y * y)
    term = (
        1.0
        + 1.0 / (2.0 * y)
        + iyy * (1.0 / 6.0 - iyy * (1.0 / 30.0 - iyy * (1.0 / 42.0)))
    ) / y
    z = term - comp
    t = result + z
    result = t
    sign = tl.where(reflect_mask, -1.0, 1.0)
    return sign * result


@triton.jit
def _lgamma_pos(z):
    # Lanczos approximation of log-gamma for z > 0 (g=7, n=9 coefficients).
    # XPU Triton has no `lgamma` intrinsic (undefined symbol at link time),
    # so it is evaluated inline in fp32. polygamma feeds s = n + 1 >= 2,
    # always positive, so no reflection branch is needed.
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


# ---------------------------------------------------------------------------
# pointwise_dynamic kernels (generic fallback math, XPU-safe intrinsics).
# ---------------------------------------------------------------------------


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def _polygamma_digamma_kernel_fn(x):
    return _digamma_body(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=config_)
@triton.jit
def _polygamma_trigamma_kernel_fn(x):
    return _trigamma_body(x.to(tl.float32))


@triton.jit
def _pow2i(e):
    # exact 2^e for integer e in [-126, 127] (and inf overflow beyond),
    # built from the fp32 exponent bits; ld.pow(2.0, k) on XPU has up to
    # ~4e-6 relative error (probed), which shows up on the zeta pole lanes.
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _pown_s(a, s):
    # a^-s for |a| < 1: libdevice pow loses up to ~1.8e-6 relative accuracy
    # on small |a| (probed; the pole-adjacent zeta lanes then fail ~3.5e-6
    # rel vs the 1.3e-6 test tolerance). Instead scale the base into [1, 2)
    # with an exact power of two, where libdevice pow is bit-exact (probed
    # over 1M samples), then scale the result back with exact powers of two
    # (split in two multiplies so intermediates cannot overflow to inf; the
    # real 1/|a|^s below fp32 max would otherwise become inf*small = inf ->
    # inf - inf = NaN in the zeta sum, 46/1M lanes probed at n=7). libdevice
    # pow handles the (-1)^s sign of negative bases itself.
    t = a.to(tl.uint32, bitcast=True)
    eb = (t >> 23) & 0xFF
    eb = tl.maximum(eb, 1)
    k = (127 - eb).to(tl.int32)
    m = a * _pow2i(k)
    p = _pow(m, -s)
    # a = m * 2^-k  =>  a^-s = m^-s * 2^(k*s)
    si = tl.full((1,), 0, tl.int32) + s.to(tl.int32)
    ks = k * si
    e1 = ks >> 1
    e2 = ks - e1
    return (p * _pow2i(e1)) * _pow2i(e2)


@triton.jit
def _zeta_pow(a, s):
    # zeta-series power term a^-s: exact binary exponentiation on the
    # near-pole terms (|a| < 1), libdevice pow elsewhere (= [-1] * s).
    return tl.where(tl.abs(a) < 1.0, _pown_s(a, s), _pow(a, -s))


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    config=config_,
)
@triton.jit
def _polygamma_zeta_kernel_fn(x, s, scale):
    # polygamma(n, x) = (-1)^(n+1) * n! * zeta(n + 1, x) for n >= 2, with
    # s = n + 1 and sign = (-1)^(n+1). Cephes Euler-Maclaurin Hurwitz zeta in
    # float32, the same algorithm the generic kernel (and PyTorch's CPU/CUDA
    # kernels) use. The n! factor is exp(lgamma(n + 1)) computed inline via
    # Lanczos (`_lgamma_pos`); s >= 2 keeps its argument positive.
    #
    # Compensated (Kahan-style) accumulation: the direct sum adds ~16 pow
    # terms of magnitude <= 1 to a running total, and at n=8 the result
    # (40320 * zeta) sits right at the max-allowed fp32 rtol of the test
    # (1.3e-6), the naive sum's rounding was ~1.7e-6 (probed).
    q = x.to(tl.float32)
    s = s.to(tl.float32)
    total = _zeta_pow(q, s)
    comp = tl.zeros_like(q)
    a = q
    for _ in range(9):
        a += 1.0
        term = _zeta_pow(a, s)
        z = term - comp
        t = total + z
        cn = (t - total) - z
        comp = tl.where(cn != cn, tl.zeros_like(cn), cn)
        total = t
    for _ in range(7):
        cont = a <= 9.0
        a = tl.where(cont, a + 1.0, a)
        term = tl.where(cont, _zeta_pow(a, s), 0.0)
        z = term - comp
        t = total + z
        cn = tl.where(cont, (t - total) - z, comp)
        comp = tl.where(cn != cn, tl.zeros_like(cn), cn)
        total = tl.where(cont, t, total)
    w = a
    w2 = w * w
    b = _pow(w, -s)
    z = b * w / (s - 1.0) - 0.5 * b
    t = total + z
    comp = (t - total) - z
    total = t
    ap = s
    b = b / w
    z = tl.where(b > 0.0, ap * b / 12.0, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 1.0) * (s + 2.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / -720.0, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 3.0) * (s + 4.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / 30240.0, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 5.0) * (s + 6.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / -1209600.0, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 7.0) * (s + 8.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / 47900160.0, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 9.0) * (s + 10.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / -1.8924375803183791606e9, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 11.0) * (s + 12.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / 7.47242496e10, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 13.0) * (s + 14.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / -2.950130727918164224e12, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 15.0) * (s + 16.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / 1.1646782814350067249e14, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 17.0) * (s + 18.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / -4.5979787224074726105e15, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 19.0) * (s + 20.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / 1.8152105401943546773e17, 0.0)
    t = total + z
    comp = (t - total) - z
    total = t
    ap = ap * (s + 21.0) * (s + 22.0)
    b = b / w2
    z = tl.where(b > 0.0, ap * b / -7.1661652561756670113e18, 0.0)
    t = total + z
    total = t
    return scale * total


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def polygamma(n, A):
    logger.debug("GEMS_KUNLUNXIN POLYGAMMA")
    if n < 0:
        raise RuntimeError("polygamma(n, x) does not support negative n.")
    if n == 0:
        return _polygamma_digamma_kernel_fn(A)
    if n == 1:
        return _polygamma_trigamma_kernel_fn(A)
    # s = n + 1; the n! * (-1)^(n+1) factor computed exactly in python (an
    # integer factorial is exactly representable in float up to 2^24, and
    # exp(lgamma) inside the kernel would add ~1e-6 relative error at the
    # 40320 scale that sits right at the test tolerance).
    import math as _math

    s = float(n + 1)
    scale = float(_math.factorial(n))
    if n % 2 == 0:
        scale = -scale
    return _polygamma_zeta_kernel_fn(A, s, scale)


def polygamma_(A, n):
    # In-place variant (aten::polygamma_), same kernels as the out-of-place
    # entry with out0=A: the pointwise_dynamic dispatch writes the result
    # straight into the input storage (no temp tensor, no nested copy_).
    logger.debug("GEMS_KUNLUNXIN POLYGAMMA_")
    if n < 0:
        raise RuntimeError("polygamma(n, x) does not support negative n.")
    if n == 0:
        _polygamma_digamma_kernel_fn(A, out0=A)
    elif n == 1:
        _polygamma_trigamma_kernel_fn(A, out0=A)
    else:
        import math as _math

        s = float(n + 1)
        scale = float(_math.factorial(n))
        if n % 2 == 0:
            scale = -scale
        _polygamma_zeta_kernel_fn(A, s, scale, out0=A)
    return A


def polygamma_out(n, A, out):
    # Out variant (aten::polygamma.out): same kernels as the out-of-place
    # entry, writing the result directly into the caller-provided `out`
    # (out0=out, no temp tensor, no nested copy_). The pointwise_dynamic
    # codegen handles strided/non-contiguous outs at runtime.
    logger.debug("GEMS_KUNLUNXIN POLYGAMMA_OUT")
    if n < 0:
        raise RuntimeError("polygamma(n, x) does not support negative n.")
    if n == 0:
        _polygamma_digamma_kernel_fn(A, out0=out)
    elif n == 1:
        _polygamma_trigamma_kernel_fn(A, out0=out)
    else:
        import math as _math

        s = float(n + 1)
        scale = float(_math.factorial(n))
        if n % 2 == 0:
            scale = -scale
        _polygamma_zeta_kernel_fn(A, s, scale, out0=out)
    return out
