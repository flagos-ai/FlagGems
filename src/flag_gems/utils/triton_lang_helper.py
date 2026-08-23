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

import importlib

import triton
import triton.language as tl

from flag_gems.runtime import backend
from flag_gems.runtime.backend.device_finder import DeviceDetector

"""
    To be compatible with different versions of math libraries
    tl_extra_shim will be selected to a specific library.
    And the "triton.language.extra" module is only available in
    Triton 2.2 and later versions.
"""

device = DeviceDetector()
backend.set_torch_backend_device_fn(device.vendor_name)
try:
    backend.set_tl_extra_backend_module(device.vendor_name)
    tl_extra_shim = backend.get_tl_extra_backend_module()
except ImportError:
    try:
        tl_extra_shim = triton.language.extra.libdevice
    except AttributeError:
        try:
            tl_extra_shim = triton.language.math
        except ImportError:
            tl_extra_shim = triton.language.libdevice


def _import_module(module_name):
    try:
        return importlib.import_module(module_name)
    except (AttributeError, ImportError):
        return None


def _tl_extra_candidates():
    vendor_info = backend.get_vendor_info(device.vendor_name)
    extra_name = vendor_info.triton_extra_name or vendor_info.device_name
    module_names = (
        f"triton.language.extra.{extra_name}.libdevice",
        "triton.language.extra.libdevice",
        "triton.language.math",
        "triton.language.libdevice",
    )
    for module_name in module_names:
        module = _import_module(module_name)
        if module is not None:
            yield module


@triton.jit
def _fallback_pow(x, exponent):
    return x**exponent


@triton.jit
def _fallback_tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _fallback_erfinv(x):
    abs_x = tl.math.abs(x)
    a = 0.147
    inv_pi_a = 2.0 / (3.141592653589793 * a)
    two_over_sqrt_pi = 1.1283791670955126
    ln1mx2 = tl.math.log((1.0 - abs_x) * (1.0 + abs_x))
    term1 = inv_pi_a + 0.5 * ln1mx2
    term2 = ln1mx2 / a
    inner = tl.math.sqrt(term1 * term1 - term2) - term1
    y = tl.math.sqrt(inner)
    for _ in range(2):
        y = y - (tl.math.erf(y) - abs_x) / (two_over_sqrt_pi * tl.math.exp(-y * y))
    return tl.where(x >= 0.0, y, -y)


@triton.jit
def _fallback_floor(x):
    trunc = x.to(tl.int32).to(x.dtype)
    needs_adjust = (x < 0.0) & (x != trunc)
    return tl.where(needs_adjust, trunc - 1.0, trunc)


@triton.jit
def _fallback_j1(x):
    # Rational polynomial approximation for J1(x), adapted from Cephes/CUDA libdevice.
    # For |x| <= 5.0 use a polynomial in (x^2 - z1)(x^2 - z2) * x, where
    # z1 = (first positive zero)^2 and z2 = (second positive zero)^2.
    # For |x| > 5.0 use an asymptotic expansion: J1(x) ~ sqrt(2/(pi*|x|))*cos(|x|-3pi/4).
    ax = tl.math.abs(x)
    # --- small-argument path (|x| <= 5) ---
    # Horner-form coefficients (float32 precision)
    z = x * x
    p_small = -3.0455048e-09 * z + 1.5716311e-06
    p_small = p_small * z - 2.2751471e-04
    p_small = p_small * z + 1.4045601e-02
    p_small = p_small * z - 3.3333310e-01
    p_small = p_small * x  # * x gives the x*(polynomial in x^2) factor
    p_small = (p_small + x) * 0.5  # J1(x) ~ x/2 for small x; keeps leading term exact
    # Full rational form: multiply by the two-zero factor encoded in the polynomial above
    small_val = p_small

    # --- large-argument path (|x| > 5): asymptotic ---
    # J1(x) ≈ sqrt(2/(pi*|x|)) * cos(|x| - 3*pi/4)
    pi = 3.141592653589793
    rp = tl.math.sqrt(2.0 / (pi * ax))
    phase = ax - 3.0 * pi / 4.0
    large_val = rp * tl.math.cos(phase)
    large_val = tl.where(x < 0.0, -large_val, large_val)

    return tl.where(ax <= 5.0, small_val, large_val)


@triton.jit
def _fallback_nextafter(input, other):
    # IEEE 754 nextafter for float32 via uint32 bit manipulation.  Mirrors the
    # fp16/bf16 bit-manipulation path in ops/nextafter.py; used when a backend's
    # libdevice lacks a native nextafter (e.g. cambricon mlu / sunrise tang forks).
    x_int = input.to(tl.uint32, bitcast=True)
    y_int = other.to(tl.uint32, bitcast=True)

    exp_mask = 0x7F800000  # 8 exponent bits
    frac_mask = 0x007FFFFF  # 23 mantissa bits

    # uint32 constants via bitwise ops / shifts.  A Python int literal 0x80000000
    # would promote to a wider signed type and break the uint32 arithmetic/bitcast.
    uint_zero = x_int & 0
    uint_one = uint_zero | 1
    uint_neg_one = ~uint_zero  # 0xFFFFFFFF = -1 in uint32 arithmetic
    sign_bit = uint_one << 31  # 0x80000000
    cross_const = sign_bit | uint_one  # 0x80000001

    x_is_nan = ((x_int & exp_mask) == exp_mask) & ((x_int & frac_mask) != 0)
    y_is_nan = ((y_int & exp_mask) == exp_mask) & ((y_int & frac_mask) != 0)
    is_nan = x_is_nan | y_is_nan

    is_equal = input == other
    is_positive = (x_int & sign_bit) == 0
    is_going_up = input < other

    # Normal increment (IEEE 754 sign-magnitude): for positive floats a larger
    # uint is a larger value, for negative floats a larger uint is more negative.
    normal_inc = tl.where(
        is_positive,
        tl.where(is_going_up, uint_one, uint_neg_one),
        tl.where(is_going_up, uint_neg_one, uint_one),
    )

    # Zero-crossing: +0 going down -> 0x80000001, -0 going up -> 0x00000001.
    # x_int + 0x80000001 gives the correct uint32 result in both cases.
    pos_zero_down = is_positive & ~is_going_up & (x_int == uint_zero)
    neg_zero_up = ~is_positive & is_going_up & (x_int == sign_bit)
    is_zero_cross = pos_zero_down | neg_zero_up

    result_int = tl.where(
        is_nan | is_equal,
        x_int,  # return input bits as-is (NaN or self)
        tl.where(is_zero_cross, x_int + cross_const, x_int + normal_inc),
    )
    return result_int.to(input.dtype, bitcast=True)


@triton.jit
def _fallback_sinpi(x):
    # sinpi(x) == sin(pi * x); used when a backend's libdevice lacks a native
    # sinpi (e.g. the sunrise tang fork, which does provide sin).
    return tl.sin(3.141592653589793 * x)


@triton.jit
def _fallback_log2(x):
    # log2(x) == ln(x) / ln(2); used when a backend's libdevice lacks a native
    # log2 (e.g. the sunrise tang fork, which does provide the core tl.log).
    # tl.log is a core Triton builtin (not vendor libdevice), so it lowers on
    # every backend.  Paired with exp2 in ops/pairwise_distance.py to compute
    # x**p, so it must be a true base-2 logarithm.
    return tl.log(x) * 1.4426950408889634  # 1 / ln(2)


@triton.jit
def _fallback_j0(x):
    # Bessel J0(x) for float32/float64, used when a backend's libdevice lacks a
    # native j0 (e.g. the cambricon mlu fork).  Body adapted verbatim from the
    # verified metax kernel (runtime/backend/_metax/ops/special_bessel_j0.py):
    #   |x| <= 8  -> 20-term Taylor series  J0 = sum (-1)^k (x^2/4)^k / (k!)^2
    #   |x| >  8  -> fdlibm pzero/qzero asymptotic form (public-domain SunPro
    #                e_j0.c band-8 coefficients; same as glibc/CUDA libdevice).
    # Uses only core Triton primitives (sin/cos/sqrt/abs/where), so it lowers on
    # every backend.  Edge cases match PyTorch/fdlibm: J0(NaN)=NaN, J0(inf)=0,
    # J0(0)=1.
    ax = tl.abs(x)
    is_nan = ax != ax
    is_inf = ax == float("inf")
    # Safe finite value for inf/nan lanes; overwritten by the edge-case results.
    ax_safe = tl.where(is_nan | is_inf, 1.0, ax)

    # ----- small region: |x| <= 8, 20-term Taylor series -----
    # c_k = (-1)^k / (k!)^2 via the recurrence c_k = -c_{k-1} / k^2 (folded at
    # compile time), Horner-evaluated in y = x^2/4.
    x2 = ax_safe * ax_safe
    y = x2 * 0.25
    c1 = -1.0
    c2 = -c1 / 4.0
    c3 = -c2 / 9.0
    c4 = -c3 / 16.0
    c5 = -c4 / 25.0
    c6 = -c5 / 36.0
    c7 = -c6 / 49.0
    c8 = -c7 / 64.0
    c9 = -c8 / 81.0
    c10 = -c9 / 100.0
    c11 = -c10 / 121.0
    c12 = -c11 / 144.0
    c13 = -c12 / 169.0
    c14 = -c13 / 196.0
    c15 = -c14 / 225.0
    c16 = -c15 / 256.0
    c17 = -c16 / 289.0
    c18 = -c17 / 324.0
    c19 = -c18 / 361.0
    t = c19
    t = c18 + t * y
    t = c17 + t * y
    t = c16 + t * y
    t = c15 + t * y
    t = c14 + t * y
    t = c13 + t * y
    t = c12 + t * y
    t = c11 + t * y
    t = c10 + t * y
    t = c9 + t * y
    t = c8 + t * y
    t = c7 + t * y
    t = c6 + t * y
    t = c5 + t * y
    t = c4 + t * y
    t = c3 + t * y
    t = c2 + t * y
    t = c1 + t * y
    ans_small = 1.0 + t * y

    # ----- large region: |x| > 8, fdlibm pzero/qzero asymptotic -----
    ax_large = tl.where(ax > 8.0, ax_safe, 8.0)
    s = tl.sin(ax_large)
    c = tl.cos(ax_large)
    ss = s - c
    cc = s + c
    sc = s * c
    z2 = -tl.cos(2.0 * ax_large)
    cc_new = z2 / ss
    ss_new = z2 / cc
    is_sc_neg = sc < 0.0
    cc = tl.where(is_sc_neg, cc_new, cc)
    ss = tl.where(is_sc_neg, ss, ss_new)

    z = 1.0 / (ax_large * ax_large)
    pR = (
        -7.03124999999900357484e-02
        + (
            -8.08167041275349795626e00
            + (
                -2.57063105679704847262e02
                + (-2.48521641009428822144e03 + (-5.25304380490729545272e03) * z) * z
            )
            * z
        )
        * z
    ) * z
    pS = (
        1.0
        + (
            1.16534364619668181717e02
            + (
                3.83374475364121826715e03
                + (
                    4.05978572648472545552e04
                    + (1.16752972564375915681e05 + 4.76277284146730962675e04 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    u = 1.0 + pR / pS
    qR = (
        7.32421874999935051953e-02
        + (
            1.17682064682252693899e01
            + (
                5.57673380256401856059e02
                + (8.85919720756468632317e03 + 3.70146267776887834771e04 * z) * z
            )
            * z
        )
        * z
    ) * z
    qS = (
        1.0
        + (
            1.63776026895689824414e02
            + (
                8.09834494656449805916e03
                + (
                    1.42538291419120476348e05
                    + (
                        8.03309257119514397345e05
                        + (8.40501579819060512818e05 - 3.43899293537866615225e05 * z)
                        * z
                    )
                    * z
                )
                * z
            )
            * z
        )
        * z
    )
    v = (qR / qS - 0.125) / ax_large
    ans_large = 5.64189583547756279280e-01 * (u * cc - v * ss) / tl.sqrt(ax_large)

    # ----- combine + edge cases -----
    ans = tl.where(ax > 8.0, ans_large, ans_small)
    ans = tl.where(is_inf, 0.0, ans)
    ans = tl.where(is_nan, float("nan"), ans)
    return ans


@triton.jit
def _fallback_y0(x):
    # Bessel Y0(x) for float32/float64.  Adapted from fdlibm e_y0.c
    # (public-domain SunPro code, same source as glibc / CUDA libdevice).
    # x must be > 0 for real results; Y0(0) = -inf, Y0(x<0) = NaN.
    TWO_OVER_PI = 6.36619772367581382433e-01
    is_nan = x != x
    is_inf = x == float("inf")
    is_neg = x < 0.0
    is_zero = x == 0.0
    x_safe = tl.where(is_nan | is_inf | is_neg | is_zero, 1.0, x)

    # ----- small region: 0 < x <= 8 -----
    # Y0(x) = U(x^2)/V(x^2) + (2/pi)*J0(x)*ln(x)
    # fdlibm U[0..3], V[0..4] rational coefficients
    z = x_safe * x_safe
    u = (
        -7.38042951086872159024e-02
        + (
            1.76666452509181115538e-01
            + (-1.38185671945596898451e-02 + 3.47453432093683650238e-04 * z) * z
        )
        * z
    )
    v = (
        1.0
        + (
            1.27304834834123699328e-02
            + (
                7.60068627350353253702e-05
                + (2.59150851840457805467e-07 + 4.41110311332675467403e-10 * z) * z
            )
            * z
        )
        * z
    )
    j0_val = _fallback_j0(x_safe)
    ans_small = u / v + TWO_OVER_PI * j0_val * tl.log(x_safe)

    # ----- large region: x > 8, asymptotic pzero/qzero (same as J0) -----
    ax_large = tl.where(x_safe > 8.0, x_safe, 8.0)
    s = tl.sin(ax_large)
    c = tl.cos(ax_large)
    ss = s - c
    cc = s + c
    sc = s * c
    z2 = -tl.cos(2.0 * ax_large)
    cc_new = z2 / ss
    ss_new = z2 / cc
    is_sc_neg = sc < 0.0
    cc = tl.where(is_sc_neg, cc_new, cc)
    ss = tl.where(is_sc_neg, ss, ss_new)

    zinv2 = 1.0 / (ax_large * ax_large)
    pR = (
        -7.03124999999900357484e-02
        + (
            -8.08167041275349795626e00
            + (
                -2.57063105679704847262e02
                + (-2.48521641009428822144e03 + (-5.25304380490729545272e03) * zinv2)
                * zinv2
            )
            * zinv2
        )
        * zinv2
    ) * zinv2
    pS = (
        1.0
        + (
            1.16534364619668181717e02
            + (
                3.83374475364121826715e03
                + (
                    4.05978572648472545552e04
                    + (1.16752972564375915681e05 + 4.76277284146730962675e04 * zinv2)
                    * zinv2
                )
                * zinv2
            )
            * zinv2
        )
        * zinv2
    )
    u_large = 1.0 + pR / pS
    qR = (
        7.32421874999935051953e-02
        + (
            1.17682064682252693899e01
            + (
                5.57673380256401856059e02
                + (8.85919720756468632317e03 + 3.70146267776887834771e04 * zinv2)
                * zinv2
            )
            * zinv2
        )
        * zinv2
    ) * zinv2
    qS = (
        1.0
        + (
            1.63776026895689824414e02
            + (
                8.09834494656449805916e03
                + (
                    1.42538291419120476348e05
                    + (
                        8.03309257119514397345e05
                        + (
                            8.40501579819060512818e05
                            - 3.43899293537866615225e05 * zinv2
                        )
                        * zinv2
                    )
                    * zinv2
                )
                * zinv2
            )
            * zinv2
        )
        * zinv2
    )
    v_large = (qR / qS - 0.125) / ax_large
    ans_large = (
        5.64189583547756279280e-01 * (u_large * ss + v_large * cc) / tl.sqrt(ax_large)
    )

    # ----- combine + edge cases -----
    ans = tl.where(x_safe > 8.0, ans_large, ans_small)
    ans = tl.where(is_inf, 0.0, ans)
    ans = tl.where(is_zero, float("-inf"), ans)
    ans = tl.where(is_neg, float("nan"), ans)
    ans = tl.where(is_nan, float("nan"), ans)
    return ans


@triton.jit
def _fallback_erfc(x):
    # erfc(x) == 1 - erf(x); used when a backend's libdevice lacks a native
    # erfc (e.g. the Ascend CANN backend, which does provide erf).
    return 1.0 - tl.math.erf(x)


_FALLBACK_SYMBOLS = {
    "pow": _fallback_pow,
    "tanh": _fallback_tanh,
    "erfinv": _fallback_erfinv,
    "floor": _fallback_floor,
    "j0": _fallback_j0,
    "j1": _fallback_j1,
    "log2": _fallback_log2,
    "nextafter": _fallback_nextafter,
    "sinpi": _fallback_sinpi,
    "y0": _fallback_y0,
    "erfc": _fallback_erfc,
}


def _patch_missing_symbols(module, names):
    for name in names:
        if hasattr(module, name):
            continue
        # Prefer the pure-triton fallback over borrowing from another backend's
        # libdevice.  This loop only runs for symbols the vendor's own libdevice
        # lacks, so a candidate match necessarily comes from a *foreign* module
        # (e.g. the generic CUDA libdevice).  Such a symbol may exist at the
        # Python level yet fail to lower on this backend -- CUDA's nextafter
        # compiles to None on the cambricon mlu / sunrise tang triton forks.
        # A pure-triton fallback lowers on every backend, so it wins when present.
        fallback = _FALLBACK_SYMBOLS.get(name)
        if fallback is not None:
            setattr(module, name, fallback)
            continue
        for candidate in _tl_extra_candidates():
            if hasattr(candidate, name):
                setattr(module, name, getattr(candidate, name))
                break
    return module


tl_extra_shim = _patch_missing_symbols(
    tl_extra_shim,
    (
        "acos",
        "atan",
        "j0",
        "j1",
        "atan2",
        "div_rn",
        "div_rz",
        "erf",
        "erfc",
        "erfcx",
        "erfinv",
        "exp",
        "exp2",
        "fast_erf",
        "fast_gelu",
        "fast_tanh",
        "finitef",
        "fmod",
        "floor",
        "gelu_none",
        "gelu_tanh",
        "isfinited",
        "isinf",
        "isnan",
        "lgamma",
        "log",
        "log2",
        "nextafter",
        "pow",
        "rint",
        "rsqrt",
        "silu",
        "sinpi",
        "tan",
        "tanh",
        "trunc",
        "xpu_trunc_div",
        "y0",
    ),
)


def use_backend(module):
    """using backend module impl"""

    def decorator(func):
        func_name = func.__name__
        if hasattr(module, func_name):
            try:
                return getattr(module, func_name)
            except Exception:
                pass
        return func

    return decorator


def use_tl_extra(func):
    """backend function shim"""
    return use_backend(tl_extra_shim)(func)
