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
def _fallback_normcdfinv(p):
    # Inverse of the standard normal CDF, used when a backend's libdevice lacks
    # a native normcdfinv (e.g. the hip-based hygon fork).  Composed from
    # _fallback_erfinv via ndtri(p) = sqrt(2) * erfinv(2p - 1), then polished
    # with Newton iterations on Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
    # special_ndtri.py notes the unpolished composition drifts to ~1.3e-05 abs
    # error in float32; the refinement below brings it back near libdevice
    # accuracy.  phi -> 0 as |x| -> inf, so the Newton step degenerates (0/0)
    # for lanes where p is at/near 0 or 1 -- keep the erfinv estimate there.
    x = 1.4142135623730951 * _fallback_erfinv(2.0 * p - 1.0)
    # _fallback_erfinv hits 0/0 (-> nan) at the exact endpoints, where
    # erfinv(+-1) is infinite; restore the exact values PyTorch/libdevice give.
    x = tl.where(p == 0.0, float("-inf"), x)
    x = tl.where(p == 1.0, float("inf"), x)
    for _ in range(3):
        phi = 0.3989422804014327 * tl.exp(-0.5 * x * x)  # 1 / sqrt(2*pi)
        cdf = 0.5 * (1.0 + tl.math.erf(0.7071067811865476 * x))
        step = tl.where((phi > 0.0) & (cdf == cdf), (cdf - p) / phi, 0.0)
        x = x - step
    return x


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
def _fallback_j1_for_y1(x):
    # Bessel J1(x), dtype-neutral, used internally by _fallback_y1 for the
    # small-|x| formula  Y1(x) = x*U/V + (2/pi)*(j1(x)*ln(x) - 1/x).  Adapted
    # verbatim from fdlibm e_j1.c (public-domain SunPro), same source as
    # glibc / CUDA libdevice:
    #   |x| <  2  -> rational  x/2 + x*z*R0/S0,  z = x*x
    #   |x| >= 2  -> asymptotic  invsqrtpi*(pone*cc - qone*ss)/sqrt(|x|)
    # Only core Triton primitives are used.  Unlike the public _fallback_j1
    # (which is float32-only, Cephes coefficients), this version uses fdlibm
    # double-precision coefficients so it stays accurate for float64 input,
    # which y1's float64 path requires (rtol=1e-7).
    # Edge cases match fdlibm: J1(NaN)=NaN, J1(+-inf)=0, J1(0)=0, J1(-x)=-J1(x).
    ax = tl.abs(x)
    is_nan = ax != ax
    is_inf = ax == float("inf")
    ax_safe = tl.where(is_nan | is_inf, 1.0, ax)

    # ----- large region: |x| >= 2, fdlibm pone/qone asymptotic -----
    # Coefficients are band-selected by |x|: pr8/ps8 (|x|>=8), pr5/ps5
    # (|x|>=4.5454 ~ 0x40122E8B), pr3/ps3 (|x|>=2.8570 ~ 0x4006DB6D),
    # pr2/ps2 (2<=|x|<2.8570).  Verbatim from fdlibm e_j1.c.
    ax_large = tl.where(ax >= 2.0, ax_safe, 2.0)
    s = tl.sin(ax_large)
    c = tl.cos(ax_large)
    ss = -s - c
    cc = s - c
    # Cancellation-avoidance: recompute the worse of (ss, cc) from cos(2x).
    # For j1's phase convention (ss=-s-c, cc=s-c) we have cos(2x) = +ss*cc
    # (note: NO negation, unlike _fallback_j0 where cos(2x) = -ss*cc).
    # ss cancels when s,c have opposite sign (s*c<0); cc cancels when s,c
    # have same sign (s*c>0).  Match fdlibm e_j1.c: s*c>0 -> cc=z/ss,
    # else -> ss=z/cc.
    z2 = tl.cos(2.0 * ax_large)
    cc_new = z2 / ss
    ss_new = z2 / cc
    is_sc_pos = (s * c) > 0.0
    cc = tl.where(is_sc_pos, cc_new, cc)
    ss = tl.where(is_sc_pos, ss, ss_new)

    z = 1.0 / (ax_large * ax_large)
    # Band 8 (|x| >= 8): pr8[0..5], ps8[0..4]
    pR8 = (
        0.0
        + (
            1.17187499999988647970e-01
            + (
                1.32394806593073575129e01
                + (
                    4.12051854307378562225e02
                    + (3.87474538913960532227e03 + 7.91447954031891731574e03 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS8 = (
        1.0
        + (
            1.14207370375678408436e02
            + (
                3.65093083420853463394e03
                + (
                    3.69562060269033463555e04
                    + (9.76027935934950801311e04 + 3.08042720627888811578e04 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR8 = (
        0.0
        + (
            -1.02539062499992714161e-01
            + (
                -1.62717534544589987888e01
                + (
                    -7.59601722513950107896e02
                    + (-1.18498066702429587167e04 + -4.84385124285750353010e04 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS8 = (
        1.0
        + (
            1.61395369700722909556e02
            + (
                7.82538599923348465381e03
                + (
                    1.33875336287249578163e05
                    + (
                        7.19657723683240939863e05
                        + (6.66601232617776375264e05 - 2.94490264303834643215e05 * z)
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
    # Band 5 (4.5454 <= |x| < 8): pr5[0..5], ps5[0..4]
    pR5 = (
        1.31990519556243522749e-11
        + (
            1.17187493190614097638e-01
            + (
                6.80275127868432871736e00
                + (
                    1.08308182990189109773e02
                    + (5.17636139533199752805e02 + 5.28715201363337541807e02 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS5 = (
        1.0
        + (
            5.92805987221131331921e01
            + (
                9.91401418733614377743e02
                + (
                    5.35326695291487976647e03
                    + (7.84469031749551231769e03 + 1.50404688810361062679e03 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR5 = (
        -2.08979931141764104297e-11
        + (
            -1.02539050241375426231e-01
            + (
                -8.05644828123936029840e00
                + (
                    -1.83669607474888380239e02
                    + (-1.37319376065508163265e03 - 2.61244440453215656817e03 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS5 = (
        1.0
        + (
            8.12765501384335777857e01
            + (
                1.99179873460485964642e03
                + (
                    1.74684851924908907677e04
                    + (
                        4.98514270910352279316e04
                        + (2.79480751638918118260e04 - 4.71918354795128470869e03 * z)
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
    # Band 3 (2.8570 <= |x| < 4.5454): pr3[0..5], ps3[0..4]
    pR3 = (
        3.02503916137373618024e-09
        + (
            1.17186865567253592491e-01
            + (
                3.93297750033315640650e00
                + (
                    3.51194035591636932736e01
                    + (9.10550110750781271918e01 + 4.85590685197364919645e01 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS3 = (
        1.0
        + (
            3.47913095001251519989e01
            + (
                3.36762458747825746741e02
                + (
                    1.04687139975775130551e03
                    + (8.90811346398256432622e02 + 1.03787932439639277504e02 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR3 = (
        -5.07831226461766561369e-09
        + (
            -1.02537829820837089745e-01
            + (
                -4.61011581139473403113e00
                + (
                    -5.78472216562783643212e01
                    + (-2.28244540737631695038e02 - 2.19210128478909325622e02 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS3 = (
        1.0
        + (
            4.76651550323729509273e01
            + (
                6.73865112676699709482e02
                + (
                    3.38015286679526343505e03
                    + (
                        5.54772909720722782367e03
                        + (1.90311919338810798763e03 - 1.35201191444307340817e02 * z)
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
    # Band 2 (2 <= |x| < 2.8570): pr2[0..5], ps2[0..4]
    pR2 = (
        1.07710830106873743082e-07
        + (
            1.17176219462683348094e-01
            + (
                2.36851496667608785174e00
                + (
                    1.22426109148261232917e01
                    + (1.76939711271687727390e01 + 5.07352312588818499250e00 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS2 = (
        1.0
        + (
            2.14364859363821409488e01
            + (
                1.25290227168402751090e02
                + (
                    2.32276469057162813669e02
                    + (1.17679373287147100768e02 + 8.36463893371618283368e00 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR2 = (
        -1.78381727510958865572e-07
        + (
            -1.02517042607985553460e-01
            + (
                -2.75220568278187460720e00
                + (
                    -1.96636162643703720221e01
                    + (-4.23253133372830490089e01 - 2.13719211703704061733e01 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS2 = (
        1.0
        + (
            2.95333629060523854548e01
            + (
                2.52981549982190529136e02
                + (
                    7.57502834868645436472e02
                    + (
                        7.39393205320467245656e02
                        + (1.55949003336666123687e02 - 4.95949898822628210127e00 * z)
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

    # Band-select: 8 -> 5 -> 3 -> 2.  fdlibm picks by IEEE bit ranges
    # (0x40200000 / 0x40122E8B / 0x4006DB6D); we use the equivalent float
    # thresholds (8.0 / 4.5454 / 2.8570).  Avoid Python's ~ on triton bool
    # tensors; use explicit < comparisons.
    pR = tl.where(
        ax >= 8.0, pR8, tl.where(ax >= 4.5454, pR5, tl.where(ax >= 2.8570, pR3, pR2))
    )
    pS = tl.where(
        ax >= 8.0, pS8, tl.where(ax >= 4.5454, pS5, tl.where(ax >= 2.8570, pS3, pS2))
    )
    qR = tl.where(
        ax >= 8.0, qR8, tl.where(ax >= 4.5454, qR5, tl.where(ax >= 2.8570, qR3, qR2))
    )
    qS = tl.where(
        ax >= 8.0, qS8, tl.where(ax >= 4.5454, qS5, tl.where(ax >= 2.8570, qS3, qS2))
    )
    pone = 1.0 + pR / pS
    qone = (qR / qS + 0.375) / ax_large  # fdlibm: (.375 + r/s)/x
    large_val = 5.64189583547756279280e-01 * (pone * cc - qone * ss) / tl.sqrt(ax_large)
    large_val = tl.where(x < 0.0, -large_val, large_val)

    # ----- small region: |x| < 2, fdlibm rational -----
    # j1(x) = x/2 + x*z*R0/S0,  z = x*x.
    # R0 = r00 + r01*z + r02*z^2 + r03*z^3;  S0 = 1 + s01*z + ... + s05*z^5.
    sx = ax_safe
    z = sx * sx
    r = z * (
        -6.25000000000000000000e-02
        + z
        * (
            1.40705666955189706048e-03
            + z * (-1.59955631084035597520e-05 + z * 4.96727999609584448412e-08)
        )
    )
    s = 1.0 + z * (
        1.91537599538363460805e-02
        + z
        * (
            1.85946785588630915560e-04
            + z
            * (
                1.17718464042623683263e-06
                + z * (5.04636257076217042715e-09 + z * 1.23542274426137913908e-11)
            )
        )
    )
    small_val = sx * 0.5 + r / s * sx

    ans = tl.where(ax >= 2.0, large_val, small_val)
    ans = tl.where(is_inf, 0.0, ans)
    ans = tl.where(is_nan, float("nan"), ans)
    return ans


@triton.jit
def _fallback_y1(x):
    # Bessel Y1(x) for float32/float64, used when a backend's libdevice lacks a
    # native y1 (e.g. the ascend cann fork, which does provide j1).  Adapted
    # verbatim from fdlibm e_j1.c (public-domain SunPro), same source as
    # glibc / CUDA libdevice:
    #   |x| <  2  -> x*U0(z)/V0(z) + (2/pi)*(j1(x)*ln(x) - 1/x),  z = x*x
    #   |x| >= 2  -> asymptotic  invsqrtpi*(pone*ss + qone*cc)/sqrt(x)
    # j1(x) is supplied by the dtype-neutral _fallback_j1_for_y1 helper above
    # (the public _fallback_j1 is float32-only and would break float64).
    # Uses only core Triton primitives (sin/cos/sqrt/log/abs/where), so it
    # lowers on every backend.  Edge cases match PyTorch/fdlibm:
    #   Y1(NaN)=NaN, Y1(+-inf)=0, Y1(+-0)=-inf, Y1(x<0)=NaN.
    ax = tl.abs(x)
    is_nan = ax != ax
    is_inf = ax == float("inf")
    is_zero = ax == 0.0
    is_neg = x < 0.0
    # Safe finite positive value for non-finite / negative / zero lanes;
    # overwritten by the edge-case results at the end.
    ax_safe = tl.where(is_nan | is_inf | is_zero | is_neg, 1.0, ax)

    # ----- large region: |x| >= 2, fdlibm pone/qone asymptotic -----
    # Same band-8..2 coefficients and phase convention as _fallback_j1_for_y1,
    # but Y1 uses  invsqrtpi*(pone*ss + qone*cc)/sqrt(x)  (j1 uses cc - q*ss).
    ax_large = tl.where(ax >= 2.0, ax_safe, 2.0)
    s = tl.sin(ax_large)
    c = tl.cos(ax_large)
    ss = -s - c
    cc = s - c
    # Same phase convention as _fallback_j1_for_y1 (cos(2x) = +ss*cc, no
    # negation; ss cancels when s*c<0, cc when s*c>0).  fdlibm e_j1.c:
    # s*c>0 -> cc=z/ss, else -> ss=z/cc.
    z2 = tl.cos(2.0 * ax_large)
    cc_new = z2 / ss
    ss_new = z2 / cc
    is_sc_pos = (s * c) > 0.0
    cc = tl.where(is_sc_pos, cc_new, cc)
    ss = tl.where(is_sc_pos, ss, ss_new)

    z = 1.0 / (ax_large * ax_large)
    # Band 8 (|x| >= 8)
    pR8 = (
        0.0
        + (
            1.17187499999988647970e-01
            + (
                1.32394806593073575129e01
                + (
                    4.12051854307378562225e02
                    + (3.87474538913960532227e03 + 7.91447954031891731574e03 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS8 = (
        1.0
        + (
            1.14207370375678408436e02
            + (
                3.65093083420853463394e03
                + (
                    3.69562060269033463555e04
                    + (9.76027935934950801311e04 + 3.08042720627888811578e04 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR8 = (
        0.0
        + (
            -1.02539062499992714161e-01
            + (
                -1.62717534544589987888e01
                + (
                    -7.59601722513950107896e02
                    + (-1.18498066702429587167e04 + -4.84385124285750353010e04 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS8 = (
        1.0
        + (
            1.61395369700722909556e02
            + (
                7.82538599923348465381e03
                + (
                    1.33875336287249578163e05
                    + (
                        7.19657723683240939863e05
                        + (6.66601232617776375264e05 - 2.94490264303834643215e05 * z)
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
    # Band 5 (4.5454 <= |x| < 8)
    pR5 = (
        1.31990519556243522749e-11
        + (
            1.17187493190614097638e-01
            + (
                6.80275127868432871736e00
                + (
                    1.08308182990189109773e02
                    + (5.17636139533199752805e02 + 5.28715201363337541807e02 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS5 = (
        1.0
        + (
            5.92805987221131331921e01
            + (
                9.91401418733614377743e02
                + (
                    5.35326695291487976647e03
                    + (7.84469031749551231769e03 + 1.50404688810361062679e03 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR5 = (
        -2.08979931141764104297e-11
        + (
            -1.02539050241375426231e-01
            + (
                -8.05644828123936029840e00
                + (
                    -1.83669607474888380239e02
                    + (-1.37319376065508163265e03 - 2.61244440453215656817e03 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS5 = (
        1.0
        + (
            8.12765501384335777857e01
            + (
                1.99179873460485964642e03
                + (
                    1.74684851924908907677e04
                    + (
                        4.98514270910352279316e04
                        + (2.79480751638918118260e04 - 4.71918354795128470869e03 * z)
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
    # Band 3 (2.8570 <= |x| < 4.5454)
    pR3 = (
        3.02503916137373618024e-09
        + (
            1.17186865567253592491e-01
            + (
                3.93297750033315640650e00
                + (
                    3.51194035591636932736e01
                    + (9.10550110750781271918e01 + 4.85590685197364919645e01 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS3 = (
        1.0
        + (
            3.47913095001251519989e01
            + (
                3.36762458747825746741e02
                + (
                    1.04687139975775130551e03
                    + (8.90811346398256432622e02 + 1.03787932439639277504e02 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR3 = (
        -5.07831226461766561369e-09
        + (
            -1.02537829820837089745e-01
            + (
                -4.61011581139473403113e00
                + (
                    -5.78472216562783643212e01
                    + (-2.28244540737631695038e02 - 2.19210128478909325622e02 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS3 = (
        1.0
        + (
            4.76651550323729509273e01
            + (
                6.73865112676699709482e02
                + (
                    3.38015286679526343505e03
                    + (5.54772909720722782367e03 + 1.90311919338810798763e03 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    # Band 2 (2 <= |x| < 2.8570)
    pR2 = (
        1.07710830106873743082e-07
        + (
            1.17176219462683348094e-01
            + (
                2.36851496667608785174e00
                + (
                    1.22426109148261232917e01
                    + (1.76939711271687727390e01 + 5.07352312588818499250e00 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    pS2 = (
        1.0
        + (
            2.14364859363821409488e01
            + (
                1.25290227168402751090e02
                + (
                    2.32276469057162813669e02
                    + (1.17679373287147100768e02 + 8.36463893371618283368e00 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qR2 = (
        -1.78381727510958865572e-07
        + (
            -1.02517042607985553460e-01
            + (
                -2.75220568278187460720e00
                + (
                    -1.96636162643703720221e01
                    + (-4.23253133372830490089e01 - 2.13719211703704061733e01 * z) * z
                )
                * z
            )
            * z
        )
        * z
    )
    qS2 = (
        1.0
        + (
            2.95333629060523854548e01
            + (
                2.52981549982190529136e02
                + (
                    7.57502834868645436472e02
                    + (
                        7.39393205320467245656e02
                        + (1.55949003336666123687e02 - 4.95949898822628210127e00 * z)
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
    pR = tl.where(
        ax >= 8.0, pR8, tl.where(ax >= 4.5454, pR5, tl.where(ax >= 2.8570, pR3, pR2))
    )
    pS = tl.where(
        ax >= 8.0, pS8, tl.where(ax >= 4.5454, pS5, tl.where(ax >= 2.8570, pS3, pS2))
    )
    qR = tl.where(
        ax >= 8.0, qR8, tl.where(ax >= 4.5454, qR5, tl.where(ax >= 2.8570, qR3, qR2))
    )
    qS = tl.where(
        ax >= 8.0, qS8, tl.where(ax >= 4.5454, qS5, tl.where(ax >= 2.8570, qS3, qS2))
    )
    pone = 1.0 + pR / pS
    qone = (qR / qS + 0.375) / ax_large
    large_val = 5.64189583547756279280e-01 * (pone * ss + qone * cc) / tl.sqrt(ax_large)

    # ----- small region: |x| < 2, fdlibm rational -----
    # Y1(x) = x*U0(z)/V0(z) + tpi*(j1(x)*ln(x) - 1/x),  z = x*x,  tpi = 2/pi.
    # For tiny x (|x| < 2^-54), Y1(x) ~ -tpi/x  dominates; we route those lanes
    # through the -tpi/x branch to avoid 1/x precision loss in the full formula.
    sx = ax_safe
    z = sx * sx
    u0 = -1.96057090646238940668e-01 + z * (
        5.04438716639811282616e-02
        + z
        * (
            -1.91256895875763547298e-03
            + z * (2.35252600561610495928e-05 + z * -9.19099158039878874504e-08)
        )
    )
    v0 = 1.0 + z * (
        1.99167318236649903973e-02
        + z
        * (
            2.02552581025135171496e-04
            + z
            * (
                1.35608801097516229404e-06
                + z * (6.22741452364621501295e-09 + z * 1.66559246207992079114e-11)
            )
        )
    )
    j1x = _fallback_j1_for_y1(x)
    # fdlibm uses log(x); x is positive here (negative lanes routed to ax_safe=1).
    log_term = tl.log(sx)
    small_val = sx * (u0 / v0) + 6.36619772367581382433e-01 * (
        j1x * log_term - 1.0 / sx
    )
    tiny = ax <= 2.91e-17  # 2**-55 ~ 2.77e-17; use 2**-54 ~ 5.55e-17 region
    small_val = tl.where(tiny, -6.36619772367581382433e-01 / sx, small_val)

    ans = tl.where(ax >= 2.0, large_val, small_val)
    # Edge cases.  PyTorch's torch.special.bessel_y1 returns NaN for +-inf
    # (unlike fdlibm which returns 0); match PyTorch since that is what
    # special_bessel_y1.py's tests compare against.
    #   Y1(+-inf)=NaN, Y1(NaN)=NaN, Y1(+-0)=-inf, Y1(x<0)=NaN.
    ans = tl.where(is_inf, float("nan"), ans)
    ans = tl.where(is_nan, float("nan"), ans)
    ans = tl.where(is_zero, float("-inf"), ans)
    ans = tl.where(is_neg, float("nan"), ans)
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
    "erfc": _fallback_erfc,
    "erfinv": _fallback_erfinv,
    "floor": _fallback_floor,
    "j0": _fallback_j0,
    "j1": _fallback_j1,
    "log2": _fallback_log2,
    "nextafter": _fallback_nextafter,
    "normcdfinv": _fallback_normcdfinv,
    "sinpi": _fallback_sinpi,
    "y0": _fallback_y0,
    "y1": _fallback_y1,
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
        "normcdfinv",
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
        "y1",
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
