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


_FALLBACK_SYMBOLS = {
    "pow": _fallback_pow,
    "tanh": _fallback_tanh,
    "erfinv": _fallback_erfinv,
    "floor": _fallback_floor,
    "j1": _fallback_j1,
    "nextafter": _fallback_nextafter,
    "sinpi": _fallback_sinpi,
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
        "j1",
        "atan2",
        "div_rn",
        "div_rz",
        "erf",
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
