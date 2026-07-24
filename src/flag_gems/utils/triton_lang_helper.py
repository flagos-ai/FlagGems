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


_FALLBACK_SYMBOLS = {
    "pow": _fallback_pow,
    "tanh": _fallback_tanh,
    "erfinv": _fallback_erfinv,
    "floor": _fallback_floor,
}


def _patch_missing_symbols(module, names):
    for name in names:
        if hasattr(module, name):
            continue
        for candidate in _tl_extra_candidates():
            if hasattr(candidate, name):
                setattr(module, name, getattr(candidate, name))
                break
        else:
            fallback = _FALLBACK_SYMBOLS.get(name)
            if fallback is not None:
                setattr(module, name, fallback)
    return module


tl_extra_shim = _patch_missing_symbols(
    tl_extra_shim,
    (
        "acos",
        "atan",
        "atan2",
        "div_rn",
        "div_rz",
        "erf",
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
        "pow",
        "rint",
        "rsqrt",
        "silu",
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
