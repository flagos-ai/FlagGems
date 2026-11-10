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

# NOTE(KUNLUNXIN, 2026-11-09): the generic ldexp kernels rely on `tl.exp2` and
# the xpu libdevice `ldexp`, both of which are wrong on this stack:
#   * `tl.exp2` lowers to the xpu `exp` implementation (e^x) -- exp2(1.0) is
#     2.718, not 2.0 -- so every float-exponent path computes x * e^y.
#   * the fp64 libdevice `ldexp` returns 0.0 for min-normal inputs scaled by a
#     large exponent (2^-1022 * 2^1024 -> 0.0 instead of 4.0).
# Both are replaced here with exact power-of-two bit scaling (2^+-126 / 2^+-512
# chunks whose bit patterns are exact), so x * 2^y never materializes an
# overflowing 2^y (which is what makes x * exp2(y) fail for x = 2^-126,
# y = 128: exp2(128) = +inf, x * inf = +inf instead of 4.0).

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.ldexp import (
    _complex_parts,
    _copy_result,
    _result_device,
    _result_dtype,
    _validate_out,
)
from flag_gems.ops.copy import copy_
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_WRAP_INTEGER_EXPONENT = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)


@triton.jit
def _ldexp_fallback(x, n):
    return x * tl.exp2(n.to(x.dtype))


# The xpu libdevice ldexp (fp32) is correct on this stack; fp64 is not (see
# _ldexp64 below).  getattr fallback keeps the vendor module importable on
# other backends / triton versions (mirrors the generic ldexp.py).
_ldexp = getattr(tl_extra_shim, "ldexp", _ldexp_fallback)


@triton.jit
def _ldexp32(x, n):
    """Exact x * 2^n (fp32; n int32).

    Mirrors the generic `_ldexp32`: the libdevice ldexp handles normal inputs
    exactly, while subnormal inputs (whose mantissa would be flushed by the
    XPU multiply) are scaled through the integer mantissa instead.
    """
    bits = x.to(tl.int32, bitcast=True)
    mantissa = bits & 0x7FFFFF
    subnormal = ((bits & 0x7F800000) == 0) & (mantissa != 0)
    n = tl.maximum(tl.minimum(n, 2**30), -(2**30))
    normal_result = _ldexp(x, n)
    subnormal_result = _ldexp(mantissa.to(tl.float32), n - 149)
    subnormal_result = tl.where(bits < 0, -subnormal_result, subnormal_result)
    return tl.where(subnormal, subnormal_result, normal_result)


@triton.jit
def _ldexp64(x, n):
    """Exact x * 2^n (fp64; n int32).

    HARDWARE NOTE (2026-11-10): the XPU fp64 is a double-fp32 emulation which
    is only exact inside the f32 range [2^-126, 2^128); 2^+-512 (used by the
    fp32 helper) is unrepresentable and folds to inf/0 there.  Scaling is
    therefore done with 17 chunks of 2^+-126 (all exact f32 powers of two),
    which covers every n whose result is representable by the emulation
    (|n| > 2142 -> 0/+-inf for any x).  Deep-fp64 values (e.g. 2^-1022) and
    their products are simply not representable on this stack.
    """
    n = tl.maximum(tl.minimum(n, 2142), -2142)
    y = x
    for _ in tl.static_range(17):
        c = tl.maximum(tl.minimum(n, 126), -126)
        y = y * tl.where(c >= 0, 2.0**126, 2.0**-126)
        n = n - c
    return y


@triton.jit
def _ldexp_f32(x, y):
    """x * 2^y for fp32 x and floating y: 2^(n+f) = ldexp(2^f, n)."""
    n = tl.floor(y).to(tl.int32)
    f = y - n.to(tl.float32)
    scaled = x * tl.exp(f * 0.6931471805599453)
    # y = -inf: x * 0.0 on the XPU multiply path loses the sign (-0.0 -> +0.0),
    # and the libdevice ldexp also drops it, so keep the input zero bit-for-bit.
    res = _ldexp32(tl.where(y == float("-inf"), x, scaled), n)
    return tl.where((y == float("-inf")) & (x == 0), x, res)


@triton.jit
def _ldexp_f64(x, y):
    """x * 2^y for fp64 x and floating y; 2^f is computed in fp32 (exact when
    f == 0, which covers every integer exponent) and then scaled exactly."""
    n = tl.floor(y).to(tl.int32)
    f = (y - n.to(tl.float32)).to(tl.float32)
    scaled = (x * tl.exp(f * 0.6931471805599453)).to(tl.float64)
    res = _ldexp64(tl.where(y == float("-inf"), x, scaled), n)
    return tl.where((y == float("-inf")) & (x == 0), x, res)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def ldexp_func(x, y):
    if y.dtype.is_int() and x.dtype.is_floating():
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _ldexp32(x.to(tl.float32), exponent)
    return _ldexp_f32(x.to(tl.float32), y.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def ldexp_fp64_func(x, y):
    if y.dtype.is_int() and x.dtype.is_floating():
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _ldexp64(x.to(tl.float64), exponent)
    return _ldexp_f64(x.to(tl.float64), y.to(tl.float64))


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def ldexp_integral_func(x, y, wrap_exponent: tl.constexpr):
    exponent = y.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
    return _ldexp32(x.to(tl.float32), exponent)


@pointwise_dynamic(
    is_tensor=[True, True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def ldexp_integral_fp64_func(x, y, wrap_exponent: tl.constexpr):
    exponent = y.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
    return _ldexp64(x.to(tl.float64), exponent)


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def ldexp_complex_func(xr, xi, yr, yi):
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    yr = yr.to(tl.float32)
    yi = yi.to(tl.float32)
    scale = _ldexp_f32(1.0, yr)
    angle = yi * 0.6931471805599453
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, 3, "INT_TO_FLOAT"),
        (0, 1, 2, 3, "INT_TO_FLOAT"),
    ],
)
@triton.jit
def ldexp_complex_fp64_func(xr, xi, yr, yi):
    xr = xr.to(tl.float64)
    xi = xi.to(tl.float64)
    yr = yr.to(tl.float64)
    yi = yi.to(tl.float64)
    scale = _ldexp_f64(1.0, yr)
    angle = yi * 0.693147180559945309417232121458176568
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


def _complex_ldexp(self, other, result_dtype):
    real_dtype = torch.float64 if result_dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    yr, yi = _complex_parts(other, real_dtype)
    func = (
        ldexp_complex_fp64_func
        if result_dtype == torch.complex128
        else ldexp_complex_func
    )
    real, imag = func(xr, xi, yr, yi)
    result = torch.empty(real.shape, dtype=result_dtype, device=real.device)
    parts = torch.view_as_real(result)
    copy_(parts[..., 0], real)
    copy_(parts[..., 1], imag)
    return result


def _ldexp_impl(self, other, out=None):
    result_dtype = _result_dtype(self, other)
    if out is not None:
        _validate_out(self, other, out, result_dtype)
    device = _result_device(self, other)
    # CPU zero-dimensional operands are host scalar inputs, not device tensors
    # that may be passed to a GPU kernel unchanged.
    if self.device != device:
        self = torch.tensor(self.item(), dtype=self.dtype, device=device)
    if other.device != device:
        other = torch.tensor(other.item(), dtype=other.dtype, device=device)
    if self.is_complex() or other.is_complex():
        result = _complex_ldexp(self, other, result_dtype)
        if out is None:
            return result
        out.resize_(result.shape)
        return _copy_result(out, result)

    func = ldexp_fp64_func if result_dtype == torch.float64 else ldexp_func
    integral_exponent = (
        self.is_floating_point()
        and not other.is_floating_point()
        and not other.is_complex()
    )
    if integral_exponent:
        func = (
            ldexp_integral_fp64_func
            if result_dtype == torch.float64
            else ldexp_integral_func
        )
    result = (
        func(self, other, _WRAP_INTEGER_EXPONENT)
        if integral_exponent
        else func(self, other)
    )
    if out is None:
        return result
    out.resize_(result.shape)
    return _copy_result(out, result)


def ldexp(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP")
    return _ldexp_impl(self, other)


def ldexp_out(self, other, *, out):
    logger.debug("GEMS_KUNLUNXIN LDEXP_OUT")
    return _ldexp_impl(self, other, out)