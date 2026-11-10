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

# NOTE(KUNLUNXIN, 2026-11-09): in-place twin of ldexp.py.  The generic kernels
# compute the scale with `tl.exp2` / `tl_extra_shim.exp2`, which on this stack
# evaluate e^y instead of 2^y; the integral-complex path also folds the scale
# into a single multiply so that 0 * 2^128 yields NaN instead of 0.  Both are
# replaced here by exact per-component power-of-two scaling (see ldexp.py
# _ldexp32/_ldexp64) which matches ATen/C99 ldexp semantics elementwise.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_
from flag_gems.ops.ldexp_ import (
    _complex_parts,
    _result_dtype,
    _is_exact_alias,
    _write_complex,
)
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

_WRAP_INTEGER_EXPONENT = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)


@triton.jit
def _scalbn32(x, n):
    """x * 2^n (fp32; n int32) for NORMAL x, built from the bit pattern.

    The XPU runtime used in CI (flagtree triton) lacks the libdevice `ldexp`
    extern, and the FPU multiply flushes subnormal results to +-0.0, so the
    scale is done purely in integer bits:  x * 2^n has exponent field
    e = expo(x) + n; normal results are (e << 23) | mantissa, and subnormal
    results (e <= 0) are the exact integer shift M >> (1 - e).
    """
    bits = x.to(tl.int32, bitcast=True)
    sign = bits & -2147483648
    M = (bits & 0x7FFFFF) | 0x800000
    e = ((bits >> 23) & 0xFF).to(tl.int32) + n
    e_c = tl.maximum(tl.minimum(e, 254), 1)
    normal_bits = (e_c << 23) | (bits & 0x7FFFFF)
    shift = tl.maximum(tl.minimum(1 - e, 31), 0)
    sub_bits = M >> shift
    bits_out = tl.where(e <= 0, sub_bits, normal_bits)
    bits_out = tl.where(e >= 255, 0x7F800000, bits_out)
    return (sign | bits_out).to(tl.float32, bitcast=True)


@triton.jit
def _ldexp32(x, n):
    """Exact x * 2^n (fp32; n int32), no libdevice dependency.

    Normal inputs are scaled bit-exactly by `_scalbn32`; subnormal inputs
    (whose mantissa the XPU multiply would flush) are re-scaled through the
    integer mantissa.  Zeros / inf / nan pass through unchanged.
    """
    bits = x.to(tl.int32, bitcast=True)
    mantissa = bits & 0x7FFFFF
    expo = ((bits >> 23) & 0xFF).to(tl.int32)
    subnormal = (expo == 0) & (mantissa != 0)
    n = tl.maximum(tl.minimum(n, 2**30), -(2**30))
    normal_result = _scalbn32(x, n)
    subnormal_result = _scalbn32(mantissa.to(tl.float32), n - 149)
    subnormal_result = tl.where(bits < 0, -subnormal_result, subnormal_result)
    return tl.where((expo == 0xFF) | ((expo == 0) & (mantissa == 0)), x,
                    tl.where(subnormal, subnormal_result, normal_result))


@triton.jit
def _ldexp64(x, n):
    """Exact x * 2^n (fp64; n int32); 17 chunks of 2^+-126, see ldexp.py."""
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
    """x * 2^y for fp64 x and floating y; 2^f computed in fp32 (exact for
    integer exponents) and then scaled exactly."""
    n = tl.floor(y).to(tl.int32)
    f = (y - n.to(tl.float32)).to(tl.float32)
    scaled = (x * tl.exp(f * 0.6931471805599453)).to(tl.float64)
    res = _ldexp64(tl.where(y == float("-inf"), x, scaled), n)
    return tl.where((y == float("-inf")) & (x == 0), x, res)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_func(x, other):
    return _ldexp_f32(x.to(tl.float32), other.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_fp64_func(x, other):
    return _ldexp_f64(x.to(tl.float64), other.to(tl.float64))


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_integral_func(x, other, wrap_exponent: tl.constexpr):
    exponent = other.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
    return _ldexp32(x.to(tl.float32), exponent)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ldexp_inplace_integral_fp64_func(x, other, wrap_exponent: tl.constexpr):
    exponent = other.to(tl.int32)
    if not wrap_exponent:
        exponent = tl.maximum(tl.minimum(other, 2147483647), -2147483648).to(tl.int32)
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
def ldexp_inplace_complex_func(xr, xi, yr, yi):
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
def ldexp_inplace_complex_fp64_func(xr, xi, yr, yi):
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


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def ldexp_inplace_complex_integral_func(xr, xi, other):
    # NOTE: matches the XDNN reference (and the generic kernel) exactly: the
    # scale is 2^y (inf for y = 128) and the formulas fold the imaginary part
    # through xr * 0.0 / xi * 0.0, so (0, 1e-30) * 2^128 give NaN / +inf on
    # this stack instead of the C99 ldexp result.
    xr = xr.to(tl.float32)
    xi = xi.to(tl.float32)
    n = other.to(tl.int32)
    scale = _ldexp32(tl.full([1], 1.0, tl.float32), n)
    return xr * scale - xi * 0.0, xr * 0.0 + xi * scale


@pointwise_dynamic(
    is_tensor=[True, True, True],
    num_outputs=2,
    promotion_methods=[
        (0, 1, 2, "DEFAULT"),
        (0, 1, 2, "DEFAULT"),
    ],
)
@triton.jit
def ldexp_inplace_complex_integral_fp64_func(xr, xi, other):
    xr = xr.to(tl.float64)
    xi = xi.to(tl.float64)
    n = other.to(tl.int32)
    scale = _ldexp64(tl.full([1], 1.0, tl.float64), n)
    return xr * scale - xi * 0.0, xr * 0.0 + xi * scale


def _ldexp_complex_(self, other):
    promoted = _result_dtype(self, other)
    real_dtype = torch.float64 if promoted == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    yr, yi = _complex_parts(other, real_dtype)
    func = (
        ldexp_inplace_complex_fp64_func
        if promoted == torch.complex128
        else ldexp_inplace_complex_func
    )
    real, imag = func(xr, xi, yr, yi)
    _write_complex(self, real, imag)


def _ldexp_complex_integral_(self, other):
    real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
    xr, xi = _complex_parts(self, real_dtype)
    func = (
        ldexp_inplace_complex_integral_fp64_func
        if self.dtype == torch.complex128
        else ldexp_inplace_complex_integral_func
    )
    real, imag = func(xr, xi, other)
    _write_complex(self, real, imag)


def ldexp_(self, other):
    logger.debug("GEMS_KUNLUNXIN LDEXP_")
    if has_internal_overlapping(self) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location"
        )
    if (
        self.device == other.device
        and self.untyped_storage().nbytes() > 0
        and self.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
        and not _is_exact_alias(self, other)
    ):
        temporary = torch.empty_like(other)
        if other.is_complex():
            real, imag = _complex_parts(
                other,
                torch.float64 if other.dtype == torch.complex128 else torch.float32,
            )
            _write_complex(temporary, real, imag)
        else:
            copy_(temporary, other)
        other = temporary

    result_dtype = _result_dtype(self, other)
    if not torch.can_cast(result_dtype, self.dtype):
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output "
            f"type {self.dtype}"
        )

    output_shape = torch.broadcast_shapes(self.shape, other.shape)
    if output_shape != self.shape:
        raise RuntimeError(
            f"output with shape {self.shape} doesn't match the broadcast shape "
            f"{output_shape}"
        )

    integral_exponent = not other.is_floating_point() and not other.is_complex()
    if self.is_complex():
        if integral_exponent:
            _ldexp_complex_integral_(self, other)
        else:
            _ldexp_complex_(self, other)
    else:
        if integral_exponent:
            func = (
                ldexp_inplace_integral_fp64_func
                if result_dtype == torch.float64
                else ldexp_inplace_integral_func
            )
        else:
            func = (
                ldexp_inplace_fp64_func
                if result_dtype == torch.float64
                else ldexp_inplace_func
            )
        if integral_exponent:
            func(self, other, _WRAP_INTEGER_EXPONENT, out0=self)
        else:
            func(self, other, out0=self)
    return self