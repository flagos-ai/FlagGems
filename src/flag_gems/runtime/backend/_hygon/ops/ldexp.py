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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.copy import copy_
from flag_gems.ops.neg import neg
from flag_gems.ops.zeros_like import zeros_like
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

logger = logging.getLogger(__name__)

_WRAP_INTEGER_EXPONENT = tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)


@triton.jit
def _ldexp_fallback(x, exponent):
    return x * tl.exp2(exponent.to(x.dtype))


_ldexp = getattr(tl_extra_shim, "ldexp", _ldexp_fallback)


@triton.jit
def _ldexp32(x, exponent):
    bits = x.to(tl.int32, bitcast=True)
    mantissa = bits & 0x7FFFFF
    subnormal = ((bits & 0x7F800000) == 0) & (mantissa != 0)
    exponent = tl.maximum(tl.minimum(exponent, 4096), -4096)
    normal_result = _ldexp(x, exponent)
    subnormal_result = _ldexp(mantissa.to(tl.float32), exponent - 149)
    subnormal_result = tl.where(bits < 0, -subnormal_result, subnormal_result)
    return tl.where(subnormal, subnormal_result, normal_result)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def ldexp_func(x, y):
    if y.dtype.is_int() and x.dtype.is_floating():
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _ldexp(x.to(tl.float32), exponent)
    xf = x.to(tl.float32)
    result = xf * tl.exp2(y.to(tl.float32))
    if x.dtype == tl.float16 and y.dtype == tl.float16:
        # Preserve the operand sign across AMD's narrowing/underflow to fp16.
        half = result.to(tl.float16)
        bits = half.to(tl.uint16, bitcast=True)
        sign = ((xf.to(tl.uint32, bitcast=True) >> 16) & 0x8000).to(tl.uint16)
        return (bits | sign).to(tl.float16, bitcast=True).to(tl.float32)
    return result


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def ldexp_fp64_func(x, y):
    if y.dtype.is_int() and x.dtype.is_floating():
        exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        return _ldexp(x.to(tl.float64), exponent)
    return x.to(tl.float64) * tl.exp2(y.to(tl.float64))


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
    return _ldexp(x.to(tl.float64), exponent)


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
    scale = tl.exp2(yr)
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
    scale = tl.exp2(yr)
    angle = yi * 0.693147180559945309417232121458176568
    cos_angle = tl.cos(angle)
    sin_angle = tl.sin(angle)
    pr = scale * cos_angle
    pi = tl.where(yi == 0, 0, scale * sin_angle)
    return xr * pr - xi * pi, xr * pi + xi * pr


def _result_dtype(self, other):
    _, result_dtype = type_promotion(
        self,
        other,
        type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    return result_dtype


def _result_device(self, other):
    if self.device == other.device:
        return self.device
    if (
        _WRAP_INTEGER_EXPONENT
        and self.is_floating_point()
        and not other.is_floating_point()
        and not other.is_complex()
    ):
        raise RuntimeError("Expected all tensors to be on the same device")
    if self.ndim == 0 and self.device.type == "cpu":
        return other.device
    if other.ndim == 0 and other.device.type == "cpu":
        return self.device
    raise RuntimeError(
        f"Expected all tensors to be on the same device, but found {self.device} "
        f"and {other.device}"
    )


def _is_exact_alias(left, right):
    return (
        left.data_ptr() == right.data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
    )


def _validate_out(self, other, out, result_dtype):
    result_device = _result_device(self, other)
    if out.device != result_device:
        raise RuntimeError(
            f"Expected out tensor to be on {result_device}, but got {out.device}"
        )
    if not torch.can_cast(result_dtype, out.dtype):
        raise RuntimeError(
            f"result type {result_dtype} can't be cast to the desired output "
            f"type {out.dtype}"
        )
    for operand in (self, other):
        if (
            operand.device == out.device
            and operand.untyped_storage().nbytes() > 0
            and operand.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        ):
            if not _is_exact_alias(out, operand):
                raise RuntimeError(
                    "some elements of the input tensor and the written-to tensor "
                    "refer to a single memory location"
                )


def _complex_parts(tensor, real_dtype):
    if tensor.is_complex():
        physical = tensor.conj() if tensor.is_conj() else tensor
        parts = torch.view_as_real(physical)
        real = torch.empty(tensor.shape, dtype=real_dtype, device=tensor.device)
        imag = torch.empty_like(real)
        copy_(real, parts[..., 0])
        copy_(imag, neg(parts[..., 1]) if tensor.is_conj() else parts[..., 1])
        return real, imag
    real = torch.empty(tensor.shape, dtype=real_dtype, device=tensor.device)
    copy_(real, tensor)
    return real, zeros_like(real)


def _copy_result(out, result):
    if out.is_complex():
        physical = out.conj() if out.is_conj() else out
        parts = torch.view_as_real(physical)
        real, imag = _complex_parts(result, parts.dtype)
        copy_(parts[..., 0], real)
        copy_(parts[..., 1], neg(imag) if out.is_conj() else imag)
    else:
        copy_(out, result)
    return out


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


_LDEXP_KERNELS = {}


@libentry()
@triton.jit
def _ldexp_flat(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr, WRAP: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if N % BLOCK == 0:
        x = tl.load(X + offsets)
        y = tl.load(Y + offsets)
    else:
        x = tl.load(X + offsets, offsets < N, 0)
        y = tl.load(Y + offsets, offsets < N, 0)
    if y.dtype.is_int():
        exponent = y.to(tl.int32)
        if not WRAP:
            exponent = tl.maximum(tl.minimum(y, 2147483647), -2147483648).to(tl.int32)
        value = _ldexp32(x.to(tl.float32), exponent)
    else:
        value = x.to(tl.float32) * tl.exp2(y.to(tl.float32))
        if x.dtype == tl.float16:
            bits = value.to(tl.float16).to(tl.uint16, bitcast=True)
            sign = ((x.to(tl.float32).to(tl.uint32, bitcast=True) >> 16) & 32768).to(
                tl.uint16
            )
            value = (bits | sign).to(tl.float16, bitcast=True)
    if N % BLOCK == 0:
        tl.store(Out + offsets, value)
    else:
        tl.store(Out + offsets, value, offsets < N)


def _dense_ldexp(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        block = 2048 if x.element_size() <= 2 else 1024
        args = (x, y, out, n, block, _WRAP_INTEGER_EXPONENT)
        key = (
            x.get_device(),
            x.dtype,
            y.dtype,
            n,
            block,
            x.data_ptr() % 16 == 0,
            y.data_ptr() % 16 == 0,
        )
        grid = (triton.cdiv(n, block), 1, 1)
        kernel = _LDEXP_KERNELS.get(key)
        if kernel is None:
            compiled = _ldexp_flat[grid](*args, num_warps=4)
            kernel = compiled[0] if isinstance(compiled, tuple) else compiled
            if kernel is not None:
                if len(_LDEXP_KERNELS) >= 128:
                    _LDEXP_KERNELS.pop(next(iter(_LDEXP_KERNELS)))
                _LDEXP_KERNELS[key] = kernel
        else:
            kernel[grid](*args, stream=torch.cuda.current_stream().cuda_stream)
    return out


def ldexp(self, other):
    logger.debug("GEMS_HYGON LDEXP")
    if (
        isinstance(self, torch.Tensor)
        and isinstance(other, torch.Tensor)
        and self.device.type != "cpu"
        and self.device == other.device
        and self.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (other.dtype == self.dtype or other.dtype in (torch.int32, torch.int64))
        and self.shape == other.shape
        and self.is_contiguous()
        and other.is_contiguous()
    ):
        return _dense_ldexp(self, other)
    return _ldexp_impl(self, other)


def ldexp_out(self, other, *, out):
    logger.debug("GEMS_HYGON LDEXP_OUT")
    return _ldexp_impl(self, other, out)
