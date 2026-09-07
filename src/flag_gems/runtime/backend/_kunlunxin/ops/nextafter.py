import logging

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

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


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _nextafter_fp16_kernel(input, other):
    x = input.to(tl.float32)
    y = other.to(tl.float32)
    is_nan = (x != x) | (y != y)
    is_equal = x == y
    is_zero = x == 0.0
    abs_x = tl.abs(x)
    is_infinite = abs_x > 65504.0
    toward_up = y > x
    moving_toward_zero = ((x > 0.0) & ~toward_up) | ((x < 0.0) & toward_up)

    x_bits = x.to(tl.int32, bitcast=True)
    abs_bits = x_bits & 0x7FFFFFFF
    exponent = ((abs_bits >> 23) & 0xFF) - 127
    spacing_bits = (exponent - 10 + 127) << 23
    spacing = spacing_bits.to(tl.float32, bitcast=True)
    is_power_of_two = (abs_bits & 0x7FFFFF) == 0
    spacing = tl.where(abs_x < 6.103515625e-05, 5.960464477539063e-08, spacing)
    spacing = tl.where(
        moving_toward_zero & is_power_of_two & (abs_x > 6.103515625e-05),
        spacing * 0.5,
        spacing,
    )
    stepped = x + tl.where(toward_up, spacing, -spacing)
    zero_result = tl.where(y > 0.0, 5.960464477539063e-08, -5.960464477539063e-08)
    infinite_result = tl.where(x > 0.0, 65504.0, -65504.0)
    result = tl.where(is_zero, zero_result, stepped)
    result = tl.where(is_infinite & ~is_equal, infinite_result, result)
    result = tl.where(is_equal, y, result)
    return tl.where(is_nan, x + y, result).to(input.dtype)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _nextafter_bf16_kernel(input, other):
    x = input.to(tl.float32)
    y = other.to(tl.float32)
    is_nan = (x != x) | (y != y)
    is_equal = x == y
    is_zero = x == 0.0
    abs_x = tl.abs(x)
    is_infinite = abs_x > 3.3895313892515355e38
    toward_up = y > x
    moving_toward_zero = ((x > 0.0) & ~toward_up) | ((x < 0.0) & toward_up)

    x_bits = x.to(tl.int32, bitcast=True)
    abs_bits = x_bits & 0x7FFFFFFF
    exponent = ((abs_bits >> 23) & 0xFF) - 127
    spacing_exponent = exponent - 7
    normal_spacing_bits = (spacing_exponent + 127) << 23
    subnormal_spacing_bits = 1 << tl.maximum(spacing_exponent + 149, 0)
    spacing_bits = tl.where(
        spacing_exponent >= -126, normal_spacing_bits, subnormal_spacing_bits
    )
    spacing = spacing_bits.to(tl.float32, bitcast=True)
    is_power_of_two = (abs_bits & 0x7FFFFF) == 0
    spacing = tl.where(abs_x < 1.1754943508222875e-38, 9.183549615799121e-41, spacing)
    spacing = tl.where(
        moving_toward_zero & is_power_of_two & (abs_x > 1.1754943508222875e-38),
        spacing * 0.5,
        spacing,
    )
    stepped = x + tl.where(toward_up, spacing, -spacing)
    zero_result = tl.where(y > 0.0, 9.183549615799121e-41, -9.183549615799121e-41)
    infinite_result = tl.where(x > 0.0, 3.3895313892515355e38, -3.3895313892515355e38)
    result = tl.where(is_zero, zero_result, stepped)
    result = tl.where(is_infinite & ~is_equal, infinite_result, result)
    result = tl.where(is_equal, y, result)
    return tl.where(is_nan, x + y, result).to(input.dtype)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _nextafter_fp32_kernel(input, other):
    x = input.to(tl.float32)
    y = other.to(tl.float32)
    x_bits = x.to(tl.int32, bitcast=True)
    y_bits = y.to(tl.int32, bitcast=True)
    x_abs = x_bits & 0x7FFFFFFF
    y_abs = y_bits & 0x7FFFFFFF

    # Everything is decided on the raw IEEE-754 bit pattern.  The XPU FPU
    # flushes subnormal operands to zero (FTZ) in floating-point arithmetic
    # AND comparisons, so `x + spacing`, `x == 0.0` and `x < y` are all wrong
    # for subnormal inputs (|x| < 2^-126) or subnormal spacings (|x| < 2^-103).
    # Sign-magnitude integer stepping (bits +/- 1) is exact for every finite
    # non-zero value: for positive x a larger bit pattern is a larger float,
    # for negative x it is reversed.
    is_nan = (x_abs > 0x7F800000) | (y_abs > 0x7F800000)
    is_zero = x_abs == 0
    is_infinite = x_abs == 0x7F800000
    is_equal = (x_bits == y_bits) | ((x_abs | y_abs) == 0)
    x_negative = x_bits < 0  # int32 sign-bit test (no FPU involved)
    y_negative = y_bits < 0
    toward_up = (
        (~x_negative & ~y_negative & (y_bits > x_bits))
        | (x_negative & y_negative & (y_bits < x_bits))
        | (x_negative & ~y_negative)
    )
    inc = tl.where(x_negative, tl.where(toward_up, -1, 1), tl.where(toward_up, 1, -1))
    stepped = (x_bits + inc).to(tl.float32, bitcast=True)
    minimum = (x_bits * 0 + 1).to(tl.float32, bitcast=True)
    zero_result = tl.where(y_negative, -minimum, minimum)
    infinite_result = tl.where(x_negative, -3.4028234663852886e38, 3.4028234663852886e38)
    result = tl.where(is_zero, zero_result, stepped)
    result = tl.where(is_infinite & ~is_equal, infinite_result, result)
    result = tl.where(is_equal, y, result)
    return tl.where(is_nan, x + y, result)


def _kernel_for(dtype):
    if dtype == torch.float16:
        return _nextafter_fp16_kernel
    if dtype == torch.bfloat16:
        return _nextafter_bf16_kernel
    if dtype == torch.float32:
        return _nextafter_fp32_kernel
    raise NotImplementedError(
        f"Kunlunxin nextafter only supports float16, bfloat16, and float32; got {dtype}."
    )


def nextafter(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN NEXTAFTER")
    kernel = _kernel_for(input.dtype)
    if out is not None:
        return kernel(input, other, out0=out)
    return kernel(input, other)


def nextafter_(input, other):
    logger.debug("GEMS_KUNLUNXIN NEXTAFTER_")
    return _kernel_for(input.dtype)(input, other, out0=input)
