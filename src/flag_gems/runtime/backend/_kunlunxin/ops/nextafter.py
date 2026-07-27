import logging

import torch
import triton
import triton.language as tl

from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

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
        moving_toward_zero
        & is_power_of_two
        & (abs_x > 1.1754943508222875e-38),
        spacing * 0.5,
        spacing,
    )
    stepped = x + tl.where(toward_up, spacing, -spacing)
    zero_result = tl.where(
        y > 0.0, 9.183549615799121e-41, -9.183549615799121e-41
    )
    infinite_result = tl.where(
        x > 0.0, 3.3895313892515355e38, -3.3895313892515355e38
    )
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
    is_nan = (x != x) | (y != y)
    is_equal = x == y
    is_zero = x == 0.0
    abs_x = tl.abs(x)
    is_infinite = abs_x > 3.4028234663852886e38
    toward_up = y > x
    moving_toward_zero = ((x > 0.0) & ~toward_up) | ((x < 0.0) & toward_up)

    x_bits = x.to(tl.int32, bitcast=True)
    abs_bits = x_bits & 0x7FFFFFFF
    exponent = ((abs_bits >> 23) & 0xFF) - 127
    spacing_exponent = exponent - 23
    normal_spacing_bits = (spacing_exponent + 127) << 23
    subnormal_spacing_bits = 1 << tl.maximum(spacing_exponent + 149, 0)
    spacing_bits = tl.where(
        spacing_exponent >= -126, normal_spacing_bits, subnormal_spacing_bits
    )
    spacing = spacing_bits.to(tl.float32, bitcast=True)
    minimum = (x_bits * 0 + 1).to(tl.float32, bitcast=True)
    is_power_of_two = (abs_bits & 0x7FFFFF) == 0
    spacing = tl.where(abs_x < 1.1754943508222875e-38, minimum, spacing)
    spacing = tl.where(
        moving_toward_zero
        & is_power_of_two
        & (abs_x > 1.1754943508222875e-38),
        spacing * 0.5,
        spacing,
    )
    stepped = x + tl.where(toward_up, spacing, -spacing)
    zero_result = tl.where(y > 0.0, minimum, -minimum)
    infinite_result = tl.where(
        x > 0.0, 3.4028234663852886e38, -3.4028234663852886e38
    )
    result = tl.where(is_zero, zero_result, stepped)
    result = tl.where(is_infinite & ~is_equal, infinite_result, result)
    result = tl.where(is_equal, y, result)
    return tl.where(is_nan, x + y, result)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def _nextafter_native_kernel(input, other):
    return tl_extra_shim.nextafter(input, other)


def _kernel_for(dtype):
    if dtype == torch.float16:
        return _nextafter_fp16_kernel
    if dtype == torch.bfloat16:
        return _nextafter_bf16_kernel
    if dtype == torch.float32:
        return _nextafter_fp32_kernel
    return _nextafter_native_kernel


def nextafter(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN NEXTAFTER")
    kernel = _kernel_for(input.dtype)
    if out is not None:
        result = kernel(input, other)
        out.copy_(result)
        return out
    return kernel(input, other)


def nextafter_(input, other):
    logger.debug("GEMS_KUNLUNXIN NEXTAFTER_")
    return _kernel_for(input.dtype)(input, other, out0=input)
