import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

BF16_EXP = 0x7F80
BF16_FRAC = 0x007F


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def nextafter_func(input, other):
    dtype = input.dtype
    if tl.constexpr(dtype == tl.float16):
        sign_mask = 0x8000
        cross_const = 0x8001
        zero_minus = 0x8000

        x_int = input.to(tl.uint16, bitcast=True).to(tl.int32)
        y_int = other.to(tl.uint16, bitcast=True).to(tl.int32)

        xnan = ((x_int & 0x7C00) == 0x7C00) & ((x_int & 0x03FF) != 0)
        ynan = ((y_int & 0x7C00) == 0x7C00) & ((y_int & 0x03FF) != 0)

        is_equal = x_int == y_int
        is_positive = ((x_int & sign_mask) == 0).to(tl.int32)
        x_is_zero = (x_int == 0).to(tl.int32)
        x_is_zm = (x_int == zero_minus).to(tl.int32)

        kx = tl.where(x_int & sign_mask != 0, 0xFFFF - x_int, x_int + 0x8000)
        ky = tl.where(y_int & sign_mask != 0, 0xFFFF - y_int, y_int + 0x8000)
        is_going_up = (kx < ky).to(tl.int32)

        inc = (is_going_up * 2 - 1) * (is_positive * 2 - 1)

        zc = (
            is_positive * (1 - is_going_up) * x_is_zero
            + (1 - is_positive) * is_going_up * x_is_zm
        )

        r = tl.where(
            xnan,
            x_int,
            tl.where(
                ynan,
                y_int,
                tl.where(
                    is_equal,
                    x_int,
                    tl.where(zc == 1, x_int + cross_const, x_int + inc),
                ),
            ),
        )
        r16 = r.to(tl.uint16)
        return r16.to(input.dtype, bitcast=True)
    elif tl.constexpr(dtype == tl.float64):
        exp_mask = 9218868437227405312
        frac_mask = 4503599627370495
        sign_bit = -9223372036854775808
        cross_const = -9223372036854775807
        zero_minus = -9223372036854775808

        x_int = input.to(tl.int64, bitcast=True)
        y_int = other.to(tl.int64, bitcast=True)

        xnan = ((x_int & exp_mask) == exp_mask) & ((x_int & frac_mask) != 0)
        ynan = ((y_int & exp_mask) == exp_mask) & ((y_int & frac_mask) != 0)

        is_equal = x_int == y_int
        is_positive = ((x_int & sign_bit) == 0).to(tl.int32)
        x_is_zero = (x_int == 0).to(tl.int32)
        x_is_zm = (x_int == zero_minus).to(tl.int32)

        kx = tl.where(x_int < 0, (~x_int) ^ sign_bit, x_int)
        ky = tl.where(y_int < 0, (~y_int) ^ sign_bit, y_int)
        is_going_up = (kx < ky).to(tl.int32)

        inc = (is_going_up * 2 - 1) * (is_positive * 2 - 1)

        zc = (
            is_positive * (1 - is_going_up) * x_is_zero
            + (1 - is_positive) * is_going_up * x_is_zm
        )

        r = tl.where(
            xnan,
            x_int,
            tl.where(
                ynan,
                y_int,
                tl.where(
                    is_equal,
                    x_int,
                    tl.where(zc == 1, x_int + cross_const, x_int + inc),
                ),
            ),
        )
        return r.to(input.dtype, bitcast=True)
    else:
        exp_mask = 2139095040
        frac_mask = 8388607
        sign_bit = -2147483648
        cross_const = -2147483647
        zero_minus = -2147483648

        x_int = input.to(tl.int32, bitcast=True)
        y_int = other.to(tl.int32, bitcast=True)

        xnan = ((x_int & exp_mask) == exp_mask) & ((x_int & frac_mask) != 0)
        ynan = ((y_int & exp_mask) == exp_mask) & ((y_int & frac_mask) != 0)

        is_equal = x_int == y_int
        is_positive = ((x_int & sign_bit) == 0).to(tl.int32)
        x_is_zero = (x_int == 0).to(tl.int32)
        x_is_zm = (x_int == zero_minus).to(tl.int32)

        kx = tl.where(x_int < 0, (~x_int) ^ sign_bit, x_int)
        ky = tl.where(y_int < 0, (~y_int) ^ sign_bit, y_int)
        is_going_up = (kx < ky).to(tl.int32)

        inc = (is_going_up * 2 - 1) * (is_positive * 2 - 1)

        zc = (
            is_positive * (1 - is_going_up) * x_is_zero
            + (1 - is_positive) * is_going_up * x_is_zm
        )

        r = tl.where(
            xnan,
            x_int,
            tl.where(
                ynan,
                y_int,
                tl.where(
                    is_equal,
                    x_int,
                    tl.where(zc == 1, x_int + cross_const, x_int + inc),
                ),
            ),
        )
        return r.to(input.dtype, bitcast=True)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def nextafter_func_bf16(input, other):
    dtype = input.dtype
    if tl.constexpr(dtype == tl.float16):
        sign_mask = 0x8000
        cross_const = 0x8001
        zero_minus = 0x8000

        x_int = input.to(tl.uint16, bitcast=True).to(tl.int32)
        y_int = other.to(tl.uint16, bitcast=True).to(tl.int32)

        xnan = ((x_int & 0x7F80) == 0x7F80) & ((x_int & 0x007F) != 0)
        ynan = ((y_int & 0x7F80) == 0x7F80) & ((y_int & 0x007F) != 0)

        is_equal = x_int == y_int
        is_positive = ((x_int & sign_mask) == 0).to(tl.int32)
        x_is_zero = (x_int == 0).to(tl.int32)
        x_is_zm = (x_int == zero_minus).to(tl.int32)

        kx = tl.where(x_int & sign_mask != 0, 0xFFFF - x_int, x_int + 0x8000)
        ky = tl.where(y_int & sign_mask != 0, 0xFFFF - y_int, y_int + 0x8000)
        is_going_up = (kx < ky).to(tl.int32)

        inc = (is_going_up * 2 - 1) * (is_positive * 2 - 1)

        zc = (
            is_positive * (1 - is_going_up) * x_is_zero
            + (1 - is_positive) * is_going_up * x_is_zm
        )

        r = tl.where(
            xnan,
            x_int,
            tl.where(
                ynan,
                y_int,
                tl.where(
                    is_equal,
                    x_int,
                    tl.where(zc == 1, x_int + cross_const, x_int + inc),
                ),
            ),
        )
        r16 = r.to(tl.uint16)
        return r16.to(input.dtype, bitcast=True)
    else:
        return input


def nextafter(input, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN NEXTAFTER")
    if input.dtype == torch.bfloat16:
        oth = other.view(torch.float16)
        if out is not None:
            nextafter_func_bf16(
                input.view(torch.float16), oth, out0=out.view(torch.float16)
            )
            return out
        return nextafter_func_bf16(input.view(torch.float16), oth).view(torch.bfloat16)
    if out is not None:
        return nextafter_func(input, other, out0=out)
    return nextafter_func(input, other)


def nextafter_(input, other):
    logger.debug("GEMS_KUNLUNXIN NEXTAFTER_")
    if input.dtype == torch.bfloat16:
        nextafter_func_bf16(
            input.view(torch.float16),
            other.view(torch.float16),
            out0=input.view(torch.float16),
        )
        return input
    return nextafter_func(input, other, out0=input)
