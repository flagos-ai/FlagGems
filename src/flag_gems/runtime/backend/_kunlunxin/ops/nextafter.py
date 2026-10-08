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
    # kunlunxin override of the generic bit-manipulation nextafter:
    #  * the generic fp16/bf16 branch assumes uint16 arithmetic stays 16-bit,
    #    but this triton build promotes it to int32, so the final uint16
    #    bitcast raises "Cannot bitcast data-type of size 32 to 16";
    #  * the generic fp32 branch calls libdevice nextafter, which the XPU
    #    elfconv toolchain fails to convert (xpu3-elfconv exit 1);
    #  * ORing an integer-derived boolean with a float-compare boolean makes
    #    the XPU triton compiler exit(1) silently after SYMBOL_REWRITE;
    #  * a bf16<->uint16 bitcast inside the kernel trips an LLVM assertion
    #    ("Invalid cast!", Instructions.cpp:3720) - bf16 is therefore routed
    #    through a fp16 view (same width, same bit patterns) by the wrapper,
    #    with the bf16 exponent/fraction masks passed as constexpr.
    # The kernel itself is fully integer: ordering is decided with the
    # sign-magnitude key transform, NaN/equal/zero-cross tests are integer
    # bit tests, and the result is assembled with nested tl.where.
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

        # sign-magnitude -> monotone integer key (keys stay in [0, 0xFFFF]):
        #   negative u: key = 0xFFFF - u;  positive u: key = u + 0x8000
        kx = tl.where(x_int & sign_mask != 0, 0xFFFF - x_int, x_int + 0x8000)
        ky = tl.where(y_int & sign_mask != 0, 0xFFFF - y_int, y_int + 0x8000)
        is_going_up = (kx < ky).to(tl.int32)

        # direction: for positive floats a larger bit pattern is a larger
        # value; for negative floats it is the other way round.
        inc = (is_going_up * 2 - 1) * (is_positive * 2 - 1)

        # +0 going down wraps to 0x8001 (largest negative subnormal);
        # -0 going up wraps to 0x0001 (smallest positive subnormal).
        # Detected with int arithmetic instead of boolean algebra (the two
        # cases are mutually exclusive, so the sum is 0/1).
        zc = (
            is_positive * (1 - is_going_up) * x_is_zero
            + (1 - is_positive) * is_going_up * x_is_zm
        )

        # NaN propagates (x-NaN keeps x, y-NaN yields y's NaN bits);
        # equal operands return input unchanged.
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
        # float64: same algorithm on the int64 bit pattern.
        exp_mask = 9218868437227405312  # 0x7FF0000000000000
        frac_mask = 4503599627370495  # 0x000FFFFFFFFFFFFF
        sign_bit = -9223372036854775808  # 0x8000000000000000 (INT64_MIN)
        cross_const = -9223372036854775807  # 0x8000000000000001
        zero_minus = -9223372036854775808  # 0x8000000000000000

        x_int = input.to(tl.int64, bitcast=True)
        y_int = other.to(tl.int64, bitcast=True)

        xnan = ((x_int & exp_mask) == exp_mask) & ((x_int & frac_mask) != 0)
        ynan = ((y_int & exp_mask) == exp_mask) & ((y_int & frac_mask) != 0)

        is_equal = x_int == y_int
        is_positive = ((x_int & sign_bit) == 0).to(tl.int32)
        x_is_zero = (x_int == 0).to(tl.int32)
        x_is_zm = (x_int == zero_minus).to(tl.int32)

        # sign-magnitude -> monotone signed int64 key
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
        # float32: same algorithm on the int32 bit pattern (two's complement).
        exp_mask = 2139095040  # 0x7F800000
        frac_mask = 8388607  # 0x007FFFFF
        sign_bit = -2147483648  # 0x80000000 (INT_MIN)
        cross_const = -2147483647  # 0x80000001
        zero_minus = -2147483648  # 0x80000000

        x_int = input.to(tl.int32, bitcast=True)
        y_int = other.to(tl.int32, bitcast=True)

        xnan = ((x_int & exp_mask) == exp_mask) & ((x_int & frac_mask) != 0)
        ynan = ((y_int & exp_mask) == exp_mask) & ((y_int & frac_mask) != 0)

        is_equal = x_int == y_int
        is_positive = ((x_int & sign_bit) == 0).to(tl.int32)
        x_is_zero = (x_int == 0).to(tl.int32)
        x_is_zm = (x_int == zero_minus).to(tl.int32)

        # sign-magnitude -> monotone *signed* key:
        #   negative (x_int < 0): key = ~x_int ^ INT_MIN;  positive: key = x_int
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
    # Same algorithm as nextafter_func but with the bfloat16 NaN masks; the
    # wrapper feeds bf16 tensors through a fp16 view (identical bit layout)
    # because a kernel-side bf16 bitcast asserts inside the XPU LLVM pass.
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
        # kernel-side bf16 bitcast is broken (LLVM "Invalid cast!" assert);
        # run the same-width fp16 view through the bf16-mask kernel instead.
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
