import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    promotion_methods=[
        (0, "DEFAULT"),
        # The actual dtype of the second output is determined by the
        # preallocated int32 output tensor passed through out1.
        (0, "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def _frexp_func(x):
    # Decompose each input value into a mantissa and an integral exponent.
    #
    # For finite nonzero values:
    #
    #     x = mantissa * 2 ** exponent
    #
    # where the absolute value of mantissa is in the range [0.5, 1.0).
    #
    # Special cases:
    # - frexp(+0.0) returns (+0.0, 0)
    # - frexp(-0.0) returns (-0.0, 0)
    # - frexp(+inf) returns (+inf, 0)
    # - frexp(-inf) returns (-inf, 0)
    # - frexp(nan) returns (nan, 0)

    # Perform the computation in float32 for float16, bfloat16, and
    # float32 inputs.
    x_fp32 = x.to(tl.float32)
    abs_x = tl.abs(x_fp32)

    # Detect IEEE-754 special values.
    # NaN is the only floating-point value that is not equal to itself.
    is_nan = x_fp32 != x_fp32
    is_inf = abs_x == float("inf")
    is_zero = x_fp32 == 0.0
    is_special = is_nan | is_inf | is_zero

    # Avoid evaluating log2 on zero, infinity, or NaN.
    # The selected value is irrelevant for special inputs because their
    # outputs are overwritten below.
    safe_abs_x = tl.where(is_special, 1.0, abs_x)

    # For a finite nonzero value:
    #
    # exponent = floor(log2(abs(x))) + 1
    # mantissa = x / 2 ** exponent
    #
    # This places the mantissa in [0.5, 1.0) for positive inputs and
    # (-1.0, -0.5] for negative inputs.
    exponent_fp32 = tl.floor(tl.log2(safe_abs_x)) + 1.0
    mantissa = x_fp32 / tl.exp2(exponent_fp32)

    # Preserve the original value for signed zero, infinity, and NaN.
    # In particular, this preserves the sign bit of negative zero.
    mantissa = tl.where(is_special, x_fp32, mantissa)

    # Special values must have an exponent of zero.
    # Replace special exponents before converting to int32 to avoid
    # converting NaN or infinity to an integer.
    exponent_fp32 = tl.where(is_special, 0.0, exponent_fp32)
    exponent = exponent_fp32.to(tl.int32)

    # The mantissa uses the same dtype as the input, while the exponent
    # is always represented as int32.
    return mantissa.to(x.dtype), exponent


def frexp(A):
    logger.debug("GEMS FREXP")

    if not A.is_floating_point():
        raise RuntimeError(
            f"frexp(): expected a floating-point tensor, but got {A.dtype}"
        )

    # This implementation converts values to float32 internally.
    # Therefore, it cannot correctly preserve the precision and range
    # of float64 inputs.
    if A.dtype == torch.float64:
        raise RuntimeError("FlagGems frexp currently does not support float64")

    # Preallocate the outputs explicitly because frexp returns tensors
    # with different dtypes:
    # - mantissa: same dtype as the input
    # - exponent: int32
    mantissa = torch.empty_like(A)
    exponent = torch.empty_like(A, dtype=torch.int32)

    return _frexp_func(
        A,
        out0=mantissa,
        out1=exponent,
    )
