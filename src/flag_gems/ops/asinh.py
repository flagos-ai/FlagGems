import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_kernel(x):
    # Compute asinh(x) = log(x + sqrt(x*x + 1))
    # This formula is numerically stable for all input values.
    # For better precision, we compute in float32 and then cast back.
    x_fp32 = x.to(tl.float32)

    # Handle special cases: inf, -inf, nan
    x_is_posinf = x_fp32 == float("inf")
    x_is_neginf = x_fp32 == -float("inf")
    x_is_nan = x_fp32 != x_fp32  # NaN check: nan != nan

    # Compute asinh using the formula: log(x + sqrt(x*x + 1))
    result = tl.log(x_fp32 + tl.sqrt(x_fp32 * x_fp32 + 1.0))

    # Fix special cases:
    # asinh(+inf) = +inf
    result = tl.where(x_is_posinf, float("inf"), result)
    # asinh(-inf) = -inf
    result = tl.where(x_is_neginf, float("-inf"), result)
    # asinh(nan) = nan (propagate NaN)
    result = tl.where(x_is_nan, float("nan"), result)

    return result


def asinh(input, *, out=None):
    """
    Returns a new tensor with the inverse hyperbolic sine of the elements of input.

    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor

    Returns:
        Tensor: a new tensor with the inverse hyperbolic sine of the elements of input

    Example:
        >>> a = torch.randn(4)
        >>> torch.asinh(a)
        tensor([ 0.1599, -1.1534, -0.9435, -0.8990 ])
    """
    logger.debug("GEMS ASINH FORWARD")
    if out is None:
        return asinh_kernel(input)
    else:
        asinh_kernel(input, out0=out)
        return out


def asinh_(input):
    """
    In-place version of asinh.

    Args:
        input (Tensor): the input tensor (will be modified in-place)

    Returns:
        Tensor: the input tensor (modified in-place)
    """
    logger.debug("GEMS ASINH_")
    asinh_kernel(input, out0=input)
    return input
