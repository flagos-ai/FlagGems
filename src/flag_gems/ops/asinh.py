"""
Asinh operator implementation for FlagGems.

This module provides the inverse hyperbolic sine (asinh) operation using native Triton kernels.
asinh(x) = ln(x + sqrt(x^2 + 1))
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def asinh_func(x):
    # Convert to float32 for numerical stability
    if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        x_fp32 = x.to(tl.float32)
    else:
        x_fp32 = x

    abs_x = tl.abs(x_fp32)

    # For large |x| (|x| > 1e4): Use simplified formula
    # asinh(x) ≈ sign(x) * ln(2*|x|)
    is_very_large = abs_x > 1e4
    log_very_large = tl.log(2.0 * abs_x)

    # For normal values: asinh(x) = ln(|x| + sqrt(x^2 + 1))
    x_squared = abs_x * abs_x
    sqrt_val = tl.sqrt(x_squared + 1.0)
    log_normal = tl.log(abs_x + sqrt_val)

    log_val = tl.where(is_very_large, log_very_large, log_normal)
    sign_x = tl.where(x_fp32 >= 0, 1.0, -1.0)
    y = sign_x * log_val

    return y.to(x.dtype)


def asinh(A):
    """
    Compute the inverse hyperbolic sine element-wise.

    Args:
        A (Tensor): Input tensor

    Returns:
        Tensor: Tensor containing asinh(A) values
    """
    logger.debug("GEMS ASINH")
    return asinh_func(A)


def asinh_(A):
    """
    In-place inverse hyperbolic sine operation.

    Args:
        A (Tensor): Input tensor (modified in-place)

    Returns:
        Tensor: The same tensor A containing asinh(A) values
    """
    logger.debug("GEMS ASINH_")
    asinh_func(A, out0=A)
    return A
