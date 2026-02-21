"""
LeakyReLU operator implementation for FlagGems.

This module provides the LeakyReLU activation function using native Triton kernels.
LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def leaky_relu_func(x, negative_slope):
    y = tl.where(x >= 0, x, x * negative_slope)
    return y


def leaky_relu(A, negative_slope=0.01):
    """
    Apply LeakyReLU activation function element-wise.

    Args:
        A (Tensor): Input tensor
        negative_slope (float): Slope for negative values (default: 0.01)

    Returns:
        Tensor: Output tensor with LeakyReLU applied
    """
    logger.debug("GEMS LEAKY_RELU")
    return leaky_relu_func(A, negative_slope)


def leaky_relu_(A, negative_slope=0.01):
    """
    In-place LeakyReLU activation function.

    Args:
        A (Tensor): Input tensor (modified in-place)
        negative_slope (float): Slope for negative values (default: 0.01)

    Returns:
        Tensor: The same tensor A containing LeakyReLU applied
    """
    logger.debug("GEMS LEAKY_RELU_")
    leaky_relu_func(A, negative_slope, out0=A)
    return A
