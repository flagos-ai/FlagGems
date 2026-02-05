"""
Cosh operator implementation for FlagGems.

This module provides the cosh operation using native Triton kernels.
Implements: cosh(x) = (e^x + e^(-x)) / 2
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_func(x):
    # Preserve float64 precision, upgrade float16/bfloat16 to float32
    if x.dtype == tl.float64:
        x_calc = x  # Keep float64 precision
    elif x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        x_calc = x.to(tl.float32)
    else:
        x_calc = x

    # cosh(x) = (e^x + e^(-x)) / 2
    exp_x = tl.exp(x_calc)
    exp_neg_x = tl.exp(-x_calc)
    result = 0.5 * (exp_x + exp_neg_x)

    return result.to(x.dtype)


def cosh(A):
    """
    Compute the hyperbolic cosine element-wise.

    This function provides the cosh operation for FlagGems, optimized
    for GPU execution using native Triton kernels.

    The formula used is: cosh(x) = (e^x + e^(-x)) / 2

    Args:
        A (Tensor): Input tensor

    Returns:
        Tensor: Tensor containing cosh(A) values

    Examples:
        >>> import torch
        >>> from flag_gems import cosh
        >>> x = torch.tensor([0.0, 1.0, 2.0], device='cuda')
        >>> y = cosh(x)
        >>> print(y)
        tensor([1.0000, 1.5431, 3.7622], device='cuda')
    """
    logger.debug("GEMS COSH")
    return cosh_func(A)


def cosh_(A):
    """
    In-place hyperbolic cosine operation.

    Args:
        A (Tensor): Input tensor (modified in-place)

    Returns:
        Tensor: The same tensor A containing cosh(A) values

    Examples:
        >>> import torch
        >>> from flag_gems import cosh_
        >>> x = torch.tensor([0.0, 1.0, 2.0], device='cuda')
        >>> cosh_(x)
        tensor([1.0000, 1.5431, 3.7622], device='cuda')
        >>> print(x)
        tensor([1.0000, 1.5431, 3.7622], device='cuda')
    """
    logger.debug("GEMS COSH_")
    cosh_func(A, out0=A)
    return A
