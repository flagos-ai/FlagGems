"""
Log10 operator implementation for FlagGems.

This module provides the log10 operation using native Triton kernels.
Implements log10(x) = log2(x) * log10(2) where log10(2) = 0.30102999566398114
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log10_func(x):
    # Determine computation dtype based on input dtype
    # Match PyTorch behavior: integers -> float64, float16/bfloat16 -> float32, preserve others
    if x.dtype == tl.int32 or x.dtype == tl.int64:
        x_calc = x.to(tl.float64)
        result_dtype = tl.float64
    elif x.dtype == tl.float16 or x.dtype == tl.bfloat16:
        x_calc = x.to(tl.float32)
        result_dtype = x.dtype
    elif x.dtype == tl.float64:
        x_calc = x
        result_dtype = tl.float64
    else:
        x_calc = x
        result_dtype = x.dtype

    # Compute log10 using log2 for better performance
    # log10(x) = log2(x) * log10(2)
    y = tl.math.log2(x_calc) * 0.30102999566398114

    return y.to(result_dtype)


def log10(A):
    """
    Compute log10 of input tensor element-wise.

    Args:
        A (Tensor): Input tensor

    Returns:
        Tensor: Tensor containing log10(A) values
    """
    logger.debug("GEMS LOG10")
    # Handle integer dtypes to match PyTorch behavior (int -> float32)
    if A.dtype in (torch.int32, torch.int64):
        A_float = A.to(torch.float32)
        return log10_func(A_float)
    return log10_func(A)


def log10_(A):
    """
    In-place log10 operation.

    Args:
        A (Tensor): Input tensor (modified in-place)

    Returns:
        Tensor: The same tensor A containing log10(A) values
    """
    logger.debug("GEMS LOG10_")
    if A.dtype in (torch.int32, torch.int64):
        A_float = A.to(torch.float32)
        return log10_func(A_float, out0=A_float)
    return log10_func(A, out0=A)
