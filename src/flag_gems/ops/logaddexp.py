"""
LogAddExp operator implementation for FlagGems.

This module provides the logaddexp operation using native Triton kernels.
Implements numerically stable: log(exp(x) + exp(y)) = max(x,y) + log(1 + exp(-|x-y|))
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True], num_outputs=1, promotion_methods=[(0, 1, "INT_TO_FLOAT")]
)
@triton.jit
def logaddexp_func(x, y):
    # Convert to float32 for computation (better numerical stability)
    x_float = x.to(tl.float32)
    y_float = y.to(tl.float32)

    # Numerically stable implementation: log(exp(x) + exp(y))
    # = max(x, y) + log(1 + exp(-|x - y|))
    m = tl.maximum(x_float, y_float)
    abs_diff = tl.abs(x_float - y_float)
    result_float = m + tl.log(1 + tl.exp(-abs_diff))

    return result_float.to(x.dtype)


def logaddexp(x, y):
    """
    Compute log(add(exp(x), exp(y))) element-wise in a numerically stable way.

    Args:
        x (Tensor): First input tensor
        y (Tensor): Second input tensor

    Returns:
        Tensor: Tensor containing logaddexp(x, y) values

    Raises:
        NotImplementedError: If either input is an integer dtype (matches PyTorch behavior)
    """
    logger.debug("GEMS LOGADDEXP")

    # Check for integer dtypes (match PyTorch behavior)
    # PyTorch raises NotImplementedError for integer inputs
    if x.dtype in (torch.int32, torch.int64) or y.dtype in (torch.int32, torch.int64):
        raise NotImplementedError(
            "logaddexp() not implemented for integer dtypes. "
            "Use float tensors instead."
        )

    return logaddexp_func(x, y)
