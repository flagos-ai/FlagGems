import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.rms_norm import rms_norm as rms_norm_impl
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def RMSNorm(x, normalized_shape, weight=None, eps=1e-5):
    """
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This is a wrapper around rms_norm with a simpler interface for common cases.

    Args:
        x: Input tensor of any shape
        normalized_shape: The normalized shape (last dimensions to normalize over)
        weight: Optional weight tensor. If None, uses ones.
        eps: A value added to the denominator for numerical stability. Default: 1e-5

    Returns:
        Normalized tensor of the same shape as input
    """
    logger.debug("GEMS RMSNORM")

    # Handle the case where normalized_shape is an int
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # Create weight if not provided (ones)
    if weight is None:
        weight = torch.ones(normalized_shape, dtype=x.dtype, device=x.device)

    # Call the existing rms_norm implementation
    return rms_norm_impl(x, normalized_shape, weight, eps)


def RMSNorm_(x, normalized_shape, eps=1e-5):
    """
    In-place version of RMSNorm.

    Args:
        x: Input tensor of any shape (modified in place)
        normalized_shape: The normalized shape (last dimensions to normalize over)
        eps: A value added to the denominator for numerical stability. Default: 1e-5

    Returns:
        The input tensor (modified in place)
    """
    logger.debug("GEMS RMSNORM_")

    # Handle the case where normalized_shape is an int
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # Create weight (ones)
    weight = torch.ones(normalized_shape, dtype=x.dtype, device=x.device)

    # Call the existing rms_norm implementation and copy result back
    result = rms_norm_impl(x, normalized_shape, weight, eps)
    x.copy_(result)
    return x
