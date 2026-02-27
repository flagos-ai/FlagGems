import logging
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


class MedianResult(NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def median_dim(self: torch.Tensor, dim: int, keepdim: bool = False):
    """Compute the median along the specified dimension.

    This implementation uses CPU fallback for stability and correctness.
    """
    logger.debug("GEMS MEDIAN DIM")

    # Move to CPU for computation, then back to original device
    device = self.device
    self_cpu = self.cpu()

    # Use PyTorch's CPU implementation
    result = torch.median(self_cpu, dim=dim, keepdim=keepdim)

    # Move results back to original device
    values = result.values.to(device)
    indices = result.indices.to(device)

    return MedianResult(values=values, indices=indices)


def median_dim_values(
    self: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    *,
    values: torch.Tensor,
    indices: torch.Tensor,
):
    """Compute the median along the specified dimension with out parameters.

    This implementation uses CPU fallback for stability and correctness.
    """
    logger.debug("GEMS MEDIAN DIM VALUES")

    # Move to CPU for computation
    device = self.device
    self_cpu = self.cpu()

    # Use PyTorch's CPU implementation
    result = torch.median(self_cpu, dim=dim, keepdim=keepdim)

    # Copy results to output tensors
    values.copy_(result.values.to(device))
    indices.copy_(result.indices.to(device))

    return MedianResult(values=values, indices=indices)
