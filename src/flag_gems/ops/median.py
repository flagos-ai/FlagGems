import logging
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


class MedianResult(NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def median(input: torch.Tensor, dim: int, keepdim: bool = False) -> MedianResult:
    """Compute the median along a dimension, returning (values, indices).

    Uses PyTorch's sort-based approach via Triton-friendly operations.
    For each slice along `dim`, the median is the element at position n//2
    in the sorted order (lower median for even-length slices, matching PyTorch).

    Args:
        input: Input tensor.
        dim: Dimension to reduce.
        keepdim: Whether to keep the reduced dimension.

    Returns:
        Named tuple (values, indices).
    """
    logger.debug("GEMS MEDIAN")
    ndim = input.ndim
    if ndim == 0:
        return MedianResult(input.clone(), torch.zeros((), dtype=torch.long, device=input.device))

    dim = dim % ndim

    # Sort along dim; median index = n // 2 (lower median)
    sorted_vals, sorted_idx = torch.sort(input, dim=dim)
    n = input.shape[dim]
    mid = n // 2

    # Gather median slice
    idx_tuple = [slice(None)] * ndim
    idx_tuple[dim] = mid
    idx_tuple = tuple(idx_tuple)

    values = sorted_vals[idx_tuple]
    indices = sorted_idx[idx_tuple]

    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    return MedianResult(values, indices)
