import logging
from typing import List

import torch

from flag_gems.ops.scatter import scatter_

logger = logging.getLogger(__name__)


def value_selecting_reduction_backward(
    grad: torch.Tensor,
    dim: int,
    indices: torch.Tensor,
    sizes: List[int],
    keepdim: bool,
) -> torch.Tensor:
    """
    Backward pass for value-selecting reduction operations (max.dim, min.dim, etc.).

    This operation scatters the gradient values back to the positions indicated
    by the indices from the forward pass.

    Args:
        grad: Gradient with respect to the reduced values
        dim: The dimension that was reduced in the forward pass
        indices: Indices of the selected values (e.g., argmax/argmin results)
        sizes: Original input tensor shape
        keepdim: Whether keepdim was used in the forward pass

    Returns:
        Gradient with respect to the original input tensor
    """
    logger.debug("GEMS VALUE_SELECTING_REDUCTION_BACKWARD")

    # Normalize dim to positive
    ndim = len(sizes)
    if dim < 0:
        dim = dim + ndim

    # Create output tensor filled with zeros
    result = grad.new_zeros(sizes)

    # If keepdim was False, we need to unsqueeze grad and indices to match
    # the expected shape for scatter operation
    if not keepdim:
        grad = grad.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    # Scatter the gradients to the positions indicated by indices
    # For value-selecting reductions (max/min), each index appears at most once
    # per reduction group, so we don't need reduce="add"
    scatter_(result, dim, indices, grad)

    return result
