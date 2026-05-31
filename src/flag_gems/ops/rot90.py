import logging

import torch

import flag_gems

logger = logging.getLogger(__name__)


def rot90(input: torch.Tensor, k: int = 1, dims = None):
    logger.debug("GEMS ROT90")

    if dims is None:
        dims = (0, 1)
    else:
        if not isinstance(dims, (list, tuple)) or len(dims) != 2:
            raise ValueError("dims must be a tuple of two dimensions")

    if len(dims) != 2:
        raise ValueError("dims must be a tuple of two dimensions")

    dim0, dim1 = dims

    # Handle negative dimension indices
    if dim0 < 0:
        dim0 = dim0 + input.dim()
    if dim1 < 0:
        dim1 = dim1 + input.dim()

    # Validate dimensions
    if dim0 < 0 or dim0 >= input.dim():
        raise ValueError(f"dim0 {dims[0]} out of range for tensor of dimension {input.dim()}")
    if dim1 < 0 or dim1 >= input.dim():
        raise ValueError(f"dim1 {dims[1]} out of range for tensor of dimension {input.dim()}")
    if dim0 == dim1:
        raise ValueError(f"dims must be different, got {dims}")

    # Normalize k to be in range [0, 3]
    k = k % 4

    if k == 0:
        return input.clone()

    # Perform rotation using flip and transpose
    x = input
    for _ in range(k):
        # Rotate 90 degrees counter-clockwise:
        # 1. Transpose the two dimensions
        # 2. Flip the first dimension
        x = x.transpose(dim0, dim1)
        x = torch.flip(x, [dim0])

    return x
