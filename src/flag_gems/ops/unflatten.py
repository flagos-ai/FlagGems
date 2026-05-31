import logging

import torch

import flag_gems

logger = logging.getLogger(__name__)


def unflatten(input: torch.Tensor, dim: int, sizes):
    logger.debug("GEMS UNFLATTEN")

    if not isinstance(sizes, (list, tuple)):
        raise ValueError("sizes must be a tuple or list")

    if dim < 0:
        dim = dim + input.dim()

    if dim < 0 or dim >= input.dim():
        raise ValueError(f"dim {dim} is out of range for tensor of dimension {input.dim()}")

    # Check if the unflatten size matches the original dimension size
    original_dim_size = input.shape[dim]
    if len(sizes) == 0:
        raise ValueError("sizes cannot be empty")

    # Handle -1 in sizes (infer from context)
    sizes = list(sizes)
    inferred = -1
    known_product = 1
    for i, size in enumerate(sizes):
        if size == -1:
            if inferred != -1:
                raise ValueError("only one dimension can be inferred")
            inferred = i
        elif size < 0:
            raise ValueError("invalid size dimension")
        else:
            known_product *= size

    if inferred != -1:
        # Infer the -1 dimension
        if known_product == 0:
            raise ValueError("cannot infer size when other dimensions are 0")
        if original_dim_size % known_product != 0:
            raise ValueError(
                f"cannot unflatten tensor of size {original_dim_size} into "
                f"sizes {sizes} because dimension {original_dim_size} is not "
                f"divisible by {known_product}"
            )
        sizes[inferred] = original_dim_size // known_product
    else:
        # Check if the total size matches
        total_size = 1
        for size in sizes:
            total_size *= size
        if total_size != original_dim_size:
            raise ValueError(
                f"cannot unflatten tensor of size {original_dim_size} into "
                f"sizes {sizes} because total size {total_size} != {original_dim_size}"
            )

    # Build output shape
    shape = list(input.shape)
    shape[dim:dim + 1] = sizes

    return input.view(shape)
