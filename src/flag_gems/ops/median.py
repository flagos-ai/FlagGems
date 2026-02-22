import logging

import torch

from flag_gems.ops.sort import sort_stable

logger = logging.getLogger(__name__)


def median(inp, dim=-1, keepdim=False):
    """torch.median(input, dim, keepdim) -> (values, indices)

    Returns the median value along *dim* and the index of that element in
    the original (unsorted) tensor.

    For a dimension of length N the median sits at sorted position
    ``k = (N - 1) // 2`` (lower-median convention, matching PyTorch).

    Delegates sorting to the existing FlagGems radix-sort back-end so no
    new Triton kernel is needed.
    """
    logger.debug("GEMS MEDIAN DIM")

    if dim < 0:
        dim = dim + inp.ndim

    N = inp.shape[dim]
    k = (N - 1) // 2  # lower-median index in sorted order

    sorted_vals, sorted_indices = sort_stable(
        inp, stable=True, dim=dim, descending=False
    )

    # select() removes the reduced dimension
    values = sorted_vals.select(dim, k)
    indices = sorted_indices.select(dim, k)

    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    return values, indices


def median_scalar(inp):
    logger.debug("GEMS MEDIAN SCALAR")
    flat = inp.flatten()
    k = (flat.numel() - 1) // 2
    sorted_flat, _ = sort_stable(flat, stable=True, dim=0, descending=False)
    return sorted_flat[k]


def median_dim_values(inp, dim, keepdim=False, *, values, indices):
    logger.debug("GEMS MEDIAN DIM VALUES")
    vals, inds = median(inp, dim, keepdim)
    values.copy_(vals)
    indices.copy_(inds)
    return values, indices


def median_out(inp, *, out):
    logger.debug("GEMS MEDIAN OUT")
    result = median_scalar(inp)
    out.copy_(result)
    return out
