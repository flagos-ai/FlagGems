import logging

import torch

from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

from ..ops.copy import copy

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def select_scatter(inp, src, dim, index):
    logger.debug("GEMS SELECT_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index >= -inp.size(dim) and index < inp.size(dim), "Invalid index"
    dim = dim % inp.ndim
    index = index % inp.size(dim)

    valid_shape = list(inp.shape)
    del valid_shape[dim]
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    if has_internal_overlapping(inp) == MemOverlap.Yes:
        out = torch.empty(inp.size(), dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty_strided(
            inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
        )

    copy(inp, out0=out)
    indices = [slice(None)] * inp.ndim
    indices[dim] = index
    copy(src, out0=out[indices])

    return out
