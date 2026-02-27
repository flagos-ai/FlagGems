import logging

import torch

from flag_gems.ops.sort import sort_stable

logger = logging.getLogger(__name__)


def argsort(inp, dim=-1, descending=False):
    """Returns the indices that sort a tensor along a given dimension.

    This is equivalent to calling torch.sort and returning only the indices.
    """
    logger.debug("GEMS ARGSORT")
    _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
    return indices
