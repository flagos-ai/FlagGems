import logging

from .sort import sort_stable

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def argsort(inp, dim=-1, descending=False):
    """Returns the indices that sort a tensor along a given dimension."""
    logger.debug("GEMS_CAMBRICON ARGSORT")
    _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
    return indices
