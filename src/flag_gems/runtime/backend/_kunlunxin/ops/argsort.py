import logging

from .sort import sort_stable

logger = logging.getLogger(__name__)


def argsort(inp, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN ARGSORT")
    _, indices = sort_stable(inp, stable=True, dim=dim, descending=descending)
    return indices


def argsort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN ARGSORT_STABLE")
    _, indices = sort_stable(inp, stable=stable, dim=dim, descending=descending)
    return indices
