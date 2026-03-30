import logging

from .scatter import scatter_

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


def scatter_add_(x, dim, index, src):
    logger.debug("GEMS_ASCEND SCATTER_ADD_")
    return scatter_(x, dim, index, src, reduce="add")
