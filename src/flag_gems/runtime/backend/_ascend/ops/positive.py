import logging

from flag_gems.runtime import device

device_ = device
logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


def positive(x):
    logger.debug("GEMS_ASCEND POSITIVE")
    return x
