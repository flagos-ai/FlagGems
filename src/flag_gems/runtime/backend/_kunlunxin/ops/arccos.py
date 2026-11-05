import logging

from .acos import acos, acos_

logger = logging.getLogger(__name__)


def arccos(A):
    logger.debug("GEMS_KUNLUNXIN ARCCOS")
    return acos(A)


def arccos_(A):
    logger.debug("GEMS_KUNLUNXIN ARCCOS_")
    return acos_(A)
