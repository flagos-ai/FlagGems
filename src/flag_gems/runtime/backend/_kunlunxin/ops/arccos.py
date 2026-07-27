import logging

from .acos import acos_kernel

logger = logging.getLogger(__name__)


def arccos(A):
    logger.debug("GEMS_KUNLUNXIN ARCCOS")
    return acos_kernel(A)


def arccos_(A):
    logger.debug("GEMS_KUNLUNXIN ARCCOS_")
    acos_kernel(A, out0=A)
    return A
