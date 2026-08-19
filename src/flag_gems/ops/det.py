import logging

from flag_gems.ops.linalg_det import _linalg_det_impl

logger = logging.getLogger(__name__)


def det(A):
    logger.debug("GEMS DET")
    return _linalg_det_impl(A)
