import logging

import torch  # noqa: F401

logger = logging.getLogger(__name__)


def conj_physical(A):
    logger.debug("GEMS_ENFLAME CONJ_PHYSICAL")
    return A.clone()
